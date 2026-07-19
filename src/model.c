/* model.c — forward pass for LLaMA-style decoder transformers.
 *
 * The matmul, quantization, and threading live in kernels.c/threads.c; this
 * file is the architecture: embedding, RMSNorm, RoPE, GQA attention, SwiGLU.
 * Weights carry a dtype so the same code runs fp32, Q8_0, or Q4_0 models.
 */
#include "ember.h"
#include "kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const void *data; int dtype; } Wt; /* a weight + its encoding */

struct EmberState {
    const EmberModel *m;
    int dim, hidden, n_layers, n_heads, n_kv, head_dim, kv_dim, vocab, ctx;
    float rope_theta, norm_eps;
    int   rope_style, qk_norm;

    Wt tok_emb, lm_head;
    Wt *wq, *wk, *wv, *wo, *w1, *w2, *w3;
    const float *final_norm;
    const float **attn_norm, **ffn_norm, **q_norm, **k_norm;

    float *x, *xb, *xb2, *hb, *hb2, *q, *att, *logits;
    float *key_cache, *value_cache;
    BlockQ8_0 *xq; /* scratch: quantized activation for q8/q4 matmuls */
};

static void rmsnorm(float *out, const float *x, const float *weight, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];
}

static void softmax(float *x, int n) {
    float max = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

static void rope(float *vec, int n, int head_dim, int pos, float theta, int style) {
    for (int h = 0; h < n; h += head_dim) {
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / head_dim);
            float val = pos * freq;
            float fcr = cosf(val), fci = sinf(val);
            int a, b;
            if (style == EMBER_ROPE_NEOX) { a = h + i; b = h + i + head_dim / 2; }
            else                         { a = h + 2 * i; b = h + 2 * i + 1; }
            float v0 = vec[a], v1 = vec[b];
            vec[a] = v0 * fcr - v1 * fci;
            vec[b] = v0 * fci + v1 * fcr;
        }
    }
}

/* dequantize one Q8_0 row into fp32 (used for embedding lookup). */
static void dequant_q8_row(const BlockQ8_0 *row, int n, float *out) {
    int nb = n / EMBER_Q_BLOCK;
    for (int b = 0; b < nb; b++) {
        float d = ember_f16_to_f32(row[b].d);
        for (int i = 0; i < EMBER_Q_BLOCK; i++)
            out[b * EMBER_Q_BLOCK + i] = row[b].qs[i] * d;
    }
}

static void embed_row(const Wt *w, int tok, int dim, float *out) {
    if (w->dtype == EMBER_DT_F32) {
        memcpy(out, (const float *)w->data + (size_t)tok * dim, sizeof(float) * dim);
    } else { /* Q8_0 embedding table */
        int nb = dim / EMBER_Q_BLOCK;
        dequant_q8_row((const BlockQ8_0 *)w->data + (size_t)tok * nb, dim, out);
    }
}

static Wt tref(const EmberModel *m, const char *fmt, int l) {
    char name[48];
    if (l >= 0) snprintf(name, sizeof(name), fmt, l);
    else        snprintf(name, sizeof(name), "%s", fmt);
    const EmberTensor *t = ember_model_tensor(m, name);
    Wt w = { NULL, EMBER_DT_F32 };
    if (t) { w.data = ember_model_data(m, t); w.dtype = t->dtype; }
    return w;
}

static const float *tref_f32(const EmberModel *m, const char *fmt, int l) {
    return (const float *)tref(m, fmt, l).data;
}

EmberState *ember_state_new(const EmberModel *m, int ctx_len) {
    const EmberHeader *h = ember_model_header(m);
    EmberState *s = calloc(1, sizeof(*s));
    s->m = m;
    s->dim = h->dim; s->hidden = h->hidden_dim; s->n_layers = h->n_layers;
    s->n_heads = h->n_heads; s->n_kv = h->n_kv_heads; s->head_dim = h->head_dim;
    s->kv_dim = h->n_kv_heads * h->head_dim; s->vocab = h->vocab_size;
    s->rope_theta = h->rope_theta; s->norm_eps = h->norm_eps;
    s->rope_style = h->rope_style; s->qk_norm = (h->flags & EMBER_FLAG_QK_NORM) != 0;
    s->ctx = (ctx_len > 0 && ctx_len < h->max_seq_len) ? ctx_len : h->max_seq_len;

    int L = s->n_layers;
    s->wq = malloc(sizeof(Wt) * L); s->wk = malloc(sizeof(Wt) * L);
    s->wv = malloc(sizeof(Wt) * L); s->wo = malloc(sizeof(Wt) * L);
    s->w1 = malloc(sizeof(Wt) * L); s->w2 = malloc(sizeof(Wt) * L);
    s->w3 = malloc(sizeof(Wt) * L);
    s->attn_norm = malloc(sizeof(void *) * L); s->ffn_norm = malloc(sizeof(void *) * L);
    s->q_norm = malloc(sizeof(void *) * L); s->k_norm = malloc(sizeof(void *) * L);
    for (int l = 0; l < L; l++) {
        s->wq[l] = tref(m, "layers.%d.wq", l); s->wk[l] = tref(m, "layers.%d.wk", l);
        s->wv[l] = tref(m, "layers.%d.wv", l); s->wo[l] = tref(m, "layers.%d.wo", l);
        s->w1[l] = tref(m, "layers.%d.w1", l); s->w2[l] = tref(m, "layers.%d.w2", l);
        s->w3[l] = tref(m, "layers.%d.w3", l);
        s->attn_norm[l] = tref_f32(m, "layers.%d.attn_norm", l);
        s->ffn_norm[l]  = tref_f32(m, "layers.%d.ffn_norm", l);
        s->q_norm[l]    = tref_f32(m, "layers.%d.q_norm", l);
        s->k_norm[l]    = tref_f32(m, "layers.%d.k_norm", l);
    }
    s->tok_emb = tref(m, "tok_embeddings", -1);
    s->final_norm = tref_f32(m, "final_norm", -1);
    s->lm_head = (h->flags & EMBER_FLAG_TIED_EMBED) ? s->tok_emb : tref(m, "lm_head", -1);

    int hd = s->head_dim, nh = s->n_heads;
    int wide = s->hidden > s->dim ? s->hidden : s->dim;
    s->x = malloc(sizeof(float) * s->dim);
    s->xb = malloc(sizeof(float) * s->dim);
    s->xb2 = malloc(sizeof(float) * s->dim);
    s->hb = malloc(sizeof(float) * s->hidden);
    s->hb2 = malloc(sizeof(float) * s->hidden);
    s->q = malloc(sizeof(float) * nh * hd);
    s->att = malloc(sizeof(float) * nh * s->ctx);
    s->logits = malloc(sizeof(float) * s->vocab);
    s->key_cache = malloc(sizeof(float) * (size_t)L * s->ctx * s->kv_dim);
    s->value_cache = malloc(sizeof(float) * (size_t)L * s->ctx * s->kv_dim);
    s->xq = malloc(sizeof(BlockQ8_0) * (wide / EMBER_Q_BLOCK + 1));
    return s;
}

void ember_state_free(EmberState *s) {
    if (!s) return;
    free(s->wq); free(s->wk); free(s->wv); free(s->wo);
    free(s->w1); free(s->w2); free(s->w3);
    free(s->attn_norm); free(s->ffn_norm); free(s->q_norm); free(s->k_norm);
    free(s->x); free(s->xb); free(s->xb2); free(s->hb); free(s->hb2);
    free(s->q); free(s->att); free(s->logits);
    free(s->key_cache); free(s->value_cache); free(s->xq);
    free(s);
}

/* Batched prefill: process ids[0..n) at positions start..start+n-1, filling the
 * KV cache, and return logits for the LAST token only. Streams each weight row
 * once for all n tokens (GEMM), which is far faster than n single-token passes. */
float *ember_prefill(EmberModel *m, EmberState *s, const int *ids, int n, int start) {
    (void)m;
    if (n <= 0) return s->logits;
    int dim = s->dim, hd = s->head_dim, nh = s->n_heads, nkv = s->n_kv;
    int kv_dim = s->kv_dim, kv_mul = nh / nkv, hidden = s->hidden;
    float eps = s->norm_eps;
    int qd = nh * hd;

    int wide = dim; if (hidden > wide) wide = hidden; if (qd > wide) wide = qd;
    float *X   = malloc(sizeof(float) * (size_t)n * dim);
    float *Xn  = malloc(sizeof(float) * (size_t)n * dim);
    float *Tmp = malloc(sizeof(float) * (size_t)n * dim);
    float *Q   = malloc(sizeof(float) * (size_t)n * qd);
    float *Kb  = malloc(sizeof(float) * (size_t)n * kv_dim);
    float *Vb  = malloc(sizeof(float) * (size_t)n * kv_dim);
    float *Ob  = malloc(sizeof(float) * (size_t)n * qd);
    float *H1  = malloc(sizeof(float) * (size_t)n * hidden);
    float *H2  = malloc(sizeof(float) * (size_t)n * hidden);
    BlockQ8_0 *xqb = malloc(sizeof(BlockQ8_0) * (size_t)n * (wide / EMBER_Q_BLOCK + 1));

    for (int b = 0; b < n; b++) embed_row(&s->tok_emb, ids[b], dim, X + (size_t)b * dim);

    for (int l = 0; l < s->n_layers; l++) {
        for (int b = 0; b < n; b++)
            rmsnorm(Xn + (size_t)b * dim, X + (size_t)b * dim, s->attn_norm[l], dim, eps);

        ember_matmul_batch(Q,  Xn, s->wq[l].data, s->wq[l].dtype, dim, qd,     n, xqb);
        ember_matmul_batch(Kb, Xn, s->wk[l].data, s->wk[l].dtype, dim, kv_dim, n, xqb);
        ember_matmul_batch(Vb, Xn, s->wv[l].data, s->wv[l].dtype, dim, kv_dim, n, xqb);

        for (int b = 0; b < n; b++) {
            int pos = start + b;
            float *qb = Q + (size_t)b * qd, *kb = Kb + (size_t)b * kv_dim;
            if (s->qk_norm && s->q_norm[l]) {
                for (int h = 0; h < nh; h++)  rmsnorm(qb + h * hd, qb + h * hd, s->q_norm[l], hd, eps);
                for (int h = 0; h < nkv; h++) rmsnorm(kb + h * hd, kb + h * hd, s->k_norm[l], hd, eps);
            }
            rope(qb, qd, hd, pos, s->rope_theta, s->rope_style);
            rope(kb, kv_dim, hd, pos, s->rope_theta, s->rope_style);
            memcpy(s->key_cache + ((size_t)l * s->ctx + pos) * kv_dim, kb, sizeof(float) * kv_dim);
            memcpy(s->value_cache + ((size_t)l * s->ctx + pos) * kv_dim,
                   Vb + (size_t)b * kv_dim, sizeof(float) * kv_dim);
        }

        float inv_sqrt = 1.0f / sqrtf((float)hd);
        for (int b = 0; b < n; b++) {
            int pos = start + b;
            for (int h = 0; h < nh; h++) {
                const float *qh = Q + (size_t)b * qd + h * hd;
                float *att = s->att + h * s->ctx;
                int kvh = h / kv_mul;
                for (int t = 0; t <= pos; t++) {
                    const float *kt = s->key_cache + ((size_t)l * s->ctx + t) * kv_dim + kvh * hd;
                    float sc = 0.0f;
                    for (int i = 0; i < hd; i++) sc += qh[i] * kt[i];
                    att[t] = sc * inv_sqrt;
                }
                softmax(att, pos + 1);
                float *out = Ob + (size_t)b * qd + h * hd;
                for (int i = 0; i < hd; i++) out[i] = 0.0f;
                for (int t = 0; t <= pos; t++) {
                    const float *vt = s->value_cache + ((size_t)l * s->ctx + t) * kv_dim + kvh * hd;
                    float a = att[t];
                    for (int i = 0; i < hd; i++) out[i] += a * vt[i];
                }
            }
        }

        ember_matmul_batch(Tmp, Ob, s->wo[l].data, s->wo[l].dtype, qd, dim, n, xqb);
        for (size_t i = 0; i < (size_t)n * dim; i++) X[i] += Tmp[i];

        for (int b = 0; b < n; b++)
            rmsnorm(Xn + (size_t)b * dim, X + (size_t)b * dim, s->ffn_norm[l], dim, eps);
        ember_matmul_batch(H1, Xn, s->w1[l].data, s->w1[l].dtype, dim, hidden, n, xqb);
        ember_matmul_batch(H2, Xn, s->w3[l].data, s->w3[l].dtype, dim, hidden, n, xqb);
        for (size_t i = 0; i < (size_t)n * hidden; i++) {
            float g = H1[i];
            H1[i] = g / (1.0f + expf(-g)) * H2[i];
        }
        ember_matmul_batch(Tmp, H1, s->w2[l].data, s->w2[l].dtype, hidden, dim, n, xqb);
        for (size_t i = 0; i < (size_t)n * dim; i++) X[i] += Tmp[i];
    }

    /* only the last token's logits are needed to continue generation */
    rmsnorm(s->x, X + (size_t)(n - 1) * dim, s->final_norm, dim, eps);
    ember_matmul(s->logits, s->x, s->lm_head.data, s->lm_head.dtype, dim, s->vocab, s->xq);

    free(X); free(Xn); free(Tmp); free(Q); free(Kb); free(Vb); free(Ob); free(H1); free(H2); free(xqb);
    return s->logits;
}

float *ember_forward(EmberModel *m, EmberState *s, int tok, int pos) {
    (void)m;
    int dim = s->dim, hd = s->head_dim, nh = s->n_heads, nkv = s->n_kv;
    int kv_dim = s->kv_dim, kv_mul = nh / nkv;
    float eps = s->norm_eps;

    embed_row(&s->tok_emb, tok, dim, s->x);

    for (int l = 0; l < s->n_layers; l++) {
        rmsnorm(s->xb, s->x, s->attn_norm[l], dim, eps);

        float *k = s->key_cache + ((size_t)l * s->ctx + pos) * kv_dim;
        float *v = s->value_cache + ((size_t)l * s->ctx + pos) * kv_dim;
        ember_matmul(s->q, s->xb, s->wq[l].data, s->wq[l].dtype, dim, nh * hd, s->xq);
        ember_matmul(k,    s->xb, s->wk[l].data, s->wk[l].dtype, dim, kv_dim, s->xq);
        ember_matmul(v,    s->xb, s->wv[l].data, s->wv[l].dtype, dim, kv_dim, s->xq);

        if (s->qk_norm && s->q_norm[l]) {
            for (int h = 0; h < nh; h++)  rmsnorm(s->q + h * hd, s->q + h * hd, s->q_norm[l], hd, eps);
            for (int h = 0; h < nkv; h++) rmsnorm(k + h * hd, k + h * hd, s->k_norm[l], hd, eps);
        }

        rope(s->q, nh * hd, hd, pos, s->rope_theta, s->rope_style);
        rope(k, kv_dim, hd, pos, s->rope_theta, s->rope_style);

        float inv_sqrt = 1.0f / sqrtf((float)hd);
        for (int h = 0; h < nh; h++) {
            const float *qh = s->q + h * hd;
            float *att = s->att + h * s->ctx;
            int kvh = h / kv_mul;
            for (int t = 0; t <= pos; t++) {
                const float *kt = s->key_cache + ((size_t)l * s->ctx + t) * kv_dim + kvh * hd;
                float score = 0.0f;
                for (int i = 0; i < hd; i++) score += qh[i] * kt[i];
                att[t] = score * inv_sqrt;
            }
            softmax(att, pos + 1);
            float *out = s->xb + h * hd;
            for (int i = 0; i < hd; i++) out[i] = 0.0f;
            for (int t = 0; t <= pos; t++) {
                const float *vt = s->value_cache + ((size_t)l * s->ctx + t) * kv_dim + kvh * hd;
                float a = att[t];
                for (int i = 0; i < hd; i++) out[i] += a * vt[i];
            }
        }

        ember_matmul(s->xb2, s->xb, s->wo[l].data, s->wo[l].dtype, nh * hd, dim, s->xq);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];

        rmsnorm(s->xb, s->x, s->ffn_norm[l], dim, eps);
        ember_matmul(s->hb,  s->xb, s->w1[l].data, s->w1[l].dtype, dim, s->hidden, s->xq);
        ember_matmul(s->hb2, s->xb, s->w3[l].data, s->w3[l].dtype, dim, s->hidden, s->xq);
        for (int i = 0; i < s->hidden; i++) {
            float g = s->hb[i];
            s->hb[i] = g / (1.0f + expf(-g)) * s->hb2[i];
        }
        ember_matmul(s->xb, s->hb, s->w2[l].data, s->w2[l].dtype, s->hidden, dim, s->xq);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
    }

    rmsnorm(s->x, s->x, s->final_norm, dim, eps);
    ember_matmul(s->logits, s->x, s->lm_head.data, s->lm_head.dtype, dim, s->vocab, s->xq);
    return s->logits;
}
