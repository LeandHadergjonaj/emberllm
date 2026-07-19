/* model.c — fp32 forward pass for LLaMA-style decoder transformers.
 *
 * Stage 1: correct but slow. Everything is scalar fp32; the matmul is a plain
 * row-wise dot product. Later stages replace `matmul` with threaded / NEON /
 * quantized kernels without touching the surrounding logic.
 */
#include "ember.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct EmberState {
    const EmberModel *m;
    int dim, hidden, n_layers, n_heads, n_kv, head_dim, kv_dim, vocab, ctx;
    float rope_theta, norm_eps;
    int   rope_style, qk_norm;

    /* resolved weight pointers (into the mmap) */
    const float  *tok_emb, *final_norm, *lm_head;
    const float **attn_norm, **wq, **wk, **wv, **wo;
    const float **ffn_norm, **w1, **w2, **w3;
    const float **q_norm, **k_norm; /* per-head RMSNorm weights, Stage 3 */

    /* activation scratch */
    float *x, *xb, *xb2, *hb, *hb2, *q, *att, *logits;
    float *key_cache, *value_cache; /* [n_layers * ctx * kv_dim] */
};

/* out[d_out] = W[d_out, n_in] * x[n_in], row-major weights. */
static void matmul(float *out, const float *x, const float *w, int n_in, int d_out) {
    for (int i = 0; i < d_out; i++) {
        const float *row = w + (size_t)i * n_in;
        float sum = 0.0f;
        for (int j = 0; j < n_in; j++) sum += row[j] * x[j];
        out[i] = sum;
    }
}

/* RMSNorm with fp32 accumulation; epsilon inside the sqrt. */
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

/* Apply RoPE in-place to a vector of `n` values grouped into head_dim chunks. */
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

static const float *tref(const EmberModel *m, const char *fmt, int l) {
    char name[48];
    if (l >= 0) snprintf(name, sizeof(name), fmt, l);
    else        snprintf(name, sizeof(name), "%s", fmt);
    const EmberTensor *t = ember_model_tensor(m, name);
    return t ? (const float *)ember_model_data(m, t) : NULL;
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
    s->attn_norm = malloc(sizeof(void *) * L); s->ffn_norm = malloc(sizeof(void *) * L);
    s->wq = malloc(sizeof(void *) * L); s->wk = malloc(sizeof(void *) * L);
    s->wv = malloc(sizeof(void *) * L); s->wo = malloc(sizeof(void *) * L);
    s->w1 = malloc(sizeof(void *) * L); s->w2 = malloc(sizeof(void *) * L);
    s->w3 = malloc(sizeof(void *) * L);
    s->q_norm = malloc(sizeof(void *) * L); s->k_norm = malloc(sizeof(void *) * L);
    for (int l = 0; l < L; l++) {
        s->attn_norm[l] = tref(m, "layers.%d.attn_norm", l);
        s->wq[l] = tref(m, "layers.%d.wq", l); s->wk[l] = tref(m, "layers.%d.wk", l);
        s->wv[l] = tref(m, "layers.%d.wv", l); s->wo[l] = tref(m, "layers.%d.wo", l);
        s->ffn_norm[l] = tref(m, "layers.%d.ffn_norm", l);
        s->w1[l] = tref(m, "layers.%d.w1", l); s->w2[l] = tref(m, "layers.%d.w2", l);
        s->w3[l] = tref(m, "layers.%d.w3", l);
        s->q_norm[l] = tref(m, "layers.%d.q_norm", l);
        s->k_norm[l] = tref(m, "layers.%d.k_norm", l);
    }
    s->tok_emb = tref(m, "tok_embeddings", -1);
    s->final_norm = tref(m, "final_norm", -1);
    s->lm_head = (h->flags & EMBER_FLAG_TIED_EMBED) ? s->tok_emb : tref(m, "lm_head", -1);

    int hd = s->head_dim, nh = s->n_heads;
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
    return s;
}

void ember_state_free(EmberState *s) {
    if (!s) return;
    free(s->attn_norm); free(s->ffn_norm);
    free(s->wq); free(s->wk); free(s->wv); free(s->wo);
    free(s->w1); free(s->w2); free(s->w3); free(s->q_norm); free(s->k_norm);
    free(s->x); free(s->xb); free(s->xb2); free(s->hb); free(s->hb2);
    free(s->q); free(s->att); free(s->logits);
    free(s->key_cache); free(s->value_cache);
    free(s);
}

float *ember_forward(EmberModel *m, EmberState *s, int tok, int pos) {
    (void)m;
    int dim = s->dim, hd = s->head_dim, nh = s->n_heads, nkv = s->n_kv;
    int kv_dim = s->kv_dim, kv_mul = nh / nkv;
    float eps = s->norm_eps;

    memcpy(s->x, s->tok_emb + (size_t)tok * dim, sizeof(float) * dim);

    for (int l = 0; l < s->n_layers; l++) {
        rmsnorm(s->xb, s->x, s->attn_norm[l], dim, eps);

        float *k = s->key_cache + ((size_t)l * s->ctx + pos) * kv_dim;
        float *v = s->value_cache + ((size_t)l * s->ctx + pos) * kv_dim;
        matmul(s->q, s->xb, s->wq[l], dim, nh * hd);
        matmul(k, s->xb, s->wk[l], dim, kv_dim);
        matmul(v, s->xb, s->wv[l], dim, kv_dim);

        /* optional per-head QK-RMSNorm before RoPE (Stage 3 models) */
        if (s->qk_norm && s->q_norm[l]) {
            for (int h = 0; h < nh; h++)  rmsnorm(s->q + h * hd, s->q + h * hd, s->q_norm[l], hd, eps);
            for (int h = 0; h < nkv; h++) rmsnorm(k + h * hd, k + h * hd, s->k_norm[l], hd, eps);
        }

        rope(s->q, nh * hd, hd, pos, s->rope_theta, s->rope_style);
        rope(k, kv_dim, hd, pos, s->rope_theta, s->rope_style);

        /* multi-head attention over positions 0..pos */
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

        matmul(s->xb2, s->xb, s->wo[l], nh * hd, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];

        /* FFN: w2( silu(w1 x) * w3 x ) */
        rmsnorm(s->xb, s->x, s->ffn_norm[l], dim, eps);
        matmul(s->hb, s->xb, s->w1[l], dim, s->hidden);
        matmul(s->hb2, s->xb, s->w3[l], dim, s->hidden);
        for (int i = 0; i < s->hidden; i++) {
            float g = s->hb[i];
            s->hb[i] = g / (1.0f + expf(-g)) * s->hb2[i];
        }
        matmul(s->xb, s->hb, s->w2[l], s->hidden, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
    }

    rmsnorm(s->x, s->x, s->final_norm, dim, eps);
    matmul(s->logits, s->x, s->lm_head, dim, s->vocab);
    return s->logits;
}
