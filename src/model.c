/* model.c — forward pass for LLaMA-style decoder transformers.
 *
 * The matmul, quantization, and threading live in kernels.c/threads.c; this
 * file is the architecture: embedding, RMSNorm, RoPE, GQA attention, SwiGLU.
 * Weights carry a dtype so the same code runs fp32, Q8_0, or Q4_0 models.
 */
#include "ember.h"
#include "kernels.h"
#include "util.h"

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
    const float **b_q, **b_k, **b_v; /* optional attention biases (Qwen2.5); NULL if absent */

    float *x, *xb, *xb2, *hb, *hb2, *q, *att, *logits;
    void  *key_cache, *value_cache; /* float* (fp32) or uint16_t* (fp16), per kv_fp16 */
    int    kv_fp16;                 /* 1 -> KV cache stored as fp16 (half the memory)  */
    float *kbuf, *vbuf;             /* per-step K/V scratch (decode), always fp32       */
    BlockQ8_0 *xq; /* scratch: quantized activation for q8/q4 matmuls */
};

/* Write one position's K and V (kv_dim floats each) into the cache for layer l,
 * converting to fp16 when the cache is half-precision. */
static void kv_store(EmberState *s, int l, int pos, const float *k, const float *v) {
    size_t off = ((size_t)l * s->ctx + pos) * s->kv_dim;
    if (s->kv_fp16) {
        uint16_t *kc = (uint16_t *)s->key_cache + off, *vc = (uint16_t *)s->value_cache + off;
        for (int i = 0; i < s->kv_dim; i++) { kc[i] = ember_f16c(k[i]); vc[i] = ember_f16c(v[i]); }
    } else {
        memcpy((float *)s->key_cache + off,   k, sizeof(float) * s->kv_dim);
        memcpy((float *)s->value_cache + off, v, sizeof(float) * s->kv_dim);
    }
}

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
    if (!w->data) ember_die("model is missing its token-embedding tensor");
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
    return ember_state_new_ex(m, ctx_len, 0);
}

EmberState *ember_state_new_ex(const EmberModel *m, int ctx_len, int kv_fp16) {
    const EmberHeader *h = ember_model_header(m);
    EmberState *s = ember_xcalloc(1, sizeof(*s), "EmberState");
    s->m = m;
    s->kv_fp16 = kv_fp16 ? 1 : 0;
    s->dim = h->dim; s->hidden = h->hidden_dim; s->n_layers = h->n_layers;
    s->n_heads = h->n_heads; s->n_kv = h->n_kv_heads; s->head_dim = h->head_dim;
    s->kv_dim = h->n_kv_heads * h->head_dim; s->vocab = h->vocab_size;
    s->rope_theta = h->rope_theta; s->norm_eps = h->norm_eps;
    s->rope_style = h->rope_style; s->qk_norm = (h->flags & EMBER_FLAG_QK_NORM) != 0;
    s->ctx = (ctx_len > 0 && ctx_len < h->max_seq_len) ? ctx_len : h->max_seq_len;

    int L = s->n_layers;
    s->wq = ember_xmalloc(sizeof(Wt) * L, "wq"); s->wk = ember_xmalloc(sizeof(Wt) * L, "wk");
    s->wv = ember_xmalloc(sizeof(Wt) * L, "wv"); s->wo = ember_xmalloc(sizeof(Wt) * L, "wo");
    s->w1 = ember_xmalloc(sizeof(Wt) * L, "w1"); s->w2 = ember_xmalloc(sizeof(Wt) * L, "w2");
    s->w3 = ember_xmalloc(sizeof(Wt) * L, "w3");
    s->attn_norm = ember_xmalloc(sizeof(void *) * L, "attn_norm");
    s->ffn_norm = ember_xmalloc(sizeof(void *) * L, "ffn_norm");
    s->q_norm = ember_xmalloc(sizeof(void *) * L, "q_norm");
    s->k_norm = ember_xmalloc(sizeof(void *) * L, "k_norm");
    s->b_q = ember_xmalloc(sizeof(void *) * L, "b_q");
    s->b_k = ember_xmalloc(sizeof(void *) * L, "b_k");
    s->b_v = ember_xmalloc(sizeof(void *) * L, "b_v");
    for (int l = 0; l < L; l++) {
        s->wq[l] = tref(m, "layers.%d.wq", l); s->wk[l] = tref(m, "layers.%d.wk", l);
        s->wv[l] = tref(m, "layers.%d.wv", l); s->wo[l] = tref(m, "layers.%d.wo", l);
        s->w1[l] = tref(m, "layers.%d.w1", l); s->w2[l] = tref(m, "layers.%d.w2", l);
        s->w3[l] = tref(m, "layers.%d.w3", l);
        s->attn_norm[l] = tref_f32(m, "layers.%d.attn_norm", l);
        s->ffn_norm[l]  = tref_f32(m, "layers.%d.ffn_norm", l);
        s->q_norm[l]    = tref_f32(m, "layers.%d.q_norm", l);
        s->k_norm[l]    = tref_f32(m, "layers.%d.k_norm", l);
        s->b_q[l]       = tref_f32(m, "layers.%d.wq_bias", l); /* optional */
        s->b_k[l]       = tref_f32(m, "layers.%d.wk_bias", l);
        s->b_v[l]       = tref_f32(m, "layers.%d.wv_bias", l);
    }
    s->tok_emb = tref(m, "tok_embeddings", -1);
    s->final_norm = tref_f32(m, "final_norm", -1);
    s->lm_head = (h->flags & EMBER_FLAG_TIED_EMBED) ? s->tok_emb : tref(m, "lm_head", -1);

    /* Fail loudly (not with a NULL deref mid-forward) if the file is missing a
     * tensor the forward pass will dereference. */
    if (!s->tok_emb.data)   ember_die("model is missing tensor 'tok_embeddings'");
    if (!s->final_norm)     ember_die("model is missing tensor 'final_norm'");
    if (!s->lm_head.data)   ember_die("model is missing tensor 'lm_head' (and is not tied)");
    for (int l = 0; l < L; l++) {
        if (!s->wq[l].data || !s->wk[l].data || !s->wv[l].data || !s->wo[l].data ||
            !s->w1[l].data || !s->w2[l].data || !s->w3[l].data ||
            !s->attn_norm[l] || !s->ffn_norm[l])
            ember_die("model is missing a required tensor in layer %d", l);
    }

    int hd = s->head_dim, nh = s->n_heads;
    int qd = nh * hd; /* query width; with decoupled head_dim this can exceed dim */
    int wide = s->hidden > s->dim ? s->hidden : s->dim;
    if (qd > wide) wide = qd;
    s->x = ember_xmalloc(sizeof(float) * s->dim, "activation x");
    s->xb = ember_xmalloc(sizeof(float) * (qd > s->dim ? qd : s->dim), "activation xb");
    s->xb2 = ember_xmalloc(sizeof(float) * s->dim, "activation xb2");
    s->hb = ember_xmalloc(sizeof(float) * s->hidden, "activation hb");
    s->hb2 = ember_xmalloc(sizeof(float) * s->hidden, "activation hb2");
    s->q = ember_xmalloc(sizeof(float) * nh * hd, "activation q");
    s->att = ember_xmalloc(sizeof(float) * nh * s->ctx, "attention scores");
    s->logits = ember_xmalloc(sizeof(float) * s->vocab, "logits");
    size_t kv_elems = (size_t)L * s->ctx * s->kv_dim;
    size_t kv_esize = s->kv_fp16 ? sizeof(uint16_t) : sizeof(float);
    s->key_cache   = ember_xmalloc(kv_esize * kv_elems, "key cache");
    s->value_cache = ember_xmalloc(kv_esize * kv_elems, "value cache");
    s->kbuf = ember_xmalloc(sizeof(float) * s->kv_dim, "K scratch");
    s->vbuf = ember_xmalloc(sizeof(float) * s->kv_dim, "V scratch");
    s->xq = ember_xmalloc(sizeof(BlockQ8_0) * (wide / EMBER_Q_BLOCK + 1), "activation quant scratch");
    return s;
}

void ember_state_free(EmberState *s) {
    if (!s) return;
    free(s->wq); free(s->wk); free(s->wv); free(s->wo);
    free(s->w1); free(s->w2); free(s->w3);
    free(s->attn_norm); free(s->ffn_norm); free(s->q_norm); free(s->k_norm);
    free(s->b_q); free(s->b_k); free(s->b_v);
    free(s->x); free(s->xb); free(s->xb2); free(s->hb); free(s->hb2);
    free(s->q); free(s->att); free(s->logits);
    free(s->key_cache); free(s->value_cache); free(s->kbuf); free(s->vbuf); free(s->xq);
    free(s);
}

/* One query position's attention over the cache, parallelized across heads.
 * Heads are independent (each owns its att[h*ctx] slice and out[h*hd] slice),
 * so this is a clean data-parallel loop. Threading it is what keeps long-context
 * decode from becoming serial-attention-bound (see report.md). */
typedef struct {
    const float *q;      /* query, [nh*hd]                              */
    float *att;          /* scratch scores, [nh*ctx]                    */
    float *out;          /* attention output, [nh*hd]                   */
    const void *kcache;  /* this layer's key cache base (fp32 or fp16)  */
    const void *vcache;  /* this layer's value cache base (fp32 or fp16)*/
    int pos, hd, kv_dim, kv_mul, ctx, kv_fp16;
    float inv_sqrt;
} AttnCtx;

static void attn_heads(void *c, int h0, int h1) {
    AttnCtx *a = c;
    for (int h = h0; h < h1; h++) {
        const float *qh = a->q + (size_t)h * a->hd;
        float *att = a->att + (size_t)h * a->ctx;
        float *out = a->out + (size_t)h * a->hd;
        int kvh = h / a->kv_mul;
        for (int i = 0; i < a->hd; i++) out[i] = 0.0f;
        if (a->kv_fp16) {
            const uint16_t *kc = (const uint16_t *)a->kcache, *vc = (const uint16_t *)a->vcache;
            for (int t = 0; t <= a->pos; t++) {
                const uint16_t *kt = kc + (size_t)t * a->kv_dim + kvh * a->hd;
                float sc = 0.0f;
                for (int i = 0; i < a->hd; i++) sc += qh[i] * ember_f16f(kt[i]);
                att[t] = sc * a->inv_sqrt;
            }
            softmax(att, a->pos + 1);
            for (int t = 0; t <= a->pos; t++) {
                const uint16_t *vt = vc + (size_t)t * a->kv_dim + kvh * a->hd;
                float w = att[t];
                for (int i = 0; i < a->hd; i++) out[i] += w * ember_f16f(vt[i]);
            }
        } else {
            const float *kc = (const float *)a->kcache, *vc = (const float *)a->vcache;
            for (int t = 0; t <= a->pos; t++) {
                const float *kt = kc + (size_t)t * a->kv_dim + kvh * a->hd;
                float sc = 0.0f;
                for (int i = 0; i < a->hd; i++) sc += qh[i] * kt[i];
                att[t] = sc * a->inv_sqrt;
            }
            softmax(att, a->pos + 1);
            for (int t = 0; t <= a->pos; t++) {
                const float *vt = vc + (size_t)t * a->kv_dim + kvh * a->hd;
                float w = att[t];
                for (int i = 0; i < a->hd; i++) out[i] += w * vt[i];
            }
        }
    }
}

/* Run one query position's attention, threading over heads only when the work
 * (≈ nh * context * head_dim) is large enough to amortize dispatch. */
static void attention(EmberState *s, const float *q, float *out, int l, int pos,
                      int nh, int hd, int kv_dim, int kv_mul) {
    size_t off = (size_t)l * s->ctx * kv_dim;
    const void *kbase = s->kv_fp16 ? (const void *)((const uint16_t *)s->key_cache + off)
                                   : (const void *)((const float *)s->key_cache + off);
    const void *vbase = s->kv_fp16 ? (const void *)((const uint16_t *)s->value_cache + off)
                                   : (const void *)((const float *)s->value_cache + off);
    AttnCtx ac = {
        q, s->att, out, kbase, vbase,
        pos, hd, kv_dim, kv_mul, s->ctx, s->kv_fp16, 1.0f / sqrtf((float)hd)
    };
    if (ember_should_parallel((long)nh * (pos + 1) * hd))
        ember_parallel_for(attn_heads, &ac, nh);
    else
        attn_heads(&ac, 0, nh);
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

    int wide = dim;
    if (hidden > wide) wide = hidden;
    if (qd > wide) wide = qd;
    float *X   = ember_xmalloc(sizeof(float) * (size_t)n * dim, "prefill X");
    float *Xn  = ember_xmalloc(sizeof(float) * (size_t)n * dim, "prefill Xn");
    float *Tmp = ember_xmalloc(sizeof(float) * (size_t)n * dim, "prefill Tmp");
    float *Q   = ember_xmalloc(sizeof(float) * (size_t)n * qd, "prefill Q");
    float *Kb  = ember_xmalloc(sizeof(float) * (size_t)n * kv_dim, "prefill K");
    float *Vb  = ember_xmalloc(sizeof(float) * (size_t)n * kv_dim, "prefill V");
    float *Ob  = ember_xmalloc(sizeof(float) * (size_t)n * qd, "prefill O");
    float *H1  = ember_xmalloc(sizeof(float) * (size_t)n * hidden, "prefill H1");
    float *H2  = ember_xmalloc(sizeof(float) * (size_t)n * hidden, "prefill H2");
    BlockQ8_0 *xqb = ember_xmalloc(sizeof(BlockQ8_0) * (size_t)n * (wide / EMBER_Q_BLOCK + 1), "prefill quant scratch");

    for (int b = 0; b < n; b++) {
        if (ids[b] < 0 || ids[b] >= s->vocab)
            ember_die("token id %d out of range [0,%d)", ids[b], s->vocab);
        embed_row(&s->tok_emb, ids[b], dim, X + (size_t)b * dim);
    }

    for (int l = 0; l < s->n_layers; l++) {
        for (int b = 0; b < n; b++)
            rmsnorm(Xn + (size_t)b * dim, X + (size_t)b * dim, s->attn_norm[l], dim, eps);

        ember_matmul_batch(Q,  Xn, s->wq[l].data, s->wq[l].dtype, dim, qd,     n, xqb);
        ember_matmul_batch(Kb, Xn, s->wk[l].data, s->wk[l].dtype, dim, kv_dim, n, xqb);
        ember_matmul_batch(Vb, Xn, s->wv[l].data, s->wv[l].dtype, dim, kv_dim, n, xqb);

        if (s->b_q[l]) for (int b = 0; b < n; b++) for (int i = 0; i < qd; i++)     Q[(size_t)b*qd+i]     += s->b_q[l][i];
        if (s->b_k[l]) for (int b = 0; b < n; b++) for (int i = 0; i < kv_dim; i++) Kb[(size_t)b*kv_dim+i] += s->b_k[l][i];
        if (s->b_v[l]) for (int b = 0; b < n; b++) for (int i = 0; i < kv_dim; i++) Vb[(size_t)b*kv_dim+i] += s->b_v[l][i];

        for (int b = 0; b < n; b++) {
            int pos = start + b;
            float *qb = Q + (size_t)b * qd, *kb = Kb + (size_t)b * kv_dim;
            if (s->qk_norm && s->q_norm[l]) {
                for (int h = 0; h < nh; h++)  rmsnorm(qb + h * hd, qb + h * hd, s->q_norm[l], hd, eps);
                for (int h = 0; h < nkv; h++) rmsnorm(kb + h * hd, kb + h * hd, s->k_norm[l], hd, eps);
            }
            rope(qb, qd, hd, pos, s->rope_theta, s->rope_style);
            rope(kb, kv_dim, hd, pos, s->rope_theta, s->rope_style);
            kv_store(s, l, pos, kb, Vb + (size_t)b * kv_dim);
        }

        for (int b = 0; b < n; b++)
            attention(s, Q + (size_t)b * qd, Ob + (size_t)b * qd, l,
                      start + b, nh, hd, kv_dim, kv_mul);

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

    if (tok < 0 || tok >= s->vocab)
        ember_die("token id %d out of range [0,%d)", tok, s->vocab);
    if (pos < 0 || pos >= s->ctx)
        ember_die("position %d out of range [0,%d)", pos, s->ctx);
    embed_row(&s->tok_emb, tok, dim, s->x);

    for (int l = 0; l < s->n_layers; l++) {
        rmsnorm(s->xb, s->x, s->attn_norm[l], dim, eps);

        float *k = s->kbuf, *v = s->vbuf; /* fp32 scratch; kv_store writes the cache */
        ember_matmul(s->q, s->xb, s->wq[l].data, s->wq[l].dtype, dim, nh * hd, s->xq);
        ember_matmul(k,    s->xb, s->wk[l].data, s->wk[l].dtype, dim, kv_dim, s->xq);
        ember_matmul(v,    s->xb, s->wv[l].data, s->wv[l].dtype, dim, kv_dim, s->xq);

        if (s->b_q[l]) for (int i = 0; i < nh * hd; i++) s->q[i] += s->b_q[l][i];
        if (s->b_k[l]) for (int i = 0; i < kv_dim; i++)  k[i]    += s->b_k[l][i];
        if (s->b_v[l]) for (int i = 0; i < kv_dim; i++)  v[i]    += s->b_v[l][i];

        if (s->qk_norm && s->q_norm[l]) {
            for (int h = 0; h < nh; h++)  rmsnorm(s->q + h * hd, s->q + h * hd, s->q_norm[l], hd, eps);
            for (int h = 0; h < nkv; h++) rmsnorm(k + h * hd, k + h * hd, s->k_norm[l], hd, eps);
        }

        rope(s->q, nh * hd, hd, pos, s->rope_theta, s->rope_style);
        rope(k, kv_dim, hd, pos, s->rope_theta, s->rope_style);
        kv_store(s, l, pos, k, v);

        attention(s, s->q, s->xb, l, pos, nh, hd, kv_dim, kv_mul);

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
