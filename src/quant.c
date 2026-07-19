/* quant.c — offline quantization: read an fp32 .ember, write a Q8_0/Q4_0 one.
 *
 * Doing this in C (rather than the Python converter) keeps convert.py free of
 * numpy and is fast. Rules:
 *   - 1D tensors (norms) stay fp32.
 *   - 2D tensors are quantized. In Q4_0 mode, the embedding/lm_head table is
 *     kept at Q8_0 (mixed precision — it is the accuracy-sensitive tensor and,
 *     via weight tying, doubles as the output projection).
 *   - a row length not divisible by 32 falls back to fp32 for that tensor.
 */
#include "ember.h"
#include "kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t align_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

typedef struct {
    EmberTensor meta;
    void   *data;
    uint64_t nbytes;
} OutTensor;

static void quantize_tensor(const EmberTensor *src, const float *fdata,
                            int want, OutTensor *dst) {
    int out_rows = src->shape[0];
    int in_len   = src->shape[1];
    dst->meta = *src;

    int can_quant = (src->ndim >= 2) && (in_len % EMBER_Q_BLOCK == 0) && (want != EMBER_DT_F32);
    if (!can_quant) {
        dst->meta.dtype = EMBER_DT_F32;
        dst->nbytes = src->nbytes;
        dst->data = malloc(src->nbytes);
        memcpy(dst->data, fdata, src->nbytes);
        return;
    }

    int nb = in_len / EMBER_Q_BLOCK;
    if (want == EMBER_DT_Q4_0) {
        dst->meta.dtype = EMBER_DT_Q4_0;
        dst->nbytes = (uint64_t)out_rows * nb * sizeof(BlockQ4_0);
        BlockQ4_0 *q = malloc(dst->nbytes);
        for (int r = 0; r < out_rows; r++)
            ember_quantize_q4_0(fdata + (size_t)r * in_len, q + (size_t)r * nb, in_len);
        dst->data = q;
    } else {
        dst->meta.dtype = EMBER_DT_Q8_0;
        dst->nbytes = (uint64_t)out_rows * nb * sizeof(BlockQ8_0);
        BlockQ8_0 *q = malloc(dst->nbytes);
        for (int r = 0; r < out_rows; r++)
            ember_quantize_q8_0(fdata + (size_t)r * in_len, q + (size_t)r * nb, in_len);
        dst->data = q;
    }
}

int ember_quantize_file(const char *in_path, const char *out_path, int target) {
    EmberModel *m = ember_model_load(in_path);
    if (!m) return 1;
    const EmberHeader *h = ember_model_header(m);
    const EmberTensor *table =
        (const EmberTensor *)((const uint8_t *)h + h->tensor_table_offset);
    int n = h->n_tensors;

    OutTensor *out = calloc(n, sizeof(OutTensor));
    for (int i = 0; i < n; i++) {
        const EmberTensor *t = &table[i];
        const float *fdata = (const float *)ember_model_data(m, t);
        /* keep the embedding/lm_head table at Q8_0 even in Q4 mode */
        int want = target;
        if (target == EMBER_DT_Q4_0 &&
            (strcmp(t->name, "tok_embeddings") == 0 || strcmp(t->name, "lm_head") == 0))
            want = EMBER_DT_Q8_0;
        if (t->dtype != EMBER_DT_F32) { /* already quantized: copy through */
            out[i].meta = *t;
            out[i].nbytes = t->nbytes;
            out[i].data = malloc(t->nbytes);
            memcpy(out[i].data, fdata, t->nbytes);
        } else {
            quantize_tensor(t, fdata, want, &out[i]);
        }
    }

    /* recompute layout */
    const uint8_t *tok_blob = (const uint8_t *)h + h->tokenizer_offset;
    uint64_t tok_off = sizeof(EmberHeader);
    uint64_t tok_size = h->tokenizer_size;
    uint64_t table_off = align_up(tok_off + tok_size, 8);
    uint64_t table_size = (uint64_t)n * sizeof(EmberTensor);
    uint64_t data_start = align_up(table_off + table_size, 64);
    uint64_t cur = data_start;
    for (int i = 0; i < n; i++) {
        cur = align_up(cur, 64);
        out[i].meta.offset = cur;
        out[i].meta.nbytes = out[i].nbytes;
        cur += out[i].nbytes;
    }

    EmberHeader nh = *h;
    nh.header_bytes = data_start;
    nh.tokenizer_offset = tok_off;
    nh.tensor_table_offset = table_off;
    nh.tensor_table_size = table_size;

    FILE *f = fopen(out_path, "wb");
    if (!f) { ember_model_free(m); return 1; }
    fwrite(&nh, sizeof(nh), 1, f);
    fwrite(tok_blob, 1, tok_size, f);
    while ((uint64_t)ftell(f) < table_off) fputc(0, f);
    for (int i = 0; i < n; i++) fwrite(&out[i].meta, sizeof(EmberTensor), 1, f);
    while ((uint64_t)ftell(f) < data_start) fputc(0, f);
    for (int i = 0; i < n; i++) {
        while ((uint64_t)ftell(f) < out[i].meta.offset) fputc(0, f);
        fwrite(out[i].data, 1, out[i].nbytes, f);
    }
    fclose(f);

    fprintf(stderr, "quantized %s -> %s (%.1f MB)\n", in_path, out_path, cur / 1e6);
    for (int i = 0; i < n; i++) free(out[i].data);
    free(out);
    ember_model_free(m);
    return 0;
}
