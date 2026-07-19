/* kernels.c — the hot loops: fp16 conversion, quantization, and matmul.
 *
 * Each kernel has a portable scalar version and, where it matters, a NEON
 * (Apple Silicon / ARM) and AVX2 (x86) path selected at compile time. The
 * matmul is memory-bandwidth-bound during decode, so the quantized paths —
 * which shrink the weight bytes streamed per token — are the real speedups;
 * the SIMD mostly keeps a single core from becoming compute-bound on prefill.
 */
#include "kernels.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#if defined(__ARM_NEON)
#  include <arm_neon.h>
#endif
#if defined(__AVX2__)
#  include <immintrin.h>
#endif

/* -------------------------------------------------------------------------- */
/* half precision                                                             */
/* -------------------------------------------------------------------------- */
float ember_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else { /* subnormal */
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &bits, 4);
    return f;
}

uint16_t ember_f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign; /* underflow to zero */
        mant |= 0x800000;
        int shift = 14 - exp;
        uint32_t half = mant >> shift;
        uint32_t rem  = mant & ((1u << shift) - 1);
        if (rem > (1u << (shift - 1)) || (rem == (1u << (shift - 1)) && (half & 1))) half++;
        return (uint16_t)(sign | half);
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00); /* overflow to inf */
    }
    uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
    uint32_t rem = mant & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
    return h;
}

/* -------------------------------------------------------------------------- */
/* quantization                                                               */
/* -------------------------------------------------------------------------- */
void ember_quantize_q8_0(const float *x, BlockQ8_0 *out, int n) {
    int nb = n / EMBER_Q_BLOCK;
    for (int b = 0; b < nb; b++) {
        const float *xb = x + b * EMBER_Q_BLOCK;
        float amax = 0.0f;
        for (int i = 0; i < EMBER_Q_BLOCK; i++) {
            float a = fabsf(xb[i]);
            if (a > amax) amax = a;
        }
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        out[b].d = ember_f32_to_f16(d);
        for (int i = 0; i < EMBER_Q_BLOCK; i++) {
            int q = (int)lroundf(xb[i] * id);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            out[b].qs[i] = (int8_t)q;
        }
    }
}

/* ggml-style Q4_0: nibbles centered at 8, values in [-8,7], one fp16 scale. */
void ember_quantize_q4_0(const float *x, BlockQ4_0 *out, int n) {
    int nb = n / EMBER_Q_BLOCK;
    for (int b = 0; b < nb; b++) {
        const float *xb = x + b * EMBER_Q_BLOCK;
        float amax = 0.0f, vmax = 0.0f;
        for (int i = 0; i < EMBER_Q_BLOCK; i++) {
            float a = fabsf(xb[i]);
            if (a > amax) { amax = a; vmax = xb[i]; }
        }
        float d = vmax / -8.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        out[b].d = ember_f32_to_f16(d);
        for (int i = 0; i < EMBER_Q_BLOCK / 2; i++) {
            int q0 = (int)(xb[i] * id + 8.5f);
            int q1 = (int)(xb[i + EMBER_Q_BLOCK / 2] * id + 8.5f);
            if (q0 < 0) q0 = 0; if (q0 > 15) q0 = 15;
            if (q1 < 0) q1 = 0; if (q1 > 15) q1 = 15;
            out[b].qs[i] = (uint8_t)(q0 | (q1 << 4));
        }
    }
}

/* -------------------------------------------------------------------------- */
/* dot products                                                               */
/* -------------------------------------------------------------------------- */
static float dot_f32(const float *a, const float *b, int n) {
#if defined(__ARM_NEON)
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i),     vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    float s = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
#elif defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    __m128 lo = _mm256_castps256_ps128(acc), hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    float s = _mm_cvtss_f32(lo);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
#else
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
#endif
}

/* Fast half->float for the hot loop: one hardware instruction on ARM, the
 * portable bit-twiddling elsewhere. */
static inline float f16_fast(uint16_t h) {
#if defined(__ARM_NEON)
    __fp16 v;
    memcpy(&v, &h, 2);
    return (float)v;
#elif defined(__F16C__)
    return _cvtsh_ss(h);
#else
    return ember_f16_to_f32(h);
#endif
}

/* dot of a Q8_0 weight row with a Q8_0 activation row (nb blocks). */
static float vecdot_q8(const BlockQ8_0 *w, const BlockQ8_0 *x, int nb) {
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        int32_t acc = 0;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        int32x4_t v = vdupq_n_s32(0);
        v = vdotq_s32(v, vld1q_s8(w[b].qs),      vld1q_s8(x[b].qs));
        v = vdotq_s32(v, vld1q_s8(w[b].qs + 16), vld1q_s8(x[b].qs + 16));
        acc = vaddvq_s32(v);
#else
        for (int i = 0; i < EMBER_Q_BLOCK; i++) acc += (int)w[b].qs[i] * (int)x[b].qs[i];
#endif
        sum += (float)acc * f16_fast(w[b].d) * f16_fast(x[b].d);
    }
    return sum;
}

/* dot of a Q4_0 weight row with a Q8_0 activation row (nb blocks). */
static float vecdot_q4(const BlockQ4_0 *w, const BlockQ8_0 *x, int nb) {
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        int32_t acc = 0;
        const uint8_t *q = w[b].qs;
        const int8_t  *a = x[b].qs;
        for (int i = 0; i < EMBER_Q_BLOCK / 2; i++) {
            int w0 = (int)(q[i] & 0x0F) - 8;
            int w1 = (int)(q[i] >> 4) - 8;
            acc += w0 * (int)a[i] + w1 * (int)a[i + EMBER_Q_BLOCK / 2];
        }
        sum += (float)acc * f16_fast(w[b].d) * f16_fast(x[b].d);
    }
    return sum;
}

/* -------------------------------------------------------------------------- */
/* matmul                                                                     */
/* -------------------------------------------------------------------------- */
typedef struct { float *out; const void *x; const void *w; int n_in; int dtype; } MMCtx;

static void mm_rows(void *c, int i0, int i1) {
    MMCtx *m = c;
    if (m->dtype == EMBER_DT_F32) {
        const float *x = m->x, *w = m->w;
        for (int i = i0; i < i1; i++) m->out[i] = dot_f32(w + (size_t)i * m->n_in, x, m->n_in);
    } else if (m->dtype == EMBER_DT_Q8_0) {
        const BlockQ8_0 *x = m->x, *w = m->w;
        int nb = m->n_in / EMBER_Q_BLOCK;
        for (int i = i0; i < i1; i++) m->out[i] = vecdot_q8(w + (size_t)i * nb, x, nb);
    } else { /* Q4_0 */
        const BlockQ8_0 *x = m->x;
        const BlockQ4_0 *w = m->w;
        int nb = m->n_in / EMBER_Q_BLOCK;
        for (int i = i0; i < i1; i++) m->out[i] = vecdot_q4(w + (size_t)i * nb, x, nb);
    }
}

void ember_matmul(float *out, const float *x, const void *w, int dtype,
                  int n_in, int d_out, BlockQ8_0 *xq) {
    const void *xin = x;
    if (dtype != EMBER_DT_F32) {
        ember_quantize_q8_0(x, xq, n_in); /* quantize activation once, reuse per row */
        xin = xq;
    }
    MMCtx c = { out, xin, w, n_in, dtype };
    /* Parallelize only when there is enough work to amortize dispatch. */
    if (ember_nthreads() > 1 && (long)n_in * d_out >= (1 << 15))
        ember_parallel_for(mm_rows, &c, d_out);
    else
        mm_rows(&c, 0, d_out);
}
