/* kernels.h — internal compute kernels and the thread pool.
 *
 * These are not part of the public API in ember.h; they are shared between
 * model.c (the forward pass), quant.c (offline quantization), and bench code.
 */
#ifndef EMBER_KERNELS_H
#define EMBER_KERNELS_H

#include "ember.h"
#include <string.h>

#if defined(__ARM_NEON)
#  include <arm_neon.h>
#endif
#if defined(__F16C__) || defined(__AVX2__)
#  include <immintrin.h>
#endif

/* --- half precision (scalar, portable bit-twiddling) --------------------- */
uint16_t ember_f32_to_f16(float f);
float    ember_f16_to_f32(uint16_t h);

/* Hot-loop fp16<->fp32 using one hardware instruction where available (NEON
 * __fp16, x86 F16C), falling back to the portable bit-twiddling. Used by the
 * matmul and the optional fp16 KV cache. */
static inline float ember_f16f(uint16_t h) {
#if defined(__ARM_NEON)
    __fp16 v; memcpy(&v, &h, 2); return (float)v;
#elif defined(__F16C__)
    return _cvtsh_ss(h);
#else
    return ember_f16_to_f32(h);
#endif
}
static inline uint16_t ember_f16c(float f) {
#if defined(__ARM_NEON)
    __fp16 v = (__fp16)f; uint16_t h; memcpy(&h, &v, 2); return h;
#elif defined(__F16C__)
    return (uint16_t)_cvtss_sh(f, 0);
#else
    return ember_f32_to_f16(f);
#endif
}

/* --- Q8_0: 32-weight blocks, one fp16 scale + 32 int8 -------------------- */
#pragma pack(push, 1)
typedef struct { uint16_t d; int8_t qs[EMBER_Q_BLOCK]; } BlockQ8_0; /* 34 bytes */
typedef struct { uint16_t d; uint8_t qs[EMBER_Q_BLOCK / 2]; } BlockQ4_0; /* 18 bytes */
#pragma pack(pop)
_Static_assert(sizeof(BlockQ8_0) == 34, "BlockQ8_0 must be 34 bytes");
_Static_assert(sizeof(BlockQ4_0) == 18, "BlockQ4_0 must be 18 bytes");

/* Quantize one row of `n` floats (n must be a multiple of 32). */
void ember_quantize_q8_0(const float *x, BlockQ8_0 *out, int n);
void ember_quantize_q4_0(const float *x, BlockQ4_0 *out, int n);

/* --- matmul: out[d_out] = W[d_out, n_in] * x[n_in] ----------------------- */
/* dtype selects the weight encoding; `xq` is scratch for the quantized
 * activation (sized n_in/32 BlockQ8_0), reused across all output rows. */
void ember_matmul(float *out, const float *x, const void *w, int dtype,
                  int n_in, int d_out, BlockQ8_0 *xq);

/* Batched matmul for prefill: out[B][d_out] = x[B][n_in] * W[d_out,n_in]^T.
 * Loops output rows outermost so each weight row is streamed once for all B
 * tokens — the bandwidth win over calling ember_matmul B times. `xq` is scratch
 * of B*(n_in/32) BlockQ8_0 for the quantized paths. */
void ember_matmul_batch(float *out, const float *x, const void *w, int dtype,
                        int n_in, int d_out, int B, BlockQ8_0 *xq);

/* --- thread pool --------------------------------------------------------- */
typedef void (*EmberRangeFn)(void *ctx, int i0, int i1);
void ember_threads_init(int nthreads);   /* nthreads<=1 disables the pool */
void ember_threads_shutdown(void);
int  ember_nthreads(void);
void ember_parallel_for(EmberRangeFn fn, void *ctx, int n);

/* Minimum work (≈ multiply-accumulates) per thread to make splitting a loop
 * worth the dispatch + barrier cost. Below this, running serially is faster. */
#define EMBER_PAR_MIN_PER_THREAD 32768
/* Should a loop of the given total work be parallelized at the current thread
 * count? True only when each thread would get at least the floor above, so tiny
 * memory-bound ops (small/quantized decode matmuls) stay serial as threads rise. */
int  ember_should_parallel(long work);

#endif /* EMBER_KERNELS_H */
