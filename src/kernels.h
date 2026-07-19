/* kernels.h — internal compute kernels and the thread pool.
 *
 * These are not part of the public API in ember.h; they are shared between
 * model.c (the forward pass), quant.c (offline quantization), and bench code.
 */
#ifndef EMBER_KERNELS_H
#define EMBER_KERNELS_H

#include "ember.h"

/* --- half precision (scalar, portable bit-twiddling) --------------------- */
uint16_t ember_f32_to_f16(float f);
float    ember_f16_to_f32(uint16_t h);

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

/* --- thread pool --------------------------------------------------------- */
typedef void (*EmberRangeFn)(void *ctx, int i0, int i1);
void ember_threads_init(int nthreads);   /* nthreads<=1 disables the pool */
void ember_threads_shutdown(void);
int  ember_nthreads(void);
void ember_parallel_for(EmberRangeFn fn, void *ctx, int n);

#endif /* EMBER_KERNELS_H */
