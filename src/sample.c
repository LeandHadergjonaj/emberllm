/* sample.c — logit sampling: greedy, temperature, top-k, top-p (nucleus).
 *
 * The RNG is a seeded xorshift64* so a given --seed reproduces a run exactly
 * (within one build; float reductions are not bit-stable across builds).
 */
#include "ember.h"

#include <math.h>
#include <stdlib.h>

void ember_sampler_init(EmberSampler *s, float temp, float top_p, int top_k, uint64_t seed) {
    s->temperature = temp;
    s->top_p = top_p;
    s->top_k = top_k;
    s->rng_state = seed ? seed : 0x9E3779B97F4A7C15ULL;
}

static uint32_t rng_u32(EmberSampler *s) {
    uint64_t x = s->rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    s->rng_state = x;
    return (uint32_t)((x * 0x2545F4914F6CDD1DULL) >> 32);
}

static float rng_f32(EmberSampler *s) { return (rng_u32(s) >> 8) / 16777216.0f; }

typedef struct { float prob; int idx; } ProbIdx;

static int cmp_desc(const void *a, const void *b) {
    float pa = ((const ProbIdx *)a)->prob, pb = ((const ProbIdx *)b)->prob;
    return (pa < pb) - (pa > pb);
}

static int argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

int ember_sample(EmberSampler *s, float *logits, int n) {
    if (s->temperature <= 0.0f) return argmax(logits, n);

    for (int i = 0; i < n; i++) logits[i] /= s->temperature;

    /* softmax */
    float max = logits[argmax(logits, n)];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { logits[i] = expf(logits[i] - max); sum += logits[i]; }
    for (int i = 0; i < n; i++) logits[i] /= sum;

    /* candidate set, optionally capped by top_k */
    ProbIdx *cand = malloc(sizeof(ProbIdx) * n);
    for (int i = 0; i < n; i++) { cand[i].prob = logits[i]; cand[i].idx = i; }
    qsort(cand, n, sizeof(ProbIdx), cmp_desc);

    int m = n;
    if (s->top_k > 0 && s->top_k < m) m = s->top_k;

    /* top-p (nucleus): keep the smallest prefix whose mass exceeds top_p */
    float cutoff = 1.0f;
    if (s->top_p > 0.0f && s->top_p < 1.0f) {
        float cum = 0.0f;
        int last = m - 1;
        for (int i = 0; i < m; i++) { cum += cand[i].prob; if (cum >= s->top_p) { last = i; break; } }
        m = last + 1;
        cutoff = cum;
    } else {
        cutoff = 0.0f;
        for (int i = 0; i < m; i++) cutoff += cand[i].prob;
    }

    /* sample within the truncated, renormalized distribution */
    float r = rng_f32(s) * cutoff;
    float cum = 0.0f;
    int chosen = cand[m - 1].idx;
    for (int i = 0; i < m; i++) { cum += cand[i].prob; if (r < cum) { chosen = cand[i].idx; break; } }
    free(cand);
    return chosen;
}
