/* sample.c — logit sampling: greedy, temperature, top-k, top-p (nucleus),
 * min-p, and repetition / presence / frequency penalties.
 *
 * The RNG is a seeded xorshift64* so a given --seed reproduces a run exactly
 * (within one build; float reductions are not bit-stable across builds).
 *
 * Penalties act on the logits before temperature/softmax, using a ring buffer
 * of recently emitted tokens. All penalty controls default to no-ops, so a
 * bare temperature/top-k/top-p run is byte-identical to the pre-penalty engine.
 */
#include "ember.h"
#include "util.h"

#include <math.h>
#include <stdlib.h>

void ember_sampler_init(EmberSampler *s, float temp, float top_p, int top_k, uint64_t seed) {
    s->temperature = temp;
    s->top_p = top_p;
    s->top_k = top_k;
    s->min_p = 0.0f;
    s->repeat_penalty = 1.0f;
    s->presence_penalty = 0.0f;
    s->frequency_penalty = 0.0f;
    s->repeat_last_n = 0;
    s->rng_state = seed ? seed : 0x9E3779B97F4A7C15ULL;
    s->history = NULL;
    s->hist_cap = s->hist_len = s->hist_pos = 0;
}

void ember_sampler_set_penalties(EmberSampler *s, float repeat_penalty, int repeat_last_n,
                                 float presence_penalty, float frequency_penalty, float min_p) {
    s->repeat_penalty = repeat_penalty;
    s->presence_penalty = presence_penalty;
    s->frequency_penalty = frequency_penalty;
    s->min_p = min_p;
    s->repeat_last_n = repeat_last_n;

    int need_history = repeat_last_n > 0 &&
        (repeat_penalty != 1.0f || presence_penalty != 0.0f || frequency_penalty != 0.0f);
    if (need_history) {
        if (s->hist_cap != repeat_last_n) {
            s->history = ember_xrealloc(s->history, sizeof(int) * (size_t)repeat_last_n,
                                        "sampler history");
            s->hist_cap = repeat_last_n;
        }
    } else {
        free(s->history);
        s->history = NULL;
        s->hist_cap = 0;
    }
    s->hist_len = s->hist_pos = 0;
}

void ember_sampler_accept(EmberSampler *s, int token) {
    if (!s->history || s->hist_cap == 0) return;
    s->history[s->hist_pos] = token;
    s->hist_pos = (s->hist_pos + 1) % s->hist_cap;
    if (s->hist_len < s->hist_cap) s->hist_len++;
}

void ember_sampler_reset(EmberSampler *s) { s->hist_len = s->hist_pos = 0; }

void ember_sampler_free(EmberSampler *s) {
    if (!s) return;
    free(s->history);
    s->history = NULL;
    s->hist_cap = s->hist_len = s->hist_pos = 0;
}

/* Apply repetition/presence/frequency penalties to the raw logits, using the
 * emitted-token history. Repetition and presence hit a token once; frequency
 * scales with how many times it appears in the window. When hist_len <= hist_cap
 * the valid entries always occupy history[0..hist_len), so a linear scan is exact
 * (the ring only wraps once it is full, at which point every slot is valid). */
static void apply_penalties(EmberSampler *s, float *logits, int n) {
    if (!s->history || s->hist_len == 0) return;
    int has_rep = s->repeat_penalty != 1.0f;
    int has_freq = s->frequency_penalty != 0.0f;
    int has_pres = s->presence_penalty != 0.0f;
    if (!has_rep && !has_freq && !has_pres) return;

    for (int i = 0; i < s->hist_len; i++) {
        int t = s->history[i];
        if (t < 0 || t >= n) continue;
        int first = 1;
        for (int j = 0; j < i; j++) if (s->history[j] == t) { first = 0; break; }
        if (first) {
            if (has_rep)
                logits[t] = logits[t] > 0.0f ? logits[t] / s->repeat_penalty
                                             : logits[t] * s->repeat_penalty;
            if (has_pres) logits[t] -= s->presence_penalty;
        }
        if (has_freq) logits[t] -= s->frequency_penalty;
    }
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
    apply_penalties(s, logits, n);

    if (s->temperature <= 0.0f) {
        int chosen = argmax(logits, n);
        ember_sampler_accept(s, chosen);
        return chosen;
    }

    for (int i = 0; i < n; i++) logits[i] /= s->temperature;

    /* softmax */
    float max = logits[argmax(logits, n)];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { logits[i] = expf(logits[i] - max); sum += logits[i]; }
    for (int i = 0; i < n; i++) logits[i] /= sum;

    /* candidate set, optionally capped by top_k */
    ProbIdx *cand = ember_xmalloc(sizeof(ProbIdx) * (size_t)n, "sampler candidates");
    for (int i = 0; i < n; i++) { cand[i].prob = logits[i]; cand[i].idx = i; }
    qsort(cand, n, sizeof(ProbIdx), cmp_desc);

    int m = n;
    if (s->top_k > 0 && s->top_k < m) m = s->top_k;

    /* min-p: drop tail below a fraction of the peak probability (cand[0] is max) */
    if (s->min_p > 0.0f && s->min_p < 1.0f) {
        float thresh = s->min_p * cand[0].prob;
        int keep = 1;
        while (keep < m && cand[keep].prob >= thresh) keep++;
        m = keep;
    }

    /* top-p (nucleus): keep the smallest prefix whose mass exceeds top_p */
    float cutoff;
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

    ember_sampler_accept(s, chosen);
    return chosen;
}
