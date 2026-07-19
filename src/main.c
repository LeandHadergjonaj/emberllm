/* main.c — command-line front end for the ember engine.
 *
 * Modes:
 *   ember info     <model.ember>
 *   ember tokenize <model.ember> "text"
 *   ember generate <model.ember> [opts]
 *   ember bench    <model.ember> [opts]
 *   ember quantize <in.ember> <out.ember> [q8_0|q4_0]
 */
#include "ember.h"
#include "kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>

static const char *DTYPE_NAME[] = { "f32", "q8_0", "q4_0", "f16" };

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static long peak_rss_kb(void) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
    return ru.ru_maxrss / 1024; /* bytes on macOS */
#else
    return ru.ru_maxrss;        /* KiB on Linux   */
#endif
}

static int cmd_info(const char *path) {
    EmberModel *m = ember_model_load(path);
    if (!m) return 1;
    const EmberHeader *h = ember_model_header(m);
    printf("emberllm model: %s\n", path);
    printf("  dim %d  hidden %d  layers %d  heads %d/%d  head_dim %d\n",
           h->dim, h->hidden_dim, h->n_layers, h->n_heads, h->n_kv_heads, h->head_dim);
    printf("  vocab %d  max_seq_len %d\n", h->vocab_size, h->max_seq_len);
    printf("  rope_theta %g (%s)  norm_eps %g\n", h->rope_theta,
           h->rope_style == EMBER_ROPE_NEOX ? "neox" : "interleaved", h->norm_eps);
    printf("  flags %s%s\n",
           (h->flags & EMBER_FLAG_TIED_EMBED) ? "tied_embed " : "",
           (h->flags & EMBER_FLAG_QK_NORM) ? "qk_norm " : "");
    printf("  tokenizer %s (%llu B)  bos/eos %d/%d%s\n",
           h->tokenizer_type == EMBER_TOK_BYTE_BPE ? "byte_bpe" : "llama_sp",
           (unsigned long long)h->tokenizer_size, h->bos_token_id, h->eos_token_id,
           h->eos_token_id2 >= 0 ? " (+2nd stop)" : "");
    printf("  sampling temp=%g top_p=%g top_k=%d\n",
           h->default_temp, h->default_top_p, h->default_top_k);
    printf("  tensors %d\n", h->n_tensors);
    const EmberTensor *table =
        (const EmberTensor *)((const uint8_t *)h + h->tensor_table_offset);
    for (int i = 0; i < h->n_tensors && i < 6; i++) {
        const EmberTensor *t = &table[i];
        printf("    %-22s %-4s [%d,%d] @%llu (%llu B)\n", t->name, DTYPE_NAME[t->dtype],
               t->shape[0], t->shape[1], (unsigned long long)t->offset,
               (unsigned long long)t->nbytes);
    }
    if (h->n_tensors > 6) printf("    ... (%d more)\n", h->n_tensors - 6);
    ember_model_free(m);
    return 0;
}

static int cmd_tokenize(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: ember tokenize <model> \"text\"\n"); return 1; }
    EmberModel *m = ember_model_load(argv[0]);
    if (!m) return 1;
    EmberTokenizer *tok = ember_tokenizer_new(m);
    int *ids, n = ember_encode(tok, argv[1], 1, &ids);
    for (int i = 0; i < n; i++) printf("%d%s", ids[i], i + 1 < n ? " " : "\n");
    free(ids);
    ember_tokenizer_free(tok);
    ember_model_free(m);
    return 0;
}

static int cmd_generate(int argc, char **argv) {
    const char *path = argv[0], *prompt = "";
    int max_new = 256, ctx = 0, threads = 1, top_k = -1;
    float temp = -1.0f, top_p = -1.0f;
    uint64_t seed = 0;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) max_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) temp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-p") && i + 1 < argc) top_p = atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-k") && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    }

    EmberModel *m = ember_model_load(path);
    if (!m) return 1;
    const EmberHeader *h = ember_model_header(m);
    if (temp  < 0) temp  = h->default_temp;
    if (top_p < 0) top_p = h->default_top_p;
    if (top_k < 0) top_k = h->default_top_k;
    if (seed == 0) seed = 0xC0FFEEULL;

    ember_threads_init(threads);
    EmberTokenizer *tok = ember_tokenizer_new(m);
    EmberState *st = ember_state_new(m, ctx);
    EmberSampler smp;
    ember_sampler_init(&smp, temp, top_p, top_k, seed);

    int *ids, n_prompt = ember_encode(tok, prompt, 1, &ids);
    if (n_prompt < 1) { ids = malloc(sizeof(int)); ids[0] = h->bos_token_id; n_prompt = 1; }

    double t0 = now_sec();
    float *logits = NULL;
    int pos = 0;
    for (; pos < n_prompt; pos++) logits = ember_forward(m, st, ids[pos], pos);
    double t_prefill = now_sec() - t0;

    fputs(prompt, stdout);
    double t1 = now_sec();
    int prev = ids[n_prompt - 1], generated = 0;
    int limit = (ctx > 0 ? ctx : h->max_seq_len);
    for (int i = 0; i < max_new; i++) {
        int next = ember_sample(&smp, logits, h->vocab_size);
        if (next == h->eos_token_id || next == h->eos_token_id2 || next == h->bos_token_id) break;
        fputs(ember_decode(tok, prev, next), stdout);
        fflush(stdout);
        prev = next;
        generated++;
        if (n_prompt + i >= limit - 1) break;
        logits = ember_forward(m, st, next, n_prompt + i);
    }
    double t_decode = now_sec() - t1;
    printf("\n\n");
    fprintf(stderr,
            "[prefill %d tok @ %.1f tok/s | decode %d tok @ %.1f tok/s | %d thread(s)]\n",
            n_prompt, n_prompt / (t_prefill + 1e-9),
            generated, generated / (t_decode + 1e-9), threads);

    free(ids);
    ember_state_free(st);
    ember_tokenizer_free(tok);
    ember_threads_shutdown();
    ember_model_free(m);
    return 0;
}

static double stddev(const double *v, int n, double mean) {
    if (n < 2) return 0.0;
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (v[i] - mean) * (v[i] - mean);
    return sqrt(s / (n - 1));
}

static int cmd_bench(int argc, char **argv) {
    const char *path = argv[0];
    int pp = 128, tg = 128, threads = 1, reps = 5, warmup = 1;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--pp") && i + 1 < argc) pp = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tg") && i + 1 < argc) tg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
    }

    EmberModel *m = ember_model_load(path);
    if (!m) return 1;
    const EmberHeader *h = ember_model_header(m);
    ember_threads_init(threads);
    EmberState *st = ember_state_new(m, pp + tg + 2);

    int ctx = h->max_seq_len;
    if (pp + tg + 1 > ctx) { pp = ctx / 2; tg = ctx / 2 - 1; }

    double *pp_rate = malloc(sizeof(double) * reps);
    double *tg_rate = malloc(sizeof(double) * reps);
    /* a fixed synthetic token stream keeps the measurement reproducible */
    for (int r = 0; r < reps + warmup; r++) {
        double t0 = now_sec();
        int pos = 0;
        for (; pos < pp; pos++) ember_forward(m, st, (pos * 131 + 7) % h->vocab_size, pos);
        double t_pp = now_sec() - t0;

        double t1 = now_sec();
        for (int i = 0; i < tg; i++, pos++) ember_forward(m, st, (pos * 131 + 7) % h->vocab_size, pos);
        double t_tg = now_sec() - t1;

        if (r >= warmup) {
            pp_rate[r - warmup] = pp / t_pp;
            tg_rate[r - warmup] = tg / t_tg;
        }
    }

    double pp_mean = 0, tg_mean = 0;
    for (int r = 0; r < reps; r++) { pp_mean += pp_rate[r]; tg_mean += tg_rate[r]; }
    pp_mean /= reps; tg_mean /= reps;

    /* work out the dominant weight dtype for the report */
    const EmberTensor *table =
        (const EmberTensor *)((const uint8_t *)h + h->tensor_table_offset);
    int qtype = EMBER_DT_F32;
    for (int i = 0; i < h->n_tensors; i++)
        if (table[i].dtype != EMBER_DT_F32) { qtype = table[i].dtype; break; }

    printf("model=%s params~%dL/%dd quant=%s threads=%d\n",
           path, h->n_layers, h->dim, DTYPE_NAME[qtype], threads);
    printf("  prefill (pp%d): %8.1f tok/s  (+/- %.1f)\n", pp, pp_mean, stddev(pp_rate, reps, pp_mean));
    printf("  decode  (tg%d): %8.1f tok/s  (+/- %.1f)\n", tg, tg_mean, stddev(tg_rate, reps, tg_mean));
    printf("  peak RSS: %ld MB  (note: mmap'd weight pages may not count toward RSS)\n",
           peak_rss_kb() / 1024);

    free(pp_rate); free(tg_rate);
    ember_state_free(st);
    ember_threads_shutdown();
    ember_model_free(m);
    return 0;
}

static int cmd_quantize(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: ember quantize <in.ember> <out.ember> [q8_0|q4_0]\n"); return 1; }
    int target = EMBER_DT_Q8_0;
    if (argc >= 3 && !strcmp(argv[2], "q4_0")) target = EMBER_DT_Q4_0;
    return ember_quantize_file(argv[0], argv[1], target);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
            "usage:\n"
            "  ember info     <model.ember>\n"
            "  ember tokenize <model.ember> \"text\"\n"
            "  ember generate <model.ember> [-p PROMPT] [-n N] [-t TEMP]\n"
            "                 [--top-p P] [--top-k K] [--seed S] [--ctx C] [--threads T]\n"
            "  ember bench    <model.ember> [--pp P] [--tg N] [--threads T] [--reps R]\n"
            "  ember quantize <in.ember> <out.ember> [q8_0|q4_0]\n");
        return 1;
    }
    if (!strcmp(argv[1], "info"))     return cmd_info(argv[2]);
    if (!strcmp(argv[1], "tokenize")) return cmd_tokenize(argc - 2, argv + 2);
    if (!strcmp(argv[1], "generate")) return cmd_generate(argc - 2, argv + 2);
    if (!strcmp(argv[1], "bench"))    return cmd_bench(argc - 2, argv + 2);
    if (!strcmp(argv[1], "quantize")) return cmd_quantize(argc - 2, argv + 2);
    fprintf(stderr, "ember: unknown command '%s'\n", argv[1]);
    return 1;
}
