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
#include "server.h"

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
    printf("  flags %s%s%s\n",
           (h->flags & EMBER_FLAG_TIED_EMBED) ? "tied_embed " : "",
           (h->flags & EMBER_FLAG_QK_NORM) ? "qk_norm " : "",
           (h->flags & EMBER_FLAG_THINKING) ? "thinking " : "");
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
    if (argc < 2) { fprintf(stderr, "usage: ember tokenize <model> \"text\" [--no-bos]\n"); return 1; }
    int add_bos = 1;
    for (int i = 2; i < argc; i++) if (!strcmp(argv[i], "--no-bos")) add_bos = 0;
    EmberModel *m = ember_model_load(argv[0]);
    if (!m) return 1;
    EmberTokenizer *tok = ember_tokenizer_new(m);
    int *ids, n = ember_encode(tok, argv[1], add_bos, &ids);
    for (int i = 0; i < n; i++) printf("%d%s", ids[i], i + 1 < n ? " " : "\n");
    free(ids);
    ember_tokenizer_free(tok);
    ember_model_free(m);
    return 0;
}

/* Auto-tune the decode thread count by measuring: a fixed flop-based gate can't
 * tell that small quantized decode matmuls are memory-bound and don't parallelize,
 * so we just time a few forward() passes at each candidate count and keep the
 * fastest. The warm-up passes write throwaway KV entries at low positions, which
 * real generation overwrites from pos 0 (and never reads past its own position),
 * so this is safe to run on a fresh state before prefill. */
static int autotune_threads(EmberModel *m, EmberState *s) {
    static const int cand[] = { 1, 2, 4, 6 };
    int best = 1;
    double best_rate = 0.0;
    for (size_t i = 0; i < sizeof(cand) / sizeof(cand[0]); i++) {
        int c = cand[i];
        ember_threads_init(c);
        for (int p = 0; p < 3; p++) ember_forward(m, s, 0, p);       /* warm up */
        double t0 = now_sec();
        int N = 12;
        for (int k = 0; k < N; k++) ember_forward(m, s, 0, 3 + (k % 28));
        double rate = N / (now_sec() - t0);
        ember_threads_shutdown();
        if (rate > best_rate) { best_rate = rate; best = c; }
    }
    ember_threads_init(best);
    fprintf(stderr, "[autotuned decode: %d thread(s), ~%.0f tok/s]\n", best, best_rate);
    return best;
}

/* Parse a --threads value that may be the literal "auto" (sentinel -1). */
static int parse_threads(const char *v) { return strcmp(v, "auto") == 0 ? -1 : atoi(v); }

/* Parse a --kv-type value: "f16" -> fp16 cache, anything else -> fp32. */
static int parse_kv_type(const char *v) { return strcmp(v, "f16") == 0; }

static int has_help(int argc, char **argv) {
    for (int i = 0; i < argc; i++)
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) return 1;
    return 0;
}

static const char GENERATE_HELP[] =
    "usage: ember generate <model.ember> [options]\n"
    "  -p PROMPT            prompt text (default: empty -> BOS only)\n"
    "  -n N                 max new tokens (default 256)\n"
    "  -t TEMP              temperature; 0 = greedy (default: model's)\n"
    "  --top-p P            nucleus sampling threshold\n"
    "  --top-k K            keep only the K most likely tokens\n"
    "  --min-p P            keep tokens with prob >= P * peak (0 = off)\n"
    "  --repeat-penalty R   penalize repeats; >1 discourages (1.0 = off)\n"
    "  --repeat-last-n N    penalty window in tokens (default 64)\n"
    "  --presence-penalty P subtract P from any token seen in the window\n"
    "  --frequency-penalty F subtract F per occurrence in the window\n"
    "  --seed S             RNG seed for reproducible sampling\n"
    "  --ctx C              cap the context length\n"
    "  --kv-type f32|f16    KV cache precision (f16 halves its memory; default f32)\n"
    "  --threads T|auto     decode threads (default 1)\n";

static int cmd_generate(int argc, char **argv) {
    if (argc < 1 || has_help(argc, argv)) { fputs(GENERATE_HELP, stdout); return argc < 1; }
    const char *path = argv[0], *prompt = "";
    int max_new = 256, ctx = 0, threads = 1, top_k = -1, repeat_last_n = 64, kv_fp16 = 0;
    float temp = -1.0f, top_p = -1.0f, min_p = 0.0f;
    float repeat_penalty = 1.0f, presence_penalty = 0.0f, frequency_penalty = 0.0f;
    uint64_t seed = 0;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--kv-type") && i + 1 < argc) kv_fp16 = parse_kv_type(argv[++i]);
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) max_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) temp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-p") && i + 1 < argc) top_p = atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-k") && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--min-p") && i + 1 < argc) min_p = atof(argv[++i]);
        else if (!strcmp(argv[i], "--repeat-penalty") && i + 1 < argc) repeat_penalty = atof(argv[++i]);
        else if (!strcmp(argv[i], "--repeat-last-n") && i + 1 < argc) repeat_last_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--presence-penalty") && i + 1 < argc) presence_penalty = atof(argv[++i]);
        else if (!strcmp(argv[i], "--frequency-penalty") && i + 1 < argc) frequency_penalty = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = parse_threads(argv[++i]);
        else { fprintf(stderr, "ember: unknown generate option '%s'\n", argv[i]); return 1; }
    }

    EmberModel *m = ember_model_load(path);
    if (!m) return 1;
    const EmberHeader *h = ember_model_header(m);
    if (temp  < 0) temp  = h->default_temp;
    if (top_p < 0) top_p = h->default_top_p;
    if (top_k < 0) top_k = h->default_top_k;
    if (seed == 0) seed = 0xC0FFEEULL;

    EmberTokenizer *tok = ember_tokenizer_new(m);
    EmberState *st = ember_state_new_ex(m, ctx, kv_fp16);
    if (threads < 0) threads = autotune_threads(m, st); /* --threads auto */
    else ember_threads_init(threads);
    EmberSampler smp;
    ember_sampler_init(&smp, temp, top_p, top_k, seed);
    ember_sampler_set_penalties(&smp, repeat_penalty, repeat_last_n,
                                presence_penalty, frequency_penalty, min_p);

    int *ids, n_prompt = ember_encode(tok, prompt, 1, &ids);
    if (n_prompt < 1) { ids = malloc(sizeof(int)); ids[0] = h->bos_token_id; n_prompt = 1; }
    for (int i = 0; i < n_prompt; i++) ember_sampler_accept(&smp, ids[i]); /* seed penalty window */

    double t0 = now_sec();
    float *logits = ember_prefill(m, st, ids, n_prompt, 0);
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
    ember_sampler_free(&smp);
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
    int pp = 128, tg = 128, threads = 1, reps = 5, warmup = 1, kv_fp16 = 0;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--pp") && i + 1 < argc) pp = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tg") && i + 1 < argc) tg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kv-type") && i + 1 < argc) kv_fp16 = parse_kv_type(argv[++i]);
    }

    EmberModel *m = ember_model_load(path);
    if (!m) return 1;
    const EmberHeader *h = ember_model_header(m);
    ember_threads_init(threads);
    EmberState *st = ember_state_new_ex(m, pp + tg + 2, kv_fp16);

    int ctx = h->max_seq_len;
    if (pp + tg + 1 > ctx) { pp = ctx / 2; tg = ctx / 2 - 1; }

    double *pp_rate = malloc(sizeof(double) * reps);
    double *tg_rate = malloc(sizeof(double) * reps);
    int *pp_ids = malloc(sizeof(int) * pp);
    for (int i = 0; i < pp; i++) pp_ids[i] = (i * 131 + 7) % h->vocab_size;
    /* a fixed synthetic token stream keeps the measurement reproducible */
    for (int r = 0; r < reps + warmup; r++) {
        double t0 = now_sec();
        ember_prefill(m, st, pp_ids, pp, 0);  /* batched prefill */
        int pos = pp;
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

    free(pp_rate); free(tg_rate); free(pp_ids);
    ember_state_free(st);
    ember_threads_shutdown();
    ember_model_free(m);
    return 0;
}

/* Encode `text` and run it through the model, advancing *pos; returns the
 * logits after the last token (NULL if nothing was fed). */
static float *feed(EmberModel *m, EmberState *s, EmberTokenizer *tok,
                   const char *text, int *pos, int ctx) {
    int *ids, n = ember_encode(tok, text, 0, &ids);
    if (*pos + n > ctx) n = ctx - *pos;
    float *logits = (n > 0) ? ember_prefill(m, s, ids, n, *pos) : NULL;
    *pos += n;
    free(ids);
    return logits;
}

static const char CHAT_HELP[] =
    "usage: ember chat <model.ember> [options]\n"
    "  --system S           system prompt\n"
    "  --think              let the model emit <think> reasoning\n"
    "  -n N                 max new tokens per reply (default 512)\n"
    "  -t TEMP              temperature (default: model's)\n"
    "  --top-p P            nucleus sampling threshold\n"
    "  --top-k K            keep only the K most likely tokens\n"
    "  --min-p P            keep tokens with prob >= P * peak\n"
    "  --repeat-penalty R   penalize repeats (default 1.1; 1.0 = off)\n"
    "  --repeat-last-n N    penalty window in tokens (default 256)\n"
    "  --presence-penalty P subtract P from any token seen in the window\n"
    "  --frequency-penalty F subtract F per occurrence in the window\n"
    "  --seed S             RNG seed\n"
    "  --ctx C              context length (default 4096)\n"
    "  --kv-type f32|f16    KV cache precision (f16 halves its memory; default f32)\n"
    "  --threads T|auto     decode threads (default 1)\n";

static int cmd_chat(int argc, char **argv) {
    if (argc < 1 || has_help(argc, argv)) { fputs(CHAT_HELP, stdout); return argc < 1; }
    const char *path = argv[0], *system = NULL;
    int threads = 1, ctx = 4096, max_new = 512, think = 0, top_k = -1, repeat_last_n = 256, kv_fp16 = 0;
    float temp = -1.0f, top_p = -1.0f, min_p = 0.0f;
    /* small chat models loop badly without a repeat penalty, so chat defaults it on */
    float repeat_penalty = 1.1f, presence_penalty = 0.0f, frequency_penalty = 0.0f;
    uint64_t seed = 0;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = parse_threads(argv[++i]);
        else if (!strcmp(argv[i], "--kv-type") && i + 1 < argc) kv_fp16 = parse_kv_type(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) max_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--system") && i + 1 < argc) system = argv[++i];
        else if (!strcmp(argv[i], "--think")) think = 1;
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) temp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-p") && i + 1 < argc) top_p = atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-k") && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--min-p") && i + 1 < argc) min_p = atof(argv[++i]);
        else if (!strcmp(argv[i], "--repeat-penalty") && i + 1 < argc) repeat_penalty = atof(argv[++i]);
        else if (!strcmp(argv[i], "--repeat-last-n") && i + 1 < argc) repeat_last_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--presence-penalty") && i + 1 < argc) presence_penalty = atof(argv[++i]);
        else if (!strcmp(argv[i], "--frequency-penalty") && i + 1 < argc) frequency_penalty = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else { fprintf(stderr, "ember: unknown chat option '%s'\n", argv[i]); return 1; }
    }

    EmberModel *m = ember_model_load(path);
    if (!m) return 1;
    const EmberHeader *h = ember_model_header(m);
    if (temp  < 0) temp  = h->default_temp;
    if (top_p < 0) top_p = h->default_top_p;
    if (top_k < 0) top_k = h->default_top_k;
    if (seed == 0) seed = 0xC0FFEEULL;
    if (ctx > h->max_seq_len) ctx = h->max_seq_len;

    EmberTokenizer *tok = ember_tokenizer_new(m);
    EmberState *st = ember_state_new_ex(m, ctx, kv_fp16);
    if (threads < 0) threads = autotune_threads(m, st); /* --threads auto */
    else ember_threads_init(threads);
    EmberSampler smp;
    ember_sampler_init(&smp, temp, top_p, top_k, seed);
    ember_sampler_set_penalties(&smp, repeat_penalty, repeat_last_n,
                                presence_penalty, frequency_penalty, min_p);

    int pos = 0;
    char buf[8192];
    if (system) {
        snprintf(buf, sizeof(buf), "<|im_start|>system\n%s<|im_end|>\n", system);
        feed(m, st, tok, buf, &pos, ctx);
    }
    fprintf(stderr, "emberllm chat — %d thread(s), temp %.2f, %s. Ctrl-D to exit.\n",
            threads, temp, think ? "thinking on" : "thinking off");

    for (;;) {
        fputs("\n> ", stderr);
        fflush(stderr);
        if (!fgets(buf, sizeof(buf), stdin)) break;
        size_t L = strlen(buf);
        if (L && buf[L - 1] == '\n') buf[--L] = '\0';
        if (L == 0) continue;

        /* Qwen3's <think> block is only injected to *suppress* its reasoning
         * mode; models without that mode must not see these literal tokens. */
        int no_think = (h->flags & EMBER_FLAG_THINKING) && !think;
        char turn[8300];
        snprintf(turn, sizeof(turn),
                 "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n%s",
                 buf, no_think ? "<think>\n\n</think>\n\n" : "");
        float *logits = feed(m, st, tok, turn, &pos, ctx);

        int prev = 0;
        for (int i = 0; i < max_new && pos < ctx - 1; i++) {
            int next = ember_sample(&smp, logits, h->vocab_size);
            if (next == h->eos_token_id || next == h->eos_token_id2) break;
            fputs(ember_decode(tok, prev, next), stdout);
            fflush(stdout);
            prev = next;
            logits = ember_forward(m, st, next, pos++);
        }
        feed(m, st, tok, "<|im_end|>\n", &pos, ctx); /* keep context consistent */
        fputc('\n', stdout);
        if (pos >= ctx - 1) { fprintf(stderr, "[context full]\n"); break; }
    }

    ember_sampler_free(&smp);
    ember_state_free(st);
    ember_tokenizer_free(tok);
    ember_threads_shutdown();
    ember_model_free(m);
    return 0;
}

/* ember perplexity <model> <textfile> — mean NLL / perplexity over a corpus.
 * The quality gate for quantization: expect Q8_0 within a hair of fp32. */
static int cmd_perplexity(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: ember perplexity <model> <textfile> [--threads T] [--max N]\n"); return 1; }
    const char *path = argv[0], *file = argv[1];
    int threads = 1, maxtok = 2048, kv_fp16 = 0;
    for (int i = 2; i < argc; i++) {
        if      (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max") && i + 1 < argc) maxtok = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kv-type") && i + 1 < argc) kv_fp16 = parse_kv_type(argv[++i]);
    }

    FILE *f = fopen(file, "rb");
    if (!f) { fprintf(stderr, "ember: cannot open %s\n", file); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *text = malloc(sz + 1);
    if (fread(text, 1, sz, f) != (size_t)sz) { fclose(f); free(text); return 1; }
    text[sz] = '\0'; fclose(f);

    EmberModel *m = ember_model_load(path);
    if (!m) { free(text); return 1; }
    const EmberHeader *h = ember_model_header(m);
    ember_threads_init(threads);
    EmberTokenizer *tok = ember_tokenizer_new(m);
    int ctx = maxtok + 2 < h->max_seq_len ? maxtok + 2 : h->max_seq_len;
    EmberState *st = ember_state_new_ex(m, ctx, kv_fp16);

    int *ids, n = ember_encode(tok, text, 0, &ids);
    if (n > maxtok) n = maxtok;

    double nll = 0.0;
    int count = 0;
    double t0 = now_sec();
    for (int i = 0; i < n - 1 && i < ctx - 1; i++) {
        float *logits = ember_forward(m, st, ids[i], i);
        int tgt = ids[i + 1];
        float max = logits[0];
        for (int j = 1; j < h->vocab_size; j++) if (logits[j] > max) max = logits[j];
        double sum = 0.0;
        for (int j = 0; j < h->vocab_size; j++) sum += exp(logits[j] - max);
        nll += -((double)logits[tgt] - max - log(sum));
        count++;
    }
    double ppl = exp(nll / count);
    printf("perplexity %.4f  (nll %.4f over %d tokens, %.1fs)\n",
           ppl, nll / count, count, now_sec() - t0);

    free(ids); free(text);
    ember_state_free(st);
    ember_tokenizer_free(tok);
    ember_threads_shutdown();
    ember_model_free(m);
    return 0;
}

static const char SERVE_HELP[] =
    "usage: ember serve <model.ember> [options]\n"
    "  --host H             bind address (default 127.0.0.1)\n"
    "  --port P             listen port (default 8080)\n"
    "  --ctx C              context length (default: model max)\n"
    "  --kv-type f32|f16    KV cache precision (f16 halves its memory; default f32)\n"
    "  --threads T|auto     decode threads (default 1)\n"
    "  --think              allow <think> reasoning in replies (Qwen-style)\n"
    "  -n N                 default max_tokens when a request omits it (default 512)\n"
    "  -t TEMP              default temperature\n"
    "  --top-p P            default nucleus threshold\n"
    "  --top-k K            default top-k\n"
    "  --repeat-penalty R   default repetition penalty (default 1.1)\n"
    "  --seed S             base RNG seed\n"
    "\n"
    "Exposes an OpenAI-compatible API: POST /v1/chat/completions (SSE when\n"
    "\"stream\": true), GET /v1/models, GET /health. Single stream at a time.\n";

static int cmd_serve(int argc, char **argv) {
    if (argc < 1 || has_help(argc, argv)) { fputs(SERVE_HELP, stdout); return argc < 1; }
    ServeConfig cfg = {
        .host = "127.0.0.1", .port = 8080, .threads = 1, .ctx = 0, .kv_fp16 = 0, .think = 0,
        .max_tokens = 512, .temperature = -1.0f, .top_p = -1.0f, .top_k = -1,
        .repeat_penalty = 1.1f, .seed = 0xC0FFEEULL,
    };
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--host") && i + 1 < argc) cfg.host = argv[++i];
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) cfg.port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) cfg.ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kv-type") && i + 1 < argc) cfg.kv_fp16 = parse_kv_type(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) cfg.threads = parse_threads(argv[++i]);
        else if (!strcmp(argv[i], "--think")) cfg.think = 1;
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) cfg.max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) cfg.temperature = atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-p") && i + 1 < argc) cfg.top_p = atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-k") && i + 1 < argc) cfg.top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--repeat-penalty") && i + 1 < argc) cfg.repeat_penalty = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) cfg.seed = strtoull(argv[++i], NULL, 10);
        else { fprintf(stderr, "ember: unknown serve option '%s'\n", argv[i]); return 1; }
    }

    /* fill sampling defaults from the model header where the user left them auto */
    EmberModel *probe = ember_model_load(argv[0]);
    if (!probe) return 1;
    const EmberHeader *h = ember_model_header(probe);
    if (cfg.temperature < 0) cfg.temperature = h->default_temp;
    if (cfg.top_p < 0) cfg.top_p = h->default_top_p;
    if (cfg.top_k < 0) cfg.top_k = h->default_top_k;
    ember_model_free(probe);

    return ember_serve(argv[0], &cfg);
}

static int cmd_quantize(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: ember quantize <in.ember> <out.ember> [q8_0|q4_0]\n"); return 1; }
    int target = EMBER_DT_Q8_0;
    if (argc >= 3 && !strcmp(argv[2], "q4_0")) target = EMBER_DT_Q4_0;
    return ember_quantize_file(argv[0], argv[1], target);
}

static const char TOP_USAGE[] =
    "emberllm — a from-scratch CPU LLM inference engine.\n"
    "\n"
    "usage: ember <command> [args]   (run 'ember <command> --help' for options)\n"
    "\n"
    "  info       <model.ember>                 print model metadata\n"
    "  tokenize   <model.ember> \"text\"           encode text to token ids\n"
    "  generate   <model.ember> [options]       one-shot text completion\n"
    "  chat       <model.ember> [options]       interactive chat (ChatML)\n"
    "  serve      <model.ember> [options]       OpenAI-compatible HTTP server\n"
    "  bench      <model.ember> [options]       prefill/decode throughput\n"
    "  perplexity <model.ember> <textfile>      quality gate over a corpus\n"
    "  quantize   <in.ember> <out.ember> [q8_0|q4_0]   requantize weights\n";

int main(int argc, char **argv) {
    if (argc >= 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h") || !strcmp(argv[1], "help"))) {
        fputs(TOP_USAGE, stdout);
        return 0;
    }
    if (argc < 3) { fputs(TOP_USAGE, stderr); return 1; }
    if (!strcmp(argv[1], "info"))     return cmd_info(argv[2]);
    if (!strcmp(argv[1], "tokenize")) return cmd_tokenize(argc - 2, argv + 2);
    if (!strcmp(argv[1], "generate")) return cmd_generate(argc - 2, argv + 2);
    if (!strcmp(argv[1], "chat"))     return cmd_chat(argc - 2, argv + 2);
    if (!strcmp(argv[1], "serve"))    return cmd_serve(argc - 2, argv + 2);
    if (!strcmp(argv[1], "bench"))    return cmd_bench(argc - 2, argv + 2);
    if (!strcmp(argv[1], "perplexity")) return cmd_perplexity(argc - 2, argv + 2);
    if (!strcmp(argv[1], "quantize")) return cmd_quantize(argc - 2, argv + 2);
    fprintf(stderr, "ember: unknown command '%s'\n", argv[1]);
    return 1;
}
