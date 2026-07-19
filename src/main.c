/* main.c — command-line front end for the ember engine.
 *
 * Modes:
 *   ember info     <model.ember>              inspect header + tensor table
 *   ember generate <model.ember> [opts]       stream a completion
 *
 * (chat and bench modes are added in later stages.)
 */
#include "ember.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *DTYPE_NAME[] = { "f32", "q8_0", "q4_0", "f16" };

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int cmd_info(const char *path) {
    EmberModel *m = ember_model_load(path);
    if (!m) return 1;
    const EmberHeader *h = ember_model_header(m);
    printf("emberllm model: %s\n", path);
    printf("  version      %u\n", h->version);
    printf("  dim          %d\n", h->dim);
    printf("  hidden_dim   %d\n", h->hidden_dim);
    printf("  n_layers     %d\n", h->n_layers);
    printf("  n_heads      %d\n", h->n_heads);
    printf("  n_kv_heads   %d\n", h->n_kv_heads);
    printf("  head_dim     %d\n", h->head_dim);
    printf("  vocab_size   %d\n", h->vocab_size);
    printf("  max_seq_len  %d\n", h->max_seq_len);
    printf("  rope_theta   %g (style %s)\n", h->rope_theta,
           h->rope_style == EMBER_ROPE_NEOX ? "neox" : "interleaved");
    printf("  norm_eps     %g\n", h->norm_eps);
    printf("  flags        %s%s\n",
           (h->flags & EMBER_FLAG_TIED_EMBED) ? "tied_embed " : "",
           (h->flags & EMBER_FLAG_QK_NORM) ? "qk_norm " : "");
    printf("  tokenizer    %s (%llu bytes)\n",
           h->tokenizer_type == EMBER_TOK_BYTE_BPE ? "byte_bpe" : "llama_sp",
           (unsigned long long)h->tokenizer_size);
    printf("  bos/eos      %d / %d%s\n", h->bos_token_id, h->eos_token_id,
           h->eos_token_id2 >= 0 ? " (+2nd stop)" : "");
    printf("  sampling     temp=%g top_p=%g top_k=%d\n",
           h->default_temp, h->default_top_p, h->default_top_k);
    printf("  tensors      %d\n", h->n_tensors);
    const EmberTensor *table =
        (const EmberTensor *)((const uint8_t *)h + h->tensor_table_offset);
    for (int i = 0; i < h->n_tensors && i < 6; i++) {
        const EmberTensor *t = &table[i];
        printf("    %-22s %-4s [%d,%d,%d,%d] @%llu (%llu B)\n",
               t->name, DTYPE_NAME[t->dtype], t->shape[0], t->shape[1], t->shape[2],
               t->shape[3], (unsigned long long)t->offset, (unsigned long long)t->nbytes);
    }
    if (h->n_tensors > 6) printf("    ... (%d more)\n", h->n_tensors - 6);
    ember_model_free(m);
    return 0;
}

/* ember tokenize <model.ember> "text" — print encoded token ids (for tests). */
static int cmd_tokenize(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: ember tokenize <model> \"text\"\n"); return 1; }
    EmberModel *m = ember_model_load(argv[0]);
    if (!m) return 1;
    EmberTokenizer *tok = ember_tokenizer_new(m);
    int *ids, n = ember_encode(tok, argv[1], /*add_bos=*/1, &ids);
    for (int i = 0; i < n; i++) printf("%d%s", ids[i], i + 1 < n ? " " : "\n");
    free(ids);
    ember_tokenizer_free(tok);
    ember_model_free(m);
    return 0;
}

static int cmd_generate(int argc, char **argv) {
    const char *path = argv[0];
    const char *prompt = "";
    int   max_new = 256, ctx = 0;
    float temp = -1.0f, top_p = -1.0f;
    int   top_k = -1;
    uint64_t seed = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) max_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) temp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-p") && i + 1 < argc) top_p = atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-k") && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
    }

    EmberModel *m = ember_model_load(path);
    if (!m) return 1;
    const EmberHeader *h = ember_model_header(m);
    if (temp  < 0) temp  = h->default_temp;
    if (top_p < 0) top_p = h->default_top_p;
    if (top_k < 0) top_k = h->default_top_k;
    if (seed == 0) seed = 0xC0FFEEULL;

    EmberTokenizer *tok = ember_tokenizer_new(m);
    EmberState *st = ember_state_new(m, ctx);
    EmberSampler smp;
    ember_sampler_init(&smp, temp, top_p, top_k, seed);

    int *ids, n_prompt = ember_encode(tok, prompt, /*add_bos=*/1, &ids);
    if (n_prompt < 1) { ids = malloc(sizeof(int)); ids[0] = h->bos_token_id; n_prompt = 1; }

    int steps = max_new;

    /* --- prefill: consume the prompt --- */
    double t0 = now_sec();
    float *logits = NULL;
    int pos = 0;
    for (; pos < n_prompt; pos++) logits = ember_forward(m, st, ids[pos], pos);
    double t_prefill = now_sec() - t0;

    /* echo the prompt, then stream generation */
    fputs(prompt, stdout);

    /* --- decode --- */
    double t1 = now_sec();
    int prev = ids[n_prompt - 1], generated = 0;
    for (int i = 0; i < steps; i++) {
        int next = ember_sample(&smp, logits, h->vocab_size);
        if (next == h->eos_token_id || next == h->eos_token_id2 || next == h->bos_token_id) break;
        fputs(ember_decode(tok, prev, next), stdout);
        fflush(stdout);
        prev = next;
        generated++;
        int p = n_prompt + i;
        if (p >= (ctx > 0 ? ctx : h->max_seq_len) - 1) break;
        logits = ember_forward(m, st, next, p);
    }
    double t_decode = now_sec() - t1;
    printf("\n\n");
    fprintf(stderr,
            "[prefill %d tok in %.3fs = %.1f tok/s | decode %d tok in %.3fs = %.1f tok/s]\n",
            n_prompt, t_prefill, n_prompt / (t_prefill + 1e-9),
            generated, t_decode, generated / (t_decode + 1e-9));

    free(ids);
    ember_state_free(st);
    ember_tokenizer_free(tok);
    ember_model_free(m);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
            "usage:\n"
            "  ember info     <model.ember>\n"
            "  ember generate <model.ember> [-p PROMPT] [-n N] [-t TEMP]\n"
            "                 [--top-p P] [--top-k K] [--seed S] [--ctx C]\n");
        return 1;
    }
    if (!strcmp(argv[1], "info"))     return cmd_info(argv[2]);
    if (!strcmp(argv[1], "tokenize")) return cmd_tokenize(argc - 2, argv + 2);
    if (!strcmp(argv[1], "generate")) return cmd_generate(argc - 2, argv + 2);
    fprintf(stderr, "ember: unknown command '%s'\n", argv[1]);
    return 1;
}
