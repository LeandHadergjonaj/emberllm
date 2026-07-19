/* tokenizer.c — SentencePiece-BPE tokenizer (EMBER_TOK_LLAMA_SP).
 *
 * This reimplements the classic Llama/SentencePiece byte-BPE scheme used by the
 * TinyStories checkpoints: an initial codepoint/byte-fallback tokenization
 * followed by greedy score-ranked pair merging. The embedded tokenizer blob is
 * exactly the layout llama2.c's tokenizer.py writes:
 *
 *   uint32 max_token_length
 *   repeat vocab_size:  float32 score, uint32 len, byte[len] piece
 *
 * Byte-fallback tokens live at ids 3..258 ("<0x00>".."<0xFF>"), so an unknown
 * byte encodes to (byte + 3); decode turns "<0xXX>" pieces back into raw bytes.
 */
#include "ember.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const char *str; int id; } VocabEntry;

struct EmberTokenizer {
    int    vocab_size;
    char **vocab;     /* null-terminated copies of each piece      */
    float *scores;
    int    max_token_length;
    VocabEntry *sorted; /* sorted by str for binary search         */
    char   byte_pieces[256 * 2]; /* single-byte string per byte    */
    int    bos_id, eos_id;
};

static int vocab_cmp(const void *a, const void *b) {
    return strcmp(((const VocabEntry *)a)->str, ((const VocabEntry *)b)->str);
}

static int str_lookup(EmberTokenizer *t, const char *s) {
    VocabEntry key = { s, -1 };
    VocabEntry *r = bsearch(&key, t->sorted, t->vocab_size, sizeof(VocabEntry), vocab_cmp);
    return r ? r->id : -1;
}

EmberTokenizer *ember_tokenizer_new(const EmberModel *m) {
    const EmberHeader *h = ember_model_header(m);
    const uint8_t *base = (const uint8_t *)h;
    const uint8_t *p = base + h->tokenizer_offset;

    EmberTokenizer *t = calloc(1, sizeof(*t));
    t->vocab_size = h->vocab_size;
    t->bos_id = h->bos_token_id;
    t->eos_id = h->eos_token_id;
    for (int b = 0; b < 256; b++) {
        t->byte_pieces[b * 2] = (char)b;
        t->byte_pieces[b * 2 + 1] = '\0';
    }

    memcpy(&t->max_token_length, p, 4); p += 4;
    t->vocab  = malloc(sizeof(char *) * t->vocab_size);
    t->scores = malloc(sizeof(float) * t->vocab_size);
    t->sorted = malloc(sizeof(VocabEntry) * t->vocab_size);
    for (int i = 0; i < t->vocab_size; i++) {
        float score; int len;
        memcpy(&score, p, 4); p += 4;
        memcpy(&len,   p, 4); p += 4;
        char *s = malloc(len + 1);
        memcpy(s, p, len); s[len] = '\0'; p += len;
        t->vocab[i]  = s;
        t->scores[i] = score;
        t->sorted[i].str = s;
        t->sorted[i].id  = i;
    }
    qsort(t->sorted, t->vocab_size, sizeof(VocabEntry), vocab_cmp);
    return t;
}

void ember_tokenizer_free(EmberTokenizer *t) {
    if (!t) return;
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab); free(t->scores); free(t->sorted); free(t);
}

int ember_encode(EmberTokenizer *t, const char *text, int add_bos, int **out_ids) {
    size_t slen = strlen(text);
    /* worst case: BOS + dummy-prefix + one token per byte */
    int *tokens = malloc(sizeof(int) * (slen + 3));
    int n = 0;

    if (add_bos) tokens[n++] = t->bos_id;

    /* SentencePiece dummy prefix: a leading space piece before real text */
    if (text[0] != '\0') {
        int sp = str_lookup(t, " ");
        if (sp != -1) tokens[n++] = sp;
    }

    /* initial pass: group UTF-8 continuation bytes into codepoints, look each
     * up whole; fall back to raw bytes (id = byte + 3) when absent */
    char *cp = malloc(t->max_token_length * 2 + 2);
    for (const char *c = text; *c; ) {
        int clen = 1;
        unsigned char b0 = (unsigned char)c[0];
        if      (b0 >= 0xF0) clen = 4;
        else if (b0 >= 0xE0) clen = 3;
        else if (b0 >= 0xC0) clen = 2;
        for (int k = 0; k < clen && c[k]; k++) cp[k] = c[k];
        int used = clen; while (used > 0 && !c[used - 1]) used--;
        cp[used] = '\0';

        int id = str_lookup(t, cp);
        if (id != -1) {
            tokens[n++] = id;
        } else {
            for (int k = 0; k < used; k++)
                tokens[n++] = (unsigned char)cp[k] + 3;
        }
        c += used;
    }

    /* greedy merge: repeatedly fuse the adjacent pair with the best score */
    for (;;) {
        float best_score = -1e30f;
        int best_id = -1, best_idx = -1;
        for (int i = 0; i < n - 1; i++) {
            snprintf(cp, t->max_token_length * 2 + 2, "%s%s",
                     t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(t, cp);
            if (id != -1 && t->scores[id] > best_score) {
                best_score = t->scores[id]; best_id = id; best_idx = i;
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        memmove(&tokens[best_idx + 1], &tokens[best_idx + 2],
                (n - best_idx - 2) * sizeof(int));
        n--;
    }
    free(cp);
    *out_ids = tokens;
    return n;
}

const char *ember_decode(EmberTokenizer *t, int prev_tok, int tok) {
    const char *piece = t->vocab[tok];
    /* right after BOS, SentencePiece strips a single leading space */
    if (prev_tok == t->bos_id && piece[0] == ' ') piece++;
    /* raw byte tokens are stored as "<0xXX>" — turn back into the byte */
    unsigned int byte_val;
    if (sscanf(piece, "<0x%02X>", &byte_val) == 1)
        return &t->byte_pieces[(byte_val & 0xFF) * 2];
    return piece;
}
