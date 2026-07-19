/* tokenizer.c — both tokenizers behind one interface.
 *
 *   EMBER_TOK_LLAMA_SP  SentencePiece-BPE, score-ranked merges (TinyStories)
 *   EMBER_TOK_BYTE_BPE  GPT-2/Qwen byte-level BPE, rank-ranked merges
 *
 * The byte-level path is the harder one: text is split by an approximation of
 * the GPT-4/Qwen pre-tokenizer regex, each piece is mapped through the GPT-2
 * byte<->unicode bijection, then merged greedily by merge rank. Special tokens
 * (<|im_start|> etc.) are matched before pre-tokenization and bypass BPE.
 */
#include "ember.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* small open-addressing string hash map                              */
/* ------------------------------------------------------------------ */
typedef struct { const char *key; int klen; int val; } HEntry;
typedef struct { HEntry *e; int cap; int mask; } HMap;

static uint64_t fnv1a(const char *s, int n) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ULL; }
    return h;
}

static void hmap_init(HMap *m, int want) {
    int cap = 16;
    while (cap < want * 2) cap <<= 1;
    m->cap = cap; m->mask = cap - 1;
    m->e = calloc(cap, sizeof(HEntry));
    for (int i = 0; i < cap; i++) m->e[i].val = -1;
}

static void hmap_put(HMap *m, const char *key, int klen, int val) {
    uint64_t i = fnv1a(key, klen) & m->mask;
    while (m->e[i].key) {
        if (m->e[i].klen == klen && memcmp(m->e[i].key, key, klen) == 0) { m->e[i].val = val; return; }
        i = (i + 1) & m->mask;
    }
    m->e[i].key = key; m->e[i].klen = klen; m->e[i].val = val;
}

static int hmap_get(const HMap *m, const char *key, int klen) {
    uint64_t i = fnv1a(key, klen) & m->mask;
    while (m->e[i].key) {
        if (m->e[i].klen == klen && memcmp(m->e[i].key, key, klen) == 0) return m->e[i].val;
        i = (i + 1) & m->mask;
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/* tokenizer state                                                    */
/* ------------------------------------------------------------------ */
typedef struct { const char *str; int len; int score_or_special; } Vocab;

struct EmberTokenizer {
    int type, vocab_size, bos_id, eos_id;

    /* llama_sp */
    char  *sp_blob;        /* owned copy of tokenizer blob for null-terminated pieces */
    char **sp_vocab;
    float *sp_scores;
    int    sp_max_len;
    HMap   sp_map;
    char   byte_pieces[256 * 2];

    /* byte_bpe */
    const uint8_t *bpe_blob;  /* points into the mmap */
    Vocab *bpe_vocab;         /* by id: str (into blob), len, is_special */
    HMap   bpe_vmap;          /* byte-level string -> id            */
    HMap   bpe_ranks;         /* "A\0B" -> merge rank               */
    char **rank_keys;         /* owned "A\0B" key buffers           */
    int    n_merges;
    struct { const char *s; int len; int id; } *specials;
    int    n_special;
    char   byte2str[256][5];  /* byte -> utf8 of its byte-level char */
    int    cp2byte[512];      /* byte-level codepoint -> original byte */
};

/* ------------------------------------------------------------------ */
/* GPT-2 byte-level bijection                                          */
/* ------------------------------------------------------------------ */
static int utf8_encode(int cp, char *out) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) { out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    out[0] = (char)(0xE0 | (cp >> 12));
    out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
}

static int utf8_decode(const char *s, int *cp) {
    unsigned char c = s[0];
    if (c < 0x80) { *cp = c; return 1; }
    if ((c & 0xE0) == 0xC0) { *cp = ((c & 0x1F) << 6) | (s[1] & 0x3F); return 2; }
    if ((c & 0xF0) == 0xE0) { *cp = ((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F); return 3; }
    *cp = ((c & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F);
    return 4;
}

static void build_byte_level(EmberTokenizer *t) {
    int is_printable[256] = {0};
    for (int c = '!'; c <= '~'; c++) is_printable[c] = 1;
    for (int c = 0xA1; c <= 0xAC; c++) is_printable[c] = 1;
    for (int c = 0xAE; c <= 0xFF; c++) is_printable[c] = 1;
    for (int i = 0; i < 512; i++) t->cp2byte[i] = -1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        int cp = is_printable[b] ? b : (256 + n++); /* printables map to self, rest to 256+ */
        int len = utf8_encode(cp, t->byte2str[b]);
        t->byte2str[b][len] = '\0';
        if (cp < 512) t->cp2byte[cp] = b;
    }
}

/* ------------------------------------------------------------------ */
/* construction                                                       */
/* ------------------------------------------------------------------ */
static void build_sp(EmberTokenizer *t, const uint8_t *p) {
    memcpy(&t->sp_max_len, p, 4); p += 4;
    t->sp_vocab = malloc(sizeof(char *) * t->vocab_size);
    t->sp_scores = malloc(sizeof(float) * t->vocab_size);
    hmap_init(&t->sp_map, t->vocab_size);
    /* copy strings into an owned, null-terminated blob for safe hashing/printing */
    for (int b = 0; b < 256; b++) { t->byte_pieces[b*2] = (char)b; t->byte_pieces[b*2+1] = 0; }
    for (int i = 0; i < t->vocab_size; i++) {
        float score; int len;
        memcpy(&score, p, 4); p += 4;
        memcpy(&len, p, 4);   p += 4;
        char *s = malloc(len + 1);
        memcpy(s, p, len); s[len] = 0; p += len;
        t->sp_vocab[i] = s; t->sp_scores[i] = score;
        hmap_put(&t->sp_map, s, len, i);
    }
}

static void build_bpe(EmberTokenizer *t, const uint8_t *p) {
    build_byte_level(t);
    uint32_t vsize, nmerges, nspecial;
    memcpy(&vsize, p, 4); memcpy(&nmerges, p + 4, 4); memcpy(&nspecial, p + 8, 4);
    p += 12;
    t->n_merges = nmerges; t->n_special = nspecial;
    t->bpe_vocab = calloc(t->vocab_size, sizeof(Vocab));
    hmap_init(&t->bpe_vmap, t->vocab_size);
    for (uint32_t i = 0; i < vsize; i++) {
        uint8_t sp = p[0]; uint32_t len; memcpy(&len, p + 1, 4); p += 5;
        t->bpe_vocab[i].str = (const char *)p;
        t->bpe_vocab[i].len = len;
        t->bpe_vocab[i].score_or_special = sp;
        if (len) hmap_put(&t->bpe_vmap, (const char *)p, len, i);
        p += len;
    }
    hmap_init(&t->bpe_ranks, nmerges);
    t->rank_keys = malloc(sizeof(char *) * nmerges);
    for (uint32_t i = 0; i < nmerges; i++) {
        uint32_t la; memcpy(&la, p, 4); p += 4; const uint8_t *a = p; p += la;
        uint32_t lb; memcpy(&lb, p, 4); p += 4; const uint8_t *b = p; p += lb;
        char *key = malloc(la + 1 + lb);
        memcpy(key, a, la); key[la] = 0; memcpy(key + la + 1, b, lb);
        t->rank_keys[i] = key;
        hmap_put(&t->bpe_ranks, key, la + 1 + lb, i);
    }
    t->specials = malloc(sizeof(*t->specials) * nspecial);
    for (uint32_t i = 0; i < nspecial; i++) {
        uint32_t id, len; memcpy(&id, p, 4); memcpy(&len, p + 4, 4); p += 8;
        t->specials[i].s = (const char *)p; t->specials[i].len = len; t->specials[i].id = id;
        p += len;
    }
}

EmberTokenizer *ember_tokenizer_new(const EmberModel *m) {
    const EmberHeader *h = ember_model_header(m);
    const uint8_t *blob = (const uint8_t *)h + h->tokenizer_offset;
    EmberTokenizer *t = calloc(1, sizeof(*t));
    t->type = h->tokenizer_type;
    t->vocab_size = h->vocab_size;
    t->bos_id = h->bos_token_id;
    t->eos_id = h->eos_token_id;
    if (t->type == EMBER_TOK_BYTE_BPE) { t->bpe_blob = blob; build_bpe(t, blob); }
    else                                build_sp(t, blob);
    return t;
}

void ember_tokenizer_free(EmberTokenizer *t) {
    if (!t) return;
    if (t->type == EMBER_TOK_BYTE_BPE) {
        for (int i = 0; i < t->n_merges; i++) free(t->rank_keys[i]);
        free(t->rank_keys); free(t->bpe_vocab); free(t->specials);
        free(t->bpe_vmap.e); free(t->bpe_ranks.e);
    } else {
        for (int i = 0; i < t->vocab_size; i++) free(t->sp_vocab[i]);
        free(t->sp_vocab); free(t->sp_scores); free(t->sp_map.e);
    }
    free(t);
}

/* ------------------------------------------------------------------ */
/* SentencePiece encode/decode                                        */
/* ------------------------------------------------------------------ */
static int sp_encode(EmberTokenizer *t, const char *text, int add_bos, int **out_ids) {
    size_t slen = strlen(text);
    int *tokens = malloc(sizeof(int) * (slen + 3));
    int n = 0;
    if (add_bos) tokens[n++] = t->bos_id;
    if (text[0]) { int sp = hmap_get(&t->sp_map, " ", 1); if (sp != -1) tokens[n++] = sp; }
    char *cp = malloc(t->sp_max_len * 2 + 2);
    for (const char *c = text; *c; ) {
        int clen = 1; unsigned char b0 = (unsigned char)c[0];
        if (b0 >= 0xF0) clen = 4; else if (b0 >= 0xE0) clen = 3; else if (b0 >= 0xC0) clen = 2;
        int used = 0; for (int k = 0; k < clen && c[k]; k++) cp[used++] = c[k];
        cp[used] = 0;
        int id = hmap_get(&t->sp_map, cp, used);
        if (id != -1) tokens[n++] = id;
        else for (int k = 0; k < used; k++) tokens[n++] = (unsigned char)cp[k] + 3;
        c += used;
    }
    for (;;) {
        float best = -1e30f; int best_id = -1, best_idx = -1;
        for (int i = 0; i < n - 1; i++) {
            snprintf(cp, t->sp_max_len * 2 + 2, "%s%s", t->sp_vocab[tokens[i]], t->sp_vocab[tokens[i+1]]);
            int id = hmap_get(&t->sp_map, cp, (int)strlen(cp));
            if (id != -1 && t->sp_scores[id] > best) { best = t->sp_scores[id]; best_id = id; best_idx = i; }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        memmove(&tokens[best_idx+1], &tokens[best_idx+2], (n - best_idx - 2) * sizeof(int));
        n--;
    }
    free(cp);
    *out_ids = tokens;
    return n;
}

static const char *sp_decode(EmberTokenizer *t, int prev, int tok) {
    const char *piece = t->sp_vocab[tok];
    if (prev == t->bos_id && piece[0] == ' ') piece++;
    unsigned int byte;
    if (sscanf(piece, "<0x%02X>", &byte) == 1) return &t->byte_pieces[(byte & 0xFF) * 2];
    return piece;
}

/* ------------------------------------------------------------------ */
/* byte-level BPE encode/decode                                       */
/* ------------------------------------------------------------------ */
/* letter/number/space classifiers, ASCII-exact with UTF-8 runs as letters */
static int is_space(unsigned char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v'; }
static int is_letter(unsigned char c) { return (c|32) >= 'a' && (c|32) <= 'z'; }
static int is_digit(unsigned char c) { return c >= '0' && c <= '9'; }
static int is_letter_ext(unsigned char c) { return is_letter(c) || c >= 0x80; } /* treat UTF-8 bytes as letters */

/* Approximate the GPT-4/Qwen pre-tokenizer: return the length in bytes of the
 * next piece starting at s (s has slen bytes remaining). */
static int next_piece(const char *s, int slen) {
    unsigned char c0 = s[0];
    /* contractions: 's 't 're 've 'm 'll 'd (case-insensitive) */
    if (c0 == '\'' && slen >= 2) {
        unsigned char c1 = s[1] | 32;
        if (c1=='s'||c1=='t'||c1=='m'||c1=='d') return 2;
        if (slen >= 3) { unsigned char c2 = s[2]|32;
            if ((c1=='r'&&c2=='e')||(c1=='v'&&c2=='e')||(c1=='l'&&c2=='l')) return 3; }
    }
    /* optional single non-letter/non-number then letters:  [^\r\n\p{L}\p{N}]?\p{L}+
     * the optional leading char may be a space, tab, or punctuation */
    {
        int has_opt = (c0 != '\r' && c0 != '\n' && !is_letter_ext(c0) && !is_digit(c0));
        int i = (has_opt && slen >= 2 && is_letter_ext((unsigned char)s[1])) ? 1 : 0;
        if (i < slen && is_letter_ext((unsigned char)s[i])) {
            int j = i; while (j < slen && is_letter_ext((unsigned char)s[j])) j++;
            return j;
        }
    }
    /* single number char */
    if (is_digit(c0)) return 1;
    /* optional space then punctuation run then newlines:  ?[^\s\p{L}\p{N}]+[\r\n]* */
    {
        int i = 0;
        if (c0 == ' ') i = 1;
        if (i < slen) { unsigned char c = s[i];
            if (!is_space(c) && !is_letter_ext(c) && !is_digit(c)) {
                int j = i;
                while (j < slen) {
                    unsigned char cc = s[j];
                    if (is_space(cc) || is_letter_ext(cc) || is_digit(cc)) break;
                    j++;
                }
                while (j < slen && (s[j] == '\r' || s[j] == '\n')) j++;
                return j;
            }
        }
    }
    /* whitespace runs (\s*[\r\n]+ | \s+(?!\S) | \s+) */
    if (is_space(c0)) {
        int j = 0; while (j < slen && is_space((unsigned char)s[j])) j++;
        int last_nl = -1;
        for (int k = 0; k < j; k++) if (s[k] == '\n' || s[k] == '\r') last_nl = k;
        if (last_nl >= 0) return last_nl + 1;   /* \s*[\r\n]+ : through the last newline */
        if (j >= slen) return j;                /* trailing whitespace at end: \s+       */
        /* followed by a non-space: \s+(?!\S) groups all-but-the-last whitespace as one
         * piece; the final char is left for the next piece (a word/punct absorbs it as
         * its optional leading char, a digit takes it as a lone space token) */
        return j > 1 ? j - 1 : 1;
    }
    return 1; /* fallback: consume one byte */
}

static int bpe_merge_piece(EmberTokenizer *t, const char *bl, int bl_len, int *out, int cap) {
    /* symbols as (start,len) substrings of the byte-level string bl */
    int starts[512], lens[512], ns = 0;
    for (int i = 0; i < bl_len && ns < 512; ) {
        int cp, adv = utf8_decode(bl + i, &cp);
        starts[ns] = i; lens[ns] = adv; ns++;
        i += adv;
    }
    char key[512];
    for (;;) {
        int best = -1, best_rank = 0x7fffffff;
        for (int i = 0; i < ns - 1; i++) {
            int la = lens[i], lb = lens[i+1];
            if (la + 1 + lb > (int)sizeof(key)) continue;
            memcpy(key, bl + starts[i], la); key[la] = 0;
            memcpy(key + la + 1, bl + starts[i+1], lb);
            int r = hmap_get(&t->bpe_ranks, key, la + 1 + lb);
            if (r >= 0 && r < best_rank) { best_rank = r; best = i; }
        }
        if (best < 0) break;
        lens[best] += lens[best+1];
        for (int j = best + 1; j < ns - 1; j++) { starts[j] = starts[j+1]; lens[j] = lens[j+1]; }
        ns--;
    }
    int n = 0;
    for (int i = 0; i < ns && n < cap; i++) {
        int id = hmap_get(&t->bpe_vmap, bl + starts[i], lens[i]);
        if (id >= 0) out[n++] = id;
        /* a symbol should always resolve; if not, drop it (robustness) */
    }
    return n;
}

/* match a special token starting at s; returns its index or -1 */
static int match_special(EmberTokenizer *t, const char *s, int slen) {
    for (int i = 0; i < t->n_special; i++) {
        int L = t->specials[i].len;
        if (L <= slen && memcmp(s, t->specials[i].s, L) == 0) return i;
    }
    return -1;
}

static int bpe_encode(EmberTokenizer *t, const char *text, int add_bos, int **out_ids) {
    int slen = (int)strlen(text);
    int cap = slen + 8, n = 0;
    int *tokens = malloc(sizeof(int) * (cap > 16 ? cap : 16));
    if (add_bos && t->bos_id >= 0) tokens[n++] = t->bos_id;

    char *bl = malloc(slen * 4 + 8);      /* byte-level buffer */
    int piecebuf[512];
    int i = 0;
    while (i < slen) {
        int si = match_special(t, text + i, slen - i);
        if (si >= 0) {
            if (n + 1 >= cap) { cap *= 2; tokens = realloc(tokens, sizeof(int) * cap); }
            tokens[n++] = t->specials[si].id;
            i += t->specials[si].len;
            continue;
        }
        /* consume one pre-token piece of normal text */
        int plen = next_piece(text + i, slen - i);
        int bl_len = 0;
        for (int k = 0; k < plen; k++) {
            const char *bs = t->byte2str[(unsigned char)text[i + k]];
            int L = (int)strlen(bs);
            memcpy(bl + bl_len, bs, L); bl_len += L;
        }
        int got = bpe_merge_piece(t, bl, bl_len, piecebuf, 512);
        if (n + got >= cap) { cap = (n + got) * 2; tokens = realloc(tokens, sizeof(int) * cap); }
        for (int k = 0; k < got; k++) tokens[n++] = piecebuf[k];
        i += plen;
    }
    free(bl);
    *out_ids = tokens;
    return n;
}

static char g_decbuf[512];
static const char *bpe_decode(EmberTokenizer *t, int prev, int tok) {
    (void)prev;
    if (tok < 0 || tok >= t->vocab_size) return "";
    const Vocab *v = &t->bpe_vocab[tok];
    if (v->score_or_special) { /* special token: literal content */
        int L = v->len < 511 ? v->len : 511;
        memcpy(g_decbuf, v->str, L); g_decbuf[L] = 0; return g_decbuf;
    }
    /* byte-level: map each codepoint back to its original byte */
    int o = 0;
    for (int i = 0; i < v->len && o < 511; ) {
        int cp, adv = utf8_decode(v->str + i, &cp);
        int b = (cp >= 0 && cp < 512) ? t->cp2byte[cp] : -1;
        g_decbuf[o++] = (char)(b >= 0 ? b : '?');
        i += adv;
    }
    g_decbuf[o] = 0;
    return g_decbuf;
}

/* ------------------------------------------------------------------ */
/* dispatch                                                           */
/* ------------------------------------------------------------------ */
int ember_encode(EmberTokenizer *t, const char *text, int add_bos, int **out_ids) {
    if (t->type == EMBER_TOK_BYTE_BPE) return bpe_encode(t, text, add_bos, out_ids);
    return sp_encode(t, text, add_bos, out_ids);
}

const char *ember_decode(EmberTokenizer *t, int prev, int tok) {
    if (t->type == EMBER_TOK_BYTE_BPE) return bpe_decode(t, prev, tok);
    return sp_decode(t, prev, tok);
}
