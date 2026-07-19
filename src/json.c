/* json.c — recursive-descent JSON parser. See json.h. */
#include "json.h"
#include "util.h"

#include <stdlib.h>
#include <string.h>

typedef struct { const char *p; int ok; } Cur;

static JsonValue *parse_value(Cur *c);

static void skip_ws(Cur *c) {
    while (*c->p == ' ' || *c->p == '\t' || *c->p == '\n' || *c->p == '\r') c->p++;
}

static JsonValue *new_val(JsonType t) {
    JsonValue *v = ember_xcalloc(1, sizeof(*v), "json value");
    v->type = t;
    return v;
}

/* Append one UTF-8 encoding of a code point to buf at *len, growing not needed
 * (caller sizes buf to >= input length, and escapes never expand beyond it). */
static void emit_utf8(char *buf, int *len, unsigned cp) {
    if (cp < 0x80) {
        buf[(*len)++] = (char)cp;
    } else if (cp < 0x800) {
        buf[(*len)++] = (char)(0xC0 | (cp >> 6));
        buf[(*len)++] = (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        buf[(*len)++] = (char)(0xE0 | (cp >> 12));
        buf[(*len)++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[(*len)++] = (char)(0x80 | (cp & 0x3F));
    } else {
        buf[(*len)++] = (char)(0xF0 | (cp >> 18));
        buf[(*len)++] = (char)(0x80 | ((cp >> 12) & 0x3F));
        buf[(*len)++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[(*len)++] = (char)(0x80 | (cp & 0x3F));
    }
}

static int hex4(const char *p, unsigned *out) {
    unsigned v = 0;
    for (int i = 0; i < 4; i++) {
        char ch = p[i];
        v <<= 4;
        if      (ch >= '0' && ch <= '9') v |= (unsigned)(ch - '0');
        else if (ch >= 'a' && ch <= 'f') v |= (unsigned)(ch - 'a' + 10);
        else if (ch >= 'A' && ch <= 'F') v |= (unsigned)(ch - 'A' + 10);
        else return 0;
    }
    *out = v;
    return 1;
}

/* Parse a JSON string starting at the opening quote; returns an owned buffer. */
static char *parse_string_raw(Cur *c) {
    if (*c->p != '"') { c->ok = 0; return NULL; }
    const char *start = ++c->p;
    /* worst case the decoded string is no longer than the source span */
    const char *scan = start;
    while (*scan && *scan != '"') { if (*scan == '\\' && scan[1]) scan += 2; else scan++; }
    if (*scan != '"') { c->ok = 0; return NULL; }
    char *out = ember_xmalloc((size_t)(scan - start) + 1, "json string");
    int len = 0;
    while (*c->p && *c->p != '"') {
        if (*c->p == '\\') {
            c->p++;
            switch (*c->p) {
                case '"': out[len++] = '"';  c->p++; break;
                case '\\': out[len++] = '\\'; c->p++; break;
                case '/': out[len++] = '/';  c->p++; break;
                case 'b': out[len++] = '\b'; c->p++; break;
                case 'f': out[len++] = '\f'; c->p++; break;
                case 'n': out[len++] = '\n'; c->p++; break;
                case 'r': out[len++] = '\r'; c->p++; break;
                case 't': out[len++] = '\t'; c->p++; break;
                case 'u': {
                    unsigned cp;
                    if (!hex4(c->p + 1, &cp)) { c->ok = 0; free(out); return NULL; }
                    c->p += 5;
                    /* surrogate pair */
                    if (cp >= 0xD800 && cp <= 0xDBFF && c->p[0] == '\\' && c->p[1] == 'u') {
                        unsigned lo;
                        if (hex4(c->p + 2, &lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            c->p += 6;
                        }
                    }
                    emit_utf8(out, &len, cp);
                    break;
                }
                default: c->ok = 0; free(out); return NULL;
            }
        } else {
            out[len++] = *c->p++;
        }
    }
    if (*c->p != '"') { c->ok = 0; free(out); return NULL; }
    c->p++; /* closing quote */
    out[len] = '\0';
    return out;
}

static JsonValue *parse_string(Cur *c) {
    char *s = parse_string_raw(c);
    if (!c->ok) return NULL;
    JsonValue *v = new_val(JSON_STR);
    v->str = s;
    return v;
}

static JsonValue *parse_number(Cur *c) {
    char *end;
    double d = strtod(c->p, &end);
    if (end == c->p) { c->ok = 0; return NULL; }
    c->p = end;
    JsonValue *v = new_val(JSON_NUM);
    v->num = d;
    return v;
}

static JsonValue *parse_array(Cur *c) {
    c->p++; /* [ */
    JsonValue *v = new_val(JSON_ARR);
    int cap = 0;
    skip_ws(c);
    if (*c->p == ']') { c->p++; return v; }
    for (;;) {
        JsonValue *item = parse_value(c);
        if (!c->ok) { json_free(v); return NULL; }
        if (v->n_items == cap) {
            cap = cap ? cap * 2 : 4;
            v->items = ember_xrealloc(v->items, sizeof(JsonValue *) * (size_t)cap, "json array");
        }
        v->items[v->n_items++] = item;
        skip_ws(c);
        if (*c->p == ',') { c->p++; skip_ws(c); continue; }
        if (*c->p == ']') { c->p++; break; }
        c->ok = 0; json_free(v); return NULL;
    }
    return v;
}

static JsonValue *parse_object(Cur *c) {
    c->p++; /* { */
    JsonValue *v = new_val(JSON_OBJ);
    int cap = 0;
    skip_ws(c);
    if (*c->p == '}') { c->p++; return v; }
    for (;;) {
        skip_ws(c);
        char *key = parse_string_raw(c);
        if (!c->ok) { json_free(v); return NULL; }
        skip_ws(c);
        if (*c->p != ':') { c->ok = 0; free(key); json_free(v); return NULL; }
        c->p++;
        JsonValue *val = parse_value(c);
        if (!c->ok) { free(key); json_free(v); return NULL; }
        if (v->n_pairs == cap) {
            cap = cap ? cap * 2 : 4;
            v->keys = ember_xrealloc(v->keys, sizeof(char *) * (size_t)cap, "json keys");
            v->vals = ember_xrealloc(v->vals, sizeof(JsonValue *) * (size_t)cap, "json vals");
        }
        v->keys[v->n_pairs] = key;
        v->vals[v->n_pairs] = val;
        v->n_pairs++;
        skip_ws(c);
        if (*c->p == ',') { c->p++; continue; }
        if (*c->p == '}') { c->p++; break; }
        c->ok = 0; json_free(v); return NULL;
    }
    return v;
}

static JsonValue *parse_literal(Cur *c, const char *word, JsonType t, double num) {
    size_t n = strlen(word);
    if (strncmp(c->p, word, n) != 0) { c->ok = 0; return NULL; }
    c->p += n;
    JsonValue *v = new_val(t);
    v->num = num;
    return v;
}

static JsonValue *parse_value(Cur *c) {
    skip_ws(c);
    switch (*c->p) {
        case '"': return parse_string(c);
        case '{': return parse_object(c);
        case '[': return parse_array(c);
        case 't': return parse_literal(c, "true",  JSON_BOOL, 1);
        case 'f': return parse_literal(c, "false", JSON_BOOL, 0);
        case 'n': return parse_literal(c, "null",  JSON_NULL, 0);
        default:
            if (*c->p == '-' || (*c->p >= '0' && *c->p <= '9')) return parse_number(c);
            c->ok = 0;
            return NULL;
    }
}

JsonValue *json_parse(const char *text) {
    Cur c = { text, 1 };
    JsonValue *v = parse_value(&c);
    if (!c.ok) { json_free(v); return NULL; }
    skip_ws(&c);
    if (*c.p != '\0') { json_free(v); return NULL; } /* trailing garbage */
    return v;
}

void json_free(JsonValue *v) {
    if (!v) return;
    free(v->str);
    for (int i = 0; i < v->n_items; i++) json_free(v->items[i]);
    free(v->items);
    for (int i = 0; i < v->n_pairs; i++) { free(v->keys[i]); json_free(v->vals[i]); }
    free(v->keys);
    free(v->vals);
    free(v);
}

const JsonValue *json_get(const JsonValue *v, const char *key) {
    if (!v || v->type != JSON_OBJ) return NULL;
    for (int i = 0; i < v->n_pairs; i++)
        if (strcmp(v->keys[i], key) == 0) return v->vals[i];
    return NULL;
}

const char *json_str(const JsonValue *v, const char *fallback) {
    return (v && v->type == JSON_STR) ? v->str : fallback;
}

double json_num(const JsonValue *v, double fallback) {
    return (v && (v->type == JSON_NUM || v->type == JSON_BOOL)) ? v->num : fallback;
}

int json_bool(const JsonValue *v, int fallback) {
    if (!v) return fallback;
    if (v->type == JSON_BOOL || v->type == JSON_NUM) return v->num != 0;
    return fallback;
}
