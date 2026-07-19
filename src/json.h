/* json.h — a tiny, dependency-free JSON reader.
 *
 * Just enough to parse an OpenAI-style chat request body: objects, arrays,
 * strings (with \uXXXX -> UTF-8), numbers, booleans, null. It builds a small
 * owned DOM; there is no serializer here (server.c writes JSON by hand, which is
 * trivial for the fixed response shapes). Not a general-purpose library — it
 * favours being short and obviously correct over speed.
 */
#ifndef EMBER_JSON_H
#define EMBER_JSON_H

typedef enum { JSON_NULL, JSON_BOOL, JSON_NUM, JSON_STR, JSON_ARR, JSON_OBJ } JsonType;

typedef struct JsonValue JsonValue;
struct JsonValue {
    JsonType type;
    double   num;             /* JSON_NUM / JSON_BOOL (0 or 1)                  */
    char    *str;             /* JSON_STR: NUL-terminated, unescaped, owned     */
    JsonValue **items;        /* JSON_ARR                                       */
    int        n_items;
    char      **keys;         /* JSON_OBJ: parallel keys/vals                   */
    JsonValue **vals;
    int         n_pairs;
};

/* Parse a NUL-terminated JSON document. Returns NULL on syntax error. */
JsonValue *json_parse(const char *text);
void       json_free(JsonValue *v);

/* Object member lookup by key (NULL if absent or v is not an object). */
const JsonValue *json_get(const JsonValue *v, const char *key);

/* Typed accessors with defaults; tolerant of NULL/wrong-type. */
const char *json_str(const JsonValue *v, const char *fallback);
double      json_num(const JsonValue *v, double fallback);
int         json_bool(const JsonValue *v, int fallback);

#endif /* EMBER_JSON_H */
