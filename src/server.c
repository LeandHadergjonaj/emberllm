/* server.c — the hand-rolled OpenAI-compatible HTTP server. See server.h.
 *
 * No framework: raw BSD sockets, a serial accept loop (single-stream scope),
 * a from-scratch request reader, and JSON responses built by hand. Requests are
 * parsed with json.c. The model, tokenizer, and KV-cache state are created once
 * and reused across requests; each request re-prefills from position 0, so the
 * server holds no conversation state — clients send the full message list, as
 * the OpenAI API prescribes.
 */
#include "ember.h"
#include "kernels.h"
#include "json.h"
#include "server.h"
#include "util.h"

#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

/* ------------------------------------------------------------------ */
/* growable byte buffer                                               */
/* ------------------------------------------------------------------ */
typedef struct { char *data; size_t len, cap; } Buf;

static void buf_reserve(Buf *b, size_t extra) {
    if (b->len + extra + 1 > b->cap) {
        b->cap = (b->len + extra + 1) * 2;
        b->data = ember_xrealloc(b->data, b->cap, "http buffer");
    }
}
static void buf_append(Buf *b, const char *s, size_t n) {
    buf_reserve(b, n);
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = '\0';
}
static void buf_puts(Buf *b, const char *s) { buf_append(b, s, strlen(s)); }
static void buf_free(Buf *b) { free(b->data); b->data = NULL; b->len = b->cap = 0; }

/* Append s with JSON string-escaping (no surrounding quotes). */
static void buf_json_escape(Buf *b, const char *s) {
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        switch (c) {
            case '"':  buf_puts(b, "\\\""); break;
            case '\\': buf_puts(b, "\\\\"); break;
            case '\n': buf_puts(b, "\\n"); break;
            case '\r': buf_puts(b, "\\r"); break;
            case '\t': buf_puts(b, "\\t"); break;
            case '\b': buf_puts(b, "\\b"); break;
            case '\f': buf_puts(b, "\\f"); break;
            default:
                if (c < 0x20) { char t[8]; snprintf(t, sizeof t, "\\u%04x", c); buf_puts(b, t); }
                else { char ch = (char)c; buf_append(b, &ch, 1); }
        }
    }
}

/* ------------------------------------------------------------------ */
/* signals: flip a flag so the accept loop exits cleanly              */
/* ------------------------------------------------------------------ */
static volatile sig_atomic_t g_stop = 0;
static void on_signal(int s) { (void)s; g_stop = 1; }

/* ------------------------------------------------------------------ */
/* socket helpers                                                     */
/* ------------------------------------------------------------------ */
static int send_all(int fd, const char *data, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t k = send(fd, data + off, len - off, 0);
        if (k < 0) { if (errno == EINTR) continue; return -1; }
        off += (size_t)k;
    }
    return 0;
}

static const char *find_seq(const char *hay, size_t hlen, const char *needle) {
    size_t nlen = strlen(needle);
    if (nlen == 0 || hlen < nlen) return NULL;
    for (size_t i = 0; i + nlen <= hlen; i++)
        if (memcmp(hay + i, needle, nlen) == 0) return hay + i;
    return NULL;
}

/* Read one full HTTP request (headers + Content-Length body). Fills method/path
 * and returns the body pointer (into *raw). Returns 0 on success, -1 on error. */
typedef struct { char method[8]; char path[512]; const char *body; size_t body_len; } Request;

static long parse_content_length(const char *hdr, size_t hlen) {
    /* case-insensitive scan for "content-length:" */
    static const char *key = "content-length:";
    size_t klen = strlen(key);
    for (size_t i = 0; i + klen <= hlen; i++) {
        size_t j = 0;
        while (j < klen && (hdr[i + j] | 0x20) == key[j]) j++;
        if (j == klen) {
            const char *p = hdr + i + klen;
            while (*p == ' ') p++;
            return strtol(p, NULL, 10);
        }
    }
    return -1;
}

static int read_request(int fd, Buf *raw, Request *req) {
    memset(req, 0, sizeof *req);
    char tmp[8192];
    const char *hdr_end = NULL;
    long content_len = -1;
    for (;;) {
        if (!hdr_end) {
            hdr_end = find_seq(raw->data ? raw->data : "", raw->len, "\r\n\r\n");
            if (hdr_end) {
                size_t hlen = (size_t)(hdr_end - raw->data);
                content_len = parse_content_length(raw->data, hlen);
                if (sscanf(raw->data, "%7s %511s", req->method, req->path) != 2) return -1;
            }
        }
        if (hdr_end) {
            size_t body_off = (size_t)(hdr_end - raw->data) + 4;
            size_t have = raw->len - body_off;
            if (content_len < 0 || have >= (size_t)content_len) {
                req->body = raw->data + body_off;
                req->body_len = content_len < 0 ? have : (size_t)content_len;
                return 0;
            }
        }
        ssize_t k = recv(fd, tmp, sizeof tmp, 0);
        if (k <= 0) { if (k < 0 && errno == EINTR) continue; return -1; }
        buf_append(raw, tmp, (size_t)k);
        if (raw->len > 64u * 1024 * 1024) return -1; /* runaway-request guard */
    }
}

static void http_respond(int fd, int status, const char *status_text,
                         const char *ctype, const char *body, size_t blen) {
    Buf h = {0};
    char line[128];
    snprintf(line, sizeof line, "HTTP/1.1 %d %s\r\n", status, status_text);
    buf_puts(&h, line);
    buf_puts(&h, "Access-Control-Allow-Origin: *\r\n");
    buf_puts(&h, "Access-Control-Allow-Headers: *\r\n");
    buf_puts(&h, "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n");
    if (ctype) { buf_puts(&h, "Content-Type: "); buf_puts(&h, ctype); buf_puts(&h, "\r\n"); }
    snprintf(line, sizeof line, "Content-Length: %zu\r\n", blen);
    buf_puts(&h, line);
    buf_puts(&h, "Connection: close\r\n\r\n");
    send_all(fd, h.data, h.len);
    if (body && blen) send_all(fd, body, blen);
    buf_free(&h);
}

static void http_error(int fd, int status, const char *status_text, const char *message) {
    Buf b = {0};
    buf_puts(&b, "{\"error\":{\"message\":\"");
    buf_json_escape(&b, message);
    buf_puts(&b, "\",\"type\":\"invalid_request_error\"}}");
    http_respond(fd, status, status_text, "application/json", b.data, b.len);
    buf_free(&b);
}

/* ------------------------------------------------------------------ */
/* the server object shared across requests                           */
/* ------------------------------------------------------------------ */
typedef struct {
    EmberModel      *m;
    const EmberHeader *h;
    EmberTokenizer  *tok;
    EmberState      *st;
    const ServeConfig *cfg;
    char             model_name[256];
    unsigned long    req_id;
    int              ctx;
} Server;

/* Build the ChatML prompt from an OpenAI "messages" array. */
static void build_prompt(const JsonValue *messages, int think, Buf *out) {
    for (int i = 0; i < messages->n_items; i++) {
        const JsonValue *msg = messages->items[i];
        const char *role = json_str(json_get(msg, "role"), "user");
        const char *content = json_str(json_get(msg, "content"), "");
        buf_puts(out, "<|im_start|>");
        buf_puts(out, role);
        buf_puts(out, "\n");
        buf_puts(out, content);
        buf_puts(out, "<|im_end|>\n");
    }
    buf_puts(out, "<|im_start|>assistant\n");
    if (!think) buf_puts(out, "<think>\n\n</think>\n\n");
}

/* Collect stop strings from the request's "stop" field (string or array). */
typedef struct { char **s; int n; } StopList;
static StopList collect_stops(const JsonValue *stop) {
    StopList sl = {0};
    if (!stop) return sl;
    if (stop->type == JSON_STR) {
        sl.s = ember_xmalloc(sizeof(char *), "stops");
        sl.s[0] = stop->str; /* borrowed from the DOM; valid for the request */
        sl.n = 1;
    } else if (stop->type == JSON_ARR) {
        sl.s = ember_xmalloc(sizeof(char *) * (size_t)stop->n_items, "stops");
        for (int i = 0; i < stop->n_items; i++)
            if (stop->items[i]->type == JSON_STR) sl.s[sl.n++] = stop->items[i]->str;
    }
    return sl;
}

/* Earliest index in `text` at which any stop string begins, or -1. */
static long find_stop(const char *text, const StopList *sl) {
    long best = -1;
    for (int i = 0; i < sl->n; i++) {
        const char *hit = strstr(text, sl->s[i]);
        if (hit) { long at = hit - text; if (best < 0 || at < best) best = at; }
    }
    return best;
}

/* One chat completion. Returns after fully responding (or on client hangup). */
static void handle_chat(Server *sv, int fd, const char *body, size_t body_len) {
    JsonValue *req = NULL;
    if (body_len == 0 || !(req = json_parse(body))) {
        http_error(fd, 400, "Bad Request", "invalid JSON body");
        json_free(req);
        return;
    }
    const JsonValue *messages = json_get(req, "messages");
    if (!messages || messages->type != JSON_ARR || messages->n_items == 0) {
        http_error(fd, 400, "Bad Request", "missing 'messages' array");
        json_free(req);
        return;
    }

    const ServeConfig *cfg = sv->cfg;
    int   stream    = json_bool(json_get(req, "stream"), 0);
    float temp      = (float)json_num(json_get(req, "temperature"), cfg->temperature);
    float top_p     = (float)json_num(json_get(req, "top_p"), cfg->top_p);
    int   top_k     = (int)json_num(json_get(req, "top_k"), cfg->top_k);
    int   max_tok   = (int)json_num(json_get(req, "max_tokens"), cfg->max_tokens);
    float rep_pen   = (float)json_num(json_get(req, "repeat_penalty"), cfg->repeat_penalty);
    float freq_pen  = (float)json_num(json_get(req, "frequency_penalty"), 0.0);
    float pres_pen  = (float)json_num(json_get(req, "presence_penalty"), 0.0);
    StopList stops  = collect_stops(json_get(req, "stop"));
    int   think     = cfg->think && json_bool(json_get(req, "think"), 1);

    Buf prompt = {0};
    build_prompt(messages, think, &prompt);

    int *ids, n_prompt = ember_encode(sv->tok, prompt.data, 0, &ids);
    buf_free(&prompt);
    if (n_prompt < 1 || n_prompt >= sv->ctx - 1) {
        free(ids);
        http_error(fd, 400, "Bad Request", "prompt is empty or exceeds the context window");
        json_free(req);
        free(stops.s);
        return;
    }

    /* response identity */
    char id[64];
    snprintf(id, sizeof id, "chatcmpl-%lu", ++sv->req_id);
    long created = (long)time(NULL);

    EmberSampler smp;
    ember_sampler_init(&smp, temp, top_p, top_k, cfg->seed + sv->req_id);
    ember_sampler_set_penalties(&smp, rep_pen, 256, pres_pen, freq_pen, 0.0f);
    for (int i = 0; i < n_prompt; i++) ember_sampler_accept(&smp, ids[i]);

    /* streaming preamble: SSE headers + a role delta */
    if (stream) {
        const char *hdr =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Connection: close\r\n\r\n";
        send_all(fd, hdr, strlen(hdr));
        Buf c = {0};
        buf_puts(&c, "data: {\"id\":\"");
        buf_puts(&c, id);
        buf_puts(&c, "\",\"object\":\"chat.completion.chunk\",\"created\":");
        char t[32]; snprintf(t, sizeof t, "%ld", created); buf_puts(&c, t);
        buf_puts(&c, ",\"model\":\"");
        buf_json_escape(&c, sv->model_name);
        buf_puts(&c, "\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n");
        send_all(fd, c.data, c.len);
        buf_free(&c);
    }

    /* decode */
    float *logits = ember_prefill(sv->m, sv->st, ids, n_prompt, 0);
    int prev = ids[n_prompt - 1];
    free(ids);

    Buf out = {0};                 /* full decoded text so far */
    size_t streamed = 0;           /* bytes already sent to the client (streaming) */
    long max_stop = 0;
    for (int i = 0; i < stops.n; i++) { long L = (long)strlen(stops.s[i]); if (L > max_stop) max_stop = L; }
    long hold = max_stop > 0 ? max_stop - 1 : 0; /* retain a possible partial stop tail */

    const char *finish = "length";
    int completion_tokens = 0;
    int broken = 0;

    for (int i = 0; i < max_tok; i++) {
        int pos = n_prompt + i;
        if (pos >= sv->ctx - 1) { finish = "length"; break; }
        int next = ember_sample(&smp, logits, sv->h->vocab_size);
        if (next == sv->h->eos_token_id || next == sv->h->eos_token_id2) { finish = "stop"; break; }
        buf_puts(&out, ember_decode(sv->tok, prev, next));
        prev = next;
        completion_tokens++;

        long sp = find_stop(out.data, &stops);
        if (sp >= 0) {
            out.data[sp] = '\0'; out.len = (size_t)sp;
            finish = "stop";
            if (stream && out.len > streamed) {
                Buf c = {0};
                buf_puts(&c, "data: {\"id\":\""); buf_puts(&c, id);
                buf_puts(&c, "\",\"object\":\"chat.completion.chunk\",\"created\":");
                char t[32]; snprintf(t, sizeof t, "%ld", created); buf_puts(&c, t);
                buf_puts(&c, ",\"model\":\""); buf_json_escape(&c, sv->model_name);
                buf_puts(&c, "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"");
                buf_json_escape(&c, out.data + streamed);
                buf_puts(&c, "\"},\"finish_reason\":null}]}\n\n");
                if (send_all(fd, c.data, c.len) < 0) broken = 1;
                buf_free(&c);
            }
            break;
        }

        if (stream) {
            long safe = (long)out.len - hold;
            if (safe > (long)streamed) {
                Buf c = {0};
                buf_puts(&c, "data: {\"id\":\""); buf_puts(&c, id);
                buf_puts(&c, "\",\"object\":\"chat.completion.chunk\",\"created\":");
                char t[32]; snprintf(t, sizeof t, "%ld", created); buf_puts(&c, t);
                buf_puts(&c, ",\"model\":\""); buf_json_escape(&c, sv->model_name);
                buf_puts(&c, "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"");
                char saved = out.data[safe];
                out.data[safe] = '\0';
                buf_json_escape(&c, out.data + streamed);
                out.data[safe] = saved;
                buf_puts(&c, "\"},\"finish_reason\":null}]}\n\n");
                if (send_all(fd, c.data, c.len) < 0) { broken = 1; buf_free(&c); break; }
                buf_free(&c);
                streamed = (size_t)safe;
            }
        }
        logits = ember_forward(sv->m, sv->st, next, pos);
    }

    if (stream && !broken) {
        /* flush any held-back tail, then the terminating chunk + [DONE] */
        if (out.len > streamed) {
            Buf c = {0};
            buf_puts(&c, "data: {\"id\":\""); buf_puts(&c, id);
            buf_puts(&c, "\",\"object\":\"chat.completion.chunk\",\"created\":");
            char t[32]; snprintf(t, sizeof t, "%ld", created); buf_puts(&c, t);
            buf_puts(&c, ",\"model\":\""); buf_json_escape(&c, sv->model_name);
            buf_puts(&c, "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"");
            buf_json_escape(&c, out.data + streamed);
            buf_puts(&c, "\"},\"finish_reason\":null}]}\n\n");
            send_all(fd, c.data, c.len);
            buf_free(&c);
        }
        Buf c = {0};
        buf_puts(&c, "data: {\"id\":\""); buf_puts(&c, id);
        buf_puts(&c, "\",\"object\":\"chat.completion.chunk\",\"created\":");
        char t[32]; snprintf(t, sizeof t, "%ld", created); buf_puts(&c, t);
        buf_puts(&c, ",\"model\":\""); buf_json_escape(&c, sv->model_name);
        buf_puts(&c, "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"");
        buf_puts(&c, finish);
        buf_puts(&c, "\"}]}\n\ndata: [DONE]\n\n");
        send_all(fd, c.data, c.len);
        buf_free(&c);
    } else if (!stream) {
        Buf c = {0};
        buf_puts(&c, "{\"id\":\""); buf_puts(&c, id);
        buf_puts(&c, "\",\"object\":\"chat.completion\",\"created\":");
        char t[32]; snprintf(t, sizeof t, "%ld", created); buf_puts(&c, t);
        buf_puts(&c, ",\"model\":\""); buf_json_escape(&c, sv->model_name);
        buf_puts(&c, "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"");
        buf_json_escape(&c, out.data ? out.data : "");
        buf_puts(&c, "\"},\"finish_reason\":\""); buf_puts(&c, finish);
        buf_puts(&c, "\"}],\"usage\":{\"prompt_tokens\":");
        snprintf(t, sizeof t, "%d", n_prompt); buf_puts(&c, t);
        buf_puts(&c, ",\"completion_tokens\":");
        snprintf(t, sizeof t, "%d", completion_tokens); buf_puts(&c, t);
        buf_puts(&c, ",\"total_tokens\":");
        snprintf(t, sizeof t, "%d", n_prompt + completion_tokens); buf_puts(&c, t);
        buf_puts(&c, "}}");
        http_respond(fd, 200, "OK", "application/json", c.data, c.len);
        buf_free(&c);
    }

    buf_free(&out);
    free(stops.s);
    ember_sampler_free(&smp);
    json_free(req);
}

static void handle_models(Server *sv, int fd) {
    Buf b = {0};
    long created = (long)time(NULL);
    buf_puts(&b, "{\"object\":\"list\",\"data\":[{\"id\":\"");
    buf_json_escape(&b, sv->model_name);
    buf_puts(&b, "\",\"object\":\"model\",\"created\":");
    char t[32]; snprintf(t, sizeof t, "%ld", created); buf_puts(&b, t);
    buf_puts(&b, ",\"owned_by\":\"emberllm\"}]}");
    http_respond(fd, 200, "OK", "application/json", b.data, b.len);
    buf_free(&b);
}

static void handle_connection(Server *sv, int fd) {
    Buf raw = {0};
    Request req;
    if (read_request(fd, &raw, &req) != 0) { buf_free(&raw); return; }

    if (strcmp(req.method, "OPTIONS") == 0) {
        http_respond(fd, 204, "No Content", NULL, NULL, 0);
    } else if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/health") == 0) {
        http_respond(fd, 200, "OK", "application/json", "{\"status\":\"ok\"}", 15);
    } else if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/v1/models") == 0) {
        handle_models(sv, fd);
    } else if (strcmp(req.method, "POST") == 0 &&
               (strcmp(req.path, "/v1/chat/completions") == 0 ||
                strcmp(req.path, "/chat/completions") == 0)) {
        handle_chat(sv, fd, req.body, req.body_len);
    } else {
        http_error(fd, 404, "Not Found", "unknown route");
    }
    buf_free(&raw);
}

/* ------------------------------------------------------------------ */
/* public entry                                                       */
/* ------------------------------------------------------------------ */
int ember_serve(const char *model_path, const ServeConfig *cfg) {
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN); /* a client hangup must not kill the server */

    Server sv = {0};
    sv.cfg = cfg;
    sv.m = ember_model_load(model_path);
    if (!sv.m) return 1;
    sv.h = ember_model_header(sv.m);
    sv.tok = ember_tokenizer_new(sv.m);
    sv.st = ember_state_new(sv.m, cfg->ctx);
    sv.ctx = cfg->ctx > 0 && cfg->ctx < sv.h->max_seq_len ? cfg->ctx : sv.h->max_seq_len;

    /* model name = basename of the path, without extension */
    const char *base = strrchr(model_path, '/');
    base = base ? base + 1 : model_path;
    snprintf(sv.model_name, sizeof sv.model_name, "%s", base);
    char *dot = strrchr(sv.model_name, '.');
    if (dot) *dot = '\0';

    int threads = cfg->threads;
    if (threads < 0) threads = 1; /* server keeps it simple; no per-run autotune */
    ember_threads_init(threads);

    int ls = socket(AF_INET, SOCK_STREAM, 0);
    if (ls < 0) { ember_die("socket() failed"); }
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)cfg->port);
    addr.sin_addr.s_addr = inet_addr(cfg->host ? cfg->host : "127.0.0.1");
    if (addr.sin_addr.s_addr == INADDR_NONE) addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    if (bind(ls, (struct sockaddr *)&addr, sizeof addr) != 0) {
        fprintf(stderr, "ember: cannot bind %s:%d (%s)\n",
                cfg->host ? cfg->host : "127.0.0.1", cfg->port, strerror(errno));
        close(ls);
        return 1;
    }
    if (listen(ls, 16) != 0) { close(ls); ember_die("listen() failed"); }

    fprintf(stderr,
        "emberllm server on http://%s:%d  (model '%s', %d thread(s), ctx %d)\n"
        "  POST /v1/chat/completions   GET /v1/models   GET /health\n"
        "  Ctrl-C to stop.\n",
        cfg->host ? cfg->host : "127.0.0.1", cfg->port, sv.model_name, threads, sv.ctx);

    while (!g_stop) {
        struct pollfd pfd = { ls, POLLIN, 0 };
        int pr = poll(&pfd, 1, 500); /* wake periodically to check g_stop */
        if (pr <= 0) { if (pr < 0 && errno != EINTR) break; continue; }
        int fd = accept(ls, NULL, NULL);
        if (fd < 0) { if (errno == EINTR) continue; continue; }
        handle_connection(&sv, fd);
        close(fd);
    }

    fprintf(stderr, "\nember: shutting down\n");
    close(ls);
    ember_state_free(sv.st);
    ember_tokenizer_free(sv.tok);
    ember_threads_shutdown();
    ember_model_free(sv.m);
    return 0;
}
