/* server.h — a tiny OpenAI-compatible HTTP server for one model, one stream.
 *
 * `ember serve <model>` exposes the loaded model over HTTP/1.1 with no external
 * dependencies (hand-rolled sockets + JSON). It speaks enough of the OpenAI API
 * to be a drop-in local backend:
 *
 *   POST /v1/chat/completions   (streaming via SSE when "stream": true)
 *   GET  /v1/models
 *   GET  /health
 *
 * Scope is deliberately single-stream: one request is served at a time. That
 * keeps the KV cache and the forward pass exactly as the CLI uses them.
 */
#ifndef EMBER_SERVER_H
#define EMBER_SERVER_H

#include <stdint.h>

typedef struct {
    const char *host;          /* bind address (default 127.0.0.1)             */
    int         port;          /* listen port (default 8080)                   */
    int         threads;       /* decode threads; -1 = autotune                */
    int         ctx;           /* context cap (0 = model max)                  */
    int         think;         /* 1 = allow <think> reasoning (Qwen-style)     */
    int         max_tokens;    /* default reply cap when the request omits it  */
    float       temperature;   /* sampling defaults when the request omits them */
    float       top_p;
    int         top_k;
    float       repeat_penalty;
    uint64_t    seed;
} ServeConfig;

/* Load `model_path` and serve until SIGINT/SIGTERM. Returns 0 on clean exit. */
int ember_serve(const char *model_path, const ServeConfig *cfg);

#endif /* EMBER_SERVER_H */
