/* threads.c — a small fork-join thread pool for data-parallel matmuls.
 *
 * Workers are spawned once and sleep on a condition variable between rounds.
 * Each ember_parallel_for splits [0,n) into one slice per thread; the main
 * thread runs slice 0 itself and then waits for the workers. Only large
 * matmuls are dispatched here (see ember_matmul), so the mutex handshake is a
 * tiny fraction of the work and correctness is worth more than shaving it.
 *
 * macOS has no pthread_barrier_t, so completion is a plain counter + condvar.
 */
#include "kernels.h"

#include <pthread.h>
#include <stdlib.h>

#if defined(__APPLE__)
#  include <pthread/qos.h>
#endif

static int   g_nthreads = 1;
static pthread_t *g_threads;              /* g_nthreads-1 workers (main is slice 0) */

static pthread_mutex_t g_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_work = PTHREAD_COND_INITIALIZER; /* workers wait for a round */
static pthread_cond_t  g_done_cv = PTHREAD_COND_INITIALIZER; /* main waits for finish */

static unsigned long g_gen;   /* incremented once per dispatched round */
static int  g_done;           /* workers finished in the current round */
static int  g_stop;
static EmberRangeFn g_fn;
static void *g_ctx;
static int   g_n;

static void run_slice(int id) {
    long n = g_n, nth = g_nthreads;
    int lo = (int)(id * n / nth), hi = (int)((id + 1) * n / nth);
    if (hi > lo) g_fn(g_ctx, lo, hi);
}

static void *worker_main(void *arg) {
    int id = (int)(long)arg; /* 1 .. nthreads-1 */
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0); /* prefer P-cores */
#endif
    unsigned long seen = 0;
    for (;;) {
        pthread_mutex_lock(&g_mtx);
        while (g_gen == seen && !g_stop) pthread_cond_wait(&g_work, &g_mtx);
        if (g_stop) { pthread_mutex_unlock(&g_mtx); return NULL; }
        seen = g_gen;
        pthread_mutex_unlock(&g_mtx);

        run_slice(id); /* g_fn/g_ctx/g_n are stable for the whole round */

        pthread_mutex_lock(&g_mtx);
        if (++g_done == g_nthreads - 1) pthread_cond_signal(&g_done_cv);
        pthread_mutex_unlock(&g_mtx);
    }
}

void ember_threads_init(int nthreads) {
    g_nthreads = nthreads < 1 ? 1 : nthreads;
    g_gen = 0; g_done = 0; g_stop = 0;
    if (g_nthreads == 1) return;
    g_threads = malloc(sizeof(pthread_t) * (g_nthreads - 1));
    for (int i = 1; i < g_nthreads; i++)
        pthread_create(&g_threads[i - 1], NULL, worker_main, (void *)(long)i);
}

void ember_threads_shutdown(void) {
    if (g_nthreads <= 1) return;
    pthread_mutex_lock(&g_mtx);
    g_stop = 1;
    pthread_cond_broadcast(&g_work);
    pthread_mutex_unlock(&g_mtx);
    for (int i = 0; i < g_nthreads - 1; i++) pthread_join(g_threads[i], NULL);
    free(g_threads);
    g_threads = NULL;
    g_nthreads = 1;
}

int ember_nthreads(void) { return g_nthreads; }

void ember_parallel_for(EmberRangeFn fn, void *ctx, int n) {
    if (g_nthreads <= 1) { fn(ctx, 0, n); return; }

    pthread_mutex_lock(&g_mtx);
    g_fn = fn; g_ctx = ctx; g_n = n;
    g_done = 0;
    g_gen++;                          /* publish the round */
    pthread_cond_broadcast(&g_work);
    pthread_mutex_unlock(&g_mtx);

    run_slice(0);                     /* main's share, in parallel with workers */

    pthread_mutex_lock(&g_mtx);
    while (g_done != g_nthreads - 1) pthread_cond_wait(&g_done_cv, &g_mtx);
    pthread_mutex_unlock(&g_mtx);
    /* all workers have returned from run_slice: ctx is safe to reuse */
}
