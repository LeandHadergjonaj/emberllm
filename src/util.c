/* util.c — fatal errors and checked allocation. See util.h. */
#include "util.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

void ember_die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("ember: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(1);
}

void *ember_xmalloc(size_t n, const char *what) {
    void *p = malloc(n ? n : 1);
    if (!p) ember_die("out of memory allocating %s (%zu bytes)", what, n);
    return p;
}

void *ember_xcalloc(size_t count, size_t size, const char *what) {
    void *p = calloc(count ? count : 1, size ? size : 1);
    if (!p) ember_die("out of memory allocating %s (%zu x %zu bytes)", what, count, size);
    return p;
}

void *ember_xrealloc(void *p, size_t n, const char *what) {
    void *q = realloc(p, n ? n : 1);
    if (!q) ember_die("out of memory reallocating %s (%zu bytes)", what, n);
    return q;
}
