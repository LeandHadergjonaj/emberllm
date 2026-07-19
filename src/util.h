/* util.h — tiny shared helpers: fatal errors and checked allocation.
 *
 * The engine mmaps multi-hundred-MB models and allocates KV caches that scale
 * with context; an unchecked malloc that returns NULL turns into a segfault far
 * from the cause. These wrappers convert an allocation failure into one clear
 * message and a clean exit, which is what the rest of the code assumes.
 */
#ifndef EMBER_UTIL_H
#define EMBER_UTIL_H

#include <stddef.h>

/* Print "ember: <msg>\n" to stderr and exit(1). printf-style. */
void  ember_die(const char *fmt, ...);

/* malloc/calloc/realloc that never return NULL: on failure they ember_die with
 * the requested size. `what` is a short label used in the error message. */
void *ember_xmalloc(size_t n, const char *what);
void *ember_xcalloc(size_t count, size_t size, const char *what);
void *ember_xrealloc(void *p, size_t n, const char *what);

#endif /* EMBER_UTIL_H */
