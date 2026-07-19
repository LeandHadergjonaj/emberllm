/* io.c — memory-mapped loading of .ember files.
 *
 * Loading is deliberately trivial: mmap the whole file, validate the header,
 * and hand back pointers straight into the mapping. Nothing is copied, so
 * startup is dominated by page faults on first access, not by parsing.
 */
#include "ember.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct EmberModel {
    int          fd;
    size_t       size;
    const uint8_t *base;      /* mmap base                                */
    const EmberHeader *hdr;
    const EmberTensor *table; /* base + hdr->tensor_table_offset          */
};

static void die(const char *msg) {
    fprintf(stderr, "ember: %s\n", msg);
}

EmberModel *ember_model_load(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { die("cannot open model file"); return NULL; }

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < (off_t)sizeof(EmberHeader)) {
        die("model file too small"); close(fd); return NULL;
    }

    void *base = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { die("mmap failed"); close(fd); return NULL; }

    const EmberHeader *hdr = (const EmberHeader *)base;
    if (memcmp(hdr->magic, "EMBR", 4) != 0 || hdr->version != EMBER_VERSION) {
        die("bad magic or version"); munmap(base, st.st_size); close(fd); return NULL;
    }
    if (hdr->tensor_table_offset + hdr->tensor_table_size > (uint64_t)st.st_size) {
        die("tensor table out of bounds"); munmap(base, st.st_size); close(fd); return NULL;
    }

    EmberModel *m = calloc(1, sizeof(*m));
    m->fd = fd;
    m->size = st.st_size;
    m->base = (const uint8_t *)base;
    m->hdr = hdr;
    m->table = (const EmberTensor *)(m->base + hdr->tensor_table_offset);

    /* Hint the kernel that we will stream the whole file; harmless if ignored. */
    madvise(base, st.st_size, MADV_WILLNEED);
    return m;
}

void ember_model_free(EmberModel *m) {
    if (!m) return;
    munmap((void *)m->base, m->size);
    close(m->fd);
    free(m);
}

const EmberHeader *ember_model_header(const EmberModel *m) { return m->hdr; }

const EmberTensor *ember_model_tensor(const EmberModel *m, const char *name) {
    for (int i = 0; i < m->hdr->n_tensors; i++) {
        /* names are null-padded to 48 bytes, so strncmp is safe */
        if (strncmp(m->table[i].name, name, sizeof(m->table[i].name)) == 0)
            return &m->table[i];
    }
    return NULL;
}

const void *ember_model_data(const EmberModel *m, const EmberTensor *t) {
    if (!t) return NULL;
    if (t->offset + t->nbytes > m->size) { die("tensor data out of bounds"); return NULL; }
    return m->base + t->offset;
}
