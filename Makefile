# emberllm — a from-scratch CPU LLM inference engine in C11.
#
#   make            optimized build (native arch, NEON/AVX where available)
#   make lib        libember.a + libember.so (embed the engine; see ember.h)
#   make baseline   scalar single-thread build, for the "before" race demo
#   make debug      warnings-as-errors, no fast-math (deterministic; for tests)
#   make test       build + run the test suite
#   make clean

CC      ?= cc
CSTD     = -std=c11
# -std=c11 is strict ISO C; glibc then hides POSIX/BSD symbols we use
# (clock_gettime, madvise/MADV_WILLNEED, getrusage/RUSAGE_SELF). _DEFAULT_SOURCE
# re-exposes them. macOS headers expose these regardless, so this is a no-op there.
DEFS     = -D_DEFAULT_SOURCE
WARN     = -Wall -Wextra
SRC      = $(wildcard src/*.c)
HDR      = $(wildcard src/*.h)
LDLIBS   = -lm
PTHREAD  = -pthread

# Optimized: let the compiler use the host's SIMD (NEON on ARM, AVX2 on x86).
CFLAGS  ?= $(CSTD) $(DEFS) $(WARN) -O3 -ffast-math -march=native -funroll-loops $(PTHREAD)

BIN      = ember

# Everything except the CLI front-end is the embeddable core (libember).
CORE_SRC = $(filter-out src/main.c,$(SRC))
CORE_OBJ = $(CORE_SRC:.c=.o)

.PHONY: all lib baseline debug test clean

all: $(BIN)

# The CLI links against the static library, which also proves the public API in
# ember.h is a complete embedding surface.
$(BIN): src/main.o libember.a
	$(CC) $(CFLAGS) -o $@ src/main.o libember.a $(LDLIBS)

lib: libember.a libember.so

libember.a: $(CORE_OBJ)
	ar rcs $@ $(CORE_OBJ)

libember.so: $(CORE_SRC) $(HDR)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $(CORE_SRC) $(LDLIBS)

# object files (used by the optimized build + static lib)
src/%.o: src/%.c $(HDR)
	$(CC) $(CFLAGS) -c -o $@ $<

# Deliberately un-optimized and single-threaded: the "before" side of the race.
baseline: $(SRC) $(HDR)
	$(CC) $(CSTD) $(DEFS) $(WARN) -O0 $(PTHREAD) -o $(BIN)-baseline $(SRC) $(LDLIBS)

# Deterministic build used by the transcript test: no FMA/fast-math reordering.
debug: $(SRC) $(HDR)
	$(CC) $(CSTD) $(DEFS) $(WARN) -Werror -O2 -fno-fast-math -ffp-contract=off \
		$(PTHREAD) -g -o $(BIN)-debug $(SRC) $(LDLIBS)

test: debug
	./tests/run_tests.sh

clean:
	rm -f $(BIN) $(BIN)-baseline $(BIN)-debug src/*.o libember.a libember.so
