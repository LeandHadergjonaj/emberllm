# emberllm — a from-scratch CPU LLM inference engine in C11.
#
#   make            optimized build (native arch, NEON/AVX where available)
#   make baseline   scalar single-thread build, for the "before" race demo
#   make debug      warnings-as-errors, no fast-math (deterministic; for tests)
#   make test       build + run the test suite
#   make clean

CC      ?= cc
CSTD     = -std=c11
WARN     = -Wall -Wextra
SRC      = $(wildcard src/*.c)
LDLIBS   = -lm
PTHREAD  = -pthread

# Optimized: let the compiler use the host's SIMD (NEON on ARM, AVX2 on x86).
CFLAGS  ?= $(CSTD) $(WARN) -O3 -ffast-math -march=native -funroll-loops $(PTHREAD)

BIN      = ember

.PHONY: all baseline debug test clean

all: $(BIN)

$(BIN): $(SRC) src/ember.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

# Deliberately un-optimized and single-threaded: the "before" side of the race.
baseline: $(SRC) src/ember.h
	$(CC) $(CSTD) $(WARN) -O0 $(PTHREAD) -o $(BIN)-baseline $(SRC) $(LDLIBS)

# Deterministic build used by the transcript test: no FMA/fast-math reordering.
debug: $(SRC) src/ember.h
	$(CC) $(CSTD) $(WARN) -Werror -O2 -fno-fast-math -ffp-contract=off \
		$(PTHREAD) -g -o $(BIN)-debug $(SRC) $(LDLIBS)

test: debug
	./tests/run_tests.sh

clean:
	rm -f $(BIN) $(BIN)-baseline $(BIN)-debug
