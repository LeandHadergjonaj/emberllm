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

# Optimized: let the compiler use the host's SIMD. Threads arrive in Stage 2.
CFLAGS  ?= $(CSTD) $(WARN) -O3 -ffast-math -march=native -funroll-loops

BIN      = ember

.PHONY: all baseline debug test clean

all: $(BIN)

$(BIN): $(SRC) src/ember.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

# Deliberately un-optimized: the slow "before" side of the race demo.
baseline: $(SRC) src/ember.h
	$(CC) $(CSTD) $(WARN) -O0 -o $(BIN)-baseline $(SRC) $(LDLIBS)

# Deterministic build used by the transcript test: no FMA/fast-math reordering.
debug: $(SRC) src/ember.h
	$(CC) $(CSTD) $(WARN) -Werror -O2 -fno-fast-math -ffp-contract=off \
		-g -o $(BIN)-debug $(SRC) $(LDLIBS)

test: debug
	./tests/run_tests.sh

clean:
	rm -f $(BIN) $(BIN)-baseline $(BIN)-debug
