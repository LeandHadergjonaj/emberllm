#!/usr/bin/env bash
# race.sh — the "before vs after" demo: the same model, same prompt, run by the
# naive scalar build and by the optimized build, with tok/s for each.
#
# This measures the *engineering*, not the model: identical weights, only the
# implementation differs. Text may differ slightly between builds because
# threaded/SIMD floating-point reductions don't reduce in the same order — the
# speed counters are the point, not bit-identical output.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_F32="${1:-models/stories110M.ember}"
MODEL_Q8="${2:-models/stories110M-q8.ember}"
PROMPT="Once upon a time"
N=100
THREADS="${THREADS:-1}"

command -v ./ember-baseline >/dev/null 2>&1 || make baseline >/dev/null
[ -x ./ember ] || make >/dev/null

echo "========================================================"
echo " emberllm race — $MODEL_F32"
echo "========================================================"
echo
echo ">> NAIVE build (-O0, scalar, fp32, 1 thread)"
./ember-baseline generate "$MODEL_F32" -p "$PROMPT" -n "$N" -t 0.8 --seed 7
echo
echo ">> OPTIMIZED build (-O3 + NEON/AVX, Q8_0, $THREADS thread)"
./ember generate "$MODEL_Q8" -p "$PROMPT" -n "$N" -t 0.8 --seed 7 --threads "$THREADS"
echo
echo "(the tok/s lines above are printed to stderr by each run)"
