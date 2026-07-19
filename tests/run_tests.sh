#!/usr/bin/env bash
# run_tests.sh — emberllm correctness tests.
#
# Self-contained: needs only the committed golden files and a stories15M.ember
# (fetched via tools/download.sh if absent). No network reference required, so
# it runs unchanged in CI.
#
# Tests:
#   1. tokenizer golden vectors  — encode() matches committed ids
#   2. greedy transcript          — temp-0 generation matches committed text
#      (transcript is bit-exact only on the reference build; see README.)
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=models/stories15M.ember
BIN=./ember-debug
GOLD=tests/golden
fail=0

if [ ! -f "$MODEL" ]; then
    echo "== fetching test model =="
    ./tools/download.sh stories15M
fi
if [ ! -x "$BIN" ]; then
    echo "== building ember-debug =="
    make debug
fi

echo "== test 1: tokenizer golden vectors =="
i=0
while IFS= read -r line && IFS= read -r expect <&3; do
    got=$("$BIN" tokenize "$MODEL" "$line" 2>/dev/null)
    if [ "$got" = "$expect" ]; then
        echo "  ok: '$line'"
    else
        echo "  FAIL: '$line'"
        echo "    expected: $expect"
        echo "    got:      $got"
        fail=1
    fi
    i=$((i + 1))
done < "$GOLD/tokenize_cases.txt" 3< "$GOLD/tokenize_expected.txt"
echo "  ($i cases)"

echo "== test 2: greedy transcript =="
if [ "${EMBER_SKIP_TRANSCRIPT:-0}" = "1" ]; then
    echo "  skip: not the reference platform (fp reductions are not bit-portable)"
    if [ "$fail" -eq 0 ]; then echo "ALL TESTS PASSED"; else echo "TESTS FAILED"; fi
    exit $fail
fi
# command substitution strips trailing newlines from both sides, so only the
# meaningful (internal) text is compared
got=$("$BIN" generate "$MODEL" -p "Once upon a time" -n 200 -t 0 2>/dev/null)
want=$(cat "$GOLD/stories15M_greedy.txt")
if [ "$got" = "$want" ]; then
    echo "  ok: transcript matches golden"
else
    echo "  FAIL: transcript diverged from golden"
    diff <(printf '%s\n' "$got") <(printf '%s\n' "$want") | head -8
    fail=1
fi

if [ "$fail" -eq 0 ]; then echo "ALL TESTS PASSED"; else echo "TESTS FAILED"; fi
exit $fail
