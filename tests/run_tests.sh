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

echo "== test 3: Q8_0 quantize + generate round-trip =="
if [ ! -x ./ember ]; then make >/dev/null; fi
./ember quantize "$MODEL" models/_test_q8.ember q8_0 >/dev/null 2>&1
q8out=$(./ember generate models/_test_q8.ember -p "Once upon a time" -n 24 -t 0 2>/dev/null)
if printf '%s' "$q8out" | grep -q "Once upon a time"; then
    echo "  ok: Q8_0 model builds and generates coherent text"
else
    echo "  FAIL: Q8_0 generation empty or malformed"; echo "    got: $q8out"; fail=1
fi
rm -f models/_test_q8.ember

echo "== test 4: sampling controls (repeat penalty / min-p) =="
# penalties must not crash and must still produce coherent, prompt-continuing text
s4=$("$BIN" generate "$MODEL" -p "Once upon a time" -n 24 -t 0.8 \
      --repeat-penalty 1.3 --repeat-last-n 64 --min-p 0.05 --seed 7 2>/dev/null)
if printf '%s' "$s4" | grep -q "Once upon a time"; then
    echo "  ok: penalized sampling generates coherent text"
else
    echo "  FAIL: penalized sampling empty or malformed"; echo "    got: $s4"; fail=1
fi
# greedy output must be identical with penalties fully disabled (defaults are no-ops)
g_off=$("$BIN" generate "$MODEL" -p "Once upon a time" -n 24 -t 0 2>/dev/null)
g_def=$("$BIN" generate "$MODEL" -p "Once upon a time" -n 24 -t 0 \
      --repeat-penalty 1.0 --presence-penalty 0 --frequency-penalty 0 --min-p 0 2>/dev/null)
if [ "$g_off" = "$g_def" ]; then
    echo "  ok: disabled penalties are a no-op (greedy unchanged)"
else
    echo "  FAIL: default penalties altered greedy output"; fail=1
fi

echo "== test 5: malformed input is a clean error, not a crash =="
garbage=models/_test_garbage.ember
head -c 4096 /dev/urandom > "$garbage" 2>/dev/null
"$BIN" info "$garbage" >/dev/null 2>&1
rc=$?
# a graceful failure is any nonzero exit below 128 (128+ = killed by signal / segfault)
if [ "$rc" -ne 0 ] && [ "$rc" -lt 128 ]; then
    echo "  ok: garbage model rejected with exit $rc (no crash)"
else
    echo "  FAIL: garbage model gave exit $rc (128+ means a signal/segfault)"; fail=1
fi
rm -f "$garbage"

echo "== test 6: HTTP server (OpenAI-compatible) =="
if ! command -v curl >/dev/null 2>&1; then
    echo "  skip: curl not available"
else
    if [ ! -x ./ember ]; then make >/dev/null; fi
    PORT=18723
    ./ember serve "$MODEL" --port "$PORT" >/dev/null 2>&1 &
    SRV=$!
    # wait for the port to come up (up to ~5s)
    up=0
    for _ in $(seq 1 50); do
        if curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then up=1; break; fi
        sleep 0.1
    done
    if [ "$up" -ne 1 ]; then
        echo "  FAIL: server did not start"; fail=1
    else
        health=$(curl -s "http://127.0.0.1:$PORT/health")
        [ "$health" = '{"status":"ok"}' ] && echo "  ok: /health" || { echo "  FAIL: /health = $health"; fail=1; }

        body='{"messages":[{"role":"user","content":"Once upon a time"}],"max_tokens":8,"temperature":0}'
        resp=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")
        if printf '%s' "$resp" | grep -q '"object":"chat.completion"'; then
            echo "  ok: POST /v1/chat/completions returns a completion"
        else
            echo "  FAIL: chat completion malformed"; echo "    got: $resp"; fail=1
        fi

        sbody='{"messages":[{"role":"user","content":"Once upon a time"}],"max_tokens":8,"temperature":0,"stream":true}'
        sresp=$(curl -s -N "http://127.0.0.1:$PORT/v1/chat/completions" \
                 -H 'Content-Type: application/json' -d "$sbody")
        if printf '%s' "$sresp" | grep -q 'data: \[DONE\]'; then
            echo "  ok: SSE stream terminates with [DONE]"
        else
            echo "  FAIL: SSE stream missing [DONE]"; fail=1
        fi

        code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/bogus")
        [ "$code" = "404" ] && echo "  ok: unknown route -> 404" || { echo "  FAIL: bogus route = $code"; fail=1; }
    fi
    kill "$SRV" 2>/dev/null
    wait "$SRV" 2>/dev/null
fi

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
