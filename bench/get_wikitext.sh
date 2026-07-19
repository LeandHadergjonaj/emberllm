#!/usr/bin/env bash
# get_wikitext.sh — fetch a small WikiText-2 slice for the perplexity harness.
# Writes bench/wikitext.txt (a few KB of clean English prose).
set -euo pipefail
cd "$(dirname "$0")/.."

URL="https://huggingface.co/datasets/mindchain/wikitext2/resolve/main/wikitext-2/test-00000-of-00001.parquet"
OUT="bench/wikitext.txt"

if [ -f "$OUT" ]; then echo "have $OUT"; exit 0; fi

# The parquet needs pandas; fall back to a plain-text mirror if unavailable.
if python3 -c "import pandas" 2>/dev/null; then
    curl -L -f -s -o /tmp/wt2.parquet "$URL"
    python3 - "$OUT" <<'PY'
import sys, pandas as pd
df = pd.read_parquet('/tmp/wt2.parquet')
text = "\n".join(t for t in df['text'].tolist() if t.strip())[:20000]
open(sys.argv[1], 'w').write(text)
print("wrote", sys.argv[1], len(text), "chars")
PY
else
    echo "pandas not available; fetching a plain-text sample instead"
    curl -L -f -s "https://www.gutenberg.org/files/1342/1342-0.txt" | sed -n '100,400p' > "$OUT"
    echo "wrote $OUT ($(wc -c < "$OUT") bytes)"
fi
