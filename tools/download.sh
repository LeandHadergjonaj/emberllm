#!/usr/bin/env bash
# download.sh — fetch upstream weights and convert them to .ember.
#
# Weights are never committed to the repo; this script pulls them on demand.
# Usage:  ./tools/download.sh [stories15M|stories110M]
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models
MODEL="${1:-stories15M}"

fetch() {  # fetch <url> <dest>
    if [ -f "$2" ]; then echo "have $2"; return; fi
    echo "downloading $2"
    if command -v curl >/dev/null; then curl -L -f -o "$2" "$1";
    elif command -v wget >/dev/null; then wget -O "$2" "$1";
    else echo "need curl or wget" >&2; exit 1; fi
}

TINYLLAMAS="https://huggingface.co/karpathy/tinyllamas/resolve/main"
TOKENIZER="https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin"

case "$MODEL" in
    stories15M|stories42M|stories110M)
        fetch "$TINYLLAMAS/${MODEL}.bin" "models/${MODEL}.bin"
        fetch "$TOKENIZER" "models/tokenizer.bin"
        python3 tools/convert.py llama2c \
            --checkpoint "models/${MODEL}.bin" \
            --tokenizer  "models/tokenizer.bin" \
            --out        "models/${MODEL}.ember"
        echo "ready: models/${MODEL}.ember"
        ;;
    *)
        echo "unknown model '$MODEL' (try: stories15M, stories42M, stories110M)" >&2
        exit 1
        ;;
esac
