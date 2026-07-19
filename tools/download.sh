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
        # also produce a Q8_0 build (needs the engine binary)
        if [ -x ./ember ]; then
            ./ember quantize "models/${MODEL}.ember" "models/${MODEL}-q8.ember" q8_0
        else
            echo "note: build the engine ('make') then run"
            echo "      ./ember quantize models/${MODEL}.ember models/${MODEL}-q8.ember q8_0"
        fi
        echo "ready: models/${MODEL}.ember"
        ;;
    qwen3-0.6b)
        # needs Python with numpy + safetensors for the one-time conversion
        BASE="https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main"
        mkdir -p models/qwen3
        for f in config.json tokenizer.json tokenizer_config.json generation_config.json model.safetensors; do
            fetch "$BASE/$f" "models/qwen3/$f"
        done
        python3 tools/convert.py qwen --model-dir models/qwen3 --out models/qwen3-0.6b.ember
        if [ -x ./ember ]; then
            ./ember quantize models/qwen3-0.6b.ember models/qwen3-0.6b-q8.ember q8_0
            echo "ready: models/qwen3-0.6b-q8.ember  (try: ./ember chat models/qwen3-0.6b-q8.ember --threads 6)"
        else
            echo "note: build the engine ('make'), then quantize:"
            echo "      ./ember quantize models/qwen3-0.6b.ember models/qwen3-0.6b-q8.ember q8_0"
        fi
        ;;
    *)
        echo "unknown model '$MODEL' (try: stories15M, stories110M, qwen3-0.6b)" >&2
        exit 1
        ;;
esac
