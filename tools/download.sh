#!/usr/bin/env bash
# download.sh — fetch upstream weights and convert them to .ember.
#
# Weights are never committed to the repo; this script pulls them on demand.
# Usage:  ./tools/download.sh <name>
#
# Supported <name> (see the model registry in the README):
#   stories15M stories42M stories110M          TinyStories (llama2.c, MIT)
#   qwen3-0.6b qwen3-1.7b                       Qwen3 (Apache-2.0)
#   qwen2.5-0.5b qwen2.5-1.5b                   Qwen2.5 (Apache-2.0, QKV bias)
#   smollm2-135m smollm2-360m smollm2-1.7b      SmolLM2 (Apache-2.0)
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

# fetch_hf <hf-repo> <local-name> — pull an HF safetensors model dir, convert to
# .ember with the general converter, and produce a Q8_0 build. Handles sharded
# checkpoints via model.safetensors.index.json.
fetch_hf() {
    local repo="$1" name="$2"
    local base="https://huggingface.co/${repo}/resolve/main"
    local dir="models/${name}-src"
    mkdir -p "$dir"
    for f in config.json tokenizer.json tokenizer_config.json generation_config.json; do
        fetch "$base/$f" "$dir/$f" || true
    done
    # single-file or sharded weights
    if curl -sL -f -o "$dir/model.safetensors.index.json" "$base/model.safetensors.index.json" 2>/dev/null; then
        for shard in $(python3 -c "import json,sys;print(' '.join(sorted(set(json.load(open('$dir/model.safetensors.index.json'))['weight_map'].values()))))"); do
            fetch "$base/$shard" "$dir/$shard"
        done
    else
        rm -f "$dir/model.safetensors.index.json"
        fetch "$base/model.safetensors" "$dir/model.safetensors"
    fi
    python3 tools/convert.py hf --model-dir "$dir" --out "models/${name}.ember"
    if [ -x ./ember ]; then
        ./ember quantize "models/${name}.ember" "models/${name}-q8.ember" q8_0
        echo "ready: models/${name}-q8.ember  (try: ./ember chat models/${name}-q8.ember --threads auto)"
    else
        echo "note: build the engine ('make'), then: ./ember quantize models/${name}.ember models/${name}-q8.ember q8_0"
    fi
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
    # needs Python with numpy + safetensors for the one-time conversion
    qwen3-0.6b)    fetch_hf "Qwen/Qwen3-0.6B"        "qwen3-0.6b" ;;
    qwen3-1.7b)    fetch_hf "Qwen/Qwen3-1.7B"        "qwen3-1.7b" ;;
    qwen2.5-0.5b)  fetch_hf "Qwen/Qwen2.5-0.5B-Instruct" "qwen2.5-0.5b" ;;
    qwen2.5-1.5b)  fetch_hf "Qwen/Qwen2.5-1.5B-Instruct" "qwen2.5-1.5b" ;;
    smollm2-135m)  fetch_hf "HuggingFaceTB/SmolLM2-135M-Instruct"  "smollm2-135m" ;;
    smollm2-360m)  fetch_hf "HuggingFaceTB/SmolLM2-360M-Instruct"  "smollm2-360m" ;;
    smollm2-1.7b)  fetch_hf "HuggingFaceTB/SmolLM2-1.7B-Instruct"  "smollm2-1.7b" ;;
    *)
        echo "unknown model '$MODEL'" >&2
        echo "try: stories15M stories110M qwen3-0.6b qwen2.5-0.5b smollm2-135m" >&2
        exit 1
        ;;
esac
