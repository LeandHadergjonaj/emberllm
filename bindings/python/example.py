#!/usr/bin/env python3
"""Minimal emberllm Python example. Run from the repo root after `make lib`:

    python3 bindings/python/example.py models/stories110M-q8.ember "Once upon a time"
"""
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from emberllm import Ember  # noqa: E402


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "models/stories15M.ember"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Once upon a time"
    with Ember(model) as m:
        h = m.header
        print(f"loaded {model}: dim={h.dim} layers={h.n_layers} vocab={h.vocab_size}",
              file=sys.stderr)
        print(prompt, end="", flush=True)
        for piece in m.generate(prompt, max_tokens=64, temperature=0.8, top_k=40, seed=42):
            print(piece, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
