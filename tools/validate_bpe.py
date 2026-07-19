#!/usr/bin/env python3
"""validate_bpe.py — fuzz the C byte-level BPE tokenizer against HF `tokenizers`.

This is the acceptance test for the pre-tokenizer regex approximation in
tokenizer.c: it compares `ember tokenize` output to the reference tokenizer over
a spread of English, code, Unicode, whitespace, and random inputs.

Requires:  pip install tokenizers   (lightweight; no torch)
Usage:     python3 tools/validate_bpe.py models/qwen3-0.6b.ember models/qwen3/tokenizer.json
"""
import random
import subprocess
import sys


def main():
    ember_model = sys.argv[1] if len(sys.argv) > 1 else "models/qwen3-0.6b.ember"
    tok_json = sys.argv[2] if len(sys.argv) > 2 else "models/qwen3/tokenizer.json"
    from tokenizers import Tokenizer

    ref = Tokenizer.from_file(tok_json)
    cases = [
        "Hello, world!", "The quick brown fox jumps over the lazy dog.",
        "def foo(x):\n    return x*2 + 1  # comment",
        "  leading spaces and\ttabs\nand newlines",
        "Contractions: I'm you're it's we'll they've don't",
        "Unicode: café résumé naïve 日本語 中文 🎉",
        "numbers 123 and 4567", "URL https://example.com/p?q=1&b=2",
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
    ]
    random.seed(0)
    chars = list("ab 12\t\n,.!()xyz'-_/AZ")
    for _ in range(500):
        cases.append("".join(random.choice(chars) for _ in range(random.randint(1, 40))))

    fails = 0
    for s in cases:
        want = ref.encode(s, add_special_tokens=False).ids
        out = subprocess.run(
            ["./ember", "tokenize", ember_model, s, "--no-bos"],
            capture_output=True, text=True).stdout.strip()
        got = [int(x) for x in out.split()] if out else []
        if got != want:
            fails += 1
            if fails <= 10:
                print(f"MISMATCH {s!r}\n  want={want}\n  got ={got}")
    total = len(cases)
    print(f"\n{total - fails}/{total} match")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
