# emberllm

A CPU-only LLM inference engine written from scratch in **~3,300 lines of
dependency-free C11**. One `Makefile`, a ~2-second build, a **123 KB binary**
(it links nothing but libSystem) that holds a real chat with Qwen3-0.6B,
serves an **OpenAI-compatible API**, and embeds as a C library or from Python
via ctypes.

And because "fast for a from-scratch engine" is meaningless without a
reference: emberllm is **benchmarked head-to-head against
[llama.cpp](https://github.com/ggml-org/llama.cpp)** — same machine, same
upstream weights, same measurements. The honest result: llama.cpp's CPU
backend is **1.8–2.5× faster at decoding** and **5–7× faster at prompt
processing**. What emberllm offers is the other side of that trade: **about
half of llama.cpp's single-stream decode speed from 1/130th as much code** —
an engine you can read in an afternoon, understand completely, and embed
anywhere a C compiler goes.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/decode-dark.png">
  <img alt="Single-stream decode benchmark: emberllm vs llama.cpp CPU on four models"
       src="docs/assets/decode-light.png">
</picture>

| model (Q8_0) | emberllm | llama.cpp (CPU) | llama.cpp (Metal GPU) |
|---|---:|---:|---:|
| stories110M | 229 tok/s | **559 tok/s** | 436 tok/s |
| SmolLM2-135M | 179 tok/s | **373 tok/s** | 245 tok/s |
| Qwen2.5-0.5B | 95 tok/s | **168 tok/s** | 143 tok/s |
| Qwen3-0.6B | 65 tok/s | 125 tok/s | **130 tok/s** |

*Single-stream decode, 128 tokens at a 128-token context depth, mean of 5
runs; the CPU engines at their best thread count from a 1–8 sweep, Metal
measured at one standard configuration. Apple M1 Pro, macOS, llama.cpp release
`b10068`. Note the Metal column: on three of these four small models,
llama.cpp's own **CPU** decode beats its GPU. Full methodology, thread-scaling
curves, prefill numbers, raw JSON, and every configuration where emberllm
loses:
**[notebooks/benchmark_vs_llamacpp.ipynb](notebooks/benchmark_vs_llamacpp.ipynb)**.*

```
$ ./ember chat models/qwen3-0.6b-q8.ember --threads auto

> Write a haiku about winter.
A frost-kissed sky
Whispers of snow on the wind,
A quiet winter's embrace.
```

## Quickstart

```sh
git clone https://github.com/LeandHadergjonaj/emberllm && cd emberllm
make                              # ~2 s; uses NEON/AVX where present

# a 110M TinyStories model — MIT-licensed, fun, and fast
./tools/download.sh stories110M   # fetches weights, converts, quantizes to Q8_0
./ember generate models/stories110M-q8.ember -p "Once upon a time"

# a real chat model — Qwen3-0.6B (needs: pip install numpy safetensors)
./tools/download.sh qwen3-0.6b
./ember chat models/qwen3-0.6b-q8.ember --threads auto
```

No weights are committed; `download.sh` fetches them from Hugging Face and the
converter turns them into a single self-describing `.ember` file. Every
subcommand documents itself: `ember generate --help`, `ember chat --help`, etc.
A [VHS](https://github.com/charmbracelet/vhs) tape for the demo GIF lives at
[`bench/demo.tape`](bench/demo.tape).

## Run it as a server

`ember serve` is a drop-in local backend for anything that speaks the OpenAI
API — chat UIs, editor plugins, agent frameworks — with no extra dependencies
(the HTTP server and JSON reader are hand-rolled in
[`src/server.c`](src/server.c) and [`src/json.c`](src/json.c)).

```sh
ember serve models/qwen3-0.6b-q8.ember --port 8080 --threads auto
```

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Write a haiku about winter."}],
  "stream": true
}'
```

It exposes **`POST /v1/chat/completions`** (with SSE streaming), **`GET
/v1/models`**, and **`GET /health`**; requests honour `temperature`, `top_p`,
`top_k`, `max_tokens`, `stop`, `repeat_penalty`, and
`presence/frequency_penalty`. The OpenAI Python client works by pointing
`base_url` at it. Scope is deliberately **single-stream** — one request at a
time, same forward pass as the CLI, ChatML prompt template (so it targets
ChatML chat models like Qwen3).

## Embed it (library + Python)

`make lib` builds `libember.a` / `libember.so` from the same core the CLI uses
— the `ember` binary is just `main.o` linked against the library, so the
~20-function C API in [`src/ember.h`](src/ember.h) is a complete embedding
surface: load, tokenize, prefill/forward, sample. Python bindings in
[`bindings/python`](bindings/python) are pure `ctypes`:

```python
from emberllm import Ember
with Ember("models/stories110M-q8.ember") as m:
    print(m.generate_str("Once upon a time", max_tokens=64, temperature=0.8))
```

New to the code? [ARCHITECTURE.md](ARCHITECTURE.md) traces a token through the
engine; [CONTRIBUTING.md](CONTRIBUTING.md) has an "add your own model" guide.

## The benchmark, honestly

The full comparison lives in
[notebooks/benchmark_vs_llamacpp.ipynb](notebooks/benchmark_vs_llamacpp.ipynb),
with raw data in [`bench/results/`](bench/results) and the driver in
[`bench/compare_llamacpp.py`](bench/compare_llamacpp.py). The design, in one
paragraph: both engines run the **same upstream weights** (Qwen3-0.6B,
Qwen2.5-0.5B-Instruct, SmolLM2-135M-Instruct, stories110M), each converted and
quantized to **Q8_0 by that engine's own tooling**; llama.cpp is a **stock
release build** at pinned tag `b10068` (CPU build with Accelerate, plus a Metal
build for context); both are measured the same way (batch prefill tok/s, and
decode tok/s at a pinned context depth; 1 warmup + 5 timed reps, mean ± sd,
serially, on a quiet machine — on battery, which Apple Silicon doesn't
throttle for) — 220 matrix measurements across 5 thread counts, 2 context
depths, and 2 quantizations, plus a 2-row end-of-run drift check that came
back ~3% slower (small, and it bounds any thermal/battery drift). Spot-check:
under greedy decoding both engines emit **identical text** for a 40-token
stories110M continuation (transcript in `bench/results/`), and near-identical
text on Qwen3-0.6B.

Where emberllm **loses**, plainly:

- **Every head-to-head throughput configuration.** Best case is single-threaded
  decode of small Q8_0 models (61–66% of llama.cpp); worst is multi-threaded
  prefill (llama.cpp's tuned GEMMs + Accelerate are 5–7× faster at pp512).
- **Q4_0 decode is ~4× slower** (44 vs 175 tok/s on Qwen3-0.6B): emberllm's
  Q4_0 kernel is scalar and unoptimized, while llama.cpp runtime-repacks Q4_0
  for NEON. Use Q8_0 with emberllm.
- **The GPU exists, and it owns prefill.** llama.cpp with Metal prefills
  3,200–15,700 tok/s — ~3× its own CPU backend, 16–22× emberllm. (Decode is
  a different story: at these model sizes Metal only beats llama.cpp's CPU
  decode on Qwen3-0.6B, and loses to it on the three smaller models — GPU
  dispatch overhead dominates tiny kernels. It still beats emberllm's decode
  everywhere.)
- **Breadth.** llama.cpp runs nearly every open model family at every size,
  with K-quants, grammars, parallel serving, and a huge ecosystem. emberllm
  runs LLaMA-family models up to ~2B parameters.

Where emberllm stands, plainly:

- **Decode holds 41–57% of llama.cpp's best-thread speed** (40–55% at the
  deeper 512-token context) — 65–95 tok/s on real 0.5–0.6B chat models,
  179–229 tok/s on 110–135M models — far above reading speed, from a codebase
  ~130× smaller (3,313 lines vs ~428k core lines at `b10068`).
- **Memory stays proportional to what you asked for**: peak RSS for a 64-token
  Qwen3-0.6B generation was **681 MB vs 3.3 GB** for llama.cpp at defaults
  (1.5 GB with `-c 512`) — one indicative config, methodology caveats in the
  notebook.
- **Zero dependencies, ~2 s build, 123 KB binary** vs a cmake build producing
  ~4.5 MB of binaries+libraries. Both are small; only one fits in your head.

To reproduce: the notebook's §11 lists the exact commands (build the pinned
llama.cpp, convert the same weights with its own tooling, run
`bench/compare_llamacpp.py` on a quiet machine, re-execute the notebook).

## Why it's as fast as it is

Single-stream decode on a CPU is **memory-bandwidth-bound**: generating one
token streams essentially the whole weight file through the cores, so

> tokens/second ≈ memory bandwidth ÷ bytes per token

Everything in emberllm follows from that one fact:

- **Quantization is the biggest lever.** `Q8_0` (8-bit, 32-weight blocks with
  an fp16 scale — the same block format llama.cpp uses) cuts bytes-per-token
  ~4× versus fp32 and is near-lossless. Done offline by `ember quantize`.
- **Threads help until bandwidth saturates — so let the engine pick.** A
  hand-rolled pool splits matmuls and attention across cores, but the useful
  count depends on the model: in the benchmark, decode peaks anywhere from
  **1 thread** (SmolLM2 on emberllm) to 6, and nearly every curve — on both
  engines — falls at 8 threads, where E-cores stall the barrier.
  **`--threads auto`** measures a few token times and picks for your machine.
  See [report.md](report.md) for the full thread-scaling investigation.
- **SIMD keeps a core from going compute-bound**: NEON (`sdot`) on Apple
  Silicon, AVX2 on x86, scalar fallback elsewhere.
- **Batched-GEMM prefill** streams each weight row once for the whole prompt;
  **mmap** loads weights with zero copies.

The same 110M model, on the same machine and protocol as the comparison
(transcript: [`bench/results/race_ladder.txt`](bench/results/race_ladder.txt)):
the naive build (`-O0`, scalar, fp32) decodes at **9.0 tok/s**; the optimized
build (Q8_0 + NEON) at **253.9 tok/s** on one thread — **28× from engineering
alone**. (Both at a 5-token prompt; decode slows as context grows, which is
why the headline table's depth-128 figure is lower.) Run it yourself:
`./bench/race.sh`.

The gap that remains against llama.cpp is also engineering: tuned/repacked
GEMM kernels, Accelerate BLAS on the prompt path, threaded decode that scales
past 2 threads. The notebook's §9 shows both engines sit in the same
memory-bandwidth regime — llama.cpp just streams bytes more efficiently.

## What's in the box

```
src/ember.h       the .ember file format + public API
src/io.c          mmap loader
src/model.c       forward pass: RMSNorm, RoPE, GQA attention, SwiGLU, KV cache,
                  QK-norm, batched prefill
src/kernels.c     matmul + dot kernels (scalar / NEON / AVX2), Q8_0/Q4_0
src/threads.c     fork-join thread pool
src/tokenizer.c   SentencePiece-BPE and byte-level BPE, both from scratch
src/quant.c       offline fp32 -> Q8_0/Q4_0
src/sample.c      greedy / temperature / top-k / top-p / min-p + repeat penalties
src/server.c      hand-rolled OpenAI-compatible HTTP/1.1 server (SSE streaming)
src/json.c        tiny dependency-free JSON reader
src/util.c        checked allocation + fatal-error helpers
src/main.c        info | tokenize | generate | chat | serve | bench | perplexity | quantize
tools/convert.py  llama2.c + LLaMA-style HF safetensors -> .ember (numpy, no torch)
bench/            benchmark driver, results (raw JSON), demo tape
notebooks/        the llama.cpp comparison notebook
```

Several model families run through **one forward pass**, distinguished only by
header fields and optional tensors — the converter detects each variant from
`config.json`:

| Model | licence | `download.sh` name | arch notes | verified |
|---|---|---|---|:-:|
| TinyStories 15M/42M/110M | MIT | `stories15M` … | LLaMA-2, SentencePiece, interleaved RoPE | ✅ |
| Qwen3 0.6B / 1.7B | Apache-2.0 | `qwen3-0.6b` `qwen3-1.7b` | GQA, QK-norm, decoupled head_dim, `<think>` | ✅ |
| Qwen2.5 0.5B / 1.5B | Apache-2.0 | `qwen2.5-0.5b` `qwen2.5-1.5b` | GQA, **QKV bias**, byte-BPE | ✅ |
| SmolLM2 135M / 360M / 1.7B | Apache-2.0 | `smollm2-135m` … | LLaMA, GQA, tied embed, byte-BPE | ✅ |

Adding a new LLaMA-style model is usually just a `download.sh` line — the
converter handles GQA, QK-norm, QKV bias, tied embeddings, and decoupled
`head_dim` automatically.

## Correctness

From-scratch inference fails on *conventions*, not math — a wrong RoPE pairing
produces fluent-looking garbage. So the engine is checked against references,
not just eyeballed:

- The TinyStories forward pass matches karpathy's `run.c` **token-for-token**
  in greedy mode (`tests/run_tests.sh`, run in CI on macOS-arm + Linux x86/arm).
- Under greedy decoding, emberllm and llama.cpp emit **identical text** for a
  40-token stories110M continuation, and near-identical text on Qwen3-0.6B
  (each engine quantized the weights independently, so low-order bits differ)
  — one prompt each, compared as text; transcripts in
  [`bench/results/correctness_spotcheck.txt`](bench/results/correctness_spotcheck.txt).
- The byte-level BPE tokenizer matches Hugging Face `tokenizers` on **512/512**
  fuzzed inputs (`tools/validate_bpe.py`).
- `ember perplexity` gates quantization; a fixed `--seed` reproduces a run
  bit-for-bit within one binary (see `make debug` for the deterministic build).

## Honest limitations

- **Slower than llama.cpp on every measured configuration** — see
  [the benchmark](#the-benchmark-honestly) for exactly how much. If you want
  maximum speed or model breadth, use llama.cpp; this project's value is a
  complete, readable, embeddable engine.
- **Tuned for Apple Silicon / NEON.** The AVX2 path is correct but untuned;
  other targets fall back to scalar. The measured comparison is one machine
  (M1 Pro) — relative results elsewhere will differ.
- **`Q4_0` is smaller but much slower** (scalar kernel; measured 4× behind
  llama.cpp). Q8_0 is the sweet spot. **K-quants are not implemented** — the
  main remaining size/quality lever, and the prerequisite for GGUF import.
- **fp16 KV cache** (`--kv-type f16`) halves KV memory; fp32 stays the default.
- **The pre-tokenizer regex is approximated** (C has no `\p{L}`); fuzz-clean on
  realistic input, may differ from HF on pathological Unicode.
- Scope is single-stream inference of LLaMA-family models up to ~2B params. No
  training, no batched serving, no speculative decoding, no GGUF import.
- Malformed models fail with a clean `ember: ...` error, but the loader trusts
  a well-formed header's internal offsets after the top-level bounds check.

## Credits

Built independently, but it stands on ideas from
[llama2.c](https://github.com/karpathy/llama2.c) (the tiny-model demo, the export
script, and the correctness oracle), [llama.cpp](https://github.com/ggml-org/llama.cpp)
(block quantization, mmap, benchmark methodology), and
[calm](https://github.com/zeux/calm) (a self-describing single-file format). The
techniques are reimplemented from scratch; no code was copied.

MIT licensed. The models it runs are MIT (TinyStories) and Apache-2.0 (Qwen3).
