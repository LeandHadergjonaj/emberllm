# emberllm

A CPU-only LLM inference engine, written from scratch in ~3,200 lines of C11.
No GPU, no runtime dependencies, one `Makefile`. It runs a 110M-parameter model
at **~260 tokens/second**, holds a real chat with **Qwen3-0.6B at ~60 tok/s** on
a plain laptop CPU, and serves an **OpenAI-compatible API** so it drops in behind
tools that already speak it.

```
$ ./ember chat models/qwen3-0.6b-q8.ember --threads auto

> Write a haiku about winter.
A frost-kissed sky
Whispers of snow on the wind,
A quiet winter's embrace.
```

```
$ ./ember generate models/stories110M-q8.ember -p "Once upon a time"
Once upon a time, there was a little boy named Timmy. Timmy loved to play
outside in the sunshine. One day, Timmy was walking in the forest when he saw
a giant. The giant was very tall and had a big smile on his face...
[prefill 5 tok @ 292 tok/s | decode 128 tok @ 259 tok/s | 1 thread]
```

A [VHS](https://github.com/charmbracelet/vhs) tape for the demo GIF lives at
[`bench/demo.tape`](bench/demo.tape) — regenerate with `vhs bench/demo.tape`.

## Quickstart

```sh
git clone https://github.com/LeandHadergjonaj/emberllm && cd emberllm
make                              # builds ./ember (C11, uses NEON/AVX where present)

# a 110M TinyStories model — MIT-licensed, fun, and fast
./tools/download.sh stories110M   # fetches weights, converts, quantizes to Q8_0
./ember generate models/stories110M-q8.ember -p "Once upon a time"

# a real chat model — Qwen3-0.6B (needs: pip install numpy safetensors)
./tools/download.sh qwen3-0.6b
./ember chat models/qwen3-0.6b-q8.ember --threads auto
```

No weights are committed; `download.sh` fetches them from Hugging Face and the
converter turns them into a single self-describing `.ember` file.

Every subcommand documents itself: `ember generate --help`, `ember chat --help`, etc.

## Run it as a server

`ember serve` turns the engine into a drop-in local backend for anything that
speaks the OpenAI API — chat UIs, editor plugins, agent frameworks — with no
extra dependencies (the HTTP server and JSON reader are hand-rolled in
[`src/server.c`](src/server.c) and [`src/json.c`](src/json.c)).

```sh
ember serve models/qwen3-0.6b-q8.ember --port 8080 --threads auto
```

```sh
# non-streaming
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Write a haiku about winter."}]
}'

# token streaming (Server-Sent Events)
curl -N http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Count to five."}],
  "stream": true
}'
```

It exposes **`POST /v1/chat/completions`** (with SSE streaming when
`"stream": true`), **`GET /v1/models`**, and **`GET /health`**. Requests honour
`temperature`, `top_p`, `top_k`, `max_tokens`, `stop` (string or array),
`repeat_penalty`, and `presence/frequency_penalty`. It also works from the OpenAI
Python client by pointing `base_url` at it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
r = client.chat.completions.create(model="ember", messages=[{"role":"user","content":"hi"}])
```

Scope is deliberately **single-stream**: one request is served at a time, using
the same KV cache and forward pass as the CLI. The prompt is assembled with the
ChatML template, so `serve` targets ChatML chat models (e.g. Qwen3).

## Sampling and reproducibility

`generate` and `chat` expose the standard sampling controls:

```sh
ember generate model.ember -p "..." \
  -t 0.8 --top-k 40 --top-p 0.95 --min-p 0.05 \
  --repeat-penalty 1.2 --repeat-last-n 256 \
  --presence-penalty 0.1 --frequency-penalty 0.1 --seed 42
```

- **`--repeat-penalty`** (`>1` discourages), **`--presence-penalty`**, and
  **`--frequency-penalty`** act over the last **`--repeat-last-n`** tokens — the
  single biggest quality win for small chat models, which otherwise loop. `chat`
  turns on a mild `--repeat-penalty 1.1` by default; `generate` leaves it off.
- **`--min-p`** keeps only tokens with probability ≥ `min_p × peak`.
- All penalty controls default to no-ops, so a plain temperature/top-k/top-p run
  is unchanged.

**Reproducibility:** a fixed `--seed` reproduces a run **bit-for-bit within one
binary**. It is *not* portable across builds or machines — the forward pass sums
floats in parallel and with `-ffast-math`, so reduction order (and thus the last
ULP of each logit) depends on thread count, SIMD width, and compiler flags. For a
byte-stable oracle use the `make debug` build single-threaded (`--threads 1`),
which is what the golden-transcript test relies on.

## Measured performance

Apple M1 Pro (6 performance cores), single stream. `pp` = prompt processing
(prefill), `tg` = token generation (decode). Numbers are `ember bench` means
over 5 runs; **measure your own hardware before quoting these.**

| Model | build | threads | decode | prefill |
|---|---|---:|---:|---:|
| stories110M | naive (`-O0`, scalar, fp32) | 1 | 8.8 tok/s | — |
| stories110M | optimized (Q8_0 + NEON) | 1 | **259 tok/s** | 292 tok/s |
| Qwen3-0.6B | Q8_0 + NEON | 1 | 50 tok/s | — |
| Qwen3-0.6B | Q8_0 + NEON | 6 | **64 tok/s** | 176 tok/s |

Those first two rows are the same 110M weights: **~29× from the naive build to
the optimized one**, purely from engineering. Run it yourself:

```sh
./bench/race.sh          # naive vs optimized, side by side
./ember bench models/qwen3-0.6b-q8.ember --pp 128 --tg 128 --threads 6
```

## How it works

Single-stream decode on a CPU is **memory-bandwidth-bound**: to generate one
token the engine streams essentially the whole weight file through the cores, so

> tokens/second ≈ memory bandwidth ÷ bytes per token

Everything in emberllm follows from that one fact:

- **Quantization is the biggest lever.** `Q8_0` (8-bit, 32-weight blocks with an
  fp16 scale) cuts bytes-per-token ~4× versus fp32 and is near-lossless — its
  perplexity is within 0.3% of fp32 on this model. `Q4_0` halves it again for
  some quality cost. Quantization is done offline by `ember quantize`.
- **Threads help until bandwidth saturates — so let the engine pick.** A hand-rolled
  pool splits each matmul (and attention over heads) across cores, but the useful
  thread count depends on the model: big models (Qwen3) scale to ~6 threads, while a
  small quantized model is bandwidth-bound and fastest at 1. Rather than guess, pass
  **`--threads auto`** and the engine measures a few token times and picks the best
  count for *your* model and machine. See [report.md](report.md) for the analysis.
- **SIMD keeps a core from going compute-bound.** NEON (`sdot`) on Apple Silicon,
  AVX2 on x86, with a scalar fallback. The single biggest kernel win was doing
  fp16→fp32 scale conversion in one hardware instruction instead of by hand.
- **Batched-GEMM prefill** streams each weight row once for the whole prompt
  (2.6× faster time-to-first-token than one-token-at-a-time).
- **mmap** loads weights with zero copies for instant startup.

The optimization ladder, on the same 110M model: naive fp32 (~9 tok/s) →
`-O3` + NEON → multithreading → **Q8_0 quantization** (~260 tok/s).

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
src/json.c        tiny dependency-free JSON reader (for request bodies)
src/util.c        checked allocation + fatal-error helpers
src/main.c        info | tokenize | generate | chat | serve | bench | perplexity | quantize
tools/convert.py  llama2.c + LLaMA-style HF safetensors -> .ember (numpy, no torch)
```

Several model families run through **one forward pass**, distinguished only by
fields in the file header and the presence of optional tensors — the converter
detects each variant from `config.json` rather than hard-coding it:

| Model | licence | `download.sh` name | arch notes | verified |
|---|---|---|---|:-:|
| TinyStories 15M/42M/110M | MIT | `stories15M` … | LLaMA-2, SentencePiece, interleaved RoPE | ✅ |
| Qwen3 0.6B / 1.7B | Apache-2.0 | `qwen3-0.6b` `qwen3-1.7b` | GQA, QK-norm, decoupled head_dim, `<think>` | ✅ |
| Qwen2.5 0.5B / 1.5B | Apache-2.0 | `qwen2.5-0.5b` `qwen2.5-1.5b` | GQA, **QKV bias**, byte-BPE | ✅ |
| SmolLM2 135M / 360M / 1.7B | Apache-2.0 | `smollm2-135m` … | LLaMA, GQA, tied embed, byte-BPE | ✅ |

"Verified" = converts and generates coherent text on this engine; the
TinyStories path is additionally checked token-for-token against `run.c`. Adding
a new LLaMA-style model is usually just a `download.sh` line — the converter
handles GQA, QK-norm, QKV bias, tied embeddings, and decoupled `head_dim`
automatically.

## Correctness

From-scratch inference fails on *conventions*, not math — a wrong RoPE pairing
or GQA index produces fluent-looking garbage. So the engine is checked against
references, not just eyeballed:

- The TinyStories forward pass matches karpathy's `run.c` **token-for-token** in
  greedy mode (`tests/run_tests.sh`, run in CI on macOS-arm + Linux x86/arm).
- The byte-level BPE tokenizer matches Hugging Face `tokenizers` on **512/512**
  fuzzed inputs (`tools/validate_bpe.py`).
- `ember perplexity` gates quantization: Q8_0 stays within a hair of fp32.

## Honest limitations

- **Tuned for Apple Silicon / NEON.** x86 has a correct AVX2 path but hasn't been
  performance-tuned; other targets fall back to a correct scalar build. Widening
  and tuning the AVX2 path is the first thing on the list.
- **`Q4_0` is smaller but not faster** here — its kernel isn't SIMD-optimized yet,
  so `Q8_0` is the sweet spot (fast *and* near-lossless). K-quants aren't
  implemented.
- **fp32 KV cache.** Fine at the default 4096-token context; fp16 KV would halve
  its memory for long contexts.
- **The pre-tokenizer regex is approximated** (C has no `\p{L}`). It's fuzz-clean
  on realistic English/code/Unicode but may differ from HF on pathological input.
- Scope is single-stream inference of LLaMA-family models up to ~2B parameters.
  No training, batching across requests, speculative decoding, or GGUF import.
- Malformed models and bad inputs fail with a clear `ember: ...` message rather
  than a crash, but the loader trusts a well-formed header's internal offsets
  once the top-level bounds check passes.

## Credits

Built independently, but it stands on ideas from
[llama2.c](https://github.com/karpathy/llama2.c) (the tiny-model demo, the export
script, and the correctness oracle), [llama.cpp](https://github.com/ggml-org/llama.cpp)
(block quantization, mmap, benchmark methodology), and
[calm](https://github.com/zeux/calm) (a self-describing single-file format). The
techniques are reimplemented from scratch; no code was copied.

MIT licensed. The models it runs are MIT (TinyStories) and Apache-2.0 (Qwen3).
