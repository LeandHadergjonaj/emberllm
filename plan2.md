# plan2 — from "a working engine" to "an engine other people use"

## Context

emberllm today is a complete, correct, fast **demo of a from-scratch CPU inference
engine**: ~2,050 lines of C11, zero runtime dependencies, running two model families
(TinyStories, Qwen3-0.6B) at good speed on Apple Silicon, with CI, tests, and honest
docs. [PLAN.md](PLAN.md) built it; [report.md](report.md) tuned its threading.

This plan is about the **next axis: generality and usefulness to other people**. The
question that frames every item below is *"what stops someone who clones this repo
from using it for their own thing?"* — not *"what's technically missing versus
llama.cpp?"*. We are not trying to become llama.cpp; we are trying to be the small,
readable engine that a person can actually **run their model in**, **plug into their
tools**, and **learn from or extend**.

### The identity constraint (do not violate)

Every addition must preserve what makes this repo worth cloning:

- **Zero runtime dependencies.** No CURL, no BLAS, no web framework, no tokenizer lib.
  A hand-rolled HTTP server and a hand-rolled GGUF reader, or it doesn't ship.
- **Readable C11, one `make`.** New code reads like the existing code. Optional
  features go behind clean seams, not `#ifdef` swamps.
- **Correctness proven against references**, the way `run.c` and HF `tokenizers`
  already gate the current code.

If a feature can't be done within those constraints, it's a non-goal (see end).

---

## The five things that block other people (and what to do)

Ordered by **usefulness-per-effort**, which is also the recommended build order.

### Stream A — Robustness & real-use quality *(do first; small, unglamorous, load-bearing)*

Right now the engine assumes a well-formed model and a cooperative user. Before
anyone relies on it, it needs to not fall over. None of this is big; all of it is
what separates "a demo" from "a tool."

- **Input & error handling.** A malformed `.ember`, a missing tensor, an out-of-range
  token, a context overflow, or an OOM allocation should produce a clear error, not a
  segfault or silent garbage. Audit every `malloc`/`mmap`/tensor lookup for a NULL
  path. Add a `--ctx`-full behavior that's graceful (already partly there in chat).
- **Better sampling.** Add **repetition penalty**, **min-p**, and **presence/frequency**
  controls to `sample.c`. Current top-k/top-p is fine but small chat models loop
  without a repeat penalty; this is the single biggest perceived-quality win for real
  use, and it's ~30 lines.
- **Deterministic, documented seeding** across a session; expose `--seed` everywhere
  (mostly done) and document reproducibility limits.
- **A `--help` per subcommand** and consistent flag names.

*Effort:* ~2–3 days. *Risk:* low. *Why first:* everything downstream (server, more
models) rides on the engine being trustworthy.

### Stream B — More models via the converter *(breadth; medium effort, mostly config)*

The forward pass already handles LLaMA-style + GQA + QK-norm + NEOX/interleaved RoPE +
decoupled head_dim. Most popular small models are minor variations. Each new model is
mostly a converter entry plus, occasionally, one architectural knob.

- **Add, in rough order of demand:** Llama-3.2-1B/3B (tiktoken-style BPE, rope_scaling),
  SmolLM2-135M/360M/1.7B (Apache-2.0, easy), Qwen2.5-0.5B/1.5B (QKV bias — a small
  forward-pass addition), larger Qwen3 (1.7B/4B — same arch, just bigger).
- **Tokenizer gaps to close:** Llama-3's tiktoken byte-BPE (very close to the current
  byte-BPE path — mostly a different merge/regex source), and a real **SentencePiece
  `.model` (protobuf) reader** so Gemma / Llama-2 originals convert without a side file.
- **Generalize `config.json` handling** in `convert.py` so a new LLaMA-ish model is a
  data entry, not new code, whenever possible. Add `rope_scaling` (Llama-3 long-context)
  to the header + RoPE.
- **A model registry + `download.sh <name>`** for each supported model, with a table in
  the README of what's verified.

*Effort:* ~1 week for 3–4 models + the tiktoken tokenizer. *Risk:* low–medium (per-arch
convention bugs; the existing logit-diff and tokenizer-fuzz harnesses catch them).

### Stream C — A local server with an OpenAI-compatible API *(integration; the biggest "now I can use it" unlock)*

A CLI is for humans at a terminal. A **server** makes emberllm a drop-in backend for
everything that already speaks the OpenAI API: chat UIs, editor plugins, scripts,
agent frameworks. This is probably the highest-visibility single feature.

- **`ember serve <model>`**: a tiny hand-rolled HTTP/1.1 server (a few hundred lines,
  `poll()`-based, no framework) exposing **`POST /v1/chat/completions`** with
  **SSE token streaming** (`stream: true`), plus `/v1/models` and `/health`.
- Map the request → the existing chat template + sampler; stream decoded tokens as SSE
  deltas. Honor `max_tokens`, `temperature`, `top_p`, `stop`.
- **One connection at a time** to start (single-stream is our scope). Document it; a
  bounded request queue is a later nicety.
- Keep it dependency-free and behind its own `src/server.c` seam so the core stays
  clean and the server is optional.

*Effort:* ~1 week. *Risk:* medium (HTTP edge cases, SSE framing). *Payoff:* huge — it
turns a demo into infrastructure people leave running.

### Stream D — Quality/size: K-quants, fp16 KV, faster Q4 *(makes real models actually good)*

The current Q4_0 is "smaller but not faster" and loses ~11% perplexity; long context
eats fp32 KV memory. These are the knobs that decide whether a 1B model is *pleasant*
to run.

- **K-quants (`Q4_K`, `Q6_K`)**: 256-weight super-blocks with 6-bit sub-scales — the
  quality-per-bit the ecosystem actually uses. This is also the **prerequisite for GGUF**
  (Stream E), since most GGUF models are K-quant. Add to `kernels.c`/`quant.c` + header.
- **fp16 KV cache**: halve KV memory and the bytes the (now-threaded) attention streams
  at long context. NEON has native fp16 converts; x86 needs F16C. A `--kv-type` flag.
- **A SIMD int8 kernel for Q4_0/Q4_K** so 4-bit is genuinely faster than Q8, not just
  smaller.
- Re-run the `perplexity` gate across all quant types and publish a quality/size/speed
  table.

*Effort:* ~1.5 weeks. *Risk:* medium (quant math is fiddly; the perplexity gate is the
safety net). *Note:* items 2–3 here are already on report.md's open list.

### Stream E — GGUF import *(the largest generality unlock; do after D)*

GGUF is how the whole local-LLM world ships quantized models. A **read-only GGUF loader**
would let users run thousands of existing models — no Python, no conversion — by pointing
`ember` at a downloaded `.gguf`. This is the single biggest step toward "useful for other
people," and deliberately sequenced last of the model-facing work because it *depends on*
K-quants (D) and benefits from the broader architecture coverage (B).

- Parse the GGUF container (magic + KV-metadata + tensor table; well-documented) directly
  in C, mapping its metadata to our runtime config. Reuse the existing mmap + forward
  pass; GGUF tensors are already in block-quant layouts close to ours.
- Support the quant types we implement (Q8_0, Q4_0, then Q4_K/Q6_K from D). Clearly error
  on unsupported types rather than mis-decoding.
- Keep `.ember` as the native format (simpler, ours); GGUF is an *import* path, so we get
  the ecosystem's models without adopting its full complexity.

*Effort:* ~1.5–2 weeks. *Risk:* medium–high (format breadth, quant-layout exactness — a
GGUF-vs-HF logit diff is the required gate). *Payoff:* the biggest of anything here.

---

## Cross-cutting streams (parallel, lower urgency)

### Stream F — Platform reach (x86 + Windows)

Half the people who clone this run x86 or Windows. Today they get a *correct* build but a
slow one (AVX2 untuned) or no build (Windows).

- **Tune the AVX2 kernels** (dot + int8) and add an **x86 perf smoke** to CI so it doesn't
  regress. Optionally runtime CPU-feature dispatch instead of `-march=native`.
- **Windows support**: either a small CMake alongside the Makefile, or an MSVC/MinGW build
  path. Replace POSIX-only calls (`mmap`, `pthread`, `getrusage`) behind a thin platform
  shim (`src/platform.h`) — a clean seam that also documents the OS surface we depend on.
- Add `windows-latest` (and a real x86 runner) to the CI matrix.

*Effort:* ~1 week. *Risk:* medium.

### Stream G — Embeddability & developer experience

Make it easy to *build on* and *contribute to*, not just run.

- **`libember`**: the public API in `ember.h` is already close to a library. Split the
  CLI (`main.c`) from a linkable core, ship a `libember.a`/`.so` target, and document the
  ~10-function C API (load, tokenize, forward/prefill, sample). This is what lets others
  embed the engine.
- **Python bindings** via `ctypes`/`cffi` over that C ABI — enormous reach for ~a day of
  work, and still zero C-side dependencies.
- **Docs for extenders**: an `ARCHITECTURE.md` (how a token flows through the code), an
  **"add your own model" guide** (the converter + header fields + how to validate), and a
  `CONTRIBUTING.md`.
- **Releases**: tagged versions with prebuilt binaries for macOS-arm / linux-x64/arm, and
  a Homebrew formula. Lowers the barrier from "clone + build" to "download + run."

*Effort:* ~1 week. *Risk:* low.

---

## Recommended sequencing

A realistic, value-front-loaded order (each phase ships independently and leaves the repo
in a better, releasable state):

1. **Stream A** (robustness + repetition penalty) — 2–3 days. *Trust.*
2. **Stream C** (OpenAI-compatible server) — 1 week. *Biggest single usefulness jump.*
3. **Stream B** (3–4 more models + tiktoken tokenizer) — 1 week. *Breadth.*
4. **Stream D** (K-quants + fp16 KV + fast Q4) — 1.5 weeks. *Quality for real models.*
5. **Stream E** (GGUF import) — 1.5–2 weeks. *The ecosystem unlock; needs D.*
6. **Streams F & G** (platform + libraries + docs) — ongoing, interleaved. *Reach & DX.*

Rationale: A and C make what already exists *trustworthy and pluggable*; B and D make
*more/better models* runnable; E then opens the floodgates; F/G widen the audience and
lower the contribution barrier. Someone could stop after phase 2 and already have a
meaningfully more useful project.

## Key decisions to make before starting

1. **GGUF as import-only, `.ember` stays native?** (Recommended: yes — get the models
   without inheriting the whole format's complexity.)
2. **Server scope**: single-stream OpenAI-compatible only, or also a small web chat UI?
   (Recommended: API only first; a static HTML page can come free later.)
3. **Windows via CMake or a platform shim + Makefile?** (Recommended: platform shim first;
   add CMake only if MSVC users actually appear — mirrors the original PLAN.md stance.)
4. **How far up in model size to officially support?** (Recommended: cap verified support
   at ~3B; larger runs but is slow on CPU and out of the "demoable" spirit.)

## Non-goals (explicitly out of scope, to keep the identity)

- Becoming a general tensor framework or matching llama.cpp's backend/feature breadth.
- GPU backends, multi-request batching/continuous batching, tensor/pipeline parallelism.
- Training or fine-tuning.
- Any dependency that breaks the zero-runtime-deps promise (BLAS, CURL, a web framework,
  a tokenizer library). If a feature needs one, it's not this project's feature.
- Exotic architectures far from the LLaMA family (MoE, Mamba/SSM, vision) — interesting,
  but they'd double the code and dilute the "one readable forward pass" value.

## Open items inherited from report.md (folded into the streams above)

- SIMD `Q4_0` kernel → **Stream D**.
- fp16 KV cache → **Stream D**.
- P/E-core-aware thread pool (residual >6-thread cliff) → **Stream A/F** (a small pool
  refinement: size to physical performance cores, pin workers).
- `--threads auto` re-tuning as context grows → minor; revisit only if profiling shows it
  matters after fp16 KV lands.
