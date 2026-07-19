# emberllm — Staged Build Plan

## Context

`emberllm` is a **CPU-only LLM inference engine written from scratch in C**, in the genre of llama.cpp / llama2.c but built independently. It must (a) be realistically implementable by one person in a few weeks part-time, (b) end in a visibly impressive terminal demo of fast text generation on ordinary no-GPU hardware, and (c) be a repo other people can build, run, and learn from.

This plan is the product of a five-track research pass (model selection, CPU optimization ladder, reference-project anatomy, demo/benchmark design, tooling audit) plus a completeness critique. Decisions locked for v1:

- **End target:** full arc — TinyStories bring-up → optimized engine → **Qwen3-0.6B instruct chat** (with SmolLM2-360M-Instruct as an optional de-risk checkpoint).
- **Language:** **pure C (C11)**, zero external runtime dependencies.
- **Hardware:** **Apple Silicon (NEON) first**; portable scalar build correct everywhere from day one; fast AVX2 x86 path is the first post-launch milestone. x86/ARM Linux CI from day one.
- **License:** MIT for the code (genre standard; both target models' licenses — MIT for TinyStories weights, Apache-2.0 for Qwen3 — are compatible with pointing users at them).

**Status:** implemented. Stages 0–4 are complete — the engine runs TinyStories and Qwen3-0.6B chat; see the [README](README.md) for measured numbers. This document is kept as the original design record.

---

## The one idea that shapes everything

Single-stream decode on CPU is **memory-bandwidth-bound**: generating one token streams essentially the entire weight file through the cores, so

> max tokens/sec ≈ memory bandwidth ÷ model bytes

Real CPU-side bandwidth (STREAM-measured, not headline): base M1 ≈ 59 GB/s, M3 ≈ 92 GB/s, M4 ≈ 103 GB/s; DDR4 laptop ≈ 40 GB/s, DDR5 ≈ 60 GB/s. That yields honest ceilings (real engines reach 50–80% of these):

| Model | fp32 | Q8_0 (~1.06 B/w) | Q4_0 (~0.56 B/w) |
|---|---|---|---|
| stories110M | ~130–210 tok/s | ~500–780 | ~940–1470 |
| Qwen3-0.6B (2.4 GB fp32) | ~25–38 | ~93–145 | ~176–274 † |

† assumes the whole file at Q4; with the planned mixed precision (Qwen3's 155M-param tied lm_head kept at Q8_0), the realistic Q4 ceiling is ~143–223 tok/s.

Consequences baked into this plan: **quantization and multithreading are the two big wins** (llama2.c measured exactly 3× from fp32→int8; threads give 3–6× then flatline at 4–8 when bandwidth saturates); SIMD mainly makes *prefill* fast; BLAS is pointless for decode; and every published number must be checked against this roofline to stay honest.

**Honest performance targets** (M-series Mac, to publish only after local measurement; stated for M3-class — scale by the roofline for older chips, e.g. on a base M1 the Q8_0 Qwen3 target is ~65% of ceiling and tight): stories110M ≥ 200 tok/s optimized; Qwen3-0.6B ≥ 60 tok/s at Q8_0, ≥ 80–150 tok/s at Q4_0. Anything ≥ 50% of roofline is a credible "fast" claim; 100+ tok/s is the visual "spew" effect (reading speed is ~6 tok/s).

## Target models

| | stories15M / stories110M | SmolLM2-360M-Instruct (optional) | Qwen3-0.6B (final demo) |
|---|---|---|---|
| Role | bring-up + first demo | de-risk checkpoint | real chat demo |
| License | MIT (vendorable) | Apache-2.0 | Apache-2.0 |
| Architecture | LLaMA-2 style: RMSNorm, RoPE θ=10k interleaved, SwiGLU, **MHA**, tied embed | + GQA 15/5, tied embed, θ=100k, rotate_half RoPE | + GQA 16/8, **QK-RMSNorm**, **head_dim=128 ≠ dim/heads**, tied embed, θ=1M, rotate_half RoPE |
| Tokenizer | SentencePiece 32k (~200 LoC in C) | byte-level BPE 49k | byte-level BPE **151,936** |
| Source format | llama2.c `.bin` (flat fp32 — no torch needed ever) | safetensors bf16 | safetensors bf16 |
| Why | llama2.c is a line-by-line correctness oracle; naive O(n²) attention fine at ctx ≤1024 | shares Stage-3 loader+BPE work, skips QK-norm & huge vocab | best sub-1B chat model as of 2026, genuinely fun to talk to |

Note: the TinyStories checkpoints tie embedding and lm_head too (llama2.c signals it via the sign of `vocab_size` in the `.bin` header — converter v1 must honor that), so tied-lm_head support lands in Stage 1 and quietly de-risks Stage 3.

Rejected: Llama 3.2 1B and Gemma 3 (gated, restrictive licenses, arch oddities), TinyLlama-1.1B (superseded in quality at 2× the size), GPT-2 (non-LLaMA arch, demo-weak). Re-check the model landscape at Stage-3 time; it moves fast.

## Repo architecture (decided up front to avoid a mid-project refactor)

- **One engine binary** (`ember`), one code path for all models. The weight file's header carries the full architecture config (dims, n_heads, n_kv_heads, head_dim, rope theta, qk_norm flag, tied-embedding flag, per-tensor quant type), so TinyStories-era and Qwen3-era files both run through the same forward pass with feature flags — no `run.c`/`runq.c` fork.
- **Single self-describing weight format** `.ember` (calm/GGUF idea, minimal execution): magic + version + config struct + **embedded tokenizer** (vocab, merges/scores, special tokens — no side files) + tensor table + 64-byte-aligned raw tensor data, laid out for mmap + pointer bumps. Quantization happens at conversion time.
- **A handful of C files, not one file** (~3–4k LoC total): `main.c` (CLI, chat/generate/bench modes), `model.c` (forward pass), `tokenizer.c`, `quant.c`, `kernels.c` (scalar + `#ifdef __ARM_NEON` / `#ifdef __AVX2__`), `threads.c` (hand-rolled pthreads pool), `io.c` (mmap loader). Single-file is iconic but unwieldy for the Qwen3 arc.
- `tools/convert.py` — the only Python, one-time use, depending on **`safetensors` + `numpy` only** (bf16 handled by viewing as uint16 and shifting; **no torch**). `tools/download.sh` wgets originals from HF.
- `tests/` (golden vectors + logit comparison), `bench/` (scripted benchmark + VHS tapes), `.github/workflows/ci.yml`.
- Build: **single portable Makefile** (`make`, `make test`, `make bench`). No CMake, no OpenMP (Apple clang still ships no libomp in 2026 — pthreads pool instead).

## Validation strategy (runs at every stage, not a stage of its own)

The genre's hard-won lesson: from-scratch failures are *convention* bugs (RoPE pairing style and HF's permuted Q/K weights, GQA head indexing, tokenizer edge cases, KV off-by-ones, double-BOS), not math bugs, and most produce plausible-looking degraded text rather than crashes. Defenses, all landing in Stage 0–1 and re-run after **every** optimization rung:

1. **Golden logits:** `tests/reference.py` (HF transformers, dev-only dep) dumps logits for fixed prompts; C test asserts max-abs-diff within tolerance (~1e-2 fp32 — threaded/SIMD reductions never bit-match; compare with tolerance, never `==`). Per-layer activation dumps for debugging; must include a **multi-token prompt at position > 0** (position 0 falsely passes RoPE bugs; single-token checks miss KV off-by-ones).
2. **Tokenizer golden vectors:** text→ids pairs generated by the HF tokenizer, checked in; plus a fuzz script diffing IDs against HF over random Unicode (critical for Stage-3 BPE).
3. **End-to-end transcript test:** greedy decode on stories15M must exactly match a checked-in transcript — but pinned to **one reference platform** (the macOS ARM runner) and a test build with `-fno-fast-math -ffp-contract=off`. Bit-exact fp32 is *not* portable: libm and FP-contraction differences across OS/arch can flip an argmax and cascade; the other CI platforms rely on the tolerance-based logit test instead.
4. **Perplexity harness** (small C or Python tool, WikiText-2 slice): lands **before** any Q4 work — logit-diff tolerance testing stops being meaningful at 4-bit; expect ≲0.1 PPL delta at Q8_0, and a big jump means a scale/rounding bug.
5. CI on `macos-15` (Apple Silicon), `ubuntu-24.04` (x86), `ubuntu-24.04-arm`: build `-Wall -Wextra -Werror`, run tests 1–2 everywhere and test 3 on the reference platform, with the 60 MB stories15M cached. Golden files are checked in, so HF transformers never runs in CI.

## Stages and milestones

### Stage 0 — Scaffolding (~half a week)
Repo layout above; Makefile; CI skeleton; `download.sh`; `convert.py` v1 reading llama2.c's `stories15M.bin` + karpathy's `tokenizer.bin` (a trivial flat score/length/string file — reading `tokenizer.model` would drag in the sentencepiece package and break the safetensors+numpy-only rule) → `.ember` v1; `tests/reference.py` producing the first golden files. **Exit:** CI green on all three runners for what exists so far (build tooling, converter run, golden-file generation — engine tests arrive with Stage 1); converter output inspected with a header-dump tool.

### Stage 1 — Correct but slow (fp32 TinyStories) (~1 week)
1. mmap loader for `.ember` (config, tokenizer, tensor pointers).
2. SentencePiece-BPE tokenizer in C (~200 LoC: byte-fallback, score-based greedy merge, `▁` dummy-prefix rules).
3. fp32 forward pass: RMSNorm (eps inside sqrt, fp32 accumulation), RoPE (**interleaved pairs** — llama2.c convention; converter must keep HF-permutation issues out by exporting from llama2.c layout), SwiGLU, naive attention with **KV cache** (fp32, flat `[layer][pos][head][head_dim]`; attention bound `<= pos`).
4. Greedy decode; then **sampling**: temperature, top-k, top-p, seeded xorshift RNG (deterministic given `--seed`). Greedy loops badly on these models — sampling is demo-critical, not polish.
5. Streaming token-by-token stdout with a final tok/s summary (the llama2.c-style minimal demo).

**Exit:** golden logits + transcript + tokenizer tests pass; stories15M tells coherent stories; measured single-thread fp32 baseline recorded (the "before" number every later stage is sold against).

### Stage 2 — Make it fast (~1–1.5 weeks) → **first shippable cut line**
Ordered by payoff-per-line, validated after each rung:
1. **pthreads pool** (~150 LoC: persistent workers, row-partitioned matvecs, spin barrier), `--threads` flag. Expect 3–6×, saturating at 4–8 threads (pin expectations to P-cores; document that SMT/E-cores hurt).
2. **NEON fp32 kernels** (`vfmaq_f32` matvec; the two hot kernels are ~95% of runtime). Expect ~2–4× single-thread.
3. **Q8_0 quantization**: 32-weight blocks, fp16 scale (stored as uint16 + NEON native converts), converter-side weight quant + runtime activation quant, `sdot` (`vdotq_s32`) int8 kernels. Norms stay fp32. Expect ~3× decode over fp32 (the single biggest win).
4. **Bench mode** (`ember bench`): llama-bench norms — separate pp512 / tg128, warmup + 5 reps, mean ± σ, reports hardware/threads/quant/commit. Peak-RSS measured properly (mmap pages don't show in RSS — never repeat the "30B in 6 GB" sin).
5. First README pass: VHS-scripted GIF (checked-in tape) of the **race mode** — `make baseline` (scalar single-thread) vs `make` (optimized) side by side, same prompt/seed, live tok/s counters. Race compares *speed counters*, not text: threaded reductions legitimately diverge; say so in the README rather than pretending bit-identity.

**Exit / cut line:** stories110M at 200+ tok/s on the M-series dev machine, honest bench table, demo GIF. If life intervenes, this is already a complete, publishable project.

### Stage 3 — Real chat: Qwen3-0.6B (~1.5–2 weeks; the schedule risk lives here)
Spike order chosen so the riskiest item starts first:
1. **Byte-level BPE tokenizer** (~300–600 LoC; *the* critical-path item — start it even while Stage 2 finishes): GPT-2 byte↔unicode map, hand-rolled pre-tokenizer splitter approximating the GPT-2 regex (no `\p{L}` in C — validate the approximation via the fuzz harness, which is the acceptance test), open-addressing hash map for the 151k vocab, special-token bypass, **streaming-safe decode** (buffer partial UTF-8 before printing).
2. **Converter v2**: safetensors (8-byte header-len + JSON + raw tensors) + bf16→fp32/Q8_0. **RoPE strategy decided here:** the engine supports two styles behind a `rope_style` header flag — interleaved pairs (llama2.c lineage) and rotate_half/neox (HF-native models; ~10 extra lines of C). The converter does **not** permute Q/K rows to unify them: HF-born models like Qwen3 have no "original" interleaved layout, and permutation would also require identically permuting Qwen3's per-head QK-RMSNorm γ vectors — a classic silent bug. A forward-pass flag is the less error-prone path.
3. **Forward-pass extensions**, each behind a config flag, each validated by per-layer logit diff vs HF: GQA (`kv_head = q_head / (n_q/n_kv)`, integer division), QK-RMSNorm per head before RoPE, explicit `head_dim=128` decoupled from `dim/n_heads` (q/k/v are 2048-wide from a 1024 model — the classic trap), tied lm_head, RoPE θ=1e6. *Optional de-risk:* bring up SmolLM2-360M first (needs only GQA + tied + BPE), then Qwen3.
4. **KV/context policy:** fp16 KV cache (NEON-native converts), `--ctx` flag defaulting to 4096 (Qwen3's 40k advertised context would naively cost ~1.8 GB fp32 — cap it), simple stop-on-full with a clear message; multi-turn cache reuse (append, don't recompute).
5. **Chat mode:** hardcoded ChatML template (`<|im_start|>…`), correct stop tokens, double-BOS check (diff templated-prompt IDs against HF `apply_chat_template`), **thinking mode disabled by default** (`enable_thinking=false` template variant / `/no_think`) — `<think>` blocks would wreck perceived latency; `--think` flag to show it deliberately. Per-model **sampling defaults shipped in the `.ember` header** (Qwen3's published: temp 0.6–0.7, top-p 0.95, top-k 20 — greedy visibly degrades it).
6. **Perplexity harness:** `ember perplexity` mode over a WikiText-2 slice fetched by `bench/get_wikitext.sh`; acceptance: Q8_0 within ≲0.1 PPL of fp32. Must merge before any Q4 code.
7. **Q4_0** (18 B per 32-block) — gated on item 6; embeddings/norms/tied lm_head stay ≥8-bit (see roofline footnote for the throughput cost of that choice).
8. **Chunked prefill:** process prompt tokens in blocks through batched GEMM-style kernels instead of one-at-a-time — without this, a 300-token templated chat prompt means multi-second TTFT, which reads as broken in a live demo. Report TTFT in bench output.

**Exit:** interactive chat with Qwen3-0.6B at ≥ 60 tok/s Q8_0 on an M3-class chip (chip-conditional — verify against the dev machine's own roofline before publishing) / target 80–150 (Q4_0), correct stops, snappy TTFT; perplexity within expected deltas; fuzz-clean tokenizer.

### Stage 4 — Launch polish (~half a week)
README anatomy that works in this genre, top to bottom: VHS GIF above the fold (chat + race mode); one-paragraph pitch ("N tok/s on a plain M-series CPU, zero dependencies, ~4k lines of C11"); copy-paste quickstart (`git clone && make && ./ember chat` after one `download.sh`); measured-numbers table regenerated by `make bench`; sample output; "how it works" section teaching the roofline model; **honest limitations** section (Apple-Silicon-optimized, scalar-only x86 for now, models up to ~1B). Credit inspirations (llama2.c, llama.cpp, calm) explicitly — techniques and format ideas are reimplemented, no code copied.

### Post-launch (explicitly out of scope for v1)
AVX2 kernels + x86 benchmarking (first follow-up — the "LLaMA goes faster on CPU" genre of before/after posts is a second launch for free), K-quants (Q4_K), speculative decoding with stories15M as draft model, GGUF import, Windows/MSVC, runtime CPU dispatch.

## Dependencies: needed vs deliberately avoided

| Needed | Avoided (and why) |
|---|---|
| C11 compiler + make + pthreads | PyTorch (2 GB for what safetensors+numpy do in 3 lines) |
| Python 3 + `safetensors` + `numpy` (converter only) | sentencepiece / HF tokenizers libs (needed subset ≈ 200–600 LoC) |
| `wget`/`curl` for weights (never committed) | CMake (single Makefile suffices at this scale), OpenMP (no libomp on Apple clang) |
| HF transformers as a *dev-only* test oracle | BLAS/Accelerate (zero decode benefit; fused dequant beats it; off-brand) |
| VHS (demo recording only), GitHub Actions | Google Highway/SIMDe, GGUF compat in v1, fat-binary dispatch |

## Schedule and risks

Total: **4–6 weeks at 10–15 h/week**, with a genuine shippable cut line at the end of Stage 2 (~2–3 weeks in). Top risks and mitigations:

1. **BPE tokenizer underestimated** → spike it first in Stage 3 (or late Stage 2), fuzz against HF continuously; fallback: ship Stage-2 repo, add chat later.
2. **Qwen3 convention bugs** (QK-norm, head_dim, permutation) → per-layer logit diffs + optional SmolLM2-360M stepping stone.
3. **Benchmark credibility** → roofline-check every number, llama-bench methodology, publish only locally measured figures.
4. **Demo latency** (TTFT, thinking mode) → chunked prefill milestone + thinking off by default.
5. **Scope creep** → post-launch list is a hard "no" list for v1.
