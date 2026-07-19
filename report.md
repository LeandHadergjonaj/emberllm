# Thread scaling: prefill vs decode

An investigation into why decode throughput *falls* as thread count rises on
small models, while prefill scales normally. The measurement pass (below) was
followed by a fixes pass; the fixes and their results are summarized in
**[Fixes applied](#fixes-applied)** immediately after the TL;DR. The rest of the
document is the original investigation that motivated them.

## TL;DR

- **Prefill scales, decode often doesn't** — and for small/quantized models decode
  gets *slower* with more threads. The original hypothesis (per-token work is too
  small to amortize thread-pool dispatch overhead) **held up**, and a controlled
  experiment pins it down exactly.
- The deciding factor is **per-token matmul work ÷ per-dispatch sync overhead**,
  not model size as such. The cleanest proof: the *same* 110M model **degrades in
  Q8_0 but scales in fp32** (81→126 tok/s), because fp32 does 4× the work per
  matmul. Quantizing a model — making it faster — makes it *less* parallelizable.
- Two effects compound the picture and are worth separating out:
  1. a **P/E-core cliff**: past the 6 performance-core count, decode drops sharply
     on this 6P+2E chip (the barrier waits on efficiency-core stragglers);
  2. a large, separate **context-length effect**: single-thread decode falls
     272→100 tok/s from ~8 to ~768 tokens of context because **attention is not
     threaded** and grows linearly with context.
- **The shipped default is already correct**: every command defaults to
  `--threads 1`. The problem is only that a user who raises `--threads` expecting
  faster decode will get the opposite on a small model.

## Fixes applied

Three changes were made after the investigation (all verified: golden tests pass,
ASan-clean on threaded decode/prefill for both model families):

**1. Attention is now threaded over heads** (`src/model.c`, shared by decode and
prefill, gated by work so short contexts stay serial). This directly removes the
serial-attention wall — the largest real-world decode cost. Long-context decode
now *scales* where it used to be flat:

| threads | stories110M-Q8 decode @768 ctx — **before** | **after** |
|--------:|--------------------------------------------:|----------:|
| 1 | 97.9 | 97.9 |
| 2 | 98.9 | 124.1 |
| 4 | 96.8 | **158.4** |
| 6 | — | 132.6 |

From flat ~100 tok/s to **1.6× at 4 threads**. (@512 ctx: 134→183, 1.4×.)

**2. The parallelization gate is now thread-aware** (`ember_should_parallel`,
`src/threads.c` + `kernels.c`). It was a fixed `work ≥ 32768`; it is now
`work ≥ 32768 × threads`, so a memory-bound op only parallelizes when each thread
gets enough work. Tiny/quantized decode matmuls fall back to serial as threads
rise instead of paying dispatch overhead, while big matmuls (Qwen3, lm_head) still
split. Qwen3 decode still scales (49→62 tok/s), both prefills still scale ~2.5×.

**3. `--threads auto`** (`src/main.c`). A flop-based gate still can't *tell* that a
small quantized matmul is bandwidth-bound, so the robust answer is to measure:
`auto` times a few `forward()` passes at 1/2/4/6 threads on a fresh state and keeps
the fastest. It picks correctly per model on this M1 Pro:

| model | `--threads auto` picks |
|---|---|
| stories15M-Q8 | 1 |
| stories110M-Q8 | 2 |
| Qwen3-0.6B-Q8 | 6 |

The README now recommends `--threads auto`; the shipped numeric default stays `1`
(safe when auto isn't requested). What remains un-done is listed in
[Next step](#next-step).

## Machine and method

- **Apple M1 Pro**, 6 performance + 2 efficiency cores (8 logical), macOS.
- `./ember bench` with a fixed synthetic token stream; each run does 1 warm-up +
  N timed reps and reports the mean. `pp` = prefill (batched GEMM over the prompt),
  `tg` = decode (one token at a time). Reps: 3–5 depending on run length.
- Decode throughput depends on **context length**, so runs that isolate the
  *threading* effect use a short prompt (`--pp 8`) to keep attention cheap; the
  context effect is measured separately.
- Caveat: the E-cores also run background macOS work, so the 7–8-thread numbers
  carry more run-to-run noise than the 1–4-thread numbers. The *shape* of every
  curve below reproduced across repeated runs.

## Results

### 1. stories110M-Q8 — full sweep (pp256 / tg256, 5 reps)

| threads | prefill (tok/s) | decode (tok/s) |
|--------:|----------------:|---------------:|
| 1 | 321.9 | **160.7** |
| 2 | 514.5 | 150.3 |
| 3 | 689.7 | 151.7 |
| 4 | **826.3** | 146.4 |
| 5 | 778.0 | 141.6 |
| 6 | 771.3 | 136.9 |
| 7 | 783.7 | 130.4 |
| 8 | 783.1 | 128.5 |

Prefill scales ~2.6× (peaks at 4 threads, then flat). Decode declines monotonically
— best at 1 thread. (This matches the reported 209→164 pattern; absolute numbers
differ because decode speed depends on the pp/tg context window and on run noise.)

### 2. Isolating the threading effect from the context effect (stories110M-Q8, decode)

Short context (`--pp 8`) makes decode matmul-dominated; long context makes it
attention-dominated.

| threads | decode @ short ctx (pp8,tg128) | decode @ long ctx (pp1024,tg128) |
|--------:|-------------------------------:|---------------------------------:|
| 1 | 247.8 | 102.7 |
| 2 | 243.7 | 98.9 |
| 3 | 242.3 | — |
| 4 | 231.7 | 96.8 |
| 6 | 167.1 | — |
| 8 | 173.3 | — |

At short context decode is **nearly flat 1→4 threads**, then **cliffs at 6–8**.
At long context it is ~2.4× slower overall and still doesn't scale.

### 3. Context length dominates single-thread decode (stories110M-Q8, 1 thread, tg64)

| start context | decode (tok/s) |
|--------------:|---------------:|
| ~8 | 271.7 |
| ~256 | 188.9 |
| ~512 | 133.3 |
| ~768 | 99.9 |

A 2.7× slowdown purely from context growth — **larger than the entire thread-count
effect** — and it is un-parallelized (see Analysis).

### 4. Controlled comparison: same model, Q8 vs fp32 (decode, pp8, short ctx)

| threads | stories110M **Q8** (tg128) | stories110M **fp32** (tg96) |
|--------:|---------------------------:|----------------------------:|
| 1 | 247.8 | 80.8 |
| 2 | 243.7 | 92.2 |
| 4 | 231.7 | **125.8** |
| 6 | 167.1 | 118.5 |

Identical architecture and dimensions; only the per-matmul work differs (fp32 = 4×
the bytes and FLOPs). **Q8 degrades, fp32 scales ~1.6×.** This isolates the cause.

### 5. Bigger model scales (Qwen3-0.6B-Q8, decode, pp8, tg64)

| threads | decode (tok/s) |
|--------:|---------------:|
| 1 | 49.5 |
| 2 | 58.5 |
| 4 | 56.0 |
| 6 | **61.1** |
| 8 | 59.0 |

Qwen3's matmuls (dim 1024, hidden 3072, 152k-row lm_head) are large enough that
decode **does** benefit — a modest ~1.23× up to the 6-core count, then flat/E-core dip.

### 6. Extreme small case (stories15M-Q8, decode, pp8, tg128)

| threads | decode (tok/s) |
|--------:|---------------:|
| 1 | **1878.8** |
| 2 | 1517.9 |
| 4 | 1303.7 |
| 8 | 659.7 |

The tiniest model is hurt most — **2.8× slower at 8 threads than at 1.** Its matmuls
are so small that thread dispatch is pure overhead.

### 7. Prefill scales regardless of model (Qwen3-0.6B-Q8, pp256, tg2)

| threads | prefill (tok/s) |
|--------:|----------------:|
| 1 | 61.9 |
| 2 | 105.7 |
| 4 | **157.3** |
| 6 | 152.4 |
| 8 | 159.3 |

~2.5×. Prefill always operates on big batched GEMMs, so there is always enough work
to amortize dispatch.

## Analysis — why decode behaves this way

Three independent mechanisms, in order of importance for the observed curves:

**(a) Per-token work vs dispatch overhead (the main effect).** Single-stream decode
is one token through a stack of matrix–*vector* products. The engine parallelizes a
matmul only when `n_in * d_out ≥ 32768` (`src/kernels.c:230`), and dispatches it to
the pool, which costs a mutex + condvar broadcast + a barrier wait on completion
(`src/threads.c`). For a small/quantized model each matmul is tiny and
memory-bandwidth-bound — splitting it across cores saves little wall-clock but pays
the dispatch cost every time, on dozens of matmuls per token. Net: flat-to-negative
scaling. Experiment 4 proves this is the mechanism: hold the model fixed and only
increase per-matmul work (fp32 vs Q8) and the sign of the scaling flips.

**(b) The P/E-core cliff.** This is a 6P+2E chip. Workers are QoS-hinted toward
P-cores, but at ≥6 threads there is no headroom left for the main thread's
coordination, and at 7–8 threads workers spill onto the slower E-cores. Because the
pool uses a **barrier** (main waits for the slowest worker on every matmul), one
E-core straggler gates the whole dispatch — hence the sharp drop from 4→6→8 threads
in experiments 1, 2, and 6, distinct from the gentle 1→4 decline.

**(c) Serial attention × context length (a separate, large effect).** Experiment 3
shows decode more than halving as context grows, and experiment 2 shows long-context
decode doesn't thread at all. The reason: **the attention loop is not parallelized**
(`src/model.c:211`, `:281` run over heads on the calling thread). Only the QKV/O/FFN
matmuls go through the pool. As context grows, per-token work shifts from
(threadable) matmuls to (serial) attention, so more threads help even less and
absolute speed drops. This is orthogonal to the threading question but matters for
any real chat session, where context is hundreds of tokens.

## Did the hypothesis hold?

**Yes**, and it is now precisely characterized. "Sync overhead outweighs per-token
work at this model size" is correct, with three refinements:

1. It's **per-token work**, not model size per se — the Q8-vs-fp32 flip on one model
   makes this exact.
2. There is a **second, distinct mechanism** (the P/E-core barrier cliff) that
   produces the steep drop at high thread counts, on top of the gradual overhead.
3. A **third, larger effect** for real workloads — serial attention vs context
   length — is not about threading at all but dominates decode speed in practice.

## Surprises

- **Quantization removes parallelism.** Making the model 4× smaller/faster (fp32→Q8)
  turned decode from thread-scaling into thread-degrading. Counterintuitive, and a
  clean example of Amdahl's law meeting the roofline.
- **Context length is a bigger decode lever than thread count** (2.7× vs ~1.3×), and
  it's currently un-parallelized — arguably the highest-value thing to fix for chat.
- The reported "sync overhead" intuition was directionally right but the **worst
  degradation is at the P→E-core boundary**, not a smooth function of thread count.

## Recommendation / guidance

For users:

- **Just use `--threads auto`** — it measures and picks the right decode count for
  your model and machine (implemented; see [Fixes applied](#fixes-applied)).
- If choosing manually: **decode** on a small/quantized model → **1 thread**;
  decode on a ≥0.6B model → up to the **performance-core count** (6 here), never
  more. **Prefill** (long prompts) → ~4–6 threads; it scales ~2.5×.
- **Rule of thumb:** the more of your time is spent generating vs. ingesting a long
  prompt, the fewer threads you want.

## Next step

Done in the fixes pass: attention threading, a thread-aware gate, and
`--threads auto` (report items below, originally listed as future work).

Still open, in priority order:

1. **A SIMD int8 kernel for `Q4_0`.** Orthogonal to threads but the other big decode
   lever; today Q4 is smaller but not faster.
2. **fp16 KV cache.** Halves KV memory and the bytes touched by (now-threaded)
   attention at long context — a further decode win beyond parallelism.
3. **Re-tune `auto` as context grows.** `auto` measures once at a moderate context;
   a very long chat shifts the optimum slightly higher. Cheap to re-measure
   periodically, but low priority — the per-op gates already keep high thread counts
   safe, so a single early measurement is close.
4. **A proper P/E-core-aware pool** (pin workers to performance cores, or size the
   pool to the P-core count) would remove the residual >6-thread cliff that the
   gate only sidesteps.
