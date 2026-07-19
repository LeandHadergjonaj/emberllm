# emberllm

A CPU-only LLM inference engine, written from scratch in C11. No GPU, no runtime dependencies — the goal is visibly fast text generation on ordinary hardware, and code small enough to read in an afternoon.

**Status: planning.** No engine code yet. The full staged build plan — target models, optimization roadmap, validation strategy, and honest performance targets — lives in **[PLAN.md](PLAN.md)**.

## What this will become

- A single `ember` binary that loads a self-describing `.ember` weight file (mmap, instant startup) and streams tokens in the terminal with a live tok/s counter.
- Bring-up on the TinyStories models (llama2.c lineage), ending at **Qwen3-0.6B instruct chat** — the best sub-1B chat model — running quantized on a plain M-series CPU.
- The optimization arc, staged and measured honestly against the memory-bandwidth roofline: fp32 baseline → pthreads → NEON kernels → Q8_0/Q4_0 quantization → chunked prefill.
- Zero runtime dependencies: hand-written tokenizers, threadpool, and SIMD kernels; one Makefile; a one-time Python conversion script (`safetensors` + `numpy` only).

Inspired by the genre of [llama2.c](https://github.com/karpathy/llama2.c), [llama.cpp](https://github.com/ggml-org/llama.cpp), and [calm](https://github.com/zeux/calm) — techniques reimplemented independently, no code copied.

License: MIT.
