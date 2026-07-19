# Architecture

How emberllm is put together, and how a token flows through it. The whole engine
is ~3,200 lines of C11 in [`src/`](src/); this document is the map.

## The pieces

| File | Responsibility |
|---|---|
| `src/ember.h` | The `.ember` file format (header + tensor table) **and** the public C API. This is the one header an embedder needs. |
| `src/io.c` | `mmap` the file, validate the header's bounds, hand back pointers into the mapping. Nothing is copied. |
| `src/model.c` | The forward pass: embedding, RMSNorm, RoPE, GQA attention, SwiGLU, KV cache, QK-norm, optional QKV bias, batched prefill. |
| `src/kernels.c` | The hot loops: fp16⇄fp32, quantize, and the matmul / dot kernels (scalar, NEON, AVX2). |
| `src/threads.c` | A fork-join thread pool. `ember_parallel_for` splits a matmul's output rows across cores. |
| `src/tokenizer.c` | Two tokenizers behind one interface: SentencePiece-BPE and byte-level BPE. |
| `src/sample.c` | Turn logits into a token: temperature, top-k, top-p, min-p, and the repetition/presence/frequency penalties. |
| `src/quant.c` | Offline requantization (`ember quantize`): fp32 → Q8_0 / Q4_0. |
| `src/server.c` | The OpenAI-compatible HTTP server (`ember serve`). |
| `src/json.c` | A tiny JSON reader for request bodies (server only). |
| `src/util.c` | `ember_die` + checked allocation. |
| `src/main.c` | The CLI front-end. Everything else compiles into `libember`. |

The `ember` binary is `main.o` linked against `libember.a`, so the API in
`ember.h` is by construction a complete embedding surface (see
[`bindings/python`](bindings/python) for a consumer).

## The `.ember` file

A single mmap-friendly container (`src/ember.h` documents the exact layout):

```
[ EmberHeader (256 B) ] [ tokenizer blob ] [ tensor table ] [ tensor data ]
```

The header is fixed-size and holds every architectural knob (dims, head counts,
RoPE style/θ, norm ε, flags for tied-embeddings / QK-norm / thinking, token ids,
sampling defaults). The tensor table maps names like `layers.7.wq` to a dtype,
shape, and 64-byte-aligned offset into the data region. Weights carry their own
dtype, so one forward pass runs fp32, Q8_0, or Q4_0 models unchanged.

## A token's journey

Take `ember generate model.ember -p "Hello"`:

1. **Load** (`ember_model_load`, io.c). `mmap` the file, check magic/version and
   that every header offset lies inside the file. Return an `EmberModel` of
   pointers into the mapping.
2. **Tokenize** (`ember_encode`, tokenizer.c). `"Hello"` → token ids. Special
   tokens (`<|im_start|>`, …) are matched first; the rest goes through
   SentencePiece or byte-level BPE.
3. **State** (`ember_state_new_ex`, model.c). Allocate activations and the KV
   cache (`n_layers × ctx × kv_dim`, fp32 or fp16). Resolve every weight pointer
   once (`layers.N.wq`, biases and norms that may be absent, …).
4. **Prefill** (`ember_prefill`, model.c). Run all prompt tokens at once as a
   batched GEMM — each weight row is streamed from memory once for the whole
   prompt — filling the KV cache and returning the last token's logits.
5. **Decode loop** (main.c). Repeatedly:
   - `ember_sample` (sample.c) picks the next token from the logits.
   - `ember_decode` (tokenizer.c) turns it back into text.
   - `ember_forward` (model.c) runs that one token at `pos`, appends its K/V to
     the cache, and returns the next logits.

   Stop on EOS or the token budget.

### Inside one layer (`ember_forward`)

```
x → RMSNorm → [Wq,Wk,Wv] (+bias) → [QK-norm] → RoPE → store K,V in cache
  → attention over cache (GQA, threaded across heads) → Wo → += residual
  → RMSNorm → SwiGLU(W1,W3) → W2 → += residual
```

Decode is **memory-bandwidth-bound**: generating one token streams essentially
the whole weight file through the cores, which is why quantization (fewer bytes
per weight) is the dominant speed lever and why threads only help until bandwidth
saturates. See [report.md](report.md) for the measurements.

## Where the variety lives

Different model families differ only in header fields and which optional tensors
are present — not in separate code paths:

- **GQA**: `n_kv_heads < n_heads`; each KV head is shared by `n_heads/n_kv_heads`
  query heads (`kv_mul` in the attention loop).
- **RoPE style**: interleaved (llama2.c) vs NEOX half-split (HF), one branch in
  `rope()`.
- **QK-norm** (Qwen3): per-head RMSNorm on Q and K before RoPE, gated by a flag.
- **QKV bias** (Qwen2.5): optional `layers.N.w{q,k,v}_bias` tensors, added after
  the projection; inert when absent.
- **Tied embeddings**: `lm_head` aliases the token-embedding matrix.

Adding a model is therefore usually a converter change, not an engine change —
see [CONTRIBUTING.md](CONTRIBUTING.md).
