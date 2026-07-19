# Contributing

emberllm is a small, readable, dependency-free CPU inference engine. Changes are
welcome as long as they keep it that way. Please read
[ARCHITECTURE.md](ARCHITECTURE.md) first.

## The three constraints (please don't break these)

1. **Zero runtime dependencies.** No BLAS, no CURL, no web framework, no
   tokenizer library. The HTTP server and GGUF/tokenizer readers are hand-rolled
   on purpose. The converter (`tools/convert.py`) may use `numpy`, but the engine
   links only `-lm -pthread`.
2. **Readable C11, one `make`.** New code should read like the code around it —
   match the comment density and naming. Optional features go behind a clean
   seam (a new `src/*.c` and a header), not `#ifdef` swamps.
3. **Correctness is proven, not asserted.** New behavior needs a check in
   `tests/run_tests.sh`, and anything touching the math needs a reference: the
   TinyStories forward pass is gated token-for-token against karpathy's `run.c`,
   and the tokenizer against Hugging Face `tokenizers`.

## Building and testing

```sh
make          # optimized ./ember
make debug    # -Werror, deterministic (no fast-math); what the tests use
make test     # build debug + run tests/run_tests.sh
make lib      # libember.a + libember.so (for embedding / bindings)
```

`make debug` is warnings-as-errors — keep it clean. `make test` must pass on the
reference platform (Apple Silicon); on other platforms the bit-exact transcript
test auto-skips because float reduction order isn't portable (`EMBER_SKIP_TRANSCRIPT`).

## Adding your own model

Most LLaMA-family models need **no engine change** — just conversion. The
converter (`tools/convert.py`, subcommand `hf`) reads a Hugging Face safetensors
checkpoint and detects the architecture from `config.json` and the tensor names.

1. **Register it.** Add a line to `tools/download.sh`:

   ```sh
   mymodel)  fetch_hf "Org/My-Model-1B" "mymodel" ;;
   ```

   then `./tools/download.sh mymodel`. `fetch_hf` pulls the repo (single-file or
   sharded), converts to `.ember`, and makes a Q8_0 build.

2. **What the converter already handles automatically:** GQA, tied embeddings,
   decoupled `head_dim`, QK-norm (Qwen3), QKV bias (Qwen2.5), NEOX RoPE, and
   `<think>` (Qwen3). It fills sampling defaults from `generation_config.json`.

3. **If it doesn't just work**, it's usually one of:
   - **A new tensor-name convention** → adjust the key strings in `convert_hf`.
   - **A genuinely new architectural op** (a norm somewhere new, a different
     activation, RoPE scaling) → add the minimal knob to `src/ember.h`'s header,
     the converter, and `src/model.c`. Keep it behind a flag so existing models
     are untouched.

4. **Prove it.** Confirm `./ember generate <model> -p "..." -t 0` produces
   coherent, on-prompt text. For a math-touching change, add a reference
   comparison (a logit diff against the HF forward pass is the gold standard) and
   run `ember perplexity` before/after.

Header fields and the tensor names the loader expects are documented inline in
[`src/ember.h`](src/ember.h) and resolved in `ember_state_new_ex` (model.c).

## Pull requests

- Keep each PR to one logical change; explain the *why*.
- Update the README / this file / ARCHITECTURE.md if you change behavior or add a
  model to the verified table.
- Make sure `make debug` and `make test` are green.
