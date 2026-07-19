#!/usr/bin/env python3
"""convert.py — turn upstream checkpoints into a single .ember file.

v1 supports the llama2.c legacy checkpoint format (the karpathy/tinyllamas
TinyStories models). It depends only on the Python standard library: the
TinyStories file is flat fp32, so no numpy/torch is needed to repack it.

Later revisions add a safetensors path for HF-native models (Stage 3); that
path will import numpy + safetensors lazily, so this stays dependency-free for
the TinyStories arc.

Usage:
    python3 tools/convert.py llama2c \
        --checkpoint models/stories15M.bin \
        --tokenizer  models/tokenizer.bin \
        --out        models/stories15M.ember
"""
import argparse
import json
import struct
import sys

# --- .ember format constants (mirror of src/ember.h) -------------------------
EMBER_MAGIC = b"EMBR"
EMBER_VERSION = 1
DT_F32, DT_Q8_0, DT_Q4_0, DT_F16 = 0, 1, 2, 3
FLAG_TIED_EMBED = 1 << 0
FLAG_QK_NORM = 1 << 1
FLAG_THINKING = 1 << 2
ROPE_INTERLEAVED, ROPE_NEOX = 0, 1
TOK_LLAMA_SP, TOK_BYTE_BPE = 0, 1

HEADER_FMT = "<4sIQ" + "i" * 8 + "ff" + "Ii" + "iiii" + "ii" + "ff" + "QQQQ" + "128s"
HEADER_SIZE = 256
TENSOR_FMT = "<48sii4iQQ"
TENSOR_SIZE = 88
ALIGN = 64


def _align(x, a=ALIGN):
    return (x + a - 1) // a * a


class TensorSpec:
    """A tensor staged for writing: name, dtype, shape, and raw bytes."""

    __slots__ = ("name", "dtype", "shape", "data")

    def __init__(self, name, dtype, shape, data):
        assert len(name.encode()) < 48, name
        self.name = name
        self.dtype = dtype
        self.shape = list(shape) + [1] * (4 - len(shape))
        self.data = data


def write_ember(out_path, cfg, tokenizer_type, tokenizer_blob, tensors):
    """Assemble header + tokenizer + tensor table + aligned tensor data."""
    n_tensors = len(tensors)
    tok_off = HEADER_SIZE
    tok_size = len(tokenizer_blob)
    table_off = _align(tok_off + tok_size, 8)
    table_size = n_tensors * TENSOR_SIZE
    data_start = _align(table_off + table_size, ALIGN)

    # assign each tensor a 64-aligned offset in the data region
    offsets = []
    cur = data_start
    for t in tensors:
        cur = _align(cur, ALIGN)
        offsets.append(cur)
        cur += len(t.data)
    header_bytes = data_start  # tensor-data region begins here

    header = struct.pack(
        HEADER_FMT,
        EMBER_MAGIC,
        EMBER_VERSION,
        header_bytes,
        cfg["dim"], cfg["hidden_dim"], cfg["n_layers"], cfg["n_heads"],
        cfg["n_kv_heads"], cfg["head_dim"], cfg["vocab_size"], cfg["max_seq_len"],
        cfg["rope_theta"], cfg["norm_eps"],
        cfg["flags"], cfg["rope_style"],
        cfg["bos_token_id"], cfg["eos_token_id"], cfg["eos_token_id2"], n_tensors,
        tokenizer_type, cfg["default_top_k"],
        cfg["default_temp"], cfg["default_top_p"],
        tok_off, tok_size, table_off, table_size,
        b"\x00" * 128,
    )
    assert len(header) == HEADER_SIZE

    table = b"".join(
        struct.pack(
            TENSOR_FMT,
            t.name.encode(),
            t.dtype, sum(1 for d in t.shape if d > 1) or 1,
            t.shape[0], t.shape[1], t.shape[2], t.shape[3],
            offsets[i], len(t.data),
        )
        for i, t in enumerate(tensors)
    )

    with open(out_path, "wb") as f:
        f.write(header)
        f.write(tokenizer_blob)
        f.write(b"\x00" * (table_off - f.tell()))
        f.write(table)
        f.write(b"\x00" * (data_start - f.tell()))
        for i, t in enumerate(tensors):
            f.write(b"\x00" * (offsets[i] - f.tell()))
            f.write(t.data)
    print(f"wrote {out_path}: {n_tensors} tensors, {cur/1e6:.1f} MB", file=sys.stderr)


# --- llama2.c legacy reader --------------------------------------------------

def convert_llama2c(checkpoint, tokenizer, out):
    with open(checkpoint, "rb") as f:
        blob = f.read()
    dim, hidden, n_layers, n_heads, n_kv, vocab, seq = struct.unpack("<7i", blob[:28])
    shared = vocab > 0
    vocab = abs(vocab)
    head_size = dim // n_heads
    kv_dim = n_kv * head_size
    print(
        f"llama2c: dim={dim} hidden={hidden} layers={n_layers} heads={n_heads} "
        f"kv={n_kv} vocab={vocab} seq={seq} shared={shared}",
        file=sys.stderr,
    )

    # sequentially consume fp32 regions in checkpoint order
    pos = 28

    def take(nfloats):
        nonlocal pos
        nbytes = nfloats * 4
        chunk = blob[pos:pos + nbytes]
        assert len(chunk) == nbytes, "checkpoint truncated"
        pos += nbytes
        return chunk

    def take_layered(per_layer):
        """Return a list of n_layers byte-chunks, each `per_layer` floats."""
        return [take(per_layer) for _ in range(n_layers)]

    tok_emb = take(vocab * dim)
    rms_att = take_layered(dim)
    wq = take_layered(dim * n_heads * head_size)
    wk = take_layered(dim * kv_dim)
    wv = take_layered(dim * kv_dim)
    wo = take_layered(n_heads * head_size * dim)
    rms_ffn = take_layered(dim)
    w1 = take_layered(hidden * dim)
    w2 = take_layered(dim * hidden)
    w3 = take_layered(hidden * dim)
    rms_final = take(dim)
    take(seq * (head_size // 2))  # freq_cis_real (unused; RoPE computed at runtime)
    take(seq * (head_size // 2))  # freq_cis_imag (unused)
    wcls = tok_emb if shared else take(vocab * dim)

    tensors = [TensorSpec("tok_embeddings", DT_F32, [vocab, dim], tok_emb)]
    for l in range(n_layers):
        tensors += [
            TensorSpec(f"layers.{l}.attn_norm", DT_F32, [dim], rms_att[l]),
            TensorSpec(f"layers.{l}.wq", DT_F32, [n_heads * head_size, dim], wq[l]),
            TensorSpec(f"layers.{l}.wk", DT_F32, [kv_dim, dim], wk[l]),
            TensorSpec(f"layers.{l}.wv", DT_F32, [kv_dim, dim], wv[l]),
            TensorSpec(f"layers.{l}.wo", DT_F32, [dim, n_heads * head_size], wo[l]),
            TensorSpec(f"layers.{l}.ffn_norm", DT_F32, [dim], rms_ffn[l]),
            TensorSpec(f"layers.{l}.w1", DT_F32, [hidden, dim], w1[l]),
            TensorSpec(f"layers.{l}.w2", DT_F32, [dim, hidden], w2[l]),
            TensorSpec(f"layers.{l}.w3", DT_F32, [hidden, dim], w3[l]),
        ]
    tensors.append(TensorSpec("final_norm", DT_F32, [dim], rms_final))
    if not shared:
        tensors.append(TensorSpec("lm_head", DT_F32, [vocab, dim], wcls))

    with open(tokenizer, "rb") as f:
        tok_blob = f.read()

    cfg = dict(
        dim=dim, hidden_dim=hidden, n_layers=n_layers, n_heads=n_heads,
        n_kv_heads=n_kv, head_dim=head_size, vocab_size=vocab, max_seq_len=seq,
        rope_theta=10000.0, norm_eps=1e-5,
        flags=(FLAG_TIED_EMBED if shared else 0),
        rope_style=ROPE_INTERLEAVED,
        bos_token_id=1, eos_token_id=2, eos_token_id2=-1,
        default_top_k=0, default_temp=1.0, default_top_p=0.9,
    )
    write_ember(out, cfg, TOK_LLAMA_SP, tok_blob, tensors)


# --- Qwen3 / safetensors reader (needs numpy + safetensors's format, not torch) ---

def _read_safetensors(path):
    """Return {name: numpy_fp32_array} for one .safetensors file, parsing bf16 by hand."""
    import numpy as np
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
        data_start = 8 + hlen
        out = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype, shape = meta["dtype"], meta["shape"]
            a, b = meta["data_offsets"]
            f.seek(data_start + a)
            raw = f.read(b - a)
            if dtype == "BF16":
                u16 = np.frombuffer(raw, dtype=np.uint16)
                arr = (u16.astype(np.uint32) << 16).view(np.float32)
            elif dtype in ("F32", "F16"):
                arr = np.frombuffer(raw, dtype=np.float16 if dtype == "F16" else np.float32).astype(np.float32)
            else:
                raise ValueError(f"unsupported dtype {dtype} for {name}")
            out[name] = arr.reshape(shape) if shape else arr
    return out


def _read_safetensors_dir(model_dir):
    """Load all tensors from a model dir: a single model.safetensors, or every
    shard listed in model.safetensors.index.json."""
    import os
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        shards = sorted(set(json.load(open(index))["weight_map"].values()))
        out = {}
        for shard in shards:
            out.update(_read_safetensors(os.path.join(model_dir, shard)))
        return out
    return _read_safetensors(os.path.join(model_dir, "model.safetensors"))


def _byte_level_maps():
    """GPT-2 byte<->unicode bijection; returns byte->unicode-codepoint list."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _build_bpe_blob(tokenizer_json, vocab_size):
    """Pack vocab + merges + specials into the EMBER_TOK_BYTE_BPE blob."""
    tj = json.load(open(tokenizer_json))
    vocab = tj["model"]["vocab"]            # token_string -> id (byte-level encoded)
    merges = tj["model"]["merges"]          # list of [a, b] or "a b" strings
    specials = {a["content"]: a["id"] for a in tj.get("added_tokens", [])}

    # tokenizers >= 0.20 stores merges as "a b" strings instead of [a, b] pairs.
    # A byte-level token never contains a literal space (spaces become 'Ġ'), so
    # splitting on the single separating space recovers the pair unambiguously.
    merges = [m.split(" ", 1) if isinstance(m, str) else m for m in merges]

    id_to_str = [b""] * vocab_size
    is_special = [0] * vocab_size
    for s, i in vocab.items():
        if 0 <= i < vocab_size:
            id_to_str[i] = s.encode("utf-8")
    for content, i in specials.items():
        if 0 <= i < vocab_size:
            id_to_str[i] = content.encode("utf-8")
            is_special[i] = 1

    out = bytearray()
    out += struct.pack("<III", vocab_size, len(merges), len(specials))
    for i in range(vocab_size):
        out += struct.pack("<BI", is_special[i], len(id_to_str[i])) + id_to_str[i]
    for a, b in merges:
        ab, bb = a.encode("utf-8"), b.encode("utf-8")
        out += struct.pack("<I", len(ab)) + ab + struct.pack("<I", len(bb)) + bb
    for content, i in specials.items():
        cb = content.encode("utf-8")
        out += struct.pack("<II", i, len(cb)) + cb
    return bytes(out)


def _special_id(model_dir, content):
    """Find the id of an added/special token by its literal content, or None."""
    try:
        tj = json.load(open(f"{model_dir}/tokenizer.json"))
    except OSError:
        return None
    for a in tj.get("added_tokens", []):
        if a.get("content") == content:
            return a["id"]
    return None


def convert_hf(model_dir, out):
    """Convert any LLaMA-style HF safetensors checkpoint (Qwen2.5/3, SmolLM2,
    Llama, ...) to .ember. Architecture variants are detected from config.json
    and the tensor names, not hard-coded per model:

      - GQA:            num_key_value_heads < num_attention_heads
      - QK-norm:        presence of self_attn.{q,k}_norm.weight (Qwen3)
      - QKV bias:       presence of self_attn.{q,k,v}_proj.bias (Qwen2.5)
      - tied lm_head:   config tie_word_embeddings
      - decoupled head: config head_dim (else hidden_size // n_heads)
    """
    import numpy as np
    cfg_j = json.load(open(f"{model_dir}/config.json"))
    st = _read_safetensors_dir(model_dir)

    dim = cfg_j["hidden_size"]; hidden = cfg_j["intermediate_size"]
    n_layers = cfg_j["num_hidden_layers"]; n_heads = cfg_j["num_attention_heads"]
    n_kv = cfg_j.get("num_key_value_heads", n_heads)
    head_dim = cfg_j.get("head_dim") or (dim // n_heads)
    vocab = cfg_j["vocab_size"]
    tied = cfg_j.get("tie_word_embeddings", False)

    has_qk_norm = "model.layers.0.self_attn.q_norm.weight" in st
    has_bias    = "model.layers.0.self_attn.q_proj.bias" in st
    # Qwen3 has a <think> reasoning mode that chat/serve suppress by default;
    # other LLaMA-style models must not get the think block injected.
    arch = (cfg_j.get("architectures") or [""])[0]
    is_thinking = arch.startswith("Qwen3")
    print(f"hf: dim={dim} hidden={hidden} layers={n_layers} heads={n_heads} kv={n_kv} "
          f"head_dim={head_dim} vocab={vocab} tied={tied} qk_norm={has_qk_norm} bias={has_bias}",
          file=sys.stderr)

    def f32(key):
        return st[key].astype(np.float32).tobytes()

    tensors = [TensorSpec("tok_embeddings", DT_F32, [vocab, dim], f32("model.embed_tokens.weight"))]
    for l in range(n_layers):
        p = f"model.layers.{l}"
        def add(name, key, shape):
            tensors.append(TensorSpec(name, DT_F32, shape, f32(key)))
        add(f"layers.{l}.attn_norm", f"{p}.input_layernorm.weight", [dim])
        add(f"layers.{l}.wq", f"{p}.self_attn.q_proj.weight", [n_heads * head_dim, dim])
        add(f"layers.{l}.wk", f"{p}.self_attn.k_proj.weight", [n_kv * head_dim, dim])
        add(f"layers.{l}.wv", f"{p}.self_attn.v_proj.weight", [n_kv * head_dim, dim])
        add(f"layers.{l}.wo", f"{p}.self_attn.o_proj.weight", [dim, n_heads * head_dim])
        if has_bias:
            add(f"layers.{l}.wq_bias", f"{p}.self_attn.q_proj.bias", [n_heads * head_dim])
            add(f"layers.{l}.wk_bias", f"{p}.self_attn.k_proj.bias", [n_kv * head_dim])
            add(f"layers.{l}.wv_bias", f"{p}.self_attn.v_proj.bias", [n_kv * head_dim])
        if has_qk_norm:
            add(f"layers.{l}.q_norm", f"{p}.self_attn.q_norm.weight", [head_dim])
            add(f"layers.{l}.k_norm", f"{p}.self_attn.k_norm.weight", [head_dim])
        add(f"layers.{l}.ffn_norm", f"{p}.post_attention_layernorm.weight", [dim])
        add(f"layers.{l}.w1", f"{p}.mlp.gate_proj.weight", [hidden, dim])
        add(f"layers.{l}.w2", f"{p}.mlp.down_proj.weight", [dim, hidden])
        add(f"layers.{l}.w3", f"{p}.mlp.up_proj.weight", [hidden, dim])
    tensors.append(TensorSpec("final_norm", DT_F32, [dim], f32("model.norm.weight")))
    if not tied:
        tensors.append(TensorSpec("lm_head", DT_F32, [vocab, dim], f32("lm_head.weight")))

    # end-of-sequence: config eos may be a scalar or a list ([eos, <|im_end|>]).
    eos = cfg_j.get("eos_token_id", -1)
    if isinstance(eos, list):
        eos_token_id = eos[0]
        eos_token_id2 = eos[1] if len(eos) > 1 else -1
    else:
        eos_token_id = eos if eos is not None else -1
        # ChatML models stop on <|im_end|>; use it as the secondary stop if present
        eos_token_id2 = _special_id(model_dir, "<|im_end|>")
        if eos_token_id2 is None:
            eos_token_id2 = -1
    bos = cfg_j.get("bos_token_id")
    bos_token_id = bos if isinstance(bos, int) else -1

    # sampling defaults: prefer the model's generation_config, else sane values
    gen = {}
    try:
        gen = json.load(open(f"{model_dir}/generation_config.json"))
    except OSError:
        pass
    default_temp = float(gen.get("temperature", 0.7))
    default_top_p = float(gen.get("top_p", 0.9))
    default_top_k = int(gen.get("top_k", 0) or 0)

    blob = _build_bpe_blob(f"{model_dir}/tokenizer.json", vocab)
    cfg = dict(
        dim=dim, hidden_dim=hidden, n_layers=n_layers, n_heads=n_heads,
        n_kv_heads=n_kv, head_dim=head_dim, vocab_size=vocab,
        max_seq_len=cfg_j["max_position_embeddings"],
        rope_theta=float(cfg_j.get("rope_theta", 10000.0)), norm_eps=float(cfg_j["rms_norm_eps"]),
        flags=(FLAG_TIED_EMBED if tied else 0) | (FLAG_QK_NORM if has_qk_norm else 0)
              | (FLAG_THINKING if is_thinking else 0),
        rope_style=ROPE_NEOX,
        bos_token_id=bos_token_id, eos_token_id=eos_token_id, eos_token_id2=eos_token_id2,
        default_top_k=default_top_k, default_temp=default_temp, default_top_p=default_top_p,
    )
    write_ember(out, cfg, TOK_BYTE_BPE, blob, tensors)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("llama2c", help="convert a llama2.c legacy .bin checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    h = sub.add_parser("hf", help="convert a LLaMA-style HF safetensors dir "
                                  "(Qwen2.5/3, SmolLM2, Llama, ...)")
    h.add_argument("--model-dir", required=True)
    h.add_argument("--out", required=True)
    # 'qwen' kept as a backwards-compatible alias for 'hf'
    q = sub.add_parser("qwen", help="alias for 'hf' (kept for compatibility)")
    q.add_argument("--model-dir", required=True)
    q.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "llama2c":
        convert_llama2c(args.checkpoint, args.tokenizer, args.out)
    elif args.cmd in ("hf", "qwen"):
        convert_hf(args.model_dir, args.out)


if __name__ == "__main__":
    main()
