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
import struct
import sys

# --- .ember format constants (mirror of src/ember.h) -------------------------
EMBER_MAGIC = b"EMBR"
EMBER_VERSION = 1
DT_F32, DT_Q8_0, DT_Q4_0, DT_F16 = 0, 1, 2, 3
FLAG_TIED_EMBED = 1 << 0
FLAG_QK_NORM = 1 << 1
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("llama2c", help="convert a llama2.c legacy .bin checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "llama2c":
        convert_llama2c(args.checkpoint, args.tokenizer, args.out)


if __name__ == "__main__":
    main()
