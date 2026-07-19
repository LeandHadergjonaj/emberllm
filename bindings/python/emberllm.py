"""emberllm — Python bindings for the emberllm C engine via ctypes.

Zero Python dependencies (stdlib only); it just loads ``libember.so``/``.dylib``
and calls the public C API from ``src/ember.h``. Build the shared library first:

    make lib          # produces ./libember.so (or libember.dylib on macOS)

Then:

    from emberllm import Ember
    m = Ember("models/stories110M-q8.ember")
    print(m.generate_str("Once upon a time", max_tokens=64, temperature=0.8))

Sampling (temperature / top-k) is done here in Python from the returned logits,
so the binding needs only the load / tokenize / forward / decode entry points.
"""
import ctypes
import math
import os
import random

# --- locate and load the shared library -------------------------------------

def _find_lib():
    if os.environ.get("EMBER_LIB"):
        return os.environ["EMBER_LIB"]
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [here, os.path.join(here, "..", ".."), os.getcwd()]
    names = ["libember.so", "libember.dylib"]
    for root in roots:
        for name in names:
            cand = os.path.normpath(os.path.join(root, name))
            if os.path.exists(cand):
                return cand
    raise FileNotFoundError(
        "libember shared library not found — run `make lib` (looked for "
        "libember.so/.dylib; set EMBER_LIB to override)."
    )


_lib = ctypes.CDLL(_find_lib())
_libc = ctypes.CDLL(None)  # for free() of the ids buffer ember_encode mallocs
_libc.free.argtypes = [ctypes.c_void_p]


# --- header struct (mirror of EmberHeader in src/ember.h, packed) ------------

class EmberHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_char * 4),
        ("version", ctypes.c_uint32),
        ("header_bytes", ctypes.c_uint64),
        ("dim", ctypes.c_int32),
        ("hidden_dim", ctypes.c_int32),
        ("n_layers", ctypes.c_int32),
        ("n_heads", ctypes.c_int32),
        ("n_kv_heads", ctypes.c_int32),
        ("head_dim", ctypes.c_int32),
        ("vocab_size", ctypes.c_int32),
        ("max_seq_len", ctypes.c_int32),
        ("rope_theta", ctypes.c_float),
        ("norm_eps", ctypes.c_float),
        ("flags", ctypes.c_uint32),
        ("rope_style", ctypes.c_int32),
        ("bos_token_id", ctypes.c_int32),
        ("eos_token_id", ctypes.c_int32),
        ("eos_token_id2", ctypes.c_int32),
        ("n_tensors", ctypes.c_int32),
        ("tokenizer_type", ctypes.c_int32),
        ("default_top_k", ctypes.c_int32),
        ("default_temp", ctypes.c_float),
        ("default_top_p", ctypes.c_float),
        ("tokenizer_offset", ctypes.c_uint64),
        ("tokenizer_size", ctypes.c_uint64),
        ("tensor_table_offset", ctypes.c_uint64),
        ("tensor_table_size", ctypes.c_uint64),
        ("reserved", ctypes.c_uint8 * 128),
    ]


assert ctypes.sizeof(EmberHeader) == 256, "EmberHeader layout drifted from ember.h"

_P = ctypes.c_void_p
_FLOATP = ctypes.POINTER(ctypes.c_float)
_INTP = ctypes.POINTER(ctypes.c_int)

# --- C prototypes ------------------------------------------------------------
_lib.ember_model_load.restype = _P
_lib.ember_model_load.argtypes = [ctypes.c_char_p]
_lib.ember_model_free.argtypes = [_P]
_lib.ember_model_header.restype = ctypes.POINTER(EmberHeader)
_lib.ember_model_header.argtypes = [_P]

_lib.ember_state_new_ex.restype = _P
_lib.ember_state_new_ex.argtypes = [_P, ctypes.c_int, ctypes.c_int]
_lib.ember_state_free.argtypes = [_P]
_lib.ember_forward.restype = _FLOATP
_lib.ember_forward.argtypes = [_P, _P, ctypes.c_int, ctypes.c_int]
_lib.ember_prefill.restype = _FLOATP
_lib.ember_prefill.argtypes = [_P, _P, _INTP, ctypes.c_int, ctypes.c_int]

_lib.ember_tokenizer_new.restype = _P
_lib.ember_tokenizer_new.argtypes = [_P]
_lib.ember_tokenizer_free.argtypes = [_P]
_lib.ember_encode.restype = ctypes.c_int
_lib.ember_encode.argtypes = [_P, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(_INTP)]
_lib.ember_decode.restype = ctypes.c_char_p
_lib.ember_decode.argtypes = [_P, ctypes.c_int, ctypes.c_int]


class Ember:
    """A loaded model plus its tokenizer and one reusable run state."""

    def __init__(self, path, ctx=0, kv_fp16=False):
        self.m = _lib.ember_model_load(path.encode())
        if not self.m:
            raise RuntimeError(f"failed to load model: {path}")
        self.header = _lib.ember_model_header(self.m).contents
        self.tok = _lib.ember_tokenizer_new(self.m)
        self.st = _lib.ember_state_new_ex(self.m, int(ctx), 1 if kv_fp16 else 0)
        maxs = self.header.max_seq_len
        self.ctx = maxs if ctx <= 0 or ctx > maxs else ctx

    # -- tokenization --------------------------------------------------------
    def encode(self, text, add_bos=True):
        out = _INTP()
        n = _lib.ember_encode(self.tok, text.encode(), 1 if add_bos else 0, ctypes.byref(out))
        ids = [out[i] for i in range(n)]
        _libc.free(out)
        return ids

    def decode(self, prev, tok):
        s = _lib.ember_decode(self.tok, prev, tok)
        return s.decode("utf-8", "replace") if s else ""

    # -- generation ----------------------------------------------------------
    def _sample(self, logits, temperature, top_k, rng):
        n = self.header.vocab_size
        if temperature <= 0.0:
            best, bi = logits[0], 0
            for i in range(1, n):
                if logits[i] > best:
                    best, bi = logits[i], i
            return bi
        scaled = [logits[i] / temperature for i in range(n)]
        order = range(n)
        if top_k and 0 < top_k < n:
            order = sorted(range(n), key=lambda i: scaled[i], reverse=True)[:top_k]
        mx = max(scaled[i] for i in order)
        exps = [(i, math.exp(scaled[i] - mx)) for i in order]
        total = sum(e for _, e in exps)
        r = rng.random() * total
        acc = 0.0
        for i, e in exps:
            acc += e
            if r < acc:
                return i
        return exps[-1][0]

    def generate(self, prompt, max_tokens=128, temperature=0.0, top_k=0, seed=None):
        """Yield decoded text pieces one token at a time."""
        ids = self.encode(prompt)
        if len(ids) >= self.ctx:
            ids = ids[: self.ctx - 1]
        arr = (ctypes.c_int * len(ids))(*ids)
        logits = _lib.ember_prefill(self.m, self.st, arr, len(ids), 0)
        rng = random.Random(seed)
        prev = ids[-1]
        pos = len(ids)
        h = self.header
        for _ in range(max_tokens):
            nxt = self._sample(logits, temperature, top_k, rng)
            if nxt in (h.eos_token_id, h.eos_token_id2, h.bos_token_id):
                break
            yield self.decode(prev, nxt)
            prev = nxt
            if pos >= self.ctx - 1:
                break
            logits = _lib.ember_forward(self.m, self.st, nxt, pos)
            pos += 1

    def generate_str(self, prompt, **kw):
        return "".join(self.generate(prompt, **kw))

    def close(self):
        if getattr(self, "st", None):
            _lib.ember_state_free(self.st); self.st = None
        if getattr(self, "tok", None):
            _lib.ember_tokenizer_free(self.tok); self.tok = None
        if getattr(self, "m", None):
            _lib.ember_model_free(self.m); self.m = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()
