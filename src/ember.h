/* ember.h — public interface and on-disk .ember format for the emberllm engine.
 *
 * The .ember file is a single, self-describing, mmap-friendly container:
 *
 *   [ EmberHeader (256 bytes) ]
 *   [ tokenizer blob         ]   (offset/size in header)
 *   [ tensor table           ]   (n_tensors * EmberTensor, offset in header)
 *   [ 64-byte pad            ]
 *   [ tensor data region     ]   (each tensor at its own 64-byte-aligned offset)
 *
 * Everything after the header is addressed by absolute byte offset from the
 * start of the file, so loading is just mmap + pointer arithmetic. All integers
 * are little-endian (the only platforms we target are LE).
 */
#ifndef EMBER_H
#define EMBER_H

#include <stdint.h>
#include <stddef.h>

#define EMBER_MAGIC   0x52424d45u /* "EMBR" */
#define EMBER_VERSION 1u

/* dtype codes for stored tensors */
enum {
    EMBER_DT_F32  = 0,
    EMBER_DT_Q8_0 = 1, /* blocks of 32: fp16 scale + 32 int8            (34 B / 32 w) */
    EMBER_DT_Q4_0 = 2, /* blocks of 32: fp16 scale + 16 packed nibbles  (18 B / 32 w) */
    EMBER_DT_F16  = 3
};

/* header flag bits */
enum {
    EMBER_FLAG_TIED_EMBED = 1u << 0, /* lm_head shares the token-embedding matrix   */
    EMBER_FLAG_QK_NORM    = 1u << 1  /* per-head RMSNorm on Q and K before RoPE      */
};

/* rope_style */
enum {
    EMBER_ROPE_INTERLEAVED = 0, /* adjacent pairs (x0,x1),(x2,x3)... — llama2.c/GPT-J */
    EMBER_ROPE_NEOX        = 1  /* half-split (x0,x_{d/2}) — HF rotate_half           */
};

/* tokenizer_type */
enum {
    EMBER_TOK_LLAMA_SP = 0, /* SentencePiece-BPE, score-ranked merges (TinyStories)   */
    EMBER_TOK_BYTE_BPE = 1  /* GPT-2/Qwen byte-level BPE, rank-ranked merges          */
};

#define EMBER_Q_BLOCK 32 /* weights per quant block for Q8_0 / Q4_0 */

#pragma pack(push, 1)
typedef struct {
    char     magic[4];             /* "EMBR"                                          */
    uint32_t version;              /* EMBER_VERSION                                   */
    uint64_t header_bytes;         /* start of tensor-data region (64-aligned)        */
    int32_t  dim;                  /* hidden size (d_model)                           */
    int32_t  hidden_dim;           /* FFN intermediate size                           */
    int32_t  n_layers;
    int32_t  n_heads;              /* query heads                                     */
    int32_t  n_kv_heads;           /* key/value heads (== n_heads for MHA)            */
    int32_t  head_dim;             /* explicit; may differ from dim / n_heads         */
    int32_t  vocab_size;
    int32_t  max_seq_len;          /* training context length                         */
    float    rope_theta;           /* RoPE base, e.g. 10000 or 1e6                    */
    float    norm_eps;             /* RMSNorm epsilon                                 */
    uint32_t flags;                /* EMBER_FLAG_*                                     */
    int32_t  rope_style;           /* EMBER_ROPE_*                                     */
    int32_t  bos_token_id;
    int32_t  eos_token_id;         /* primary end-of-sequence, -1 if none             */
    int32_t  eos_token_id2;        /* secondary stop (e.g. <|im_end|>), -1 if none    */
    int32_t  n_tensors;
    int32_t  tokenizer_type;       /* EMBER_TOK_*                                      */
    int32_t  default_top_k;
    float    default_temp;
    float    default_top_p;
    uint64_t tokenizer_offset;
    uint64_t tokenizer_size;
    uint64_t tensor_table_offset;
    uint64_t tensor_table_size;
    uint8_t  reserved[128];        /* pads header to 256 bytes                        */
} EmberHeader;

typedef struct {
    char     name[48];             /* null-padded ascii identifier                    */
    int32_t  dtype;                /* EMBER_DT_*                                       */
    int32_t  ndim;                 /* 1..4                                            */
    int32_t  shape[4];             /* logical dims (unused trailing entries = 1)      */
    uint64_t offset;               /* absolute byte offset to data (64-aligned)       */
    uint64_t nbytes;               /* stored size in bytes (post-quantization)        */
} EmberTensor;
#pragma pack(pop)

_Static_assert(sizeof(EmberHeader) == 256, "EmberHeader must be 256 bytes");
_Static_assert(sizeof(EmberTensor) == 88,  "EmberTensor must be 88 bytes");

/* -------------------------------------------------------------------------- */
/* Runtime types                                                              */
/* -------------------------------------------------------------------------- */

/* A loaded model: the mmap'd file plus resolved pointers into it. */
typedef struct EmberModel EmberModel;

/* Tokenizer over the embedded vocab. */
typedef struct EmberTokenizer EmberTokenizer;

/* Per-run transient state (activations + KV cache). */
typedef struct EmberState EmberState;

/* io.c */
EmberModel     *ember_model_load(const char *path);
void            ember_model_free(EmberModel *m);
const EmberHeader *ember_model_header(const EmberModel *m);
/* Resolve a tensor by name; returns NULL if absent. */
const EmberTensor *ember_model_tensor(const EmberModel *m, const char *name);
const void        *ember_model_data(const EmberModel *m, const EmberTensor *t);

/* model.c */
EmberState *ember_state_new(const EmberModel *m, int ctx_len);
void        ember_state_free(EmberState *s);
/* Run one forward pass for token `tok` at position `pos`; returns logits
 * (length vocab_size), owned by the state. */
float      *ember_forward(EmberModel *m, EmberState *s, int tok, int pos);

/* tokenizer.c */
EmberTokenizer *ember_tokenizer_new(const EmberModel *m);
void            ember_tokenizer_free(EmberTokenizer *t);
/* Encode `text` into token ids (caller frees *out_ids). Returns count. */
int  ember_encode(EmberTokenizer *t, const char *text, int add_bos, int **out_ids);
/* Decode a single token given the previous token (for space-stripping rules).
 * Returns a pointer to a static/internal buffer valid until the next call. */
const char *ember_decode(EmberTokenizer *t, int prev_tok, int tok);

/* sample.c */
typedef struct {
    float temperature;
    float top_p;
    int   top_k;
    uint64_t rng_state;
} EmberSampler;

void ember_sampler_init(EmberSampler *s, float temp, float top_p, int top_k, uint64_t seed);
int  ember_sample(EmberSampler *s, float *logits, int n);

/* quant.c — offline requantization (target: EMBER_DT_Q8_0 or EMBER_DT_Q4_0) */
int ember_quantize_file(const char *in_path, const char *out_path, int target);

#endif /* EMBER_H */
