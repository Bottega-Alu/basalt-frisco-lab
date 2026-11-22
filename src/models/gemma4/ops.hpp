// src/models/gemma4/ops.hpp - the Gemma 4 kernels a row pass runs that the shared kernels do not have: the embedding
// rows, the norms, the per-head norm and RoPE (plain and proportional), the K/V cache rows, the attention (a sliding
// window or the whole context, any head width up to 512), the router, the experts' weighted sum, GeGLU and the softcap.
// Every pointer is device memory; every call enqueues on `stream` and allocates nothing. Rows are [rows][width].
#pragma once

#include <cstdint>

namespace basalt::models::gemma4::ops {

/// out[r] = scale * dequant(table row ids[r]) for a Q8_0 table of `d`-wide rows.
void embed_rows(const uint8_t* q8_table, const int32_t* ids, int rows, int d, float scale, float* out, void* stream);

/// out = x / rms(x) * w (w null: no weight), per row of `cols`. out may be x.
void rms_norm(const float* x, const float* w, float* out, int rows, int cols, float eps, void* stream);

/// x += rms_norm(y) * w, per row; with `scale` (device, one float) the sum is then multiplied by *scale.
void norm_add(float* x, const float* y, const float* w, int rows, int cols, float eps, const float* scale, void* stream);

/// out = rms_norm(a) * wa + rms_norm(b) * wb, per row (the dense and the expert branch of a layer, each post-normed).
void norm2_sum(const float* a, const float* wa, const float* b, const float* wb, float* out, int rows, int cols, float eps,
               void* stream);

/// In place on heads of `hd` values (`n_heads` per row, rows `row_stride` floats apart): RMS norm with weight `w` (null:
/// none), then, when `base` > 0, NEOX RoPE over the whole head at position pos0 + row: pair (i, i + hd/2) turns by
/// pos * base^(-2i/hd) / ff[i] (ff null: 1).
void head_norm_rope(float* x, const float* w, int rows, int n_heads, int hd, int row_stride, int64_t pos0, float base,
                    const float* ff, float eps, void* stream);

/// Writes rows' K and V (f32, [rows][kvh*hd]) as f16 into the caches at position pos0 + row: cell pos % ring (ring 0:
/// cell pos).
void kv_write(const float* k, const float* v, int rows, int kvh, int hd, int64_t pos0, int ring, uint16_t* kc, uint16_t* vc,
              void* stream);

/// out[r][h] = softmax(q[r][h] . K) V over the cached positions row r sees (pos0 + r back to the window's start; window 0:
/// position 0), scale 1. q and out are [rows][n_head*hd]; head h reads K/V head h / (n_head / kvh). hd <= 512.
void attention(const float* q, const uint16_t* kc, const uint16_t* vc, int rows, int n_head, int kvh, int hd, int64_t pos0,
               int window, int ring, float* out, void* stream);

/// The router of `rows` rows: x / rms(x) / sqrt(d) * scale, its logits through w ([n_expert][d] f32), the softmax's top
/// `k` renormalised. ids[r*k + j], weights[r*k + j]. n_expert <= 256, k <= 16, d <= 4096.
/// `logits`: scratch of rows * n_expert floats (n_expert <= 256, k <= 16).
void router_topk(const float* x, const float* scale, const float* w, int rows, int d, int n_expert, int k, float eps,
                 int32_t* ids, float* weights, float* logits, void* stream);

/// out[r] = sum_j weights[r*k + j] * eo[r*k + j], rows of `d`.
void moe_combine(const float* eo, const float* weights, int rows, int k, int d, float* out, void* stream);

/// h = gelu_tanh(g) * u, n values.
void geglu(const float* g, const float* u, float* h, int64_t n, void* stream);

/// x = cap * tanh(x / cap), n values.
void softcap(float* x, int64_t n, float cap, void* stream);

/// The grouped expert kernel's plan of a decode window, built on the device from the router's picks `ids` (rows x k): one
/// group per distinct expert in order of first use, its entries in row order (dst = r * k + j, tok = r). An expert's
/// weights: its VRAM slot (`slot_of[e]` >= 0: `arena` + slot * `stride`), else its copy in mapped pinned host memory
/// (`host` + e * `blob`), read over the bus.
void expert_plan(const int32_t* ids, int rows, int k, const int32_t* slot_of, const uint8_t* arena, uint64_t stride,
                 const uint8_t* host, uint64_t blob, unsigned long long* ptr, int32_t* start, int32_t* ng, int32_t* dst,
                 int32_t* tok, void* stream);

/// dst = src, n floats.
void copy(float* dst, const float* src, int64_t n, void* stream);

}  // namespace basalt::models::gemma4::ops
