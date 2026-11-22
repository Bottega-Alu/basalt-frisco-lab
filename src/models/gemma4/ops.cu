// src/models/gemma4/ops.cu - Gemma 4's own kernels (ops.hpp).
#include "models/gemma4/ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace basalt::models::gemma4::ops {
namespace {

constexpr int kBlock = 256;
constexpr int kWarps = kBlock / 32;

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// The sum of `v` over the block (every thread gets it); `red` holds kWarps floats.
__device__ __forceinline__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float s = 0.f;
    const int nw = (blockDim.x + 31) >> 5;
    for (int w = 0; w < nw; ++w) s += red[w];
    return s;
}

__device__ __forceinline__ float gelu_tanh(float g) {
    return 0.5f * g * (1.0f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g)));
}

__global__ void embed_kernel(const uint8_t* __restrict__ table, const int32_t* __restrict__ ids, int d, float scale,
                             float* __restrict__ out) {
    const int r = blockIdx.x;
    const uint8_t* row = table + (size_t) ids[r] * (size_t) (d / 32) * 34;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const uint8_t* b = row + (size_t) (i / 32) * 34;
        const float s = __half2float(*reinterpret_cast<const __half*>(b));
        out[(size_t) r * d + i] = scale * s * (float) (int8_t) b[2 + (i & 31)];
    }
}

__global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out, int cols,
                                float eps) {
    __shared__ float red[kWarps];
    const float* xr = x + (size_t) blockIdx.x * cols;
    float ss = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) ss += xr[i] * xr[i];
    const float inv = rsqrtf(block_sum(ss, red) / (float) cols + eps);
    float* o = out + (size_t) blockIdx.x * cols;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) o[i] = xr[i] * inv * (w ? w[i] : 1.f);
}

__global__ void norm_add_kernel(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ w, int cols,
                                float eps, const float* __restrict__ scale) {
    __shared__ float red[kWarps];
    const float* yr = y + (size_t) blockIdx.x * cols;
    float ss = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) ss += yr[i] * yr[i];
    const float inv = rsqrtf(block_sum(ss, red) / (float) cols + eps);
    const float s = scale ? *scale : 1.f;
    float* xr = x + (size_t) blockIdx.x * cols;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) xr[i] = (xr[i] + yr[i] * inv * w[i]) * s;
}

__global__ void norm2_sum_kernel(const float* __restrict__ a, const float* __restrict__ wa, const float* __restrict__ b,
                                 const float* __restrict__ wb, float* __restrict__ out, int cols, float eps) {
    __shared__ float red[kWarps];
    const float* ar = a + (size_t) blockIdx.x * cols;
    const float* br = b + (size_t) blockIdx.x * cols;
    float sa = 0.f, sb = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sa += ar[i] * ar[i];
        sb += br[i] * br[i];
    }
    const float ia = rsqrtf(block_sum(sa, red) / (float) cols + eps);
    const float ib = rsqrtf(block_sum(sb, red) / (float) cols + eps);
    float* o = out + (size_t) blockIdx.x * cols;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) o[i] = ar[i] * ia * wa[i] + br[i] * ib * wb[i];
}

// a block per (head, row), hd/2 threads: thread i holds the pair (i, i + hd/2)
__global__ void head_norm_rope_kernel(float* __restrict__ x, const float* __restrict__ w, int hd, int row_stride,
                                      int64_t pos0, float base, const float* __restrict__ ff, float eps,
                                      const int64_t* __restrict__ pos_dev) {
    __shared__ float red[kWarps * 2];
    if (pos_dev) pos0 = *pos_dev;
    const int half = hd / 2, i = threadIdx.x;
    float* h = x + (size_t) blockIdx.y * row_stride + (size_t) blockIdx.x * hd;
    float a = h[i], b = h[i + half];
    const float inv = rsqrtf(block_sum(a * a + b * b, red) / (float) hd + eps);
    a *= inv * (w ? w[i] : 1.f);
    b *= inv * (w ? w[i + half] : 1.f);
    if (base > 0.f) {
        const float pos = (float) (pos0 + blockIdx.y);
        float theta = pos * powf(base, -2.0f * (float) i / (float) hd);
        if (ff) theta /= ff[i];
        float s, c;
        sincosf(theta, &s, &c);
        const float na = a * c - b * s, nb = a * s + b * c;
        a = na;
        b = nb;
    }
    h[i] = a;
    h[i + half] = b;
}

__global__ void kv_write_kernel(const float* __restrict__ k, const float* __restrict__ v, int width, int64_t pos0, int ring,
                                __half* __restrict__ kc, __half* __restrict__ vc, const int64_t* __restrict__ pos_dev) {
    if (pos_dev) pos0 = *pos_dev;
    const int64_t pos = pos0 + blockIdx.x;
    const int64_t cell = ring > 0 ? pos % ring : pos;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        kc[cell * width + i] = __float2half(k[(size_t) blockIdx.x * width + i]);
        vc[cell * width + i] = __float2half(v[(size_t) blockIdx.x * width + i]);
    }
}

// A block per (head, row), 8 warps over the positions, each with an online softmax; lane l holds values l, l + 32, ...
template<int HD>
__global__ void __launch_bounds__(kBlock) attention_kernel(const float* __restrict__ q, const __half* __restrict__ kc,
                                                          const __half* __restrict__ vc, int n_head, int kvh, int64_t pos0,
                                                          int window, int ring, float* __restrict__ out,
                                                          const int64_t* __restrict__ pos_dev) {
    constexpr int V = HD / 32;
    if (pos_dev) pos0 = *pos_dev;
    __shared__ float sm[kWarps], sl[kWarps];
    __shared__ float sacc[kWarps][HD];
    const int h = blockIdx.x, r = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int kh = h / (n_head / kvh);
    const int64_t p = pos0 + r;
    const int64_t lo = window > 0 ? (p - window + 1 > 0 ? p - window + 1 : 0) : 0;
    const float* qr = q + ((size_t) r * n_head + h) * HD;
    float qv[V], acc[V];
#pragma unroll
    for (int t = 0; t < V; ++t) {
        qv[t] = qr[t * 32 + lane];
        acc[t] = 0.f;
    }
    float m = -INFINITY, l = 0.f;
    const size_t stride = (size_t) kvh * HD;
    for (int64_t j = lo + warp; j <= p; j += kWarps) {
        const int64_t cell = ring > 0 ? j % ring : j;
        const __half* kp = kc + (size_t) cell * stride + (size_t) kh * HD;
        const __half* vp = vc + (size_t) cell * stride + (size_t) kh * HD;
        float dot = 0.f;
#pragma unroll
        for (int t = 0; t < V; ++t) dot += qv[t] * __half2float(kp[t * 32 + lane]);
        dot = warp_sum(dot);
        const float mn = fmaxf(m, dot);
        const float c = __expf(m - mn), e = __expf(dot - mn);
        l = l * c + e;
#pragma unroll
        for (int t = 0; t < V; ++t) acc[t] = acc[t] * c + e * __half2float(vp[t * 32 + lane]);
        m = mn;
    }
    if (lane == 0) {
        sm[warp] = m;
        sl[warp] = l;
    }
#pragma unroll
    for (int t = 0; t < V; ++t) sacc[warp][t * 32 + lane] = acc[t];
    __syncthreads();
    float M = -INFINITY;
    for (int w = 0; w < kWarps; ++w) M = fmaxf(M, sm[w]);
    float L = 0.f, f[kWarps];
    for (int w = 0; w < kWarps; ++w) {
        f[w] = sm[w] == -INFINITY ? 0.f : __expf(sm[w] - M);
        L += sl[w] * f[w];
    }
    float* o = out + ((size_t) r * n_head + h) * HD;
    for (int d = threadIdx.x; d < HD; d += kBlock) {
        float s = 0.f;
        for (int w = 0; w < kWarps; ++w) s += sacc[w][d] * f[w];
        o[d] = s / L;
    }
}

// The router's logits: block (b, r) takes row r and experts [8b, 8b + 8), a warp each; every block norms the row itself
// (rms(x) / sqrt(d) * scale, d <= 4096 floats in shared memory).
__global__ void router_logits_kernel(const float* __restrict__ x, const float* __restrict__ scale, const float* __restrict__ w,
                                     int d, int n_expert, float eps, float* __restrict__ logits) {
    __shared__ float xs[4096];
    __shared__ float red[kWarps];
    const int r = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float* xr = x + (size_t) r * d;
    float ss = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) ss += xr[i] * xr[i];
    const float inv = rsqrtf(block_sum(ss, red) / (float) d + eps) * rsqrtf((float) d);
    for (int i = threadIdx.x; i < d; i += blockDim.x) xs[i] = xr[i] * inv * scale[i];
    __syncthreads();
    const int e = blockIdx.x * kWarps + warp;
    if (e >= n_expert) return;
    const float4* we = (const float4*) (w + (size_t) e * d);
    const float4* x4 = (const float4*) xs;
    float s = 0.f;
    for (int i = lane; i < d / 4; i += 32) {
        const float4 a = we[i], c = x4[i];
        s += a.x * c.x + a.y * c.y + a.z * c.z + a.w * c.w;
    }
    s = warp_sum(s);
    if (lane == 0) logits[(size_t) r * n_expert + e] = s;
}

// The top k of a row's logits (a warp per row; ties to the lower expert), their softmax as the weights.
__global__ void router_pick_kernel(const float* __restrict__ logits, int rows, int n_expert, int k, int32_t* __restrict__ ids,
                                   float* __restrict__ weights) {
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= rows) return;
    constexpr int kPer = 8;   // n_expert <= 256
    float v[kPer];
    for (int t = 0; t < kPer; ++t) {
        const int e = lane + 32 * t;
        v[t] = e < n_expert ? logits[(size_t) r * n_expert + e] : -INFINITY;
    }
    float top = 0.f, sum = 0.f, lv = 0.f;
    int pick = 0;
    for (int j = 0; j < k; ++j) {
        float bv = -INFINITY;
        int be = 0x7fffffff;
        for (int t = 0; t < kPer; ++t)
            if (v[t] > bv) bv = v[t], be = lane + 32 * t;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oe = __shfl_xor_sync(0xffffffffu, be, o);
            if (ov > bv || (ov == bv && oe < be)) bv = ov, be = oe;
        }
        if (j == 0) top = bv;
        const float ex = __expf(bv - top);
        sum += ex;
        if (lane == j) lv = ex, pick = be;
        if ((be & 31) == lane) v[be >> 5] = -INFINITY;
    }
    if (lane < k) {
        ids[r * k + lane] = pick;
        weights[r * k + lane] = lv / sum;
    }
}

__global__ void moe_combine_kernel(const float* __restrict__ eo, const float* __restrict__ weights, int k, int d,
                                   float* __restrict__ out) {
    const int r = blockIdx.x;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float s = 0.f;
        for (int j = 0; j < k; ++j) s += weights[r * k + j] * eo[((size_t) r * k + j) * d + i];
        out[(size_t) r * d + i] = s;
    }
}

__global__ void geglu_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ h, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        h[i] = gelu_tanh(g[i]) * u[i];
}

__global__ void softcap_kernel(float* __restrict__ x, int64_t n, float cap) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        x[i] = cap * tanhf(x[i] / cap);
}

unsigned grid_for(int64_t n) {
    const int64_t b = (n + kBlock - 1) / kBlock;
    return (unsigned) (b < 4096 ? (b < 1 ? 1 : b) : 4096);
}

}  // namespace

void embed_rows(const uint8_t* q8_table, const int32_t* ids, int rows, int d, float scale, float* out, void* stream) {
    embed_kernel<<<rows, kBlock, 0, (cudaStream_t) stream>>>(q8_table, ids, d, scale, out);
}

void rms_norm(const float* x, const float* w, float* out, int rows, int cols, float eps, void* stream) {
    rms_norm_kernel<<<rows, kBlock, 0, (cudaStream_t) stream>>>(x, w, out, cols, eps);
}

void norm_add(float* x, const float* y, const float* w, int rows, int cols, float eps, const float* scale, void* stream) {
    norm_add_kernel<<<rows, kBlock, 0, (cudaStream_t) stream>>>(x, y, w, cols, eps, scale);
}

void norm2_sum(const float* a, const float* wa, const float* b, const float* wb, float* out, int rows, int cols, float eps,
               void* stream) {
    norm2_sum_kernel<<<rows, kBlock, 0, (cudaStream_t) stream>>>(a, wa, b, wb, out, cols, eps);
}

void head_norm_rope(float* x, const float* w, int rows, int n_heads, int hd, int row_stride, int64_t pos0, float base,
                    const float* ff, float eps, void* stream, const int64_t* pos_dev) {
    head_norm_rope_kernel<<<dim3((unsigned) n_heads, (unsigned) rows), hd / 2, 0, (cudaStream_t) stream>>>(
        x, w, hd, row_stride, pos0, base, ff, eps, pos_dev);
}

void kv_write(const float* k, const float* v, int rows, int kvh, int hd, int64_t pos0, int ring, uint16_t* kc, uint16_t* vc,
              void* stream, const int64_t* pos_dev) {
    kv_write_kernel<<<rows, kBlock, 0, (cudaStream_t) stream>>>(k, v, kvh * hd, pos0, ring, (__half*) kc, (__half*) vc, pos_dev);
}

void attention(const float* q, const uint16_t* kc, const uint16_t* vc, int rows, int n_head, int kvh, int hd, int64_t pos0,
               int window, int ring, float* out, void* stream, const int64_t* pos_dev) {
    // the tiled kernel takes pos0 from the host: prompt chunks run eagerly (graphs capture only rows <= 8), so it is current
    if (rows > 8 && attention_tiled(q, kc, vc, rows, n_head, kvh, hd, pos0, window, ring, out, stream)) return;
    attention_simple(q, kc, vc, rows, n_head, kvh, hd, pos0, window, ring, out, stream, pos_dev);
}

void attention_simple(const float* q, const uint16_t* kc, const uint16_t* vc, int rows, int n_head, int kvh, int hd,
                      int64_t pos0, int window, int ring, float* out, void* stream, const int64_t* pos_dev) {
    const dim3 grid((unsigned) n_head, (unsigned) rows);
    cudaStream_t s = (cudaStream_t) stream;
    if (hd == 256)
        attention_kernel<256><<<grid, kBlock, 0, s>>>(q, (const __half*) kc, (const __half*) vc, n_head, kvh, pos0, window, ring, out, pos_dev);
    else
        attention_kernel<512><<<grid, kBlock, 0, s>>>(q, (const __half*) kc, (const __half*) vc, n_head, kvh, pos0, window, ring, out, pos_dev);
}

void router_topk(const float* x, const float* scale, const float* w, int rows, int d, int n_expert, int k, float eps,
                 int32_t* ids, float* weights, float* logits, void* stream) {
    const cudaStream_t s = (cudaStream_t) stream;
    router_logits_kernel<<<dim3((unsigned) ((n_expert + kWarps - 1) / kWarps), (unsigned) rows), kBlock, 0, s>>>(x, scale, w, d, n_expert,
                                                                                                             eps, logits);
    router_pick_kernel<<<(rows + 7) / 8, 256, 0, s>>>(logits, rows, n_expert, k, ids, weights);
}

void moe_combine(const float* eo, const float* weights, int rows, int k, int d, float* out, void* stream) {
    moe_combine_kernel<<<rows, kBlock, 0, (cudaStream_t) stream>>>(eo, weights, k, d, out);
}

void geglu(const float* g, const float* u, float* h, int64_t n, void* stream) {
    geglu_kernel<<<grid_for(n), kBlock, 0, (cudaStream_t) stream>>>(g, u, h, n);
}

void softcap(float* x, int64_t n, float cap, void* stream) {
    softcap_kernel<<<grid_for(n), kBlock, 0, (cudaStream_t) stream>>>(x, n, cap);
}

__global__ void expert_plan_kernel(const int32_t* __restrict__ ids, int n, int k, const int32_t* __restrict__ slot_of,
                                   const uint8_t* arena, uint64_t stride, const uint8_t* host, uint64_t blob,
                                   unsigned long long* ptr, int32_t* start, int32_t* ng, int32_t* dst, int32_t* tok) {
    // one thread: a window has at most 64 picks
    int uniq[64], nu = 0;
    const int total = n * k;
    for (int i = 0; i < total; ++i) {
        const int e = ids[i];
        int g = 0;
        while (g < nu && uniq[g] != e) ++g;
        if (g == nu) uniq[nu++] = e;
    }
    int at = 0;
    for (int g = 0; g < nu; ++g) {
        const int e = uniq[g], s = slot_of[e];
        ptr[g] = (unsigned long long) (uintptr_t) (s >= 0 ? arena + (uint64_t) s * stride : host + (uint64_t) e * blob);
        start[g] = at;
        for (int i = 0; i < total; ++i)
            if (ids[i] == e) {
                dst[at] = i;
                tok[at] = i / k;
                ++at;
            }
    }
    start[nu] = at;
    ng[0] = nu;
}

void expert_plan(const int32_t* ids, int rows, int k, const int32_t* slot_of, const uint8_t* arena, uint64_t stride,
                 const uint8_t* host, uint64_t blob, unsigned long long* ptr, int32_t* start, int32_t* ng, int32_t* dst,
                 int32_t* tok, void* stream) {
    expert_plan_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(ids, rows, k, slot_of, arena, stride, host, blob, ptr, start, ng, dst,
                                                           tok);
}

void copy(float* dst, const float* src, int64_t n, void* stream) {
    cudaMemcpyAsync(dst, src, (size_t) n * sizeof(float), cudaMemcpyDeviceToDevice, (cudaStream_t) stream);
}

}  // namespace basalt::models::gemma4::ops
