// src/models/gemma4/attn.cu - Gemma 4's prompt attention on the tensor cores (ops::attention_tiled, ops.hpp).
//
// Flash-attention style, after the shared prompt_f16_kernel (src/kernels/cuda/gqa_attn.cu), which is fixed at 256-wide
// heads, scale 1/16 and a linear cache: here the head is 256 (sliding-window layers) or 512 wide (global layers), the
// scale is 1, the window masks cells older than window - 1 positions and the cache may be a ring (cell = pos % ring).
//
// Block (query tile, KV head): its 16 * NG rows are the tile's positions times the KV group's query heads (head fastest),
// so a K/V tile is read once for the whole group.  A 16-row group belongs to P = HD / 256 warps, warp slice s owning dims
// 256 s .. 256 s + 255: each warp sums its slice of Q K^T, the P partial scores are added through shared memory (in the
// same order on every warp, so both hold the same scores), the online softmax runs on each, and each warp accumulates
// its slice of the output, P V.  Cells stream in tiles of BN.  Q K^T and P V on fp16 tensor cores, fp32 accumulation;
// online softmax in fp32.  With QS the query tile is held as two fp16 parts (q = hi + lo) and Q K^T takes both, so the
// scores carry the fp32 query against the fp16 keys (an order of magnitude closer to the simple kernel than fp16 Q).
// The 512-wide tile's partial scores are exchanged through the key tile once every warp has read it.
#include "models/gemma4/ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace basalt::models::gemma4::ops {
namespace {

constexpr float kLog2e = 1.4426950408889634f;
constexpr float kNeg = -1e30f;
constexpr int DS = 256;   // dims of one warp's slice

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}
__device__ __forceinline__ void cp16(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ float quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
__device__ __forceinline__ float quad_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

template <int HD, int NG, int BN, bool QS>
struct Tile {
    static_assert(HD % DS == 0 && HD / DS <= 2, "a 256- or 512-wide head");
    static_assert(BN % 16 == 0, "cells in steps of 16");
    static constexpr int P = HD / DS;                  // warps per 16-row group
    static constexpr int NW = NG * P;                  // warps per block
    static constexpr int ROWS = 16 * NG;               // query rows per block
    static constexpr int LD = HD + 8;                  // padded fp16 row (halves)
    static constexpr int XF = (BN / 8) * 4;            // partial-score floats per thread
    static constexpr int QT = QS ? 2 : 1;              // query tiles: fp16 high part, and the low part when split
    static constexpr size_t smem = (size_t) (QT * ROWS + 2 * BN) * LD * 2;
    // the partial scores reuse the key tile once every warp has read it
    static_assert(P == 1 || (size_t) NW * 32 * XF * 4 <= (size_t) BN * LD * 2, "the score exchange fits the key tile");
};

template <int HD, int NG, int BN, bool QS>
__global__ void __launch_bounds__(Tile<HD, NG, BN, QS>::NW * 32, 1)
    tiled_kernel(const float* __restrict__ q, const __half* __restrict__ kc, const __half* __restrict__ vc, int n_head,
                 int kvh, int64_t pos0, int T, int window, int ring, float* __restrict__ out) {
    using C = Tile<HD, NG, BN, QS>;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half(*sq)[C::LD] = reinterpret_cast<__half(*)[C::LD]>(smem_raw);
    __half(*sql)[C::LD] = sq + (QS ? C::ROWS : 0);   // the low part (QS)
    __half(*sk)[C::LD] = sq + C::QT * C::ROWS;
    __half(*sv)[C::LD] = sk + BN;
    float* xs = reinterpret_cast<float*>(&sk[0][0]);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int grp = warp / C::P, sl = warp % C::P;
    const int G = n_head / kvh, PB = C::ROWS / G;   // query heads per KV head, positions per tile
    const int kh = blockIdx.y;
    const int q0 = (int) (gridDim.x - 1 - blockIdx.x) * PB;   // the longest tiles first
    const int last = min(T, q0 + PB) - 1;
    const int64_t c_hi = pos0 + last;                                                     // the newest cell any row sees
    const int64_t c_lo = window > 0 ? max((int64_t) 0, pos0 + q0 - window + 1) : 0;     // the oldest
    const int g = lane >> 2, tq = lane & 3;
    const int ra = grp * 16 + g, rb = ra + 8;   // this thread's two rows
    const int row_a = q0 + ra / G, row_b = q0 + rb / G;
    const int64_t p_a = pos0 + row_a, p_b = pos0 + row_b;
    const int head_a = kh * G + ra % G, head_b = kh * G + rb % G;

    for (int i = tid; i < C::ROWS * HD / 4; i += C::NW * 32) {
        const int r = i / (HD / 4), c = (i % (HD / 4)) * 4;
        const int row = q0 + r / G, head = kh * G + r % G;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (row < T) v = *reinterpret_cast<const float4*>(q + ((size_t) row * n_head + head) * HD + c);
        const __half2 h0 = __floats2half2_rn(v.x, v.y), h1 = __floats2half2_rn(v.z, v.w);
        *reinterpret_cast<__half2*>(&sq[r][c]) = h0;
        *reinterpret_cast<__half2*>(&sq[r][c + 2]) = h1;
        if constexpr (QS) {   // q = hi + lo: the scores carry q's fp32 precision against the fp16 keys
            const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
            *reinterpret_cast<__half2*>(&sql[r][c]) = __floats2half2_rn(v.x - f0.x, v.y - f0.y);
            *reinterpret_cast<__half2*>(&sql[r][c + 2]) = __floats2half2_rn(v.z - f1.x, v.w - f1.y);
        }
    }

    float o[DS / 8][4];
#pragma unroll
    for (int n = 0; n < DS / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float m_a = kNeg, m_b = kNeg, l_a = 0.0f, l_b = 0.0f;
    const int kb = sl * DS;   // this warp's dims

    for (int64_t c0 = c_lo / BN * BN; c0 <= c_hi; c0 += BN) {
        __syncthreads();   // the previous tile is consumed (and the query tile is written)
        for (int i = tid; i < BN * HD / 8; i += C::NW * 32) {
            const int r = i / (HD / 8), c = (i % (HD / 8)) * 8;
            const int64_t cell = c0 + r;
            const bool ok = cell >= c_lo && cell <= c_hi;   // the rest zero: no stale ring cell reaches the sums
            const int64_t slot = ok ? (ring > 0 ? cell % ring : cell) : 0;
            const size_t at = ((size_t) slot * kvh + kh) * HD + c;
            cp16(&sk[r][c], kc + at, ok);
            cp16(&sv[r][c], vc + at, ok);
        }
        asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();

        // S = Q K^T over this warp's dims, for its group's 16 rows and the tile's BN cells
        float s[BN / 8][4];
#pragma unroll
        for (int n = 0; n < BN / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll 4
        for (int k = 0; k < DS; k += 16) {
            unsigned a0, a1, a2, a3, l0 = 0, l1 = 0, l2 = 0, l3 = 0;
            ldsm_x4(a0, a1, a2, a3, &sq[grp * 16 + (lane & 15)][kb + k + (lane >> 4) * 8]);
            if constexpr (QS) ldsm_x4(l0, l1, l2, l3, &sql[grp * 16 + (lane & 15)][kb + k + (lane >> 4) * 8]);
#pragma unroll
            for (int n = 0; n < BN / 8; n += 2) {
                unsigned b0, b1, b2, b3;
                ldsm_x4(b0, b1, b2, b3, &sk[n * 8 + (lane & 7) + ((lane >> 4) << 3)][kb + k + ((lane >> 3) & 1) * 8]);
                if constexpr (QS) {
                    mma16816(s[n], l0, l1, l2, l3, b0, b1);
                    mma16816(s[n + 1], l0, l1, l2, l3, b2, b3);
                }
                mma16816(s[n], a0, a1, a2, a3, b0, b1);
                mma16816(s[n + 1], a0, a1, a2, a3, b2, b3);
            }
        }
        if constexpr (C::P > 1) {   // the two slices' partial scores, summed as mine + the other's on both warps
            __syncthreads();         // every warp is done with the key tile, which holds the exchange
            float* mine = xs + ((size_t) warp * 32 + lane) * C::XF;
#pragma unroll
            for (int n = 0; n < BN / 8; ++n)
#pragma unroll
                for (int i = 0; i < 4; ++i) mine[n * 4 + i] = s[n][i];
            __syncthreads();
            const float* other = xs + ((size_t) (warp ^ 1) * 32 + lane) * C::XF;
#pragma unroll
            for (int n = 0; n < BN / 8; ++n)
#pragma unroll
                for (int i = 0; i < 4; ++i) s[n][i] += other[n * 4 + i];
        }
        // mask and online softmax: the thread holds rows ra (s[n][0..1]) and rb (s[n][2..3])
        float mx_a = m_a, mx_b = m_b;
#pragma unroll
        for (int n = 0; n < BN / 8; ++n)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int64_t cell = c0 + n * 8 + tq * 2 + (i & 1);
                const int64_t p = i < 2 ? p_a : p_b;
                const bool vis = cell <= p && cell >= c_lo && (window <= 0 || cell > p - window);
                s[n][i] = vis ? s[n][i] * kLog2e : kNeg;
                if (i < 2) mx_a = fmaxf(mx_a, s[n][i]);
                else mx_b = fmaxf(mx_b, s[n][i]);
            }
        mx_a = quad_max(mx_a);
        mx_b = quad_max(mx_b);
        const float sc_a = exp2f(m_a - mx_a), sc_b = exp2f(m_b - mx_b);
        m_a = mx_a;
        m_b = mx_b;
        float ps_a = 0.0f, ps_b = 0.0f;
#pragma unroll
        for (int n = 0; n < BN / 8; ++n) {
            s[n][0] = s[n][0] <= kNeg ? 0.0f : exp2f(s[n][0] - m_a);
            s[n][1] = s[n][1] <= kNeg ? 0.0f : exp2f(s[n][1] - m_a);
            s[n][2] = s[n][2] <= kNeg ? 0.0f : exp2f(s[n][2] - m_b);
            s[n][3] = s[n][3] <= kNeg ? 0.0f : exp2f(s[n][3] - m_b);
            ps_a += s[n][0] + s[n][1];
            ps_b += s[n][2] + s[n][3];
        }
        l_a = l_a * sc_a + ps_a;
        l_b = l_b * sc_b + ps_b;
        // O = O * scale + P V over this warp's dims
        unsigned pa[BN / 16][4];
#pragma unroll
        for (int j = 0; j < BN / 16; ++j) {
            pa[j][0] = pack_h2(s[2 * j][0], s[2 * j][1]);
            pa[j][1] = pack_h2(s[2 * j][2], s[2 * j][3]);
            pa[j][2] = pack_h2(s[2 * j + 1][0], s[2 * j + 1][1]);
            pa[j][3] = pack_h2(s[2 * j + 1][2], s[2 * j + 1][3]);
        }
#pragma unroll
        for (int n = 0; n < DS / 8; ++n) {
            o[n][0] *= sc_a;
            o[n][1] *= sc_a;
            o[n][2] *= sc_b;
            o[n][3] *= sc_b;
        }
#pragma unroll
        for (int n = 0; n < DS / 8; n += 2) {
#pragma unroll
            for (int j = 0; j < BN / 16; ++j) {
                unsigned b0, b1, b2, b3;
                // V rows are cells (k), columns dims (n): transposed loads give the col-major B of dims n*8.., n*8+8..
                ldsm_x4_t(b0, b1, b2, b3, &sv[j * 16 + (lane & 15)][kb + n * 8 + (lane >> 4) * 8]);
                mma16816(o[n], pa[j][0], pa[j][1], pa[j][2], pa[j][3], b0, b1);
                mma16816(o[n + 1], pa[j][0], pa[j][1], pa[j][2], pa[j][3], b2, b3);
            }
        }
    }
    l_a = quad_sum(l_a);
    l_b = quad_sum(l_b);
    const float ia = l_a > 0.0f ? 1.0f / l_a : 0.0f, ib = l_b > 0.0f ? 1.0f / l_b : 0.0f;
#pragma unroll
    for (int n = 0; n < DS / 8; ++n) {
        const int col = kb + n * 8 + tq * 2;
        if (row_a < T) *reinterpret_cast<float2*>(out + ((size_t) row_a * n_head + head_a) * HD + col) = make_float2(o[n][0] * ia, o[n][1] * ia);
        if (row_b < T) *reinterpret_cast<float2*>(out + ((size_t) row_b * n_head + head_b) * HD + col) = make_float2(o[n][2] * ib, o[n][3] * ib);
    }
}

template <int HD, int NG, int BN, bool QS>
bool launch(const float* q, const uint16_t* kc, const uint16_t* vc, int rows, int n_head, int kvh, int64_t pos0, int window,
            int ring, float* out, cudaStream_t s) {
    using C = Tile<HD, NG, BN, QS>;
    const int G = n_head / kvh;
    if (C::ROWS % G) return false;
    auto* k = tiled_kernel<HD, NG, BN, QS>;
    // the opt-in is per device and cheap: set on every call rather than tracked
    if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) C::smem) != cudaSuccess) return false;
    const int pb = C::ROWS / G;
    const dim3 grid((unsigned) ((rows + pb - 1) / pb), (unsigned) kvh);
    k<<<grid, C::NW * 32, C::smem, s>>>(q, (const __half*) kc, (const __half*) vc, n_head, kvh, pos0, rows, window, ring, out);
    return true;
}

}  // namespace

bool attention_tiled(const float* q, const uint16_t* kc, const uint16_t* vc, int rows, int n_head, int kvh, int hd,
                     int64_t pos0, int window, int ring, float* out, void* stream) {
    if (rows < 1 || kvh < 1 || n_head % kvh) return false;
    const cudaStream_t s = (cudaStream_t) stream;
    // QS: the queries as fp16 high + low parts, so the scores keep q's fp32 precision (scale 1: Gemma's scores are large)
    if (hd == 256) return launch<256, 4, 32, true>(q, kc, vc, rows, n_head, kvh, pos0, window, ring, out, s);
    if (hd == 512) return launch<512, 2, 16, true>(q, kc, vc, rows, n_head, kvh, pos0, window, ring, out, s);
    return false;
}

}  // namespace basalt::models::gemma4::ops
