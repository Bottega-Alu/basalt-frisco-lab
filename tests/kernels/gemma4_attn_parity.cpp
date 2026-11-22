// tests/kernels/gemma4_attn_parity.cpp - Gemma 4's tiled prompt attention (ops::attention_tiled) against the simple
// kernel (ops::attention_simple) at the model's two geometries, a prompt fed in steps as the session does:
//
//   - sliding-window layers: 16 query heads, 8 KV heads, 256 wide, window 1024, a ring of 1024 + 512 cells (the ring wraps
//     and the window crosses it from the second step past 1536 on);
//   - global layers: 16 query heads, 2 KV heads, 512 wide, every position.
//
// Steps of 512 rows to 16000 positions (the last one short), then odd steps (9, 37, 300 rows) from a few positions.  For
// every step: the largest |difference| and the least cosine of an output row; fails past 2e-2 or under 0.9999.
// `--bench`: the last 512-row step at 16K, each kernel, each geometry (ms).  Exit 77 without a GPU.
#include "models/gemma4/ops.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace ops = basalt::models::gemma4::ops;

namespace {

constexpr int NH = 16, WINDOW = 1024, STEP = 512;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Geo {
    const char* name;
    int kvh, hd, window, ring;
};

struct Run {
    Geo g;
    int64_t cells;
    uint16_t *kc = nullptr, *vc = nullptr;
    float *q = nullptr, *k = nullptr, *v = nullptr, *o_ref = nullptr, *o_new = nullptr;
    std::mt19937 rng{1234};

    explicit Run(Geo geo, int64_t max_pos) : g(geo), cells(geo.ring > 0 ? geo.ring : max_pos) {
        const size_t kv = (size_t) cells * g.kvh * g.hd * 2;
        ck(cudaMalloc(&kc, kv), "kc");
        ck(cudaMalloc(&vc, kv), "vc");
        ck(cudaMemset(kc, 0xff, kv), "poison kc");   // f16 NaN: a stray read of an unwritten cell shows
        ck(cudaMemset(vc, 0xff, kv), "poison vc");
        ck(cudaMalloc(&q, (size_t) STEP * NH * g.hd * 4), "q");
        ck(cudaMalloc(&k, (size_t) STEP * g.kvh * g.hd * 4), "k");
        ck(cudaMalloc(&v, (size_t) STEP * g.kvh * g.hd * 4), "v");
        ck(cudaMalloc(&o_ref, (size_t) STEP * NH * g.hd * 4), "o_ref");
        ck(cudaMalloc(&o_new, (size_t) STEP * NH * g.hd * 4), "o_new");
    }
    ~Run() {
        for (void* p : {(void*) kc, (void*) vc, (void*) q, (void*) k, (void*) v, (void*) o_ref, (void*) o_new}) cudaFree(p);
    }

    // rows at pos0: fresh Q/K/V (RMS-normed rows times a Gemma-like scale), written to the cache
    void feed(int rows, int64_t pos0) {
        std::normal_distribution<float> nd(0.f, 1.f);
        auto fill = [&](float* dst, size_t n, float scale) {
            std::vector<float> h(n);
            for (auto& x : h) x = nd(rng) * scale;
            ck(cudaMemcpy(dst, h.data(), n * 4, cudaMemcpyHostToDevice), "upload");
        };
        fill(q, (size_t) rows * NH * g.hd, 0.35f);
        fill(k, (size_t) rows * g.kvh * g.hd, 1.0f);
        fill(v, (size_t) rows * g.kvh * g.hd, 1.0f);
        ops::kv_write(k, v, rows, g.kvh, g.hd, pos0, g.ring, kc, vc, nullptr);
    }

    // both kernels on the step; returns false past the bounds
    bool compare(int rows, int64_t pos0, bool print) {
        ops::attention_simple(q, kc, vc, rows, NH, g.kvh, g.hd, pos0, g.window, g.ring, o_ref, nullptr);
        if (!ops::attention_tiled(q, kc, vc, rows, NH, g.kvh, g.hd, pos0, g.window, g.ring, o_new, nullptr)) {
            std::fprintf(stderr, "%s: attention_tiled refused rows=%d\n", g.name, rows);
            return false;
        }
        ck(cudaDeviceSynchronize(), "step");
        const size_t w = (size_t) NH * g.hd, n = (size_t) rows * w;
        std::vector<float> a(n), b(n);
        ck(cudaMemcpy(a.data(), o_ref, n * 4, cudaMemcpyDeviceToHost), "down");
        ck(cudaMemcpy(b.data(), o_new, n * 4, cudaMemcpyDeviceToHost), "down");
        double max_abs = 0, min_cos = 1;
        bool finite = true;
        for (size_t r = 0; r < (size_t) rows * NH; ++r) {
            double dot = 0, na = 0, nb = 0;
            for (size_t i = r * g.hd; i < (r + 1) * g.hd; ++i) {
                if (!std::isfinite(b[i])) finite = false;
                max_abs = std::max(max_abs, (double) std::fabs(a[i] - b[i]));
                dot += (double) a[i] * b[i];
                na += (double) a[i] * a[i];
                nb += (double) b[i] * b[i];
            }
            min_cos = std::min(min_cos, dot / std::sqrt(na * nb + 1e-30));
        }
        const bool ok = finite && max_abs <= 2e-2 && min_cos >= 0.9999;
        if (print || !ok)
            std::printf("%-6s pos %6lld..%6lld (%3d rows): max|d| %.3e  min cos %.7f%s\n", g.name, (long long) pos0,
                        (long long) (pos0 + rows - 1), rows, max_abs, min_cos, ok ? "" : "  FAIL");
        return ok;
    }
};

double time_ms(auto&& f, int reps) {
    f();
    ck(cudaDeviceSynchronize(), "warm");
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) f();
    ck(cudaDeviceSynchronize(), "timed");
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
}

}  // namespace

int main(int argc, char** argv) {
    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess || nd == 0) {
        std::puts("no CUDA device: skipped");
        return 77;
    }
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    const int64_t total = 16000;
    const Geo geos[] = {{"swa", 8, 256, WINDOW, WINDOW + STEP}, {"global", 2, 512, 0, 0}};
    bool ok = true;
    for (const Geo& g : geos) {
        Run run(g, total);
        for (int64_t pos = 0; pos < total; pos += STEP) {
            const int rows = (int) std::min<int64_t>(STEP, total - pos);
            run.feed(rows, pos);
            const int64_t end = pos + rows;
            const bool print = end <= 2 * STEP || end == 1536 || end == 2048 || end == 4096 || end == 8192 || end == total ||
                               (pos < 1536 && end > 1536);
            ok = run.compare(rows, pos, print) && ok;
        }
        if (bench) {
            const int64_t pos0 = total - STEP;
            const double a = time_ms([&] { ops::attention_simple(run.q, run.kc, run.vc, STEP, NH, g.kvh, g.hd, pos0, g.window, g.ring, run.o_ref, nullptr); }, 5);
            const double b = time_ms([&] { ops::attention_tiled(run.q, run.kc, run.vc, STEP, NH, g.kvh, g.hd, pos0, g.window, g.ring, run.o_new, nullptr); }, 20);
            std::printf("bench %-6s 512 rows at %lld: simple %.3f ms, tiled %.3f ms (x%.1f)\n", g.name, (long long) pos0, a, b, a / b);
        }
    }
    // odd steps: 9, 37 and 300 rows from a few positions, on fresh caches
    for (const Geo& g : geos) {
        Run run(g, 4096);
        int64_t pos = 0;
        for (int rows : {9, 37, 300, 512, 512, 512, 9, 300, 37, 512, 512}) {
            run.feed(rows, pos);
            ok = run.compare(rows, pos, true) && ok;
            pos += rows;
        }
    }
    std::puts(ok ? "gemma4_attn_parity: OK" : "gemma4_attn_parity: FAIL");
    return ok ? 0 : 1;
}
