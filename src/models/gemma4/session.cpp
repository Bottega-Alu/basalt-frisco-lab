// src/models/gemma4/session.cpp - one conversation of Gemma 4: its K/V caches (a ring of the window's positions on the
// sliding-window layers, every position on the others) and the forward of up to kMaxRows rows that every decode window and
// every step of a prompt goes through.
//
// Per layer: the attention (norm, Q/K/V, per-head norms and RoPE, the cache rows, the attention, the output projection and
// its post-norm into the residual), then the FFN: the dense GeGLU MLP and the routed experts side by side, each post-normed,
// their sum post-normed into the residual, which the layer's scalar multiplies. The router's picks come to the host, which
// finds each expert's VRAM slot (a miss is copied in over the bus) and launches the grouped expert kernel on them.
#include "basalt/core/checkpoint_regions.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/logprobs.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/log.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "models/gemma4/model.hpp"
#include "models/gemma4/ops.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace basalt::models::gemma4 {
namespace {

namespace bk = basalt::kernels;

namespace mmq = basalt::prefill::mmq;

constexpr int kChunk = 512;       // rows of one prompt step (MMQ products); a decode window runs kMaxRows at most
constexpr int kMaxEntries = kMaxRows * 16;
constexpr int kTypeQ8_0 = 8, kTypeNvfp4 = 40;

// One prompt pass of a layer's experts, uploaded with one copy: the (token, pick) pairs sorted by VRAM slot.
struct PromptPlan {
    int32_t tok[kChunk * 8];      // the pair's token row (the quantizer's gather)
    int32_t pair[kChunk * 8];     // the pair's index r * k + j (the down product's destination row)
    int32_t slot[kChunk * 8];     // the pair's slot (its tail scales)
    int32_t bounds[132];          // per slot: the first sorted pair (slots + 1 entries)
};

// One layer's plan for the grouped expert kernel, uploaded with one copy.
struct Plan {
    unsigned long long ptr[kMaxEntries];
    int32_t start[kMaxEntries + 2];
    int32_t ng[4];
    int32_t dst[kMaxEntries];
    int32_t tok[kMaxEntries];
};

class Session final : public model::ModelSession {
public:
    explicit Session(GemmaModel& m) : m_(m), g_(m.geo()) {}

    ~Session() override {
        if (stream_) cudaStreamSynchronize(stream_);
        for (void* p : bufs_) cudaFree(p);
        for (void* p : pinned_) cudaFreeHost(p);
        if (bias_) cudaFree(bias_);
        if (mask_) cudaFree(mask_);
        if (hist_d_) cudaFree(hist_d_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    bool open(std::string& err) {
        if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) { err = "no CUDA stream"; return false; }
        const int D = g_.n_embd, H = g_.n_head;
        // the prompt path: MMQ products of kChunk rows, when every product's shape has a tile on this card
        bool big = mmq::built() && mmq::fits(kTypeNvfp4, 2 * (int64_t) g_.n_ff_exp) && mmq::fits(kTypeNvfp4, D) &&
                   g_.n_used <= 8 && g_.n_expert <= 128 && mmq::fits(kTypeQ8_0, D) && mmq::fits(kTypeQ8_0, g_.n_ff);
        for (int l = 0; l < g_.n_layers && big; ++l)
            big = mmq::fits(kTypeQ8_0, (int64_t) H * g_.head_dim(l)) &&
                  mmq::fits(kTypeQ8_0, (int64_t) g_.kv_heads[(size_t) l] * g_.head_dim(l));
        rows_ = big ? kChunk : kMaxRows;
        if (!big) basalt::log::warn("basalt gemma4", "no MMQ tiles for these shapes: prompts run %d rows a step", kMaxRows);
        const int R = rows_;
        const int qmax = H * g_.hd_full;
        int kvmax = 0;
        for (int l = 0; l < g_.n_layers; ++l) kvmax = std::max(kvmax, g_.kv_heads[(size_t) l] * g_.head_dim(l));
        const int widest = std::max({D, qmax, g_.n_ff});
        bool ok = true;
        x_ = (float*) alloc((size_t) R * D * 4, ok);
        xn_ = (float*) alloc((size_t) R * D * 4, ok);
        tmp_ = (float*) alloc((size_t) R * D * 4, ok);
        h1_ = (float*) alloc((size_t) R * D * 4, ok);
        h2_ = (float*) alloc((size_t) R * D * 4, ok);
        q_ = (float*) alloc((size_t) R * qmax * 4, ok);
        att_ = (float*) alloc((size_t) R * qmax * 4, ok);
        k_ = (float*) alloc((size_t) R * kvmax * 4, ok);
        v_ = (float*) alloc((size_t) R * kvmax * 4, ok);
        fg_ = (float*) alloc((size_t) R * g_.n_ff * 4, ok);
        fu_ = (float*) alloc((size_t) R * g_.n_ff * 4, ok);
        xq_ = alloc(bk::native_q8_1_bytes(widest, kMaxRows), ok);
        xq2_ = alloc(bk::native_q8_1_bytes(D, kMaxRows), ok);
        ids_ = (int32_t*) alloc((size_t) R * 16 * 4, ok);
        win_ids_ = (int32_t*) alloc((size_t) g_.n_layers * kMaxEntries * 4, ok);
        slot_d_ = (int32_t*) alloc((size_t) g_.n_layers * g_.n_expert * 4, ok);
        plans_d_ = (Plan*) alloc(sizeof(Plan) * (size_t) g_.n_layers, ok);
        rw_ = (float*) alloc((size_t) R * 16 * 4, ok);
        rlog_ = (float*) alloc((size_t) R * g_.n_expert * 4, ok);
        tok_d_ = (int32_t*) alloc((size_t) R * 4, ok);
        eo_ = (float*) alloc((size_t) R * 16 * D * 4, ok);
        if (big) {
            const int64_t P = (int64_t) R * g_.n_used;
            mxq_ = alloc(mmq::q8_bytes(R, widest), ok);
            exq_ = alloc(mmq::q8_bytes(P, D), ok);
            exq2_ = alloc(mmq::q8_bytes(P, g_.n_ff_exp), ok);
            gu_ = (float*) alloc((size_t) P * 2 * g_.n_ff_exp * 4, ok);
            he_ = (float*) alloc((size_t) P * g_.n_ff_exp * 4, ok);
            ident_ = (int32_t*) alloc((size_t) P * 4, ok);
            bounds_tab_ = (int32_t*) alloc((size_t) (R + 1) * 2 * 4, ok);
            pplan_d_ = (PromptPlan*) alloc(sizeof(PromptPlan), ok);
        }
        escratch_ = alloc(bk::native_expert_scratch_bytes(kMaxEntries, g_.n_ff_exp), ok);
        logits_ = (float*) alloc((size_t) R * g_.n_vocab * 4, ok);
        lp_ = (float*) alloc(((size_t) bk::row_logprobs_scratch_floats(R, g_.n_vocab) + (size_t) R * bk::kLogprobStride) * 4, ok);
        pick_ = (int32_t*) alloc(64 * 4, ok);
        // the caches: f16 K and V, a ring of window + a step's rows on the window layers (a step's rows are written before
        // they are read)
        ring_ = g_.window + std::max(R, 16);
        kc_.assign((size_t) g_.n_layers, nullptr);
        vc_.assign((size_t) g_.n_layers, nullptr);
        for (int l = 0; l < g_.n_layers && ok; ++l) {
            const size_t cells = g_.swa[(size_t) l] ? (size_t) ring_ : (size_t) m_.context();
            const size_t bytes = cells * (size_t) g_.kv_heads[(size_t) l] * (size_t) g_.head_dim(l) * 2;
            kc_[(size_t) l] = (uint16_t*) alloc(bytes, ok);
            vc_[(size_t) l] = (uint16_t*) alloc(bytes, ok);
            kv_bytes_ += 2 * (int64_t) bytes;
        }
        if (!ok) { cudaGetLastError(); err = "out of device memory for the session's buffers and caches"; return false; }
        if (cudaHostAlloc((void**) &ids_h_, (size_t) R * 16 * 4, cudaHostAllocDefault) != cudaSuccess ||
            (big && cudaHostAlloc((void**) &pplan_h_, sizeof(PromptPlan), cudaHostAllocDefault) != cudaSuccess) ||
            cudaHostAlloc((void**) &tok_h_, (size_t) R * 4, cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc((void**) &win_ids_h_, (size_t) g_.n_layers * kMaxEntries * 4, cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc((void**) &slot_h_, (size_t) g_.n_layers * g_.n_expert * 4, cudaHostAllocDefault) != cudaSuccess) {
            cudaGetLastError();
            err = "cannot pin the session's host buffers";
            return false;
        }
        pinned_ = {ids_h_, tok_h_, win_ids_h_, slot_h_};
        if (pplan_h_) pinned_.push_back(pplan_h_);
        if (big) {
            mmq::iota(ident_, (int64_t) R * g_.n_used, stream_);
            std::vector<int32_t> tab((size_t) (R + 1) * 2);
            for (int i = 0; i <= R; ++i) tab[(size_t) 2 * i] = 0, tab[(size_t) 2 * i + 1] = i;
            cudaMemcpy(bounds_tab_, tab.data(), tab.size() * 4, cudaMemcpyHostToDevice);
            cudaStreamSynchronize(stream_);
        }
        lay_ = bk::native_expert_layout(m_.pack().experts.front().gu_type, m_.pack().experts.front().d_type, g_.n_embd, g_.n_ff_exp);
        lay_.gelu = true;
        if (!bk::native_expert_supported(lay_.gu_type, lay_.d_type, g_.n_embd, g_.n_ff_exp) ||
            lay_.bytes != m_.pack().experts.front().blob_bytes) {
            err = "the grouped expert kernels do not run these experts (types " + std::to_string(lay_.gu_type) + "/" +
                  std::to_string(lay_.d_type) + ", " + std::to_string(g_.n_embd) + " x " + std::to_string(g_.n_ff_exp) + ")";
            return false;
        }
        return true;
    }

    // ---- the engine's calls --------------------------------------------------------------------------------------
    model::Result decode_window(const model::DecodeRequest& r) override {
        if (r.n_tokens < 1) return {model::Status::Error, "gemma4: an empty window"};
        if (r.pos + r.n_tokens > m_.context()) return {model::Status::Refused, "gemma4: the context is full"};
        std::string err;
        push_slots();
        for (int t0 = 0; t0 < r.n_tokens; t0 += kMaxRows) {
            const int n = std::min(kMaxRows, r.n_tokens - t0);
            const bool last = t0 + n == r.n_tokens;
            const bool want = r.row_tokens || (last && (r.next_token || r.logits));
            const int head = !want ? 0 : r.row_tokens ? n : 1;
            if (!run_rows(r.tokens + t0, n, r.pos + t0, head, err)) return {model::Status::Error, "gemma4: " + err};
            if (!settle_experts(n, err)) return {model::Status::Error, "gemma4: " + err};
            if (!want) continue;
            if (!pick(r, t0, head, err)) return {model::Status::Error, "gemma4: " + err};
            if (r.row_tokens)
                for (int i = 0; i < n; ++i) r.row_tokens[t0 + i] = picks_[(size_t) i];
            if (last && r.next_token) *r.next_token = picks_[(size_t) head - 1];
            if (last && r.logits)
                cudaMemcpyAsync(r.logits, logits_ + (size_t) (head - 1) * g_.n_vocab, (size_t) g_.n_vocab * 4, cudaMemcpyDeviceToHost, stream_);
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "gemma4: the window failed"};
        return {};
    }

    model::Result prefill_chunk(const model::PrefillRequest& r) override {
        if (r.pos + r.n_tokens > m_.context()) return {model::Status::Refused, "gemma4: the prompt does not fit the context"};
        std::string err;
        push_slots();
        for (int t0 = 0; t0 < r.n_tokens; t0 += rows_) {
            if (r.stop && r.stop()) break;
            const int n = std::min(rows_, r.n_tokens - t0);
            if (!run_rows(r.tokens + t0, n, r.pos + t0, 0, err)) return {model::Status::Error, "gemma4: " + err};
            if (r.progress && (rows_ > kMaxRows || (t0 / kMaxRows) % 16 == 15 || t0 + n == r.n_tokens)) {
                cudaStreamSynchronize(stream_);
                r.progress(t0 + n);
            }
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "gemma4: the prompt failed"};
        return {};
    }

    int max_window() const override { return 1; }

    // The running state: the window layers' rings (their K and V), which cannot be rewound; the whole-context layers' caches
    // stay valid below any position. A checkpoint at `pos` is intact while the session has not run more than ring_ - window
    // positions past it.
    size_t running_bytes() const override {
        size_t n = 0;
        for (int l = 0; l < g_.n_layers; ++l)
            if (g_.swa[(size_t) l]) n += 2 * ring_layer_bytes(l);
        return n;
    }
    model::Result running_state(int64_t pos, bool save, core::RunningState& st) override {
        if (save && (pos > hw_ || hw_ - pos > ring_ - g_.window))
            return {model::Status::Error, "gemma4: a checkpoint at " + std::to_string(pos) + " with the rings at " + std::to_string(hw_)};
        for (int l = 0; l < g_.n_layers; ++l)
            if (g_.swa[(size_t) l]) {
                st.regions.push_back({kc_[(size_t) l], ring_layer_bytes(l)});
                st.regions.push_back({vc_[(size_t) l], ring_layer_bytes(l)});
            }
        st.stream = stream_;
        return {};
    }
    model::Result running_restored(int64_t pos, const uint8_t*) override {
        hw_ = pos;
        return {};
    }

    model::Result set_logit_bias(const float* bias) override {
        bias_on_ = bias != nullptr;
        if (!bias_on_) return {};
        std::string e;
        if (!device_vocab(bias_, bias, e)) return {model::Status::Error, "gemma4: " + e};
        return {};
    }

    model::Result resample_row(int row, const float* bias, int32_t& token) override {
        std::string e;
        if (row < 0 || row >= last_rows_) return {model::Status::Error, "gemma4: the row is not in the last window"};
        if (!device_vocab(mask_, bias, e)) return {model::Status::Error, "gemma4: " + e};
        bk::SamplerParams sp = sampling_;
        sp.logit_bias = mask_;
        sp.counter += (uint64_t) row;
        bk::sample_tokens(logits_ + (size_t) row * g_.n_vocab, 1, g_.n_vocab, hist_len_ > 0 ? hist_d_ + (size_t) row * hist_len_ : nullptr,
                          hist_len_, sp, pick_ + 24, stream_);
        int32_t t = 0;
        if (cudaMemcpyAsync(&t, pick_ + 24, 4, cudaMemcpyDeviceToHost, stream_) != cudaSuccess || cudaStreamSynchronize(stream_) != cudaSuccess)
            return {model::Status::Error, "gemma4: the masked draw failed"};
        token = t;
        return {};
    }

    model::Result copy_logits(int row, float* out) override {
        if (row < 0 || row >= last_rows_) return {model::Status::Error, "gemma4: the session keeps the last window's logits only"};
        cudaMemcpy(out, logits_ + (size_t) row * g_.n_vocab, (size_t) g_.n_vocab * 4, cudaMemcpyDeviceToHost);
        return {};
    }

    model::Result row_logprobs(int row0, int n, const int32_t* picks, int top, float* out) override {
        if (row0 < 0 || n < 1 || row0 + n > last_rows_) return {model::Status::Error, "gemma4: the session keeps the last window's logits only"};
        float* d_out = lp_ + bk::row_logprobs_scratch_floats(n, g_.n_vocab);
        int32_t* d_pick = pick_ + 32;
        cudaMemcpyAsync(d_pick, picks, (size_t) n * 4, cudaMemcpyHostToDevice, stream_);
        bk::row_logprobs(logits_ + (size_t) row0 * g_.n_vocab, n, g_.n_vocab, d_pick, top, d_out, lp_, stream_);
        cudaMemcpyAsync(out, d_out, (size_t) n * bk::kLogprobStride * 4, cudaMemcpyDeviceToHost, stream_);
        return cudaStreamSynchronize(stream_) == cudaSuccess ? model::Result{} : model::Result{model::Status::Error, "gemma4: logprobs failed"};
    }

    ExpertCounts expert_counts() const override {
        ExpertCounts c;
        c.hits = m_.cache().hits;
        c.lookups = m_.cache().lookups;
        return c;
    }
    int64_t kv_bytes() const override { return kv_bytes_; }
    std::string info_words() const override {
        return "expert_slots=" + std::to_string((int64_t) m_.cache().slots * g_.n_layers) + " kv_mib=" + std::to_string(kv_bytes_ >> 20);
    }

private:
    void* alloc(size_t bytes, bool& ok) {
        void* p = nullptr;
        if (!ok || cudaMalloc(&p, std::max<size_t>(bytes, 16)) != cudaSuccess) {
            ok = false;
            return nullptr;
        }
        bufs_.push_back(p);
        return p;
    }

    size_t ring_layer_bytes(int l) const { return (size_t) ring_ * (size_t) g_.kv_heads[(size_t) l] * (size_t) g_.head_dim(l) * 2; }

    const float* f32(int l, const char* suffix) const {
        const DevTensor* t = m_.tensor(l, suffix);
        return t ? (const float*) t->ptr : nullptr;
    }
    const void* q8(int l, const char* suffix) const {
        const DevTensor* t = m_.tensor(l, suffix);
        return t ? t->ptr : nullptr;
    }

    // The rows of x quantized for the Q8_0 products that follow (a decode window: q8_1 for MMVQ; a prompt step: MMQ's form).
    void quant(const float* x, int cols, int n) {
        if (n > kMaxRows) mmq::quantize(x, nullptr, mxq_, kTypeQ8_0, cols, cols, n, stream_);
        else bk::native_quantize_q8_1(x, xq_, cols, n, stream_);
    }
    // y[n][n_out] = W x for the rows `quant` left
    void mm(const void* w, float* y, int n_in, int n_out, int n) {
        if (n <= kMaxRows) {
            bk::native_q8_0_mmvq(w, xq_, y, n_in, n_out, n, stream_);
            return;
        }
        if (loaded_.insert({kTypeQ8_0, n_out}).second) mmq::load(kTypeQ8_0, n_out);
        mmq::Product p;
        p.w = w;
        p.type = kTypeQ8_0;
        p.w_rows = n_out;
        p.w_cols = n_in;
        p.expert_bytes = mmq::matrix_bytes(kTypeQ8_0, n_out, n_in);
        p.n = 1;
        p.xq = mxq_;
        p.bounds = bounds_tab_ + 2 * n;
        p.ids = ident_;
        p.total_rows = n;
        p.max_rows = n;
        p.dst = y;
        p.ld_dst = n_out;
        mmq_.run(p, stream_);
    }

    // The rows' attention positions: the window layers' ring holds positions back to hw_ - ring_; a call that starts further
    // back than the ring still holds is refused.
    bool check_positions(int64_t pos, std::string& err) {
        if (pos < hw_ && pos > 0 && hw_ - pos > ring_ - g_.window) {
            err = "the conversation went back " + std::to_string(hw_ - pos) + " positions, past what the sliding-window cache keeps (" +
                  std::to_string(ring_ - g_.window) + ")";
            return false;
        }
        return true;
    }

    // The forward of `n` rows (tokens at pos0..), the logits of the last `head` rows into logits_.
    bool run_rows(const int32_t* tokens, int n, int64_t pos0, int head, std::string& err) {
        if (!check_positions(pos0, err)) return false;
        hw_dbg_ = pos0 + n - 1;
        const int D = g_.n_embd, H = g_.n_head;
        const float eps = g_.rms_eps;
        std::memcpy(tok_h_, tokens, (size_t) n * 4);
        cudaMemcpyAsync(tok_d_, tok_h_, (size_t) n * 4, cudaMemcpyHostToDevice, stream_);
        ops::embed_rows((const uint8_t*) m_.tensor("token_embd.weight")->ptr, tok_d_, n, D, std::sqrt((float) D), x_, stream_);
        const float* ff = (const float*) m_.tensor("rope_freqs.weight")->ptr;
        for (int l = 0; l < g_.n_layers; ++l) {
            const bool swa = g_.swa[(size_t) l];
            const int hd = g_.head_dim(l), kvh = g_.kv_heads[(size_t) l];
            const int qw = H * hd, kw = kvh * hd;
            // attention
            ops::rms_norm(x_, f32(l, "attn_norm.weight"), xn_, n, D, eps, stream_);
            quant(xn_, D, n);
            mm(q8(l, "attn_q.weight"), q_, D, qw, n);
            mm(q8(l, "attn_k.weight"), k_, D, kw, n);
            if (const void* wv = q8(l, "attn_v.weight")) mm(wv, v_, D, kw, n);
            else ops::copy(v_, k_, (int64_t) n * kw, stream_);   // K = V: the value is the key's projection
            const float base = swa ? g_.rope_base_swa : g_.rope_base;
            ops::head_norm_rope(q_, f32(l, "attn_q_norm.weight"), n, H, hd, qw, pos0, base, swa ? nullptr : ff, eps, stream_);
            ops::head_norm_rope(v_, nullptr, n, kvh, hd, kw, pos0, 0.f, nullptr, eps, stream_);
            ops::head_norm_rope(k_, f32(l, "attn_k_norm.weight"), n, kvh, hd, kw, pos0, base, swa ? nullptr : ff, eps, stream_);
            ops::kv_write(k_, v_, n, kvh, hd, pos0, swa ? ring_ : 0, kc_[(size_t) l], vc_[(size_t) l], stream_);
            ops::attention(q_, kc_[(size_t) l], vc_[(size_t) l], n, H, kvh, hd, pos0, swa ? g_.window : 0, swa ? ring_ : 0, att_, stream_);
            quant(att_, qw, n);
            mm(q8(l, "attn_output.weight"), tmp_, qw, D, n);
            ops::norm_add(x_, tmp_, f32(l, "post_attention_norm.weight"), n, D, eps, nullptr, stream_);
            // the dense MLP
            ops::rms_norm(x_, f32(l, "ffn_norm.weight"), xn_, n, D, eps, stream_);
            quant(xn_, D, n);
            mm(q8(l, "ffn_gate.weight"), fg_, D, g_.n_ff, n);
            mm(q8(l, "ffn_up.weight"), fu_, D, g_.n_ff, n);
            ops::geglu(fg_, fu_, fg_, (int64_t) n * g_.n_ff, stream_);
            quant(fg_, g_.n_ff, n);
            mm(q8(l, "ffn_down.weight"), h1_, g_.n_ff, D, n);
            // the routed experts
            ops::router_topk(x_, f32(l, "ffn_gate_inp.scale"), f32(l, "ffn_gate_inp.weight"), n, D, g_.n_expert, g_.n_used, eps,
                             ids_, rw_, rlog_, stream_);
            ops::rms_norm(x_, f32(l, "pre_ffw_norm_2.weight"), xn_, n, D, eps, stream_);
            if (n > kMaxRows) {
                if (!prompt_experts(l, n, err)) return false;
            } else {
                bk::native_quantize_q8_1(xn_, xq2_, D, n, stream_);
                experts(l, n);
            }
            ops::moe_combine(eo_, rw_, n, g_.n_used, D, h2_, stream_);
            ops::norm2_sum(h1_, f32(l, "post_ffw_norm_1.weight"), h2_, f32(l, "post_ffw_norm_2.weight"), tmp_, n, D, eps, stream_);
            ops::norm_add(x_, tmp_, f32(l, "post_ffw_norm.weight"), n, D, eps, f32(l, "layer_output_scale.weight"), stream_);
            if (!m_.flags().diag_dump.empty()) dump_row(l, pos0 + n - 1, x_ + (size_t) (n - 1) * D);
        }
        hw_ = pos0 + n;
        if (head > kMaxRows) {
            err = "a prompt step's logits were asked for";
            return false;
        }
        if (head > 0) {
            const float* xl = x_ + (size_t) (n - head) * D;
            ops::rms_norm(xl, (const float*) m_.tensor("output_norm.weight")->ptr, xn_, head, D, eps, stream_);
            bk::native_quantize_q8_1(xn_, xq_, D, head, stream_);
            bk::native_q8_0_mmvq(m_.tensor("token_embd.weight")->ptr, xq_, logits_, D, g_.n_vocab, head, stream_);
            if (g_.softcap > 0.f) ops::softcap(logits_, (int64_t) head * g_.n_vocab, g_.softcap, stream_);
            last_rows_ = head;
        }
        if (cudaGetLastError() != cudaSuccess) {
            err = "a kernel launch failed";
            return false;
        }
        return true;
    }

    // Layer l's routed experts of a decode window's n rows: the plan built on the device from the picks and the slot table (a
    // pick outside the VRAM cache is read from pinned host memory over the bus), the grouped kernel into eo_ (row r's j-th
    // pick at eo_[r*k + j]). The picks are kept for settle_experts, which brings the misses into the cache after the window.
    void experts(int l, int n) {
        const int k = g_.n_used;
        const ExpertCache& c = m_.cache();
        Plan* pl = plans_d_ + l;
        cudaMemcpyAsync(win_ids_ + (size_t) l * kMaxEntries, ids_, (size_t) n * k * 4, cudaMemcpyDeviceToDevice, stream_);
        ops::expert_plan(ids_, n, k, slot_d_ + (size_t) l * g_.n_expert, c.slot_ptr(l, 0), c.stride, m_.host_experts_dev(l), c.blob,
                         pl->ptr, pl->start, pl->ng, pl->dst, pl->tok, stream_);
        bk::native_expert_grouped(lay_, pl->ptr, pl->start, pl->ng, pl->dst, pl->tok, kMaxEntries, kMaxEntries, xq2_, escratch_, eo_,
                                  stream_, (int64_t) n * k);
    }

    // After a decode window's rows: every layer's picks to the host (one copy, one sync), the cache's hits stamped and its
    // misses copied into the least recently used slots, the slot table updated on the device for the next window.
    bool settle_experts(int n, std::string& err) {
        const int k = g_.n_used, L = g_.n_layers;
        cudaMemcpyAsync(win_ids_h_, win_ids_, (size_t) L * kMaxEntries * 4, cudaMemcpyDeviceToHost, stream_);
        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            err = "the window's experts failed";
            return false;
        }
        ExpertCache& c = m_.cache();
        const uint64_t now = ++c.tick;
        bool moved = false;
        for (int l = 0; l < L; ++l) {
            std::vector<int>& slot_of = c.slot_of[(size_t) l];
            std::vector<int>& holds = c.holds[(size_t) l];
            std::vector<uint64_t>& used = c.used[(size_t) l];
            const int32_t* ids = win_ids_h_ + (size_t) l * kMaxEntries;
            int uniq[kMaxEntries], nu = 0;
            for (int i = 0; i < n * k; ++i) {
                const int e = ids[i];
                if (e < 0 || e >= g_.n_expert) {
                    err = "the router picked expert " + std::to_string(e);
                    return false;
                }
                int g = 0;
                while (g < nu && uniq[g] != e) ++g;
                if (g == nu) uniq[nu++] = e;
            }
            c.lookups += nu;
            for (int g = 0; g < nu; ++g)
                if (slot_of[(size_t) uniq[g]] >= 0) {
                    used[(size_t) slot_of[(size_t) uniq[g]]] = now;
                    ++c.hits;
                }
            for (int g = 0; g < nu; ++g) {
                const int e = uniq[g];
                if (slot_of[(size_t) e] >= 0) continue;
                int victim = 0;
                for (int s = 1; s < c.slots; ++s)
                    if (used[(size_t) s] < used[(size_t) victim]) victim = s;
                if (used[(size_t) victim] == now) break;   // the window's own experts fill the cache: the rest stay on the host
                if (holds[(size_t) victim] >= 0) slot_of[(size_t) holds[(size_t) victim]] = -1;
                holds[(size_t) victim] = e;
                slot_of[(size_t) e] = victim;
                used[(size_t) victim] = now;
                cudaMemcpyAsync(c.slot_ptr(l, victim), m_.host_expert(l, e), c.blob, cudaMemcpyHostToDevice, stream_);
                moved = true;
            }
        }
        if (moved) slots_dirty_ = true;
        push_slots();
        return true;
    }

    // The cache's slot table to the device when the host changed it (the next kernels on the stream read it).
    void push_slots() {
        if (!slots_dirty_) return;
        const ExpertCache& c = m_.cache();
        for (int l = 0; l < g_.n_layers; ++l)
            std::memcpy(slot_h_ + (size_t) l * g_.n_expert, c.slot_of[(size_t) l].data(), (size_t) g_.n_expert * 4);
        cudaMemcpyAsync(slot_d_, slot_h_, (size_t) g_.n_layers * g_.n_expert * 4, cudaMemcpyHostToDevice, stream_);
        // the pinned table is rewritten only after the stream's next sync (the end of a window or a prompt step)
        slots_dirty_ = false;
    }

    // A prompt step's routed experts (n rows, n > kMaxRows): the picks to the host, the experts in passes of at most the
    // cache's slots (a miss evicts the least recently used slot outside the pass, copied in over the bus), each pass one MMQ
    // gate/up product over the layer's slots, the GeGLU with the blobs' tail scales, one down product into eo_ at the pairs'
    // rows (row r's j-th pick at eo_[r*k + j]).
    bool prompt_experts(int l, int n, std::string& err) {
        const int k = g_.n_used, D = g_.n_embd, F = g_.n_ff_exp, P = n * k;
        cudaMemcpyAsync(ids_h_, ids_, (size_t) P * 4, cudaMemcpyDeviceToHost, stream_);
        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            err = "the router failed";
            return false;
        }
        ExpertCache& c = m_.cache();
        std::vector<int>& slot_of = c.slot_of[(size_t) l];
        std::vector<int>& holds = c.holds[(size_t) l];
        std::vector<uint64_t>& used = c.used[(size_t) l];
        std::vector<int> count((size_t) g_.n_expert, 0);
        for (int i = 0; i < P; ++i) {
            const int e = ids_h_[i];
            if (e < 0 || e >= g_.n_expert) {
                err = "the router picked expert " + std::to_string(e);
                return false;
            }
            ++count[(size_t) e];
        }
        // the experts this step uses, the cached ones first (they need no copy)
        std::vector<int> want;
        for (int e = 0; e < g_.n_expert; ++e)
            if (count[(size_t) e] && slot_of[(size_t) e] >= 0) want.push_back(e);
        const size_t cached = want.size();
        for (int e = 0; e < g_.n_expert; ++e)
            if (count[(size_t) e] && slot_of[(size_t) e] < 0) want.push_back(e);
        c.lookups += (int64_t) want.size();
        c.hits += (int64_t) cached;
        const size_t GUB = mmq::matrix_bytes(kTypeNvfp4, F, D);   // the gate rows; the up rows follow, then down
        std::vector<int> pass_of((size_t) g_.n_expert, -1);
        if (loaded_.insert({kTypeNvfp4, 2 * F}).second) mmq::load(kTypeNvfp4, 2 * F);
        if (loaded_.insert({kTypeNvfp4, D}).second) mmq::load(kTypeNvfp4, D);
        for (size_t at = 0; at < want.size();) {
            const size_t end = std::min(want.size(), at + (size_t) c.slots);
            const uint64_t now = ++c.tick;
            for (size_t i = at; i < end; ++i)
                if (slot_of[(size_t) want[i]] >= 0) used[(size_t) slot_of[(size_t) want[i]]] = now;
            for (size_t i = at; i < end; ++i) {
                const int e = want[i];
                if (slot_of[(size_t) e] >= 0) continue;
                int victim = 0;
                for (int s = 1; s < c.slots; ++s)
                    if (used[(size_t) s] < used[(size_t) victim]) victim = s;
                if (used[(size_t) victim] == now) {
                    err = "the expert cache ran out of slots in a prompt pass";
                    return false;
                }
                if (holds[(size_t) victim] >= 0) slot_of[(size_t) holds[(size_t) victim]] = -1;
                holds[(size_t) victim] = e;
                slot_of[(size_t) e] = victim;
                used[(size_t) victim] = now;
                cudaMemcpyAsync(c.slot_ptr(l, victim), m_.host_expert(l, e), c.blob, cudaMemcpyHostToDevice, stream_);
            }
            // the pass's pairs sorted by slot
            PromptPlan& pp = *pplan_h_;
            std::vector<int> per((size_t) c.slots + 1, 0);
            for (size_t i = at; i < end; ++i) per[(size_t) slot_of[(size_t) want[i]] + 1] = count[(size_t) want[i]];
            for (int s = 0; s < c.slots; ++s) per[(size_t) s + 1] += per[(size_t) s];
            int max_rows = 0;
            for (int s = 0; s <= c.slots; ++s) {
                pp.bounds[s] = per[(size_t) s];
                if (s > 0) max_rows = std::max(max_rows, per[(size_t) s] - per[(size_t) s - 1]);
            }
            const int rows = per[(size_t) c.slots];
            std::vector<int> fill(per.begin(), per.end() - 1);
            for (size_t i = at; i < end; ++i) pass_of[(size_t) want[i]] = (int) at;
            for (int i = 0; i < P; ++i) {
                const int e = ids_h_[i];
                if (pass_of[(size_t) e] != (int) at) continue;
                const int s = slot_of[(size_t) e], q = fill[(size_t) s]++;
                pp.tok[q] = i / k;
                pp.pair[q] = i;
                pp.slot[q] = s;
            }
            cudaMemcpyAsync(pplan_d_, pplan_h_, sizeof(PromptPlan), cudaMemcpyHostToDevice, stream_);
            const uint8_t* base = c.slot_ptr(l, 0);
            mmq::quantize(xn_, pplan_d_->tok, exq_, kTypeNvfp4, D, D, rows, stream_);
            mmq::Product gu;
            gu.w = base;
            gu.type = kTypeNvfp4;
            gu.w_rows = 2 * (int64_t) F;
            gu.w_cols = D;
            gu.expert_bytes = c.stride;
            gu.n = c.slots;
            gu.xq = exq_;
            gu.bounds = pplan_d_->bounds;
            gu.ids = ident_;
            gu.total_rows = rows;
            gu.max_rows = max_rows;
            gu.dst = gu_;
            gu.ld_dst = 2 * (int64_t) F;
            mmq_.run(gu, stream_);
            mmq::SwigluForm form;
            form.gelu = true;
            form.row_expert = pplan_d_->slot;
            form.tail_stride = c.stride;
            mmq::swiglu(gu_, he_, rows, F, false, stream_, (const float*) (base + c.blob - 16), pplan_d_->bounds, c.slots, form);
            mmq::quantize(he_, nullptr, exq2_, kTypeNvfp4, F, F, rows, stream_);
            mmq::Product dn = gu;
            dn.w = base + 2 * GUB;
            dn.w_rows = D;
            dn.w_cols = F;
            dn.xq = exq2_;
            dn.ids = pplan_d_->pair;
            dn.dst = eo_;
            dn.ld_dst = D;
            mmq_.run(dn, stream_);
            at = end;
            slots_dirty_ = true;
            if (at < want.size() && cudaStreamSynchronize(stream_) != cudaSuccess) {   // the plan buffer is reused
                err = "a prompt expert pass failed";
                return false;
            }
        }
        return true;
    }

    // diagnostic (--diag-dump DIR): a call's last row after layer l, appended to DIR/gemma4_rows.bin as {int64 pos, int32
    // layer, int32 width, width floats}
    void dump_row(int l, int64_t pos, const float* row) {
        std::vector<float> h((size_t) g_.n_embd);
        cudaMemcpyAsync(h.data(), row, h.size() * 4, cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        const std::string path = m_.flags().diag_dump + "/gemma4_rows.bin";
        if (std::FILE* f = std::fopen(path.c_str(), "ab")) {
            const int32_t meta[2] = {l, g_.n_embd};
            std::fwrite(&pos, 8, 1, f);
            std::fwrite(meta, 4, 2, f);
            std::fwrite(h.data(), 4, h.size(), f);
            std::fclose(f);
        }
    }

    bool device_vocab(float*& dev, const float* host, std::string& err) {
        const size_t bytes = (size_t) g_.n_vocab * sizeof(float);
        if (!dev && cudaMalloc((void**) &dev, bytes) != cudaSuccess) {
            cudaGetLastError();
            dev = nullptr;
            err = "out of device memory for a logit bias";
            return false;
        }
        if (cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, stream_) != cudaSuccess) { err = "the logit bias upload failed"; return false; }
        return true;
    }

    // The picks of the last `rows` rows of logits_ (row t of the window is draw counter + t0 + t).
    bool pick(const model::DecodeRequest& r, int t0, int rows, std::string& err) {
        bk::SamplerParams sp;
        if (r.sampling) sp = *r.sampling;
        else {
            sp.greedy = true;
            sp.temperature = 0.f;
        }
        sp.counter += (uint64_t) t0;
        sp.logit_bias = bias_on_ ? bias_ : nullptr;
        hist_len_ = r.history != nullptr ? r.history_len : 0;
        if (hist_len_ > 0) {
            const size_t need = (size_t) kMaxRows * hist_len_;
            if (hist_cap_ < need) {
                if (hist_d_) cudaFree(hist_d_);
                hist_d_ = nullptr;
                if (cudaMalloc((void**) &hist_d_, need * 4) != cudaSuccess) { cudaGetLastError(); err = "out of device memory for the sampling history"; return false; }
                hist_cap_ = need;
            }
            cudaMemcpyAsync(hist_d_, r.history + (size_t) t0 * hist_len_, (size_t) rows * hist_len_ * 4, cudaMemcpyHostToDevice, stream_);
        }
        sampling_ = sp;
        bk::sample_tokens(logits_, rows, g_.n_vocab, hist_len_ > 0 ? hist_d_ : nullptr, hist_len_, sp, pick_, stream_);
        picks_.assign((size_t) rows, 0);
        if (cudaMemcpyAsync(picks_.data(), pick_, (size_t) rows * 4, cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess) {
            err = "the pick failed";
            return false;
        }
        return true;
    }

    GemmaModel& m_;
    const Geometry& g_;
    cudaStream_t stream_ = nullptr;
    std::vector<void*> bufs_, pinned_;
    float *x_ = nullptr, *xn_ = nullptr, *tmp_ = nullptr, *h1_ = nullptr, *h2_ = nullptr, *q_ = nullptr, *att_ = nullptr, *k_ = nullptr,
          *v_ = nullptr, *fg_ = nullptr, *fu_ = nullptr, *rw_ = nullptr, *eo_ = nullptr, *logits_ = nullptr, *lp_ = nullptr;
    void *xq_ = nullptr, *xq2_ = nullptr, *escratch_ = nullptr;
    int32_t *ids_ = nullptr, *tok_d_ = nullptr, *pick_ = nullptr;
    int32_t *ids_h_ = nullptr, *tok_h_ = nullptr;
    std::vector<uint16_t*> kc_, vc_;
    int ring_ = 0, rows_ = kMaxRows;
    float* rlog_ = nullptr;   // the router's logits
    // the decode windows' experts: every layer's picks, the slot table on the device and its pinned staging, the plans
    int32_t *win_ids_ = nullptr, *win_ids_h_ = nullptr, *slot_d_ = nullptr, *slot_h_ = nullptr;
    Plan* plans_d_ = nullptr;
    bool slots_dirty_ = true;
    // the prompt path
    void *mxq_ = nullptr, *exq_ = nullptr, *exq2_ = nullptr;
    float *gu_ = nullptr, *he_ = nullptr;
    int32_t *ident_ = nullptr, *bounds_tab_ = nullptr;
    PromptPlan* pplan_d_ = nullptr;
    PromptPlan* pplan_h_ = nullptr;
    mmq::Context mmq_;
    std::set<std::pair<int, int64_t>> loaded_;
    int64_t hw_ = 0, kv_bytes_ = 0, hw_dbg_ = 0;
    bk::NativeExpertLayout lay_;
    int last_rows_ = 0;
    std::vector<int32_t> picks_;
    bk::SamplerParams sampling_;
    float *bias_ = nullptr, *mask_ = nullptr;
    bool bias_on_ = false;
    int32_t* hist_d_ = nullptr;
    size_t hist_cap_ = 0;
    int hist_len_ = 0;
};

}  // namespace

std::unique_ptr<model::ModelSession> make_session(GemmaModel& model, std::string& err) {
    auto s = std::make_unique<Session>(model);
    if (!s->open(err)) return nullptr;
    return s;
}

}  // namespace basalt::models::gemma4
