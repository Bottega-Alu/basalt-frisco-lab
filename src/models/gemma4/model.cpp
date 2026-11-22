// src/models/gemma4/model.cpp - loads a Gemma 4 pack: the dense tensors to the main card, the experts to pinned host memory
// and as many of each layer's as fit into the VRAM cache.
#include "models/gemma4/model.hpp"

#include "basalt/core/loader.hpp"
#include "basalt/log.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/platform/file.hpp"

#include <cuda_runtime.h>

#include <algorithm>

namespace basalt::models::gemma4 {
namespace {
constexpr uint64_t kMiB = 1ull << 20;
constexpr uint64_t kGiB = 1ull << 30;
constexpr int kTypeQ8_0 = 8, kTypeF32 = 0, kTypeNvfp4 = 40;
// VRAM the cache leaves free beyond --vram-reserve-mib: kernel modules, the driver's own, the sampler's and the experts'
// scratch
constexpr uint64_t kCacheMargin = 1280 * kMiB;
}  // namespace

GemmaModel::GemmaModel() = default;

GemmaModel::~GemmaModel() {
    for (auto& [name, t] : dev_)
        if (t.ptr) cudaFree(t.ptr);
    for (uint8_t* p : host_exp_)
        if (p) cudaFreeHost(p);
    if (cache_.arena) cudaFree(cache_.arena);
    if (fd_ >= 0) platform::close_fd(fd_);
}

const DevTensor* GemmaModel::tensor(const std::string& name) const {
    auto it = dev_.find(name);
    return it == dev_.end() ? nullptr : &it->second;
}

model::Result GemmaModel::plan(const model::LoadRequest& req, core::LoadPlan& reads,
                               std::vector<std::unique_ptr<model::ModelSession>>& sessions) {
    if (req.paths.empty()) return {model::Status::Error, "gemma4: no model path"};
    if (req.sessions > 1) return {model::Status::Error, "gemma4: one session only (--batch is not supported yet)"};
    if (req.spec > 1) basalt::log::warn("basalt gemma4", "--spec %d: this model has no drafter, it runs plain decode", req.spec);
    std::string err;
    const std::string& path = req.paths.front();
    if (!open_pack(path, pack_, err)) return {model::Status::Error, "gemma4: " + err};
    cudaGetDevice(&device_);
    flags_ = req;
    const Geometry& g = pack_.geo;
    desc_.arch = kArch;
    desc_.n_layers = g.n_layers;
    desc_.n_embd = g.n_embd;
    desc_.n_vocab = g.n_vocab;
    desc_.n_expert = g.n_expert;
    desc_.n_expert_used = g.n_used;
    desc_.n_ff_exp = g.n_ff_exp;
    desc_.context_length = g.context_length;
    for (const ExpertLayer& e : pack_.experts) {
        if (e.gu_type != kTypeNvfp4 || e.d_type != kTypeNvfp4)
            return {model::Status::Error, "gemma4: the experts are not NVFP4 (this engine runs the converter's NVFP4 packs)"};
        desc_.expert_gate_up_type.push_back(e.gu_type);
        desc_.expert_down_type.push_back(e.d_type);
    }
    ctx_ = req.max_context > 0 ? req.max_context : kDefaultContext;
    if (g.context_length > 0) ctx_ = std::min<int64_t>(ctx_, g.context_length);

    fd_ = platform::open_fd_utf8(path);
    if (fd_ < 0) return {model::Status::Error, "gemma4: cannot open " + path};
    const int file = reads.file_fd(fd_);

    // the dense tensors: Q8_0 matrices and F32 vectors, as the pack holds them
    uint64_t dense = 0;
    for (const auto& [name, t] : pack_.tensors) {
        if (t.type != kTypeQ8_0 && t.type != kTypeF32)
            return {model::Status::Error, "gemma4: tensor " + name + " is neither Q8_0 nor F32"};
        DevTensor d;
        d.type = t.type;
        d.bytes = t.bytes;
        d.shape = t.shape;
        if (cudaMalloc(&d.ptr, d.bytes) != cudaSuccess) {
            cudaGetLastError();
            return {model::Status::Error, "gemma4: out of device memory loading " + name};
        }
        core::LoadItem it;
        it.file = file;
        it.off = t.offset;
        it.bytes = t.bytes;
        it.dev = d.ptr;
        it.device = device_;
        reads.add(it);
        dense += d.bytes;
        dev_.emplace(name, std::move(d));
    }
    for (const char* n : {"token_embd.weight", "output_norm.weight", "rope_freqs.weight"})
        if (!tensor(n)) return {model::Status::Error, std::string("gemma4: the pack has no ") + n};
    if (tensor("token_embd.weight")->type != kTypeQ8_0) return {model::Status::Error, "gemma4: token_embd.weight is not Q8_0"};
    basalt::log::info("basalt gemma4", "%zu dense tensors, %.2f GiB on the GPU", dev_.size(), (double) dense / (double) kGiB);

    // every expert in pinned host memory, a layer per allocation
    uint64_t host = 0;
    host_exp_.assign((size_t) g.n_layers, nullptr);
    host_dev_.assign((size_t) g.n_layers, nullptr);
    for (int l = 0; l < g.n_layers; ++l) {
        const ExpertLayer& e = pack_.experts[(size_t) l];
        const uint64_t bytes = e.blob_bytes * (uint64_t) g.n_expert;
        if (cudaHostAlloc((void**) &host_exp_[(size_t) l], bytes, cudaHostAllocMapped) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &host_dev_[(size_t) l], host_exp_[(size_t) l], 0) != cudaSuccess) {
            cudaGetLastError();
            return {model::Status::Error, "gemma4: cannot pin " + std::to_string(bytes / kMiB) + " MiB for layer " +
                                              std::to_string(l) + "'s experts"};
        }
        core::LoadItem it;
        it.file = file;
        it.off = pack_.experts_range.offset + e.offset;
        it.bytes = bytes;
        it.host = host_exp_[(size_t) l];
        reads.add(it);
        host += bytes;
    }
    basalt::log::info("basalt gemma4", "experts: %.2f GiB in pinned host memory", (double) host / (double) kGiB);

    // the session (its caches and buffers) before the expert cache, which takes what they leave
    std::unique_ptr<model::ModelSession> s = make_session(*this, err);
    if (!s) return {model::Status::Error, "gemma4: " + err};
    sessions.push_back(std::move(s));

    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    const uint64_t blob = pack_.experts.front().blob_bytes;
    for (const ExpertLayer& e : pack_.experts)
        if (e.blob_bytes != blob) return {model::Status::Error, "gemma4: the layers' expert blobs differ in size"};
    const uint64_t reserve = kCacheMargin + (uint64_t) std::max(req.vram_reserve_mib, 0) * kMiB;
    const uint64_t stride = (blob + 143) / 144 * 144;   // lcm(36, 16)
    int slots = free_b > reserve ? (int) ((free_b - reserve) / (stride * (uint64_t) g.n_layers)) : 0;
    if (req.diag_tier_vram >= 0) slots = std::min(slots, req.diag_tier_vram);
    if (req.expert_cache > 0) slots = std::min(slots, req.expert_cache / g.n_layers);
    slots = std::min(slots, g.n_expert);
    const int need = kMaxRows * g.n_used;   // a window's distinct experts of one layer
    if (slots < std::min(need, g.n_expert))
        return {model::Status::Error, "gemma4: " + std::to_string(slots) + " VRAM expert slots per layer fit, a window needs " +
                                          std::to_string(std::min(need, g.n_expert)) + " (free " + std::to_string(free_b / kMiB) +
                                          " MiB after the dense tensors and the caches)"};
    ExpertCache& c = cache_;
    c.slots = slots;
    c.blob = blob;
    c.stride = stride;
    // the prompt path's MMQ reads the last slot's down rows (n_ff columns) on to a multiple of 256 values: zeros there
    const size_t past = std::max(prefill::mmq::read_past_bytes(kTypeNvfp4, g.n_embd), prefill::mmq::read_past_bytes(kTypeNvfp4, g.n_ff_exp));
    const size_t arena = (size_t) slots * (size_t) g.n_layers * stride;
    if (cudaMalloc((void**) &c.arena, arena + past + 16) != cudaSuccess) {
        cudaGetLastError();
        return {model::Status::Error, "gemma4: out of device memory for the expert cache"};
    }
    cudaMemset(c.arena + arena, 0, past + 16);
    c.slot_of.assign((size_t) g.n_layers, std::vector<int>((size_t) g.n_expert, -1));
    c.holds.assign((size_t) g.n_layers, std::vector<int>((size_t) slots, -1));
    c.used.assign((size_t) g.n_layers, std::vector<uint64_t>((size_t) slots, 0));
    basalt::log::info("basalt gemma4", "expert cache: %d of %d experts per layer on the GPU (%.2f GiB), the rest come over the bus",
                      slots, g.n_expert, (double) slots * (double) g.n_layers * (double) stride / (double) kGiB);
    return {};
}

model::Result GemmaModel::attach(const model::LoadRequest&, std::vector<std::unique_ptr<model::ModelSession>>&) {
    // the cache starts with each layer's first experts
    cudaSetDevice(device_);
    ExpertCache& c = cache_;
    for (int l = 0; l < geo().n_layers; ++l)
        for (int s = 0; s < c.slots; ++s) {
            if (cudaMemcpy(c.slot_ptr(l, s), host_expert(l, s), c.blob, cudaMemcpyHostToDevice) != cudaSuccess)
                return {model::Status::Error, "gemma4: the expert cache's first copy failed"};
            c.holds[(size_t) l][(size_t) s] = s;
            c.slot_of[(size_t) l][(size_t) s] = s;
        }
    return {};
}

}  // namespace basalt::models::gemma4
