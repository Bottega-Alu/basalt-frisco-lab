// src/models/gemma4/model.hpp - a loaded Gemma 4 pack: the dense tensors (the Q8_0 embedding, which is also the output
// head, among them) on the main card, every routed expert in pinned host memory, and a VRAM cache of each layer's experts
// that a miss fills over the bus. Every expert runs on the GPU.
#pragma once

#include "basalt/model/adapter.hpp"
#include "models/gemma4/pack.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::gemma4 {

struct DevTensor {
    void* ptr = nullptr;
    int type = 0;
    uint64_t bytes = 0;
    std::vector<uint64_t> shape;
};

constexpr int64_t kDefaultContext = 8192;
constexpr int kMaxRows = 8;   ///< rows of one forward call: a decode window, or a step of a prompt chunk

/// The VRAM copies of each layer's experts: `slots` per layer, least recently used gives way. One per model.
struct ExpertCache {
    int slots = 0;
    uint64_t blob = 0;                        ///< an expert's bytes (what a miss copies)
    uint64_t stride = 0;                      ///< a slot's bytes: whole NVFP4 blocks (36) and 16 bytes, so MMQ steps from
                                              ///< slot to slot in whole blocks
    uint8_t* arena = nullptr;                 ///< [layer][slot] blobs
    std::vector<std::vector<int>> slot_of;    ///< [layer][expert] -> slot or -1
    std::vector<std::vector<int>> holds;      ///< [layer][slot] -> expert or -1
    std::vector<std::vector<uint64_t>> used;  ///< [layer][slot] -> the tick of its last use
    uint64_t tick = 0;
    int64_t hits = 0, lookups = 0;
    uint8_t* slot_ptr(int layer, int slot) const { return arena + ((size_t) layer * (size_t) slots + (size_t) slot) * stride; }
};

class GemmaModel final : public model::LoadedModel {
public:
    GemmaModel();
    ~GemmaModel() override;
    model::Result plan(const model::LoadRequest& req, core::LoadPlan& reads,
                       std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;
    model::Result attach(const model::LoadRequest& req, std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;
    const model::ModelDesc& desc() const override { return desc_; }
    int64_t context() const override { return ctx_; }

    const Pack& pack() const { return pack_; }
    const Geometry& geo() const { return pack_.geo; }
    /// Device tensor `blk.<layer>.<suffix>` or a global name; null when absent.
    const DevTensor* tensor(const std::string& name) const;
    const DevTensor* tensor(int layer, const char* suffix) const { return tensor("blk." + std::to_string(layer) + "." + suffix); }
    /// Layer `l`'s expert `e` in pinned host memory.
    const uint8_t* host_expert(int l, int e) const { return host_exp_[(size_t) l] + (size_t) e * pack_.experts[(size_t) l].blob_bytes; }
    /// Layer `l`'s experts in pinned host memory as the device addresses them (mapped: a kernel reads them over the bus).
    const uint8_t* host_experts_dev(int l) const { return host_dev_[(size_t) l]; }
    ExpertCache& cache() const { return cache_; }
    int device() const { return device_; }
    const model::LoadRequest& flags() const { return flags_; }

private:
    Pack pack_;
    model::ModelDesc desc_;
    std::map<std::string, DevTensor> dev_;
    std::vector<uint8_t*> host_exp_, host_dev_;
    mutable ExpertCache cache_;
    model::LoadRequest flags_;
    int device_ = 0, fd_ = -1;
    int64_t ctx_ = kDefaultContext;
};

/// A session over `model`; null with `err` set.
std::unique_ptr<model::ModelSession> make_session(GemmaModel& model, std::string& err);

}  // namespace basalt::models::gemma4
