// src/models/gemma4/pack.hpp - the Gemma 4 .basalt pack as the loader sees it: geometry, the tensor table and the experts
// layout, from the pack's meta section (tools/convert/arch/gemma4.py and gemma4_bf16.py write it).
#pragma once

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/model/adapter.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace basalt::models::gemma4 {

inline constexpr const char* kArch = "gemma4";

struct Geometry {
    int n_layers = 0, n_embd = 0, n_vocab = 0, n_ff = 0, n_head = 0;
    int n_expert = 0, n_used = 0, n_ff_exp = 0;
    int hd_full = 0, hd_swa = 0, window = 0;
    float rope_base = 1e6f, rope_base_swa = 1e4f, rms_eps = 1e-6f, softcap = 0.f;
    int64_t context_length = 0;
    std::vector<int> kv_heads;   ///< per layer
    std::vector<bool> swa;       ///< per layer: a sliding-window layer
    int head_dim(int l) const { return swa[(size_t) l] ? hd_swa : hd_full; }
};

struct TensorRef {
    int type = 0;
    std::vector<uint64_t> shape;   ///< GGUF order, innermost first
    uint64_t offset = 0;           ///< absolute file offset
    uint64_t bytes = 0;
};

struct ExpertLayer {
    int gu_type = 0, d_type = 0;
    uint64_t offset = 0, blob_bytes = 0;   ///< offset inside the experts section
};

struct Pack {
    core::ModelFile file;
    Geometry geo;
    std::map<std::string, TensorRef> tensors;
    std::vector<ExpertLayer> experts;   ///< per layer
    core::FileRange experts_range;

    const TensorRef* find(const std::string& n) const {
        auto it = tensors.find(n);
        return it == tensors.end() ? nullptr : &it->second;
    }
};

/// Opens the pack and fills `p`. False with `err` when it is not a Gemma 4 pack, is malformed or has a layout this engine
/// does not run.
bool open_pack(const std::string& path, Pack& p, std::string& err);

/// The file's metadata (GGUF keys `gemma4.*`): refuses another model, describes this one.
std::string validate_meta(const GgufFile& g);
model::ModelDesc describe_meta(const GgufFile& g);

}  // namespace basalt::models::gemma4
