// src/models/gemma4/pack.cpp - reads the Gemma 4 pack's meta section.
#include "models/gemma4/pack.hpp"

#include <sstream>

namespace basalt::models::gemma4 {
namespace {

using core::Json;

double num_or(const Json* j, double d) { return j && j->is_number() ? j->num() : d; }

uint64_t key_u(const GgufFile& g, const char* suffix) {
    const MetaValue* v = g.get(std::string(kArch) + "." + suffix);
    return v ? v->u : 0;
}

}  // namespace

std::string validate_meta(const GgufFile& g) {
    const MetaValue* arch = g.get("general.architecture");
    if (!arch) return "missing general.architecture";
    if (arch->s != kArch) return "architecture is '" + arch->s + "', this adapter requires '" + kArch + "'";
    for (const char* k : {"block_count", "embedding_length", "expert_count", "expert_used_count", "expert_feed_forward_length"})
        if (key_u(g, k) == 0) return std::string("missing or zero ") + kArch + "." + k + " (Gemma 4 without experts is not run)";
    return {};
}

model::ModelDesc describe_meta(const GgufFile& g) {
    model::ModelDesc d;
    d.arch = kArch;
    d.n_layers = (int64_t) key_u(g, "block_count");
    d.n_embd = (int64_t) key_u(g, "embedding_length");
    d.n_expert = (int64_t) key_u(g, "expert_count");
    d.n_expert_used = (int64_t) key_u(g, "expert_used_count");
    d.n_ff_exp = (int64_t) key_u(g, "expert_feed_forward_length");
    d.context_length = (int64_t) key_u(g, "context_length");
    if (const MetaValue* t = g.get("tokenizer.ggml.tokens")) d.n_vocab = (int64_t) t->count;
    return d;
}

bool open_pack(const std::string& path, Pack& p, std::string& err) {
    if (!p.file.open(path, err) || !p.file.load_meta(err)) return false;
    const Json& meta = p.file.meta();
    const Json* arch = meta.get("arch");
    if (!arch || !arch->is_string() || arch->s != kArch) {
        err = path + ": not a gemma4 pack";
        return false;
    }
    const Json* gj = meta.get("geometry");
    const Json* tj = meta.get("tensors");
    if (!gj || !gj->is_object() || !tj || !tj->is_object()) {
        err = path + ": meta has no geometry / tensors";
        return false;
    }
    Geometry& g = p.geo;
    auto u = [&](const char* k, double d = 0) { return (int) num_or(gj->get(k), d); };
    g.n_layers = u("block_count");
    g.n_embd = u("embedding_length");
    g.n_ff = u("feed_forward_length");
    g.n_head = u("attention.head_count");
    g.n_expert = u("expert_count");
    g.n_used = u("expert_used_count");
    g.n_ff_exp = u("expert_feed_forward_length");
    g.hd_full = u("attention.key_length");
    g.hd_swa = u("attention.key_length_swa");
    g.window = u("attention.sliding_window");
    g.rope_base = (float) num_or(gj->get("rope.freq_base"), 1e6);
    g.rope_base_swa = (float) num_or(gj->get("rope.freq_base_swa"), 1e4);
    g.rms_eps = (float) num_or(gj->get("attention.layer_norm_rms_epsilon"), 1e-6);
    g.softcap = (float) num_or(gj->get("final_logit_softcapping"), 0);
    g.context_length = (int64_t) num_or(gj->get("context_length"), 0);
    if (const Json* a = gj->get("attention.head_count_kv"); a && a->is_array())
        for (const Json& x : a->items) g.kv_heads.push_back((int) x.num());
    if (const Json* a = gj->get("attention.sliding_window_pattern"); a && a->is_array())
        for (const Json& x : a->items) g.swa.push_back(x.is_number() ? x.num() != 0 : x.b);
    if (const Json* kv = meta.get("gguf_kv"))
        if (const Json* t = kv->get("tokenizer.ggml.tokens")) g.n_vocab = (int) t->count;
    if (g.n_layers <= 0 || g.n_embd <= 0 || g.n_expert <= 0 || g.n_head <= 0 || (int) g.kv_heads.size() < g.n_layers ||
        (int) g.swa.size() < g.n_layers || g.n_vocab <= 0) {
        err = path + ": meta.geometry is incomplete";
        return false;
    }
    g.kv_heads.resize((size_t) g.n_layers);
    g.swa.resize((size_t) g.n_layers);
    // the layout this engine runs: heads of 256 (window) and 512 (whole context), at most 16 routed experts of 256, widths
    // the kernels take
    if (g.hd_swa != 256 || g.hd_full != 512 || g.n_used > 16 || g.n_expert > 256 || g.n_embd > 4096 || g.n_embd % 256 != 0 ||
        g.window <= 0) {
        err = path + ": a Gemma 4 layout this engine does not run (heads of 256 and 512, at most 256 experts and 16 routed, "
                     "a width of at most 4096 in steps of 256)";
        return false;
    }
    for (int l = 0; l < g.n_layers; ++l)
        if (g.kv_heads[(size_t) l] <= 0 || g.n_head % g.kv_heads[(size_t) l] != 0) {
            err = path + ": layer " + std::to_string(l) + " has a KV head count that does not divide the heads";
            return false;
        }

    for (const auto& [name, t] : tj->members) {
        TensorRef r;
        r.type = (int) num_or(t.get("type_id"), 0);
        if (const Json* sh = t.get("shape"))
            for (const Json& d : sh->items) r.shape.push_back(d.u64());
        r.bytes = (uint64_t) num_or(t.get("bytes"), 0);
        std::string section = "gguf";
        if (const Json* s = t.get("section"); s && s->is_string()) section = s->s;
        const core::ModelFile::Section* sec = p.file.find(section);
        if (!sec || !t.get("offset")) {
            err = path + ": tensor " + name + " names a missing section " + section;
            return false;
        }
        r.offset = sec->offset + t.get("offset")->u64();
        p.tensors.emplace(name, std::move(r));
    }

    p.experts_range = p.file.range("experts");
    std::vector<uint8_t> text;
    if (!p.file.read("experts.layout", text, err)) return false;
    std::istringstream in(std::string(text.begin(), text.end()));
    std::string line;
    p.experts.assign((size_t) g.n_layers, ExpertLayer{});
    int seen = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        long long l, gu, d;
        unsigned long long off, by;
        if (!(ls >> l >> gu >> d >> off >> by) || l < 0 || l >= g.n_layers) {
            err = "experts.layout: bad line '" + line + "'";
            return false;
        }
        p.experts[(size_t) l] = {(int) gu, (int) d, off, by};
        ++seen;
    }
    if (seen != g.n_layers) {
        err = "experts.layout lists " + std::to_string(seen) + " layers, the model has " + std::to_string(g.n_layers);
        return false;
    }
    return true;
}

}  // namespace basalt::models::gemma4
