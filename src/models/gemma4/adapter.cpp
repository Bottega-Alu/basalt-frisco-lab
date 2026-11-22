// Gemma 4 adapter: registers the architecture, validates and describes the file from its `gemma4.*` metadata, and loads the
// pack into a GemmaModel. The server tokenizes Gemma's text itself (tools/tokenizer.py's SentencePiece-style mode) and hands
// the engine ids: the pack carries no merges.txt, so the engine's own tokenizer stays off.
#include "basalt/model/adapter.hpp"
#include "basalt/program/grammar/tool_form.hpp"
#include "models/gemma4/model.hpp"
#include "models/gemma4/pack.hpp"

namespace basalt::models::gemma4 {
namespace {

// The whole text is one piece: Gemma's pre-tokenizer splits nothing (spaces become U+2581 before the merges).
void pretok_whole(const char*, size_t size, std::vector<size_t>& ends) {
    ends.clear();
    if (size) ends.push_back(size);
}

class Adapter final : public model::ModelAdapter {
public:
    const char* arch() const override { return kArch; }
    std::string validate(const GgufFile& g) const override { return validate_meta(g); }
    model::ModelDesc describe(const GgufFile& g) const override { return describe_meta(g); }
    std::unique_ptr<model::LoadedModel> make_model() const override { return std::make_unique<GemmaModel>(); }
    model::TokenizerSpec tokenizer_spec() const override { return {"tokenizer.ggml.tokens", "gemma4"}; }
    PreTokenizer pretokenizer() const override { return {"gemma4", pretok_whole}; }
    int max_expert_cards() const override { return 0; }
    const char* model_name() const override { return "Gemma-4-26B-A4B-it"; }

    // <|tool_call>call:NAME{key:<|"|>text<|"|>,...}<tool_call|>
    program::grammar::ToolForm tool_form() const override {
        program::grammar::ToolForm f;
        f.call_open = "<|tool_call>";
        f.call_close = "<tool_call|>";
        f.fn_open = "call:";
        f.fn_mid = "{";
        f.fn_close = "}";
        f.param_open = "";
        f.param_mid_text = ":<|\"|>";
        f.param_mid_value = ":";
        f.param_close = ",";
        f.text_end = "<|\"|>";
        f.trigger = "<|tool_call>";
        return f;
    }

    // <turn|> ends the model's turn, <eos> the text; <|turn> opens a turn and `model` names the model's own
    model::TokenNames token_names() const override { return {{"<turn|>", "<eos>"}, "<|turn>", "", "", "model"}; }
};

}  // namespace

std::unique_ptr<model::ModelAdapter> make_adapter() { return std::make_unique<Adapter>(); }

}  // namespace basalt::models::gemma4
