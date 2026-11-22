"""Gemma 4 MoE (architecture `gemma4`; GGUF architecture `gemma4`), google/gemma-4-26B-A4B-it, from llama.cpp's BF16 GGUF.

The pack is made by tools/convert/gemma4_bf16.py (`convert.py --gguf gemma-4-26B-A4B-it-BF16.gguf --nvfp4 --hf-dir DIR`):
the routed experts (`ffn_gate_up_exps` split into gate and up, `ffn_down_exps`) rounded to NVFP4 with the router's
per-expert scale (`ffn_down_exps.scale`) folded into each expert's FP32 down scale, every BF16 projection (attention,
the dense MLP beside the experts, the tied token embedding) as Q8_0, the F32 tensors (norms, router, layer scalars, the
full-attention layers' rope_freqs) as they are. Text only: no vision or audio sections, no drafter.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import Arch, register

P = "gemma4."


class Gemma4(Arch):
    name = "gemma4"
    gguf_architecture = "gemma4"
    side_sections = False
    needs_hf_dir = True
    sampling_kv = {}

    def license(self, kv):
        return {"name": "Apache License 2.0", "url": "https://ai.google.dev/gemma/apache_2"}

    def hf_meta(self, hf_dir: Path) -> dict:
        from model_sections import ConvertError
        cfg = hf_dir / "config.json"
        if not cfg.is_file():
            raise ConvertError(f"missing {cfg} (--hf-dir is a google/gemma-4-*-A4B directory)")
        conf = json.loads(cfg.read_text(encoding="utf-8"))
        text = conf.get("text_config", conf)
        if text.get("model_type") != "gemma4_text" or not text.get("enable_moe_block"):
            raise ConvertError(f"{cfg} is model_type {text.get('model_type')!r} without the MoE block; expected gemma4_text MoE")
        return {"license": self.license({}), "hf_config": conf}

    def default_name(self, first_gguf, kv, sections_from=False):
        return "Gemma-4-26B-A4B-it NVFP4"

    def check(self, kv, own):
        from model_sections import ConvertError
        if int(kv.get(P + "expert_count", 0)) <= 0:
            raise ConvertError("the GGUF has no experts (gemma4.expert_count)")

    def meta_extra(self, kv, tensors, own):
        return {
            "arch": self.name,
            "geometry": {k[len(P):]: v for k, v in kv.items() if k.startswith(P)},
            "vision": None,
        }


register(Gemma4(), "gemma4")
