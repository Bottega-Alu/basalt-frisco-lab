"""Fase 1 — riferimento BF16 di Gemma 4 26B-A4B-it (transformers, CPU) per la parità del port Basalt.

Uso:  refvenv\\Scripts\\python.exe -I ref_forward.py MODEL_DIR OUT_DIR [--tokens 64]

Per ogni prompt (chat template del modello, generation prompt, thinking off):
  OUT_DIR/<name>.npz : prompt_ids, gen_ids (greedy), top_ids/top_logits (top-20 per step, logit finali dopo softcap),
                       hidden_last (hidden state dell'ultimo token del prompt dopo embedding e dopo ogni layer, fp32)
  OUT_DIR/summary.json : testo generato, tempi, versioni, revisione del modello
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = {
    "conv": "Ciao! Mi spieghi in poche frasi cos'è lo Stretto di Hormuz e perché conta per il petrolio?",
    "struct": "Write a Python function that returns the primes below n using the sieve of Eratosthenes. Only code.",
    "prose": "Write the opening paragraph of a short story set in a lighthouse during a storm.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--tokens", type=int, default=64)
    a = ap.parse_args()
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, (torch.get_num_threads())))
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.bfloat16, device_map="cpu")
    model.eval()
    load_s = time.time() - t0
    print(f"loaded {type(model).__name__} in {load_s:.0f}s", flush=True)
    summary = {"model_dir": a.model_dir, "torch": torch.__version__, "transformers": transformers.__version__,
               "model_class": type(model).__name__, "load_s": round(load_s, 1), "prompts": {}}
    for name, text in PROMPTS.items():
        msgs = [{"role": "user", "content": text}]
        try:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        except TypeError:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        t = time.time()
        with torch.no_grad():
            fw = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            hidden_last = np.stack([h[0, -1].float().numpy() for h in fw.hidden_states])
            g = model.generate(ids, max_new_tokens=a.tokens, do_sample=False, output_logits=True,
                               return_dict_in_generate=True)
        gen = g.sequences[0, ids.shape[1]:].numpy()
        logits = torch.stack(g.logits)[:, 0].float()          # [steps, vocab]
        top = torch.topk(logits, 20, dim=-1)
        np.savez_compressed(out / f"{name}.npz", prompt_ids=ids[0].numpy(), gen_ids=gen,
                            top_ids=top.indices.numpy(), top_logits=top.values.numpy(), hidden_last=hidden_last)
        dt = time.time() - t
        txt = tok.decode(gen, skip_special_tokens=False)
        summary["prompts"][name] = {"prompt": text, "prompt_tokens": int(ids.shape[1]), "gen_tokens": int(len(gen)),
                                    "seconds": round(dt, 1), "text": txt}
        print(f"{name}: {ids.shape[1]} prompt tok, {len(gen)} gen tok, {dt:.0f}s\n  {txt[:200]!r}", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
