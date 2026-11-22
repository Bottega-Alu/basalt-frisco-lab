# Gemma 4 26B-A4B — gate e bench

Script usati per il port di Gemma 4 26B-A4B-it (pack NVFP4) su Basalt. Output in `out/`, riferimenti in `ref/` (entrambi ignorati da git).

| Script | Cosa fa |
|---|---|
| `ref_forward.py MODEL_DIR OUT_DIR` | riferimento BF16 con transformers (CPU): prompt, gen greedy, hidden e logits → `ref/*.npz` |
| `engine_gate.py EXE PACK --gguf GGUF [--every 2]` | gate: generazione libera + teacher forcing contro `ref/`; soglie top-1 ≥ 90 %, rif. nel top-5 ≥ 99 % |
| `pp_bench.py EXE PACK GGUF [512,2048,8192,16000]` | velocità di prefill su testo reale (i `docs/` del repo) |
| `tok_parity.py HF_DIR TOOLS_DIR [--gguf FILE]` | tokenizer `gemma4` di Basalt == HF `tokenizers` |
| `template_parity.py HF_DIR REPO` | chat template di serve == `apply_chat_template` HF |
| `smoke_api.py [MODEL ...]` | smoke del router OpenAI-compatibile (`BASALT_URL`, key da `BASALT_API_KEY` o keyring) |

`BASALT_CUDA_BIN`: cartella delle DLL CUDA, se non sono già nel `PATH`.

## Risultati (RTX 5080 16 GB, Windows 11, 2026-11-22)

- Gate teacher forcing top-1: conv 97,0 %, struct 98,5 %, prose 92,9 %; riferimento nel top-5 100 %.
- Decode: ~88 tok/s in locale, 90–100 tok/s via router.
- Prefill: pp512 2379, pp2048 3075, pp8192 2248, pp16000 1650 tok/s.
