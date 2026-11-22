"""Prefill speed: GEN 1 on real text prompts of N tokens (distinct prefixes, no reuse). Corpus: the repo's docs/.

Uso:  python pp_bench.py BASALT_EXE PACK GEMMA4_BF16_GGUF [512,2048,8192,16000]
"""
import sys
import time

from engine_gate import Engine, OUT, ROOT

sys.path.insert(0, str(ROOT / "tools"))
import tokenizer as T

exe, pack, gguf = sys.argv[1:4]
tok = T.Tokenizer.from_gguf(gguf)
text = "".join(p.read_text(encoding="utf-8", errors="ignore") for p in sorted(ROOT.glob("docs/**/*.md")))
ids = tok.encode(text)
print("corpus tokens", len(ids))
lens = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [512, 2048, 8192, 16000]
eng = Engine(exe, ["--model", pack, "--max-context", str(max(lens) + 64)], OUT / "pp-bench.err")
try:
    off = 0
    for n in lens:
        p = [2] + ids[off: off + n - 1]
        off += n
        t0 = time.time()
        _, _, done = eng.gen(p, 1)
        dt = time.time() - t0
        print(f"pp{n}: wall {dt:.2f} s -> {n / dt:.0f} tok/s | {done}")
finally:
    eng.close()
