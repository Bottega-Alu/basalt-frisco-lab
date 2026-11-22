"""Gate fase 4 - il motore Basalt su Gemma 4 contro il riferimento BF16 (ref/*.npz di ref_forward.py).

Uso:  python engine_gate.py BASALT_EXE PACK --gguf GEMMA4_BF16_GGUF [--ref DIR] [--every N] [--tokens N] [-- ENGINE ARGS]
      (il GGUF serve solo al tokenizer; BASALT_CUDA_BIN = cartella delle DLL CUDA se non sono nel PATH)

Per ciascun prompt (conv, struct, prose):
  * libero: GEN greedy di N token; token uguali al riferimento fino alla prima divergenza, e il testo.
  * teacher forcing: ogni `--every` posizioni t, GEN 1 logprobs=20 sul prompt + gen_ref[:t]; il top-1 del motore contro il
    top-1 del riferimento, il riferimento nel top-5 del motore, e la logprob che il motore dà al token del riferimento.
Le soglie del gate: top-1 teacher-forced >= 90 % su ogni prompt, riferimento nel top-5 >= 99 %, nessun ERR.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]   # the repo
OUT = HERE / "out"
PROMPTS = ("conv", "struct", "prose")
CU = Path(os.environ.get("BASALT_CUDA_BIN", ROOT / "toolchain" / "venv" / "Lib" / "site-packages" / "nvidia" / "cu13" / "bin"))


class Engine:
    def __init__(self, exe: str, args: list[str], log: Path):
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join([str(CU / "x86_64"), str(CU), env.get("PATH", "")])
        log.parent.mkdir(parents=True, exist_ok=True)
        self.err = open(log, "w", encoding="utf-8")
        self.p = subprocess.Popen([exe, "--serve", *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err,
                                  text=True, encoding="utf-8", bufsize=1, env=env)
        self.info = []
        for line in self.p.stdout:
            if line.startswith("READY"):
                self.ready = line.strip()
                return
            self.info.append(line.strip())
        raise RuntimeError(f"the engine ended before READY (see {log})")

    def gen(self, ids: list[int], max_new: int, logprobs: int = -1):
        """(ids, P lines, DONE line); raises on ERR."""
        extra = f" logprobs={logprobs}" if logprobs >= 0 else ""
        self.p.stdin.write(f"GEN {max_new} temperature=0{extra} {','.join(map(str, ids))}\n")
        self.p.stdin.flush()
        got, lps = [], []
        for line in self.p.stdout:
            f = line.split()
            if not f:
                continue
            if f[0] == "T":
                got.append(int(f[1]))
            elif f[0] == "P":
                alts = [(int(a.split(":")[0]), float(a.split(":")[1])) for a in f[3:]]
                lps.append((int(f[1]), float(f[2]), alts))
            elif f[0] == "DONE":
                return got, lps, line.strip()
            elif f[0] == "ERR":
                raise RuntimeError(line.strip())
        raise RuntimeError("the engine ended mid-request")

    def close(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=60)
        except Exception:  # noqa: BLE001
            self.p.kill()
        self.err.close()


def main() -> int:
    argv = sys.argv[1:]
    extra: list[str] = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("exe")
    ap.add_argument("pack")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ref", default=str(HERE / "ref"))
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT / "tools"))
    import tokenizer as T  # noqa: E402
    tok = T.Tokenizer.from_gguf(a.gguf)
    eng = Engine(a.exe, ["--model", a.pack, "--max-context", "4096", *extra], OUT / "engine-gate.err")
    print(eng.ready, "|", " ".join(i for i in eng.info if "expert" in i or "slots" in i)[:300])
    ok = True
    try:
        for name in (a.only.split(",") if a.only else PROMPTS):
            z = np.load(Path(a.ref) / f"{name}.npz")
            prompt, ref = z["prompt_ids"].tolist(), z["gen_ids"].tolist()
            top = z["top_ids"]
            t0 = time.time()
            got, _, done = eng.gen(prompt, min(a.tokens, len(ref)))
            dt = time.time() - t0
            same = next((i for i in range(min(len(got), len(ref))) if got[i] != ref[i]), min(len(got), len(ref)))
            print(f"\n[{name}] prompt {len(prompt)} tok; libero: {len(got)} tok in {dt:.1f} s ({len(got) / dt:.1f} tok/s incl. prompt), "
                  f"uguali al riferimento per {same}/{len(ref)}  | {done}")
            print("   motore:", repr(tok.decode(got)[:300]))
            if same < len(ref):
                print("   rif.  :", repr(tok.decode(ref)[:300]))
            agree = in5 = n = 0
            lp_ref = []
            for t in range(0, len(ref), a.every):
                _, lps, _ = eng.gen(prompt + ref[:t], 1, logprobs=20)
                pick, _, alts = lps[0]
                ids = [i for i, _ in alts]
                n += 1
                agree += pick == int(top[t][0])
                in5 += int(top[t][0]) in ids[:5]
                lp_ref.append(dict(alts).get(int(top[t][0]), -99.0))
            print(f"   teacher forcing su {n} posizioni: top-1 {100 * agree / n:.1f} %, rif. nel top-5 {100 * in5 / n:.1f} %, "
                  f"logprob mediana del token rif. {float(np.median(lp_ref)):.3f}")
            ok &= agree / n >= 0.90 and in5 / n >= 0.99
    finally:
        eng.close()
    print("\nGATE", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
