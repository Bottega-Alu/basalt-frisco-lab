"""Fase 3 — parità del tokenizer `gemma4` di Basalt (tools/tokenizer.py) contro Hugging Face `tokenizers`.

Uso:  refvenv\\Scripts\\python.exe -I tok_parity.py HF_MODEL_DIR BASALT_TOOLS_DIR [--gguf FILE]

Senza --gguf il Tokenizer di Basalt è costruito dal tokenizer.json (tipi come li scrive llama.cpp: special -> CONTROL 3,
i token resi visibili da llama.cpp -> USER_DEFINED 4, <0xNN> -> BYTE 6); con --gguf dal GGUF (percorso del converter).
Confronta gli id di ogni testo del corpus (encode con parse_special) e il round trip decode(encode(s)) == s.
"""
import argparse
import json
import pathlib
import sys

from tokenizers import Tokenizer as HFTokenizer

VISIBLE = {"<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>", '<|"|>'}


def corpus(repo: pathlib.Path) -> list[str]:
    base = ["", " ", "  ", "Hello, world!", "  leading and trailing  ", "a\n\n\nb", "\t\ttabs\tand  spaces ",
            "def f(x):\n\treturn x  # comment\n", "你好，世界", "مرحبا بالعالم", "😀🚀🇺🇸👨‍👩‍👧", "é́ combining",
            "\x00\x01\x7f control", " non-breaking space", "1234567890 3.14159 1e-9", "MixedCASE_and-dashes",
            "\r\n\r\n", "  ", "x" * 5000, " " * 300, "\n" * 70, "Perché è così? Già, città, perciò.",
            "<|turn>user\nCiao<turn|>\n<|turn>model\n", "<bos><|turn>system\nYou are helpful.<turn|>",
            "<|think|> ragionamento <|channel>thought\n...<channel|>risposta", '<|tool_call>call:f{a:<|"|>x<|"|>}<tool_call|>',
            "Ѐ Ж ж Ѣ ᚠ ᛫ 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𐍈 ﷽", "```json\n{\"a\": [1, 2, {\"b\": null}]}\n```"]
    files = list(repo.glob("tools/*.py"))[:12] + list(repo.glob("docs/*.md"))[:8] + list(repo.glob("src/core/*.cpp"))[:6]
    real = [p.read_text(encoding="utf-8", errors="replace") for p in files]
    return base + real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir")
    ap.add_argument("tools_dir")
    ap.add_argument("--gguf")
    a = ap.parse_args()
    hf_dir = pathlib.Path(a.hf_dir)
    sys.path.insert(0, a.tools_dir)
    import tokenizer as T
    hf = HFTokenizer.from_file(str(hf_dir / "tokenizer.json"))
    if a.gguf:
        bt = T.Tokenizer.from_gguf(a.gguf)
        src = f"GGUF {a.gguf}"
    else:
        tj = json.loads((hf_dir / "tokenizer.json").read_text(encoding="utf-8"))
        vocab = tj["model"]["vocab"]
        tokens = [None] * len(vocab)
        for t, i in vocab.items():
            tokens[i] = t
        added = {x["id"]: x for x in tj["added_tokens"]}
        types = []
        for i, t in enumerate(tokens):
            if i in added and added[i]["special"]:
                types.append(4 if t in VISIBLE else 3)
            elif t.startswith("<0x") and len(t) == 6:
                types.append(6)
            else:
                types.append(1)
        merges = [" ".join(m) for m in tj["model"]["merges"]]
        bt = T.Tokenizer(tokens, merges, types, model="gemma4")
        src = "tokenizer.json"
    texts = corpus(pathlib.Path(a.tools_dir).parent)
    bad_ids = bad_rt = ntok = 0
    for s in texts:
        want = hf.encode(s, add_special_tokens=False).ids
        got = bt.encode(s, parse_special=True)
        ntok += len(want)
        if got != want:
            bad_ids += 1
            k = next((j for j in range(min(len(got), len(want))) if got[j] != want[j]), min(len(got), len(want)))
            print(f"IDS DIFFER {s[:50]!r}: at {k}: basalt {got[k:k+5]} hf {want[k:k+5]} (len {len(got)} vs {len(want)})")
        if bt.decode(got) != s:
            bad_rt += 1
            print(f"ROUND TRIP {s[:50]!r} -> {bt.decode(got)[:50]!r}")
    print(f"source {src}: {len(texts)} texts, {ntok} tokens, ids differ {bad_ids}, round trip failed {bad_rt}")
    return 0 if bad_ids == 0 and bad_rt == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
