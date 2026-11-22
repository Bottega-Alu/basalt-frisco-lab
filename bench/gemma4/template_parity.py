"""Fase 3 — il template di Gemma 4 reso da serve.frontend.ChatTemplate (Basalt) == apply_chat_template di Hugging Face.

Uso:  refvenv\\Scripts\\python.exe template_parity.py HF_MODEL_DIR BASALT_REPO
"""
import json
import sys

from transformers import AutoTokenizer

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city"]}}}]
CASES = {
    "base": dict(messages=[{"role": "user", "content": "Ciao, come va?"}]),
    "thinking": dict(messages=[{"role": "user", "content": "Quanto fa 17*23?"}], enable_thinking=True),
    "system": dict(messages=[{"role": "system", "content": "Rispondi in italiano."},
                             {"role": "user", "content": "Hi"}]),
    "multi": dict(messages=[{"role": "user", "content": "Uno"}, {"role": "assistant", "content": "Due"},
                            {"role": "user", "content": "Tre"}]),
    "tools": dict(messages=[
        {"role": "user", "content": "Che tempo fa a Roma?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "get_weather", "arguments": {"city": "Roma", "days": 2}}}]},
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"temp": 21, "sky": "sereno"})}], tools=TOOLS),
}


def main():
    hf_dir, repo = sys.argv[1], sys.argv[2]
    sys.path.insert(0, repo)
    from serve import frontend
    tok = AutoTokenizer.from_pretrained(hf_dir)
    src = open(f"{hf_dir}/chat_template.jinja", encoding="utf-8").read()
    tpl = frontend.ChatTemplate(src)
    bad = 0
    for name, c in CASES.items():
        kw = {k: v for k, v in c.items() if k not in ("messages", "tools")}
        want = tok.apply_chat_template(c["messages"], tools=c.get("tools"), add_generation_prompt=True, tokenize=False, **kw)
        try:
            got = tpl.render(c["messages"], tools=c.get("tools"), add_generation_prompt=True, bos_token="<bos>", **kw)
        except Exception as e:  # noqa: BLE001
            got = f"<render error {type(e).__name__}: {e}>"
        ok = got == want
        bad += not ok
        print(f"{name:9s} {'IDENTICO' if ok else 'DIVERSO'}  {len(want)} char")
        if not ok:
            k = next((i for i in range(min(len(got), len(want))) if got[i] != want[i]), min(len(got), len(want)))
            print(f"   basalt: {got[max(0, k - 40):k + 60]!r}\n   hf    : {want[max(0, k - 40):k + 60]!r}")
    print("RESULT", "ALL IDENTICAL" if not bad else f"{bad} differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
