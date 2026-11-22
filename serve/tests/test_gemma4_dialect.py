"""The Gemma 4 dialect: the template rendered as transformers renders it (with `<bos>`), and the reply's thought channel,
text and tool calls (Gemma's argument syntax) parsed whole and streamed. Text only: no engine, no pack."""
import json
import unittest
from pathlib import Path

from serve.models import dialect_for
from serve.models.gemma4 import CALL, CALL_END, CHANNEL, CHANNEL_END, QUOTE, TOOL_RESPONSE, TURN_END, Gemma4, Gemma4Parser, parse_args

FIXTURES = Path(__file__).resolve().parent / "fixtures"
OFFICIAL = (FIXTURES / "gemma4_chat_template.jinja").read_text(encoding="utf-8")
GOLDEN = json.loads((FIXTURES / "gemma4_golden.json").read_text(encoding="utf-8"))


def parse(text, chunk=None, thinking=True):
    p = Gemma4Parser(thinking=thinking)
    events = []
    if chunk is None:
        events += p.feed(text)
    else:
        for i in range(0, len(text), chunk):
            events += p.feed(text[i:i + chunk])
    events += p.finish()
    return events


def merged(events):
    out = []
    for e in events:
        if e.kind in ("reasoning", "content"):
            if out and out[-1][0] == e.kind:
                out[-1] = (e.kind, out[-1][1] + e.text)
            elif e.text:
                out.append((e.kind, e.text))
        elif e.kind == "tool_call":
            out.append(("call", e.call.name, e.call.arguments))
    return out


class Template(unittest.TestCase):
    def test_golden(self):
        tpl = Gemma4().template(OFFICIAL)
        for name, g in GOLDEN.items():
            with self.subTest(name):
                got = tpl.render(g["messages"], tools=g["tools"], add_generation_prompt=True, **g["kwargs"])
                self.assertEqual(got, g["prompt"])

    def test_dialect_for(self):
        self.assertIsInstance(dialect_for("gemma4"), Gemma4)


class Args(unittest.TestCase):
    def test_values(self):
        q = QUOTE
        self.assertEqual(parse_args("{city:%sRoma, Italia%s,days:2}" % (q, q)), {"city": "Roma, Italia", "days": 2})
        self.assertEqual(parse_args("{a:true,b:false,c:null,d:-1.5e3}"), {"a": True, "b": False, "c": None, "d": -1500.0})
        self.assertEqual(parse_args("{xs:[1,%sdue%s,{k:%sv}%s}],o:{p:{q:3}}}" % (q, q, q, q)),
                         {"xs": [1, "due", {"k": "v}"}], "o": {"p": {"q": 3}}})
        self.assertEqual(parse_args("{}"), {})


class Reply(unittest.TestCase):
    THOUGHT = CHANNEL + "thought\nPrima penso.\nPoi rispondo.\n" + CHANNEL_END

    def check_all_chunks(self, text, want, thinking=True):
        for chunk in (None, 1, 2, 3, 5, 7, 13):
            with self.subTest(chunk=chunk):
                self.assertEqual(merged(parse(text, chunk, thinking)), want)

    def test_thinking_then_answer(self):
        self.check_all_chunks(self.THOUGHT + "La risposta è 391." + TURN_END,
                              [("reasoning", "Prima penso.\nPoi rispondo."), ("content", "La risposta è 391.")])

    def test_no_thinking(self):
        self.check_all_chunks("Ciao! Tutto bene." + TURN_END, [("content", "Ciao! Tutto bene.")], thinking=False)

    def test_empty_channel_is_dropped(self):
        self.check_all_chunks(CHANNEL + "thought\n" + CHANNEL_END + "ok", [("content", "ok")])

    def test_tool_calls(self):
        q = QUOTE
        text = (self.THOUGHT + "Controllo." + CALL + "call:get_weather{city:%sRoma%s,days:2}" % (q, q) + CALL_END +
                CALL + "call:get_time{tz:%sEurope/Rome%s}" % (q, q) + CALL_END + TOOL_RESPONSE)
        self.check_all_chunks(text, [("reasoning", "Prima penso.\nPoi rispondo."), ("content", "Controllo."),
                                     ("call", "get_weather", {"city": "Roma", "days": 2}),
                                     ("call", "get_time", {"tz": "Europe/Rome"})])

    def test_streamed_call_events(self):
        q = QUOTE
        events = parse(CALL + "call:f{a:%sx%s}" % (q, q) + CALL_END, chunk=2)
        kinds = [e.kind for e in events]
        self.assertEqual(kinds, ["tool_start", "tool_args", "tool_call"])
        self.assertEqual(json.loads(events[1].text), {"a": "x"})

    def test_cut_call_is_dropped(self):
        self.assertEqual(merged(parse("a" + CALL + "call:f{a:1")), [("content", "a")])


if __name__ == "__main__":
    unittest.main()
