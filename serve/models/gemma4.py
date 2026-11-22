"""The Gemma 4 dialect: the model's Jinja chat template (with its `<bos>`), and a reply of an optional thought channel,
text and tool calls.

A model turn is `<|turn>model\n` then, with thinking, `<|channel>thought\n...<channel|>` (without it the template writes an
empty channel itself), the answer, and tool calls `<|tool_call>call:NAME{key:value,...}<tool_call|>` in Gemma's own
argument syntax: bare keys, strings between `<|"|>` marks, numbers, true/false/null, nested `{}` and `[]`. The turn ends
with `<turn|>`; after its calls the model writes `<|tool_response>` to hand over to the tools.
"""
from __future__ import annotations

import json

from serve import frontend
from serve.frontend import Event, ToolCall
from serve.models import Dialect

BOS = "<bos>"
TURN_END = "<turn|>"
CHANNEL = "<|channel>"
CHANNEL_END = "<channel|>"
THOUGHT = "thought\n"
CALL = "<|tool_call>"
CALL_END = "<tool_call|>"
TOOL_RESPONSE = "<|tool_response>"
QUOTE = '<|"|>'
OWN_TOKENS = ("<bos>", "<eos>", "<pad>", "<|turn>", TURN_END, "<|think|>", CHANNEL, CHANNEL_END, "<|tool>", "<tool|>",
              CALL, CALL_END, TOOL_RESPONSE, "<tool_response|>", QUOTE, "<|image>", "<|audio>", "<|image|>", "<|audio|>",
              "<image|>", "<audio|>", "<|video|>")
MARK0 = 0xF2600                         # the marks of the dialect's own tokens while a message is rendered


# ---------------------------------------------------------------------- Gemma's call arguments -> JSON values
class ArgsError(ValueError):
    pass


def parse_args(text: str):
    """`{key:value,...}` in Gemma's syntax to a Python value (a dict for a call's arguments)."""
    v, i = _value(text, _ws(text, 0))
    if _ws(text, i) != len(text):
        raise ArgsError(f"text after the arguments at {i}")
    return v


def _ws(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t\r\n":
        i += 1
    return i


def _value(s: str, i: int):
    if s.startswith(QUOTE, i):
        j = s.find(QUOTE, i + len(QUOTE))
        if j < 0:
            raise ArgsError("an unterminated string")
        return s[i + len(QUOTE):j], j + len(QUOTE)
    if i < len(s) and s[i] == "{":
        out, i = {}, _ws(s, i + 1)
        while i < len(s) and s[i] != "}":
            k = i
            while i < len(s) and s[i] not in ":}":
                i += 1
            if i >= len(s) or s[i] != ":":
                raise ArgsError(f"a key without a value at {k}")
            key = s[k:i].strip()
            if key.startswith(QUOTE) and key.endswith(QUOTE):
                key = key[len(QUOTE):-len(QUOTE)]
            out[key], i = _value(s, _ws(s, i + 1))
            i = _ws(s, i)
            if i < len(s) and s[i] == ",":
                i = _ws(s, i + 1)
        if i >= len(s):
            raise ArgsError("an unterminated object")
        return out, i + 1
    if i < len(s) and s[i] == "[":
        out, i = [], _ws(s, i + 1)
        while i < len(s) and s[i] != "]":
            v, i = _value(s, i)
            out.append(v)
            i = _ws(s, i)
            if i < len(s) and s[i] == ",":
                i = _ws(s, i + 1)
        if i >= len(s):
            raise ArgsError("an unterminated list")
        return out, i + 1
    j = i
    while j < len(s) and s[j] not in ",}]":
        j += 1
    word = s[i:j].strip()
    if word in ("true", "false", "null"):
        return {"true": True, "false": False, "null": None}[word], j
    try:
        return json.loads(word), j
    except ValueError:
        return word, j                  # a bare word: kept as the text the model wrote


# ---------------------------------------------------------------------- the reply parser
class Gemma4Parser:
    """Incremental parser of the model's text. `state`: "start" (a thought channel may open), "reasoning", "content",
    "call" (inside `<|tool_call>...<tool_call|>`); text that may still be the start of a marker is held back."""

    MARKERS = (CHANNEL, CHANNEL_END, CALL, TOOL_RESPONSE, TURN_END)

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = True):
        self.thinking = thinking
        self.stream_tools = stream_tools
        self.state = "start"
        self.buf = ""
        self.raw = ""
        self.call: ToolCall | None = None

    def _emit(self, out, kind, text):
        if text:
            if out and out[-1].kind == kind and out[-1].call is None:
                out[-1].text += text
            else:
                out.append(Event(kind, text))

    def _start(self, out) -> bool:
        if not self.buf:
            return False
        if self.buf.startswith(CHANNEL + THOUGHT):
            self.buf, self.state = self.buf[len(CHANNEL + THOUGHT):], "reasoning"
            return True
        if (CHANNEL + THOUGHT).startswith(self.buf):
            return False                # the channel still arriving
        if self.buf.startswith(CHANNEL + CHANNEL_END) or self.buf.startswith(CHANNEL + "thought" + CHANNEL_END):
            self.buf = self.buf[self.buf.find(CHANNEL_END) + len(CHANNEL_END):]
        self.state = "content"
        return True

    def _reasoning(self, out) -> bool:
        end = self.buf.find(CHANNEL_END)
        if end >= 0:
            text = self.buf[:end]
            self._emit(out, "reasoning", text[:-1] if text.endswith("\n") else text)
            self.buf, self.state = self.buf[end + len(CHANNEL_END):], "content"
            return True
        # the newline the template puts before <channel|> is not reasoning: hold back a possible "\n<channel|>" start
        keep = frontend.hold_back(self.buf, ("\n" + CHANNEL_END, CHANNEL_END))
        if len(self.buf) > keep:
            self._emit(out, "reasoning", self.buf[:len(self.buf) - keep])
            self.buf = self.buf[len(self.buf) - keep:]
        return False

    def _content(self, out) -> bool:
        at = self.buf.find(CALL)
        if at >= 0:
            self._emit(out, "content", self.buf[:at])
            self.buf, self.state, self.raw = self.buf[at + len(CALL):], "call", ""
            self.call = None
            return True
        for stop in (TOOL_RESPONSE, TURN_END):
            s = self.buf.find(stop)
            if s >= 0:
                self._emit(out, "content", self.buf[:s])
                self.buf = ""
                return False
        keep = frontend.hold_back(self.buf, (CALL, TOOL_RESPONSE, TURN_END))
        if len(self.buf) > keep:
            self._emit(out, "content", self.buf[:len(self.buf) - keep])
            self.buf = self.buf[len(self.buf) - keep:]
        return False

    def _call(self, out) -> bool:
        end = self.buf.find(CALL_END)
        take = self.buf if end < 0 else self.buf[:end]
        keep = 0 if end >= 0 else frontend.hold_back(take, (CALL_END,))
        self.raw += take[:len(take) - keep]
        self.buf = self.buf[end + len(CALL_END):] if end >= 0 else take[len(take) - keep:]
        if self.call is None:
            brace = self.raw.find("{")
            if brace >= 0:
                name = self.raw[:brace].strip()
                name = name[len("call:"):] if name.startswith("call:") else name
                self.call = ToolCall(name=name, arguments={})
                if self.stream_tools:
                    out.append(Event("tool_start", call=self.call))
        if end < 0:
            return False
        self._end_call(out)
        self.state = "content"
        return True

    def _end_call(self, out) -> None:
        brace = self.raw.find("{")
        head = self.raw[:brace].strip() if brace >= 0 else self.raw.strip()
        name = head[len("call:"):] if head.startswith("call:") else head
        call = self.call or ToolCall(name=name, arguments={})
        try:
            args = parse_args(self.raw[brace:]) if brace >= 0 else {}
        except ArgsError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        if self.stream_tools:
            out.append(Event("tool_args", json.dumps(args, ensure_ascii=False), call=call))
        out.append(Event("tool_call", call=ToolCall(name=name or call.name, arguments=args, id=call.id)))
        self.call, self.raw = None, ""

    def feed(self, delta: str) -> list[Event]:
        self.buf += delta
        out: list[Event] = []
        while True:
            more = {"start": self._start, "reasoning": self._reasoning, "content": self._content,
                    "call": self._call}[self.state](out)
            if not more:
                return out

    def finish(self) -> list[Event]:
        out: list[Event] = []
        if self.state == "call":
            self.call = None            # a call cut off mid-way: no tool_call
        elif self.state == "reasoning":
            self._emit(out, "reasoning", self.buf)
        elif self.buf and self.buf not in (CHANNEL, CHANNEL + THOUGHT):
            text = self.buf
            for stop in (TOOL_RESPONSE, TURN_END):
                if stop.startswith(text):
                    text = ""
            self._emit(out, "content", text)
        self.buf = ""
        return out


class _Renderer:
    """The pack's template with Gemma's `<bos>` (Basalt's renderer passes no bos_token of its own)."""

    def __init__(self, source: str):
        self.inner = frontend.ChatTemplate(source)
        self.source = source

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        kwargs.setdefault("bos_token", BOS)
        return self.inner.render(messages, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)


class Gemma4(Dialect):
    name = "gemma4"
    stop_texts = (TURN_END, TOOL_RESPONSE, "<eos>")
    reasoning_close = "\n" + CHANNEL_END
    call_newlines = 0
    # chat_template.jinja: tools and several calls per turn, system messages, thinking on or off (no effort levels),
    # and `preserve_thinking` for the reasoning of turns with calls
    caps = {"supports_tool_calls": True, "supports_parallel_tool_calls": True, "supports_reasoning_effort": False,
            "supports_system_role": True, "supports_preserve_reasoning": True}

    def effort_level(self, level):
        return level                    # the template has only enable_thinking: any effort but none is thinking on

    def template(self, source):
        return _Renderer(source)

    def parser(self, thinking, tools, stream_tools=True):
        return Gemma4Parser(thinking=thinking, tools=tools, stream_tools=stream_tools)

    def literal_tags(self, controls):
        own = {tok: chr(MARK0 + k) for k, tok in enumerate(OWN_TOKENS)}
        tags = {**frontend.literal_tags(controls), **own}
        return dict(sorted(tags.items(), key=lambda kv: -len(kv[0])))

    def forced_opening(self, name):
        return CALL + "call:" + (name + "{" if name else "")
