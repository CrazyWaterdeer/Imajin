from __future__ import annotations

import itertools
import json
import re
from collections.abc import Iterator
from typing import Any

from imajin.agent.providers.base import (
    Event,
    Stop,
    TextDelta,
    ToolUse,
    ToolUseStart,
)

# --- inline tool-call salvage ------------------------------------------------
#
# Some local models (qwen, llama, and any fine-tune trained on a different
# function-calling convention) don't put a tool call on the proper streamed
# `tool_calls` channel -- they write it as text in the content field instead,
# in one of several shapes depending on the model family. Everything down to
# _parse_inline_tool_calls recognizes and extracts those, so the turn doesn't
# silently fall through to "the model just talked instead of acting".

_ARG_KEY_MARKERS = ('"arguments"', '"parameters"', '"input"', '"args"')


def _looks_like_tool_call(text: str) -> bool:
    """Cheap, key-order-independent pre-filter so an ordinary prose reply
    skips the JSON-slicing work below.

    An earlier version gated on a single regex that required `"name"` to
    appear textually *before* the arguments-like key, which missed any model
    that happens to emit the arguments key first. Plain substring checks
    carry no such ordering assumption.
    """
    if '"tool_calls"' in text:
        return True
    return '"name"' in text and any(marker in text for marker in _ARG_KEY_MARKERS)


def _slice_first_json_span(text: str) -> tuple[int, int] | None:
    """Return the (start, end) span of the first balanced JSON array/object
    substring in `text`, or None.

    Shared by `_slice_first_json` (kept for its existing str-returning
    contract -- other tests import it directly) and `_extract_all_json_values`
    (which needs the span to know where to resume scanning for a second,
    third, ... value).
    """
    start = -1
    opener = closer = ""
    for i, ch in enumerate(text):
        if ch in "[{":
            start = i
            opener = ch
            closer = "]" if opener == "[" else "}"
            break
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return start, i + 1
    return None


def _slice_first_json(text: str) -> str | None:
    """Return the first balanced JSON array/object substring, or None."""
    span = _slice_first_json_span(text)
    return text[span[0] : span[1]] if span else None


def _try_json(text: str) -> Any | None:
    """Best-effort JSON parse: exact parse first, then the first balanced
    JSON value inside `text` (handles stray prose/backticks a model left
    around otherwise-valid JSON). Returns None rather than raising."""
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    sliced = _slice_first_json(text)
    if sliced is None:
        return None
    try:
        return json.loads(sliced)
    except json.JSONDecodeError:
        return None


def _extract_all_json_values(text: str, *, limit: int = 8) -> list[Any]:
    """Repeatedly slice the first balanced JSON value out of `text`, parse
    it, and advance past it -- handles a model that emits several separate
    JSON values back to back instead of one (with or without a wrapping
    array). Bounded so malformed/adversarial input can't loop unboundedly;
    a real tool-call reply never approaches the limit.
    """
    values: list[Any] = []
    pos = 0
    for _ in range(limit):
        span = _slice_first_json_span(text[pos:])
        if span is None:
            break
        start, end = span
        parsed = _try_json(text[pos + start : pos + end])
        if parsed is not None:
            values.append(parsed)
        pos += end
    return values


def _calls_from_json_value(
    value: Any, known_tool_names: set[str], counter: Iterator[int]
) -> list[dict[str, Any]]:
    """Turn one parsed JSON value into 0+ salvaged tool calls.

    A dict is one call, a list is that many, and a `{"tool_calls": [...]}`
    wrapper -- the shape a few local runtimes surface even *outside* the real
    tool_calls channel -- is unwrapped to its list first. Anything not naming
    a known tool (typo, hallucinated tool, or just JSON that happened to
    parse but isn't a call) is silently skipped: this is a best-effort
    salvage of a possibly-malformed response, not a trusted input.
    """
    if isinstance(value, dict) and isinstance(value.get("tool_calls"), list):
        value = value["tool_calls"]
    items = value if isinstance(value, list) else [value]
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        function = item.get("function") if isinstance(item.get("function"), dict) else {}
        name = item.get("name") or function.get("name")
        if not isinstance(name, str) or name not in known_tool_names:
            continue
        args = (
            item.get("arguments")
            or item.get("parameters")
            or item.get("input")
            or item.get("args")
            or function.get("arguments")
            or {}
        )
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        out.append({"id": f"inline_{next(counter)}", "name": name, "input": args})
    return out


_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _extract_tool_call_tag_calls(
    text: str, known_tool_names: set[str], counter: Iterator[int]
) -> list[dict[str, Any]]:
    """`<tool_call>{...}</tool_call>` -- the Qwen / Hermes tool-call format
    (this is the pinned qwen3.5:9b model's own fallback shape for a call that
    misses the real tool_calls streaming channel)."""
    out: list[dict[str, Any]] = []
    for m in _TOOL_CALL_TAG_RE.finditer(text):
        parsed = _try_json(m.group(1))
        if parsed is not None:
            out.extend(_calls_from_json_value(parsed, known_tool_names, counter))
    return out


_FUNCTION_TAG_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)


def _extract_function_tag_calls(
    text: str, known_tool_names: set[str], counter: Iterator[int]
) -> list[dict[str, Any]]:
    """`<function=name>{...}</function>` -- seen from some Llama fine-tunes."""
    out: list[dict[str, Any]] = []
    for m in _FUNCTION_TAG_RE.finditer(text):
        name = m.group(1).strip()
        if name not in known_tool_names:
            continue
        args = _try_json(m.group(2))
        out.append(
            {
                "id": f"inline_{next(counter)}",
                "name": name,
                "input": args if isinstance(args, dict) else {},
            }
        )
    return out


_PYTHON_TAG_RE = re.compile(r"<\|python_tag\|>")
_PYTHON_TAG_STOP_RE = re.compile(r"<\|eo[mt]_id\|>")


def _extract_python_tag_calls(
    text: str, known_tool_names: set[str], counter: Iterator[int]
) -> list[dict[str, Any]]:
    """`<|python_tag|>{...}` -- Llama's built-in tool-call marker. Meta's own
    format allows several `;`-separated JSON calls after one marker for a
    parallel tool-call turn, so every value up to the next end-of-turn marker
    (or end of text) is salvaged, not just the first.
    """
    match = _PYTHON_TAG_RE.search(text)
    if match is None:
        return []
    stop = _PYTHON_TAG_STOP_RE.search(text, match.end())
    segment = text[match.end() : stop.start() if stop else len(text)]
    out: list[dict[str, Any]] = []
    for value in _extract_all_json_values(segment):
        out.extend(_calls_from_json_value(value, known_tool_names, counter))
    return out


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_fenced_json_values(text: str) -> list[Any]:
    """Every fenced code block's content, each parsed independently -- a
    model that puts multiple tool calls in separate ```json fences (instead
    of one array in one fence) must not lose all but the first.
    """
    values: list[Any] = []
    for m in _FENCE_RE.finditer(text):
        values.extend(_extract_all_json_values(m.group(1)))
    return values


def _parse_inline_tool_calls(text: str, known_tool_names: set[str]) -> list[dict[str, Any]]:
    """Extract every tool call salvageable from `text` (see the section
    docstring above for why this exists). Recognizes, in this order:

      - `<tool_call>{...}</tool_call>` (Qwen / Hermes)
      - `<function=name>{...}</function>` (some Llama fine-tunes)
      - `<|python_tag|>{...}[; {...}]` (Llama's built-in marker)
      - bare or ```-fenced JSON: `{"name": ..., "arguments": {...}}`, a
        `[...]` array of those, or a `{"tool_calls": [...]}` wrapper -- any
        number of them, fenced or not, in any key order.

    Returns every call found (not just the first), each as one
    `{"id": str, "name": str, "input": dict}`. Ids are unique across the
    whole result (one counter spans every shape above), so a fenced-block
    call and a `<tool_call>` tag can never collide even in the same message.
    """
    if not text:
        return []
    counter = itertools.count()
    out: list[dict[str, Any]] = []
    out.extend(_extract_tool_call_tag_calls(text, known_tool_names, counter))
    out.extend(_extract_function_tag_calls(text, known_tool_names, counter))
    out.extend(_extract_python_tag_calls(text, known_tool_names, counter))
    if out:
        return out

    values = _extract_fenced_json_values(text) if "```" in text else []
    if not values and _looks_like_tool_call(text):
        values = _extract_all_json_values(text)
    for value in values:
        out.extend(_calls_from_json_value(value, known_tool_names, counter))
    return out


# --- Anthropic <-> OpenAI-compat message translation -------------------------


def _anthropic_to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate Anthropic-style content-block messages into OpenAI chat format."""
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for block in content:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block["text"])
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                )
            elif btype == "tool_result":
                content_val = block.get("content", "")
                if isinstance(content_val, list):
                    # This flattening always drops any image block and keeps
                    # only text. `_anthropic_to_openai_messages` has a pinned
                    # signature (slice A imports it as-is), so it cannot grow
                    # a vision-aware branch of its own. Vision instead lives
                    # in OpenAICompatProvider.stream(): when constructed with
                    # supports_vision=True it re-scans the original (pre-
                    # flattening) messages for tool_result image blocks and
                    # appends them as one trailing role:user image_url
                    # message after calling this function -- so a text-only
                    # server (supports_vision defaults off) never receives
                    # image content it would 400 on.
                    content_val = "".join(
                        b.get("text", "") for b in content_val if b.get("type") == "text"
                    )
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": str(content_val),
                    }
                )
        if role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
        else:
            if text_parts:
                out.append({"role": role, "content": "".join(text_parts)})
            out.extend(tool_results)
    return out


def _tool_result_image_url_blocks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect every tool_result image block (Anthropic base64 source shape)
    across `messages`, translated to OpenAI `image_url` content blocks, in
    conversation order.

    Only ever called when the provider was constructed with
    supports_vision=True -- a text-only server 400s on image content, so this
    must not run unless the model picker already confirmed the model is
    multimodal.
    """
    blocks: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            inner = block.get("content")
            if not isinstance(inner, list):
                continue
            for part in inner:
                if not isinstance(part, dict) or part.get("type") != "image":
                    continue
                source = part.get("source") or {}
                data = source.get("data")
                if not data:
                    continue
                media_type = source.get("media_type", "image/png")
                blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{data}"},
                    }
                )
    return blocks


def _error_text(exc: Exception) -> str:
    """Flatten a BadRequestError's message + parsed JSON body to one
    lowercase string, so callers can substring-match a rejected param name
    regardless of the server's exact error shape (OpenAI nests it under
    `error.message`; plenty of compat servers don't)."""
    parts = [str(exc)]
    body = getattr(exc, "body", None)
    if body is not None:
        try:
            parts.append(json.dumps(body, default=str))
        except TypeError:
            parts.append(str(body))
    return " ".join(parts).lower()


def _extract_usage(raw: Any) -> dict[str, int]:
    """Map an OpenAI-compat `usage` object to the runner's canonical usage
    keys -- the same names AnthropicProvider uses, and the only ones the chat
    dock's token readout looks up (see ui/chat_dock.py), so an OpenAI-compat
    turn's counts actually render instead of silently showing 0.
    """
    usage: dict[str, int] = {}
    prompt = getattr(raw, "prompt_tokens", None)
    if isinstance(prompt, int):
        usage["input_tokens"] = prompt
    completion = getattr(raw, "completion_tokens", None)
    if isinstance(completion, int):
        usage["output_tokens"] = completion
    details = getattr(raw, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details is not None else None
    if isinstance(cached, int):
        usage["cache_read_input_tokens"] = cached
    return usage


class OpenAICompatProvider:
    name = "openai-compat"

    def __init__(
        self,
        api_key: str | None,
        model: str = "gpt-5",
        base_url: str = "https://api.openai.com/v1",
        max_tokens: int = 4096,
        supports_vision: bool = False,
        timeout: float = 900.0,
    ) -> None:
        import httpx
        from openai import OpenAI

        self.model = model
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.supports_vision = supports_vision
        self._client = OpenAI(
            api_key=api_key or "ollama",
            base_url=base_url,
            # Split connect/read timeouts: a local server can spend minutes
            # just loading the model and prefilling a ~100 KB tool catalogue
            # before the first token (measured: 25.5s for a *small* prompt at
            # num_ctx=32768 -- a full-catalogue prompt is worse), so the read
            # timeout has to stay generous. A dead connect attempt should
            # still fail fast rather than hang for that same span.
            timeout=httpx.Timeout(timeout, connect=10.0),
        )
        self._resolved = False
        # Which request param carries the output-length budget, and whether
        # to ask for usage on the final chunk. Both start optimistic (the
        # common case: Ollama/LM Studio-style `max_tokens`, and a server that
        # accepts stream_options) and self-correct on the first rejection --
        # see _create_with_retries.
        self._max_tokens_param = "max_tokens"
        self._use_usage_stream_option = True

    def _build_create_kwargs(
        self, oai_messages: list[dict[str, Any]], oai_tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            self._max_tokens_param: self.max_tokens,
            "stream": True,
        }
        if oai_tools:
            kwargs["tools"] = oai_tools
        if self._use_usage_stream_option:
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    def _create_with_retries(
        self, oai_messages: list[dict[str, Any]], oai_tools: list[dict[str, Any]]
    ) -> Any:
        """`chat.completions.create()`, self-healing two known cross-server
        incompatibilities on the first 400 that names them, and remembering
        the fix on this instance so later turns pay the retry only once
        ever, not once per turn:

          - gpt-5+/o-series OpenAI models reject `max_tokens` and require
            `max_completion_tokens`; Ollama and LM Studio only understand
            `max_tokens`. There is no reliable way to tell from the model
            name alone, so probe by trying and reading the rejection.
          - `stream_options` (needed to get usage on the final chunk) is an
            OpenAI extension; several compat servers 400 on an unrecognized
            top-level param instead of ignoring it.

        A 400 from this call is always a synchronous, pre-stream rejection
        (the SDK checks the HTTP status as soon as headers arrive, before any
        chunk is handed back -- see openai._base_client.SyncAPIClient.request,
        which raises from `response.raise_for_status()` well before returning
        a Stream to its caller), so a retry here can never re-send content
        the caller already saw.
        """
        from openai import BadRequestError

        tried: set[str] = set()
        while True:
            kwargs = self._build_create_kwargs(oai_messages, oai_tools)
            try:
                return self._client.chat.completions.create(**kwargs)
            except BadRequestError as exc:
                text = _error_text(exc)
                if (
                    "tokens_param" not in tried
                    and self._max_tokens_param == "max_tokens"
                    and "max_completion_tokens" in text
                ):
                    tried.add("tokens_param")
                    self._max_tokens_param = "max_completion_tokens"
                    continue
                if (
                    "stream_options" not in tried
                    and self._use_usage_stream_option
                    and "stream_options" in text
                ):
                    tried.add("stream_options")
                    self._use_usage_stream_option = False
                    continue
                raise

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> Iterator[Event]:
        # Upgrade the "gpt" tier token to the latest flagship model once, on this
        # worker thread (best-effort, cached, falls back to gpt-5). Concrete model
        # ids — including Ollama models like "qwen3.5:9b" — pass through untouched.
        if not self._resolved:
            self._resolved = True
            if self.model == "gpt":
                from imajin.agent.model_catalog import resolve_openai_model

                self.model = resolve_openai_model(self._client)

        oai_messages = [
            {"role": "system", "content": system},
            *_anthropic_to_openai_messages(messages),
        ]
        if self.supports_vision:
            image_blocks = _tool_result_image_url_blocks(messages)
            if image_blocks:
                oai_messages.append({"role": "user", "content": image_blocks})

        oai_tools: list[dict[str, Any]] = []
        for t in tools:
            oai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t["input_schema"],
                    },
                }
            )

        response = self._create_with_retries(oai_messages, oai_tools)

        tool_buffers: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        accumulated_text = ""
        usage: dict[str, int] = {}
        known_tool_names = {t["name"] for t in tools}

        for chunk in response:
            # The usage-only final chunk (stream_options={"include_usage":
            # True}) carries an empty `choices` list, so this must be read
            # before the `not chunk.choices: continue` below or it's missed.
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = _extract_usage(chunk_usage)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            if getattr(delta, "content", None):
                accumulated_text += delta.content
                yield TextDelta(text=delta.content)

            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    idx = tc.index
                    buf = tool_buffers.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if tc.id:
                        buf["id"] = tc.id
                    if tc.function:
                        if tc.function.name and not buf["name"]:
                            buf["name"] = tc.function.name
                            yield ToolUseStart(id=buf["id"] or f"call_{idx}", name=buf["name"])
                        if tc.function.arguments:
                            buf["args"] += tc.function.arguments

            if choice.finish_reason:
                finish_reason = choice.finish_reason

        if tool_buffers:
            for idx, buf in tool_buffers.items():
                try:
                    parsed = json.loads(buf["args"]) if buf["args"] else {}
                except json.JSONDecodeError:
                    parsed = {}
                yield ToolUse(id=buf["id"] or f"call_{idx}", name=buf["name"], input=parsed)
        else:
            # Fallback: some local models (qwen, llama via Ollama) emit tool
            # calls as text in the content field instead of via the proper
            # tool_calls channel. Scan the accumulated content text for that
            # and synthesize ToolUse events.
            inline_calls = _parse_inline_tool_calls(accumulated_text, known_tool_names)
            for call in inline_calls:
                yield ToolUseStart(id=call["id"], name=call["name"])
                yield ToolUse(id=call["id"], name=call["name"], input=call["input"])
            if inline_calls:
                finish_reason = "tool_calls"

        reason = "tool_use" if finish_reason == "tool_calls" else (finish_reason or "end_turn")
        yield Stop(reason=reason, usage=usage)
