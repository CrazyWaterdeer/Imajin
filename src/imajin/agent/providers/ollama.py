"""Provider that speaks Ollama's NATIVE ``/api/chat`` rather than its OpenAI-compat
``/v1/chat/completions`` endpoint.

The OpenAI-compat endpoint ignores ``num_ctx`` no matter how it is passed —
top-level, ``extra_body``, and ``extra_body.options`` all still truncate to
Ollama's 4096-token default. With Imajin's real system prompt + 104 tool
schemas (~28,300 prompt tokens) that truncation leaves the model unable to see
its own tool list: it narrates calling a tool instead of actually emitting a
tool call. Only the native endpoint honours ``{"options": {"num_ctx": N}}``, so
this provider talks to it directly; the caller sizes ``num_ctx`` up front from
:func:`imajin.agent.local_models.choose_num_ctx` and passes it in — this class
does not decide that itself, so it stays a plain, testable ``stream()``.

Message translation reuses
:func:`imajin.agent.providers.openai_compat._anthropic_to_openai_messages`
(Ollama's native chat messages are OpenAI-shaped for role/content/tool_calls,
and a tool result is `{"role": "tool", "content": ...}` the same way). Images
are the one place that translation doesn't carry over: Ollama wants a bare
``"images": [<base64>, ...]`` field on the message rather than an OpenAI
``image_url`` content block, and it rejects http(s) URLs outright, so
:func:`_attach_images` walks the *original* Anthropic-style messages after
translation and reattaches whatever QC-overlay image
(:func:`imajin.agent.runner._maybe_overlay_block`) the OpenAI-shaped
translation dropped.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

from imajin.agent.local_models import native_base_url
from imajin.agent.providers.base import (
    Event,
    Stop,
    TextDelta,
    ToolUse,
    ToolUseStart,
)
from imajin.agent.providers.openai_compat import _anthropic_to_openai_messages


def _http_error_message(exc: urllib.error.HTTPError) -> str:
    """Fold an Ollama error response body into a readable one-line message."""
    try:
        body = exc.read().decode("utf-8", "replace").strip()
    except Exception:  # noqa: BLE001 - a body we cannot read must not mask the HTTP error
        body = ""
    detail = ""
    if body:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            detail = body
        else:
            detail = str(parsed.get("error") or body) if isinstance(parsed, dict) else body
    return f"Ollama HTTP {exc.code}: {detail[:500] or exc.reason}"


def _iter_ndjson_lines(url: str, payload: dict[str, Any], timeout: float) -> Iterator[str]:
    """POST `payload` to `url`, yield each non-blank response line, decoded.

    Ollama's native /api/chat streams one JSON object per line while
    "stream": true. Iterating the urllib response object reads it as the
    server writes rather than buffering the whole (potentially long,
    tool-heavy) response first. This is the one network seam in this module —
    tests monkeypatch this function with a canned list of JSON-line strings
    instead of touching urllib or a real socket.
    """
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    # `timeout` bounds *silence* between reads, not the whole request — a cold
    # model load (60s+ before the first token) must not trip this, which is
    # why the provider default is 900s rather than a short socket timeout.
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        # Ollama puts its real diagnosis in the response *body*
        # ({"error": "model 'x' not found"}) while HTTPError.__str__ is only
        # "HTTP Error 404: Not Found". Left unread, a user who typed a model
        # name by hand into the Ollama-model setting -- the one path that
        # reaches a model never confirmed by discovery -- gets a bare status
        # code and nothing to act on. It also strands runner's
        # _context_limit_error(), which matches on message text and so could
        # never fire (and never trigger compaction) for this provider.
        raise RuntimeError(_http_error_message(exc)) from exc
    with response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if line:
                yield line


def _attach_images(
    oai_messages: list[dict[str, Any]], original_messages: list[dict[str, Any]]
) -> None:
    """Attach Anthropic-style image blocks to their translated Ollama message, in place.

    Imajin only ever attaches an image to a tool_result block — the QC-overlay
    ROI screenshot from runner._maybe_overlay_block, shaped
    ``{"type": "image", "source": {"type": "base64", "media_type": ..., "data": ...}}``
    nested inside that tool_result's `content` list — so we match on
    `tool_use_id` / `tool_call_id` rather than tracking positional
    correspondence through `_anthropic_to_openai_messages`'s fan-out (one
    original user message with N tool_results becomes N translated "tool"
    messages plus, sometimes, a separate text message).
    """
    by_tool_call_id = {m["tool_call_id"]: m for m in oai_messages if m.get("role") == "tool"}
    if not by_tool_call_id:
        return
    for message in original_messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            target = by_tool_call_id.get(block.get("tool_use_id"))
            if target is None:
                continue
            inner = block.get("content")
            if not isinstance(inner, list):
                continue
            for sub in inner:
                if not isinstance(sub, dict) or sub.get("type") != "image":
                    continue
                data = (sub.get("source") or {}).get("data")
                if data:
                    target.setdefault("images", []).append(data)


class OllamaProvider:
    """Provider protocol (see providers/base.py) speaking Ollama's NATIVE /api/chat."""

    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        num_ctx: int | None = None,
        keep_alive: str = "30m",
        max_tokens: int | None = None,
        timeout: float = 900.0,
        supports_vision: bool = True,
        think: bool | None = None,
    ) -> None:
        self.model = model
        self.base_url = native_base_url(base_url)
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.max_tokens = max_tokens
        self.timeout = timeout
        # Gates _attach_images: a model without vision support gets no
        # "images" field at all rather than one it will error or hallucinate
        # on. The caller (model picker) decides this from LocalModel.supports_vision.
        self.supports_vision = supports_vision
        # None leaves the server default alone; False sends {"think": false}.
        # Reasoning is charged against the *output* budget, not the context: on
        # qwen3.5:9b "what is 2+2" costs 106 output tokens with thinking on and 2
        # with it off, and a num_predict small enough to be exhausted mid-reasoning
        # returns done_reason="length" with content EMPTY -- a tool call would be
        # truncated to nothing. Harmless while max_tokens is None (no num_predict,
        # so no ceiling to exhaust), which is why the default keeps thinking on:
        # it measurably helps the model pick among 104 tools.
        self.think = think

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> Iterator[Event]:
        oai_messages = [
            {"role": "system", "content": system},
            *_anthropic_to_openai_messages(messages),
        ]
        if self.supports_vision:
            _attach_images(oai_messages, messages)

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

        options: dict[str, Any] = {}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if self.max_tokens is not None:
            options["num_predict"] = self.max_tokens

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "stream": True,
            "keep_alive": self.keep_alive,
        }
        if self.think is not None:
            payload["think"] = self.think
        if oai_tools:
            payload["tools"] = oai_tools
        if options:
            payload["options"] = options

        # Keyed by a synthesized/observed index rather than by list position, so
        # a tool call that (per Ollama's usual behaviour) arrives complete in one
        # chunk still works, and one that arrives fragmented across chunks
        # (partial name, then arguments later) accumulates into the same entry.
        tool_buffers: dict[Any, dict[str, Any]] = {}
        tool_order: list[Any] = []
        done_reason: str | None = None
        usage: dict[str, int] = {}

        for line in _iter_ndjson_lines(f"{self.base_url}/api/chat", payload, self.timeout):
            chunk = json.loads(line)
            # Ollama reports mid-stream failures (e.g. a bad tool schema) inside
            # an otherwise-200 response instead of an HTTP error status.
            if "error" in chunk:
                raise RuntimeError(str(chunk["error"]))

            message = chunk.get("message") or {}
            # message["thinking"] is deliberately dropped: the native API keeps
            # chain-of-thought in its own field (it never pollutes content), and
            # Imajin's RunEvent vocabulary has no thinking channel to route it to.
            content = message.get("content")
            if content:
                yield TextDelta(text=content)

            for i, call in enumerate(message.get("tool_calls") or []):
                idx = call.get("index", i)
                buf = tool_buffers.get(idx)
                if buf is None:
                    # Ollama often omits an id; synthesize one now and never
                    # revisit it, so ToolUseStart and the later ToolUse always
                    # agree even if a real id shows up on a later fragment.
                    buf = {
                        "id": call.get("id") or f"ollama_{idx}",
                        "name": None,
                        "args_str": "",
                        "args_obj": None,
                    }
                    tool_buffers[idx] = buf
                    tool_order.append(idx)

                func = call.get("function") or {}
                name = func.get("name")
                if name and not buf["name"]:
                    buf["name"] = name
                    yield ToolUseStart(id=buf["id"], name=name)

                args = func.get("arguments")
                if isinstance(args, str):
                    buf["args_str"] += args
                elif isinstance(args, dict):
                    buf["args_obj"] = args

            if chunk.get("done"):
                done_reason = chunk.get("done_reason")
                prompt_eval = chunk.get("prompt_eval_count")
                eval_count = chunk.get("eval_count")
                if isinstance(prompt_eval, int):
                    usage["input_tokens"] = prompt_eval
                if isinstance(eval_count, int):
                    usage["output_tokens"] = eval_count
                break

        truncated_args = False
        for idx in tool_order:
            buf = tool_buffers[idx]
            if buf["args_obj"] is not None:
                parsed = buf["args_obj"]
            elif buf["args_str"]:
                try:
                    parsed = json.loads(buf["args_str"])
                except json.JSONDecodeError:
                    parsed = {}
                    truncated_args = True
            else:
                parsed = {}
            yield ToolUse(id=buf["id"], name=buf["name"] or "", input=parsed)

        # A call whose arguments didn't survive the output limit must NOT be
        # reported as a clean "tool_use": running it with {} would either raise a
        # confusing validation error or, worse, execute with defaults the model
        # never chose. Reporting "length" instead hands it to the runner's
        # truncation guard, which drops the incomplete tool_use (rather than
        # committing an orphan) and tells the user the reply was cut off.
        if tool_buffers and not (done_reason == "length" and truncated_args):
            reason = "tool_use"
        else:
            reason = done_reason or "end_turn"
        yield Stop(reason=reason, usage=usage)
