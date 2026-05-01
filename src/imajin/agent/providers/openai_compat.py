from __future__ import annotations

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


_JSON_TOOLCALL_RE = re.compile(
    r"\[?\s*\{[^{}]*\"name\"\s*:\s*\"[^\"]+\"[^{}]*\"(?:arguments|parameters|input|args)\"\s*:",
    re.DOTALL,
)


def _parse_inline_tool_calls(text: str, known_tool_names: set[str]) -> list[dict[str, Any]]:
    """Extract tool calls that some local models (qwen, llama) emit as JSON in
    the content field instead of via the proper `tool_calls` field.

    Recognizes shapes like:
        [{"name": "foo", "arguments": {...}}]
        {"name": "foo", "arguments": {...}}
        ```json {"name": "foo", "parameters": {...}} ```
    Returns a list of {"name": str, "input": dict, "id": str} dicts.
    """
    if not text or not _JSON_TOOLCALL_RE.search(text):
        return []
    # Strip code fences if present.
    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    # Try direct parse, then try to slice the first JSON object/array out.
    obj: Any = None
    for parser_input in (candidate, _slice_first_json(candidate)):
        if parser_input is None:
            continue
        try:
            obj = json.loads(parser_input)
            break
        except json.JSONDecodeError:
            continue
    if obj is None:
        return []
    items = obj if isinstance(obj, list) else [obj]
    out: list[dict[str, Any]] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        name = it.get("name") or (it.get("function") or {}).get("name")
        if not isinstance(name, str) or name not in known_tool_names:
            continue
        args = (
            it.get("arguments")
            or it.get("parameters")
            or it.get("input")
            or it.get("args")
            or (it.get("function") or {}).get("arguments")
            or {}
        )
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        out.append({"id": f"inline_{i}", "name": name, "input": args})
    return out


def _slice_first_json(text: str) -> str | None:
    """Return the first balanced JSON array/object substring, or None."""
    start = -1
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
                return text[start : i + 1]
    return None


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


class OpenAICompatProvider:
    name = "openai-compat"

    def __init__(
        self,
        api_key: str | None,
        model: str = "gpt-5",
        base_url: str = "https://api.openai.com/v1",
        max_tokens: int = 4096,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.max_tokens = max_tokens
        self.base_url = base_url
        self._client = OpenAI(api_key=api_key or "ollama", base_url=base_url)

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> Iterator[Event]:
        oai_messages = [{"role": "system", "content": system}, *_anthropic_to_openai_messages(messages)]
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

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if oai_tools:
            kwargs["tools"] = oai_tools

        tool_buffers: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        accumulated_text = ""
        known_tool_names = {t["name"] for t in tools}

        for chunk in self._client.chat.completions.create(**kwargs):
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
            # calls as JSON inside the content field instead of via the proper
            # tool_calls channel. Scan the accumulated content text for that
            # pattern and synthesize ToolUse events.
            inline_calls = _parse_inline_tool_calls(accumulated_text, known_tool_names)
            for call in inline_calls:
                yield ToolUseStart(id=call["id"], name=call["name"])
                yield ToolUse(id=call["id"], name=call["name"], input=call["input"])
            if inline_calls:
                finish_reason = "tool_calls"

        reason = "tool_use" if finish_reason == "tool_calls" else (finish_reason or "end_turn")
        yield Stop(reason=reason)
