from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from imajin.agent.providers.base import (
    Event,
    Stop,
    TextDelta,
    ToolUse,
    ToolUseStart,
)


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

        for chunk in self._client.chat.completions.create(**kwargs):
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            if getattr(delta, "content", None):
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

        for idx, buf in tool_buffers.items():
            try:
                parsed = json.loads(buf["args"]) if buf["args"] else {}
            except json.JSONDecodeError:
                parsed = {}
            yield ToolUse(id=buf["id"] or f"call_{idx}", name=buf["name"], input=parsed)

        reason = "tool_use" if finish_reason == "tool_calls" else (finish_reason or "end_turn")
        yield Stop(reason=reason)
