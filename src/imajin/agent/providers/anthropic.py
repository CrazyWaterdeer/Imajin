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


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 4096,
    ) -> None:
        from anthropic import Anthropic

        self.model = model
        self.max_tokens = max_tokens
        self._client = Anthropic(api_key=api_key)

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> Iterator[Event]:
        cached_system = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
        cached_tools: list[dict[str, Any]] = []
        for i, t in enumerate(tools):
            entry = dict(t)
            if i == len(tools) - 1:
                entry["cache_control"] = {"type": "ephemeral"}
            cached_tools.append(entry)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": cached_system,
            "messages": messages,
        }
        if cached_tools:
            kwargs["tools"] = cached_tools

        with self._client.messages.stream(**kwargs) as stream:
            tool_inputs: dict[int, dict[str, Any]] = {}
            tool_meta: dict[int, dict[str, str]] = {}

            for event in stream:
                t = event.type
                if t == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        idx = event.index
                        tool_meta[idx] = {"id": block.id, "name": block.name}
                        tool_inputs[idx] = {"_buf": ""}
                        yield ToolUseStart(id=block.id, name=block.name)
                elif t == "content_block_delta":
                    delta = event.delta
                    dt = getattr(delta, "type", None)
                    if dt == "text_delta":
                        yield TextDelta(text=delta.text)
                    elif dt == "input_json_delta":
                        idx = event.index
                        tool_inputs.setdefault(idx, {"_buf": ""})
                        tool_inputs[idx]["_buf"] += delta.partial_json
                elif t == "content_block_stop":
                    idx = event.index
                    if idx in tool_meta:
                        meta = tool_meta[idx]
                        buf = tool_inputs[idx]["_buf"]
                        try:
                            parsed = json.loads(buf) if buf else {}
                        except json.JSONDecodeError:
                            parsed = {}
                        yield ToolUse(id=meta["id"], name=meta["name"], input=parsed)

            final = stream.get_final_message()
            usage = {}
            try:
                usage = {
                    "input_tokens": final.usage.input_tokens,
                    "output_tokens": final.usage.output_tokens,
                    "cache_read_input_tokens": getattr(
                        final.usage, "cache_read_input_tokens", 0
                    ),
                    "cache_creation_input_tokens": getattr(
                        final.usage, "cache_creation_input_tokens", 0
                    ),
                }
            except AttributeError:
                pass
            yield Stop(reason=final.stop_reason or "end_turn", usage=usage)
