from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from imajin.agent.providers.base import (
    Event,
    Provider,
    Stop,
    TextDelta,
    ToolUse,
    ToolUseStart,
)


@dataclass
class ToolResult:
    tool_use_id: str
    name: str
    output: Any
    is_error: bool = False


@dataclass
class TurnDone:
    stop_reason: str
    total_usage: dict[str, int] = field(default_factory=dict)


RunEvent = TextDelta | ToolUseStart | ToolUse | ToolResult | TurnDone


def _stringify_output(output: Any) -> str:
    try:
        return json.dumps(output, default=str)
    except TypeError:
        return str(output)


class AgentRunner:
    def __init__(
        self,
        provider: Provider,
        system_prompt: str,
        max_loops: int = 12,
        tool_caller: Any | None = None,
    ) -> None:
        self.provider = provider
        self.system_prompt = system_prompt
        self.max_loops = max_loops
        self.messages: list[dict[str, Any]] = []
        self._cancelled = False
        # If unset, falls back to direct call_tool (suitable for tests/scripts).
        # In the GUI, chat dock injects a callable that marshals to the main
        # thread to avoid Qt threading violations.
        self._tool_caller = tool_caller

    def _runtime_system_prompt(self) -> str:
        try:
            from imajin.agent.context import summarize_viewer_state
            from imajin.agent.qt_dispatch import call_on_main

            context = call_on_main(summarize_viewer_state)
        except Exception:
            context = ""
        if not context:
            return self.system_prompt
        return f"{self.system_prompt}\n\nCurrent session context:\n{context}"

    def cancel(self) -> None:
        self._cancelled = True

    def reset(self) -> None:
        self.messages = []
        self._cancelled = False

    def turn(self, user_text: str) -> Iterator[RunEvent]:
        from imajin.agent.specialists.base import set_current_provider
        from imajin.tools import call_tool, tools_for_anthropic

        set_current_provider(self.provider)
        self.messages.append(
            {"role": "user", "content": [{"type": "text", "text": user_text}]}
        )

        tools_spec = tools_for_anthropic()
        total_usage: dict[str, int] = {}

        for _ in range(self.max_loops):
            if self._cancelled:
                yield TurnDone(stop_reason="cancelled", total_usage=total_usage)
                self._cancelled = False
                return

            assistant_blocks: list[dict[str, Any]] = []
            current_text = ""
            stop_reason = "end_turn"

            for event in self.provider.stream(
                self.messages, tools_spec, self._runtime_system_prompt()
            ):
                if self._cancelled:
                    break
                if isinstance(event, TextDelta):
                    current_text += event.text
                    yield event
                elif isinstance(event, ToolUseStart):
                    if current_text:
                        assistant_blocks.append({"type": "text", "text": current_text})
                        current_text = ""
                    yield event
                elif isinstance(event, ToolUse):
                    assistant_blocks.append(
                        {
                            "type": "tool_use",
                            "id": event.id,
                            "name": event.name,
                            "input": event.input,
                        }
                    )
                    yield event
                elif isinstance(event, Stop):
                    stop_reason = event.reason
                    if event.usage:
                        for k, v in event.usage.items():
                            total_usage[k] = total_usage.get(k, 0) + int(v)

            if self._cancelled:
                yield TurnDone(stop_reason="cancelled", total_usage=total_usage)
                self._cancelled = False
                return

            if current_text:
                assistant_blocks.append({"type": "text", "text": current_text})

            if assistant_blocks:
                self.messages.append({"role": "assistant", "content": assistant_blocks})

            if stop_reason != "tool_use":
                yield TurnDone(stop_reason=stop_reason, total_usage=total_usage)
                return

            tool_result_blocks: list[dict[str, Any]] = []
            for block in assistant_blocks:
                if block.get("type") != "tool_use":
                    continue
                if self._cancelled:
                    break
                from imajin.agent import provenance

                provenance.set_driver(f"llm:{self.provider.model}")
                tool_caller = self._tool_caller or call_tool
                try:
                    result = tool_caller(block["name"], **block.get("input", {}))
                    tool_result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block["id"],
                            "content": _stringify_output(result),
                        }
                    )
                    yield ToolResult(
                        tool_use_id=block["id"], name=block["name"], output=result
                    )
                except Exception as e:
                    tool_result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block["id"],
                            "content": f"ERROR: {e}",
                            "is_error": True,
                        }
                    )
                    yield ToolResult(
                        tool_use_id=block["id"],
                        name=block["name"],
                        output=str(e),
                        is_error=True,
                    )

            if tool_result_blocks:
                self.messages.append({"role": "user", "content": tool_result_blocks})

        yield TurnDone(stop_reason="max_loops", total_usage=total_usage)
