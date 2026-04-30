from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from imajin.agent.providers.base import (
    Provider,
    Stop,
    TextDelta,
    ToolUse,
    ToolUseStart,
)


@dataclass
class SubAgentResult:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: dict[str, int] = field(default_factory=dict)


_CURRENT_PROVIDER: Provider | None = None


def set_current_provider(p: Provider | None) -> None:
    global _CURRENT_PROVIDER
    _CURRENT_PROVIDER = p


def get_current_provider() -> Provider:
    if _CURRENT_PROVIDER is None:
        raise RuntimeError(
            "No active LLM provider. Specialist consult tools require an AgentRunner "
            "turn to be in progress (the runner registers its provider before tool "
            "dispatch)."
        )
    return _CURRENT_PROVIDER


class SubAgent:
    """Isolated tool-use loop. Sees only tools tagged for this specialist.

    Intentionally NOT a generator like AgentRunner — specialists are called
    synchronously from a parent tool and must return a single structured result.
    The parent can stream its own progress events; the specialist's internal
    chatter stays internal.
    """

    def __init__(
        self,
        provider: Provider,
        system_prompt: str,
        subagent_name: str,
        max_loops: int = 8,
    ) -> None:
        self.provider = provider
        self.system_prompt = system_prompt
        self.subagent_name = subagent_name
        self.max_loops = max_loops

    def run(self, user_text: str) -> SubAgentResult:
        from imajin.tools import call_tool, tools_for_anthropic

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": [{"type": "text", "text": user_text}]}
        ]
        tools_spec = tools_for_anthropic(subagent=self.subagent_name)

        accumulated_text = ""
        all_tool_calls: list[dict[str, Any]] = []
        usage: dict[str, int] = {}
        stop_reason = "end_turn"

        for _ in range(self.max_loops):
            assistant_blocks: list[dict[str, Any]] = []
            current_text = ""

            for event in self.provider.stream(messages, tools_spec, self.system_prompt):
                if isinstance(event, TextDelta):
                    current_text += event.text
                elif isinstance(event, ToolUseStart):
                    if current_text:
                        assistant_blocks.append({"type": "text", "text": current_text})
                        current_text = ""
                elif isinstance(event, ToolUse):
                    assistant_blocks.append(
                        {
                            "type": "tool_use",
                            "id": event.id,
                            "name": event.name,
                            "input": event.input,
                        }
                    )
                elif isinstance(event, Stop):
                    stop_reason = event.reason
                    if event.usage:
                        for k, v in event.usage.items():
                            usage[k] = usage.get(k, 0) + int(v)

            if current_text:
                assistant_blocks.append({"type": "text", "text": current_text})
                accumulated_text += current_text

            if assistant_blocks:
                messages.append({"role": "assistant", "content": assistant_blocks})

            if stop_reason != "tool_use":
                break

            tool_result_blocks: list[dict[str, Any]] = []
            for block in assistant_blocks:
                if block.get("type") != "tool_use":
                    continue
                from imajin.agent import provenance

                provenance.set_driver(f"specialist:{self.subagent_name}")
                try:
                    result = call_tool(block["name"], **block.get("input", {}))
                    all_tool_calls.append(
                        {"name": block["name"], "input": block["input"], "output": result, "ok": True}
                    )
                    tool_result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block["id"],
                            "content": _stringify(result),
                        }
                    )
                except Exception as e:
                    all_tool_calls.append(
                        {"name": block["name"], "input": block["input"], "output": str(e), "ok": False}
                    )
                    tool_result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block["id"],
                            "content": f"ERROR: {e}",
                            "is_error": True,
                        }
                    )

            if tool_result_blocks:
                messages.append({"role": "user", "content": tool_result_blocks})
        else:
            stop_reason = "max_loops"

        return SubAgentResult(
            text=accumulated_text,
            tool_calls=all_tool_calls,
            stop_reason=stop_reason,
            usage=usage,
        )


def _stringify(output: Any) -> str:
    import json

    try:
        return json.dumps(output, default=str)
    except TypeError:
        return str(output)
