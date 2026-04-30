from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolUseStart:
    id: str
    name: str


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Stop:
    reason: str
    usage: dict[str, Any] = field(default_factory=dict)


Event = TextDelta | ToolUseStart | ToolUse | Stop


class Provider(Protocol):
    name: str
    model: str

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> Iterator[Event]: ...
