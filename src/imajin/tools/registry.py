from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, create_model


@dataclass
class ToolEntry:
    name: str
    description: str
    func: Callable[..., Any]
    input_model: type[BaseModel]
    phase: str = ""
    vision_hint: bool = False
    subagent: str | None = None

    @property
    def json_schema(self) -> dict[str, Any]:
        return self.input_model.model_json_schema()


_REGISTRY: dict[str, ToolEntry] = {}


def _build_input_model(name: str, func: Callable[..., Any]) -> type[BaseModel]:
    sig = inspect.signature(func)
    fields: dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        if pname in {"self", "cls"}:
            continue
        annotation = (
            param.annotation if param.annotation is not inspect.Parameter.empty else Any
        )
        default = param.default if param.default is not inspect.Parameter.empty else ...
        fields[pname] = (annotation, default)
    return create_model(f"{name}Input", **fields)


def tool(
    *,
    name: str | None = None,
    description: str = "",
    phase: str = "",
    vision_hint: bool = False,
    subagent: str | None = None,
    input_model: type[BaseModel] | None = None,
) -> Callable[..., Any]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or func.__name__
        model = input_model or _build_input_model(tool_name, func)
        desc = description or (func.__doc__ or "").strip().split("\n")[0]
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            from imajin.agent.provenance import record_call

            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                inputs = dict(bound.arguments)
            except TypeError:
                inputs = {"args": args, "kwargs": kwargs}

            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                record_call(
                    tool_name, inputs, result, time.perf_counter() - t0, ok=True
                )
                return result
            except Exception as e:
                record_call(
                    tool_name, inputs, str(e), time.perf_counter() - t0, ok=False
                )
                raise

        entry = ToolEntry(
            name=tool_name,
            description=desc,
            func=wrapped,
            input_model=model,
            phase=phase,
            vision_hint=vision_hint,
            subagent=subagent,
        )
        _REGISTRY[tool_name] = entry
        wrapped.__tool_entry__ = entry  # type: ignore[attr-defined]
        return wrapped

    return decorator


def get_tool(name: str) -> ToolEntry:
    return _REGISTRY[name]


def iter_tools() -> list[ToolEntry]:
    return list(_REGISTRY.values())


def call_tool(name: str, **kwargs: Any) -> Any:
    entry = _REGISTRY[name]
    validated = entry.input_model(**kwargs)
    return entry.func(**validated.model_dump())


def _entries_for(subagent: str | None) -> list[ToolEntry]:
    return [e for e in _REGISTRY.values() if e.subagent == subagent]


def tools_for_anthropic(subagent: str | None = None) -> list[dict[str, Any]]:
    return [
        {"name": e.name, "description": e.description, "input_schema": e.json_schema}
        for e in _entries_for(subagent)
    ]


def tools_for_openai(subagent: str | None = None) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": e.name,
                "description": e.description,
                "parameters": e.json_schema,
            },
        }
        for e in _entries_for(subagent)
    ]
