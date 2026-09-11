from __future__ import annotations

import difflib
import functools
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

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
    manual: bool = True
    llm: bool = True
    worker: bool = False

    @property
    def json_schema(self) -> dict[str, Any]:
        return self.input_model.model_json_schema()


_REGISTRY: dict[str, ToolEntry] = {}


def _build_input_model(name: str, func: Callable[..., Any]) -> type[BaseModel]:
    sig = inspect.signature(func)
    try:
        type_hints = get_type_hints(func)
    except Exception:
        type_hints = {}
    fields: dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        if pname in {"self", "cls"}:
            continue
        annotation = type_hints.get(
            pname,
            param.annotation if param.annotation is not inspect.Parameter.empty else Any,
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
    manual: bool | None = None,
    llm: bool = True,
    worker: bool = False,
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
            manual=(subagent is None) if manual is None else manual,
            llm=llm,
            worker=worker,
        )
        _REGISTRY[tool_name] = entry
        wrapped.__tool_entry__ = entry  # type: ignore[attr-defined]
        return wrapped

    return decorator


class ToolNotFoundError(KeyError):
    """``get_tool``/``call_tool`` found no entry for the given name.

    Subclasses ``KeyError`` rather than a fresh ``Exception`` type on purpose:
    runner.py's vision-hint overlay check and qt_tool_runner.py's cross-thread
    dispatch both already do a defensive ``except KeyError`` around a tool
    lookup (for a name an injected or marshalled caller doesn't fully
    control) and must keep degrading gracefully instead of seeing this as an
    unrecognized type and propagating past them.
    """

    def __str__(self) -> str:
        # KeyError.__str__ reprs a lone arg -- quoting it and escaping any
        # quotes inside -- because it's built for a bare missing key, not a
        # written-out sentence; return the message verbatim instead.
        return str(self.args[0]) if self.args else super().__str__()


def _tool_not_found_message(name: str) -> str:
    """Build the error text for an unregistered tool ``name``.

    Names near matches (via difflib) so a typo or a renamed tool is
    self-correctable instead of provoking an identical retry, and says the name
    may be real but unadvertised this session (the local-model 20-tool core,
    notably) rather than misspelled or removed -- otherwise a model that knows
    the tool exists reads "unknown tool" as a transport glitch and retries the
    identical call. It does NOT send the model to get_help to find the tool:
    get_help is onboarding-scoped (imajin/tools/help.py) and returns a docs URL
    plus guide-section titles, never a tool list, so that would spend another
    turn and answer nothing.
    """
    close = difflib.get_close_matches(name, sorted(_REGISTRY), n=3)
    msg = f"Unknown tool {name!r}."
    if close:
        suggestions = ", ".join(repr(c) for c in close)
        msg += f" Did you mean: {suggestions}?"
    msg += (
        " It may be a real tool that just isn't advertised in this session"
        " (e.g. the local-model core subset) rather than misspelled or"
        " removed: only the tools in your current tool list are callable, so"
        " do not retry this name. (get_help returns the getting-started guide,"
        " not a tool list.)"
    )
    return msg


def get_tool(name: str) -> ToolEntry:
    try:
        return _REGISTRY[name]
    except KeyError:
        # from None: this *is* the KeyError (see class docstring above), so
        # chaining "during handling of the above exception" onto itself is
        # just noise for whoever reads the traceback.
        raise ToolNotFoundError(_tool_not_found_message(name)) from None


def iter_tools() -> list[ToolEntry]:
    return list(_REGISTRY.values())


def call_tool(tool_name: str, **kwargs: Any) -> Any:
    # Routed through get_tool (not a raw _REGISTRY[tool_name]) so a bad name
    # gets the same suggest-and-point-at-get_help message instead of a bare
    # KeyError -- call_tool is the dispatch path every tool_caller (the agent
    # loop, the job execution service, the MCP bridge) actually calls, so
    # this is where the model would otherwise see the unhelpful raw KeyError.
    entry = get_tool(tool_name)
    validated = entry.input_model(**kwargs)
    return entry.func(**validated.model_dump())


def _entries_for(subagent: str | None) -> list[ToolEntry]:
    return [e for e in _REGISTRY.values() if e.subagent == subagent and e.llm]


def manual_tools() -> list[ToolEntry]:
    return [e for e in _REGISTRY.values() if e.manual]


def _compact_json_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _compact_json_schema(v)
            for k, v in value.items()
            if k not in {"title"}
        }
    if isinstance(value, list):
        return [_compact_json_schema(v) for v in value]
    return value


def tools_for_anthropic(subagent: str | None = None) -> list[dict[str, Any]]:
    return [
        {
            "name": e.name,
            "description": e.description,
            "input_schema": _compact_json_schema(e.json_schema),
        }
        for e in _entries_for(subagent)
    ]


def tools_for_openai(subagent: str | None = None) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": e.name,
                "description": e.description,
                "parameters": _compact_json_schema(e.json_schema),
            },
        }
        for e in _entries_for(subagent)
    ]
