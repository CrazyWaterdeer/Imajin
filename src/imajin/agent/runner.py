from __future__ import annotations

import json
from collections.abc import Iterator
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

_MAX_TOOL_RESULT_CHARS = 6000
_MAX_STRING_CHARS = 1200
_MAX_LIST_ITEMS = 8
_MAX_DICT_ITEMS = 40
_MAX_RECENT_MESSAGES = 14
_MAX_RECENT_CHARS = 28_000
_MAX_FILE_RECORDS_IN_HISTORY = 80
_FILE_REGISTRY_TOOLS = {
    "register_files",
    "list_registered_files",
    "filter_registered_files",
    "list_experiment",
}


def _stringify_output(output: Any) -> str:
    try:
        return json.dumps(output, default=str)
    except TypeError:
        return str(output)


def _truncate_text(text: str, max_chars: int = _MAX_STRING_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [truncated {len(text) - max_chars} chars]"


def _compact_value(value: Any, depth: int = 0) -> Any:
    if isinstance(value, str):
        return _truncate_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if depth >= 4:
        return _truncate_text(str(value), 300)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        items = list(value.items())
        for key, val in items[:_MAX_DICT_ITEMS]:
            out[str(key)] = _compact_value(val, depth + 1)
        if len(items) > _MAX_DICT_ITEMS:
            out["_omitted_keys"] = len(items) - _MAX_DICT_ITEMS
        return out
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        out = [_compact_value(v, depth + 1) for v in items[:_MAX_LIST_ITEMS]]
        if len(items) > _MAX_LIST_ITEMS:
            out.append({"_omitted_items": len(items) - _MAX_LIST_ITEMS})
        return out
    return _truncate_text(str(value), 500)


def _compact_file_record(record: dict[str, Any], *, include_path: bool) -> dict[str, Any]:
    keys = (
        "file_id",
        "original_name",
        "file_type",
        "supported",
        "exists",
        "load_status",
        "path",
    )
    out: dict[str, Any] = {}
    for key in keys:
        if key == "path" and not include_path:
            continue
        if key not in record:
            continue
        value = record[key]
        out[key] = _truncate_text(str(value), 260) if isinstance(value, str) else value
    return out


def _compact_file_registry_result(tool_name: str, output: Any) -> str | None:
    if tool_name not in _FILE_REGISTRY_TOOLS:
        return None
    if not isinstance(output, dict) or not isinstance(output.get("files"), list):
        return None

    files = [rec for rec in output["files"] if isinstance(rec, dict)]
    include_path = tool_name != "register_files" and len(files) <= 30
    compact: dict[str, Any] = {}

    for key, value in output.items():
        if key in {"files", "paths"}:
            continue
        if key == "representative_file" and isinstance(value, dict):
            compact[key] = _compact_file_record(value, include_path=True)
            continue
        compact[key] = _compact_value(value)

    compact["files"] = [
        _compact_file_record(rec, include_path=include_path)
        for rec in files[:_MAX_FILE_RECORDS_IN_HISTORY]
    ]
    if len(files) > _MAX_FILE_RECORDS_IN_HISTORY:
        compact["files_omitted"] = len(files) - _MAX_FILE_RECORDS_IN_HISTORY
    if not include_path:
        compact["path_note"] = (
            "Per-file paths were omitted from this compacted history entry. "
            "Use filter_registered_files or list_registered_files with a narrower "
            "include/limit to retrieve paths for selected files."
        )

    text = _stringify_output(compact)
    if len(text) <= _MAX_TOOL_RESULT_CHARS:
        return text

    # Last-resort reduction: keep all visible file identities, but drop repeated
    # path-heavy fields so the agent can still distinguish the full file set.
    for rec in compact.get("files", []):
        if isinstance(rec, dict):
            rec.pop("path", None)
    compact.pop("paths", None)
    compact["path_note"] = (
        "Paths were omitted because the file list was long. Use "
        "filter_registered_files(include=[...], limit=...) to retrieve selected paths."
    )
    text = _stringify_output(compact)
    if len(text) <= _MAX_TOOL_RESULT_CHARS:
        return text

    compact["files"] = compact["files"][:40]
    compact["files_omitted"] = max(0, len(files) - 40)
    return _stringify_output(compact)


def _compact_tool_result(tool_name: str, output: Any) -> str:
    file_registry_result = _compact_file_registry_result(tool_name, output)
    if file_registry_result is not None:
        return file_registry_result

    compact = _compact_value(output)
    text = _stringify_output(compact)
    if len(text) <= _MAX_TOOL_RESULT_CHARS:
        return text

    fallback: dict[str, Any] = {
        "tool": tool_name,
        "result_summary": _truncate_text(text, _MAX_TOOL_RESULT_CHARS),
        "note": "Tool result was compacted to keep the conversation within context.",
    }
    if isinstance(output, dict):
        for key in (
            "path",
            "table_name",
            "labels_layer",
            "n_cells",
            "n_objects",
            "n_registered",
            "n_complete",
            "n_failed",
            "qc_png_path",
            "warnings",
        ):
            if key in output:
                fallback[key] = _compact_value(output[key])
    return _stringify_output(fallback)


def _maybe_overlay_block(
    tool_name: str, result: Any, budget: dict[str, str] | None = None
) -> dict | None:
    """Return a QC-overlay image block when a vision-hint tool produced an
    *ambiguous* ROI (Phase A, ambiguous-only gate), else ``None``.

    Gated on ``roi_confidence`` so a confident segmentation stays text-only and
    spends no image tokens; the overlay is what lets the agent judge too-wide
    vs too-narrow ROIs and decide whether to correct or escalate.
    """
    if not isinstance(result, dict):
        return None
    if result.get("roi_confidence") not in {"low", "medium"}:
        return None
    qc_png_path = result.get("qc_png_path")
    if not qc_png_path:
        return None
    # Defensive lookup (H4): an injected tool_caller may route names absent from
    # the registry; get_tool raises KeyError, which would abort the whole turn.
    from imajin.tools.registry import get_tool

    try:
        entry = get_tool(tool_name)
    except KeyError:
        return None
    if not getattr(entry, "vision_hint", False):
        return None
    # Escalation budget (E2): skip if this ROI layer was already shown at >= this
    # severity; re-attach only when confidence worsened (e.g. medium -> low), so
    # v2.1's more-frequent "medium" doesn't re-show the same overlay every turn.
    if budget is not None:
        key = str(result.get("labels_layer") or tool_name)
        conf = result.get("roi_confidence")
        rank = {"low": 0, "medium": 1}
        prev = budget.get(key)
        if prev is not None and rank.get(conf, 1) >= rank.get(prev, 1):
            return None
        budget[key] = str(conf)
    from imajin.agent.vision import overlay_image_block

    return overlay_image_block(qc_png_path)


def _tool_result_content(
    tool_name: str, result: Any, budget: dict[str, str] | None = None
) -> Any:
    """Compacted text tool-result, plus a QC overlay image block when the result
    is an ambiguous ROI from a vision-hint tool (Phase A), subject to the
    escalation budget (E2)."""
    text = _compact_tool_result(tool_name, result)
    block = _maybe_overlay_block(tool_name, result, budget)
    if block is None:
        return text
    return [{"type": "text", "text": text}, block]


_IMAGE_BLOCK_NOMINAL_CHARS = 1200


def _strip_image_data(value: Any) -> Any:
    """Replace base64 image blocks with a small placeholder for budgeting only.

    Compaction (C A.3) must count an attached QC overlay as a fixed nominal
    cost, not its full base64 length, so one overlay does not prematurely evict
    recent text context. The real image still travels to the API untouched in
    ``self.messages``; this copy is used only by :func:`_message_chars`.
    """
    if isinstance(value, dict):
        if value.get("type") == "image":
            return {"type": "image", "_nominal": "x" * _IMAGE_BLOCK_NOMINAL_CHARS}
        return {k: _strip_image_data(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_image_data(v) for v in value]
    return value


def _message_chars(message: dict[str, Any]) -> int:
    return len(_stringify_output(_strip_image_data(message)))


def _is_tool_result_message(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return (
        message.get("role") == "user"
        and isinstance(content, list)
        and bool(content)
        and all(block.get("type") == "tool_result" for block in content if isinstance(block, dict))
    )


def _summarize_messages(messages: list[dict[str, Any]]) -> str:
    lines = ["[Compacted earlier conversation]"]
    tool_counts: dict[str, int] = {}
    user_notes: list[str] = []
    assistant_notes: list[str] = []
    for message in messages:
        role = message.get("role", "?")
        content = message.get("content", [])
        if not isinstance(content, list):
            text = str(content)
            if role == "user":
                user_notes.append(_truncate_text(text, 180))
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = str(block.get("text", ""))
                if role == "user":
                    user_notes.append(_truncate_text(text, 180))
                elif role == "assistant":
                    assistant_notes.append(_truncate_text(text, 180))
            elif btype == "tool_use":
                name = str(block.get("name", "?"))
                tool_counts[name] = tool_counts.get(name, 0) + 1
            elif btype == "tool_result":
                # Tool result contents are already compacted; counting is enough here.
                tool_counts["tool_result"] = tool_counts.get("tool_result", 0) + 1
    if user_notes:
        lines.append("Recent user intents before compaction:")
        lines.extend(f"- {note}" for note in user_notes[-6:])
    if assistant_notes:
        lines.append("Assistant context before compaction:")
        lines.extend(f"- {note}" for note in assistant_notes[-4:])
    if tool_counts:
        counts = ", ".join(f"{name} x{count}" for name, count in sorted(tool_counts.items()))
        lines.append(f"Tool activity before compaction: {counts}")
    lines.append(
        "Use the current viewer and session state as authoritative; earlier raw "
        "tool outputs were omitted to stay within model context."
    )
    return "\n".join(lines)


def _context_limit_error(exc: Exception) -> bool:
    """True only when the exception is the model reporting the prompt exceeded
    its context window — the one case where force-compaction + retry is right.

    Rate-limit / quota / overload errors also mention "tokens" but are transient
    and unrelated to prompt size; compacting the conversation would silently drop
    context and mislabel the failure. They are excluded explicitly, and the
    markers are specific phrases (no bare "tokens") so a 429 never trips this.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(
        marker in text
        for marker in (
            "rate limit",
            "rate_limit",
            "quota",
            "overloaded",
            "too many requests",
        )
    ):
        return False
    markers = (
        "context length",
        "context window",
        "maximum context",
        "too many input tokens",
        "prompt is too long",
        "input length",
        "token limit",
    )
    return any(marker in text for marker in markers)


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
        # Escalation budget: last overlay confidence shown per ROI layer, so v2.1's
        # more-frequent "medium" doesn't re-attach the same overlay every turn.
        self._overlay_budget: dict[str, str] = {}
        # If unset, falls back to direct call_tool (suitable for tests/scripts).
        # In the GUI, chat dock injects a callable that marshals to the main
        # thread to avoid Qt threading violations.
        self._tool_caller = tool_caller

    def _runtime_system_prompt(self) -> str:
        from imajin.agent.qt_dispatch import call_on_main

        prompt = self.system_prompt
        # Viewer state and the batch-progress ledger are independent: a failure or
        # emptiness in one must not suppress the other, and the ledger does not need
        # a viewer. Both are rebuilt every turn from durable session state, so message
        # compaction can never erase them.
        try:
            from imajin.agent.context import summarize_viewer_state

            context = call_on_main(summarize_viewer_state)
        except Exception:
            context = ""
        if context:
            prompt = f"{prompt}\n\nCurrent session context:\n{context}"
        try:
            from imajin.agent.context import summarize_batch_progress

            ledger = call_on_main(summarize_batch_progress)
        except Exception:
            ledger = None
        if ledger:
            prompt = f"{prompt}\n\nBatch progress:\n{ledger}"
        return prompt

    def cancel(self) -> None:
        self._cancelled = True

    def reset(self) -> None:
        self.messages = []
        self._cancelled = False

    def _compact_messages(self, *, force: bool = False) -> None:
        if not self.messages:
            return
        total_chars = sum(_message_chars(m) for m in self.messages)
        if (
            not force
            and len(self.messages) <= _MAX_RECENT_MESSAGES
            and total_chars <= _MAX_RECENT_CHARS
        ):
            return

        chars = 0
        start = len(self.messages)
        while start > 0 and (len(self.messages) - start) < _MAX_RECENT_MESSAGES:
            next_chars = _message_chars(self.messages[start - 1])
            if chars + next_chars > _MAX_RECENT_CHARS and start < len(self.messages):
                break
            chars += next_chars
            start -= 1

        if start > 0 and _is_tool_result_message(self.messages[start]):
            # Keep the assistant tool_use immediately before a tool_result block
            # so provider APIs do not see orphaned tool results.
            start -= 1

        if start <= 0:
            return
        summary = _summarize_messages(self.messages[:start])
        self.messages = [
            {"role": "user", "content": [{"type": "text", "text": summary}]},
            *self.messages[start:],
        ]

    def turn(self, user_text: str) -> Iterator[RunEvent]:
        from imajin.agent.specialists.base import set_current_provider
        from imajin.tools import call_tool, tools_for_anthropic

        set_current_provider(self.provider)
        self.messages.append(
            {"role": "user", "content": [{"type": "text", "text": user_text}]}
        )
        self._compact_messages()

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

            stream_attempt = 0
            while True:
                try:
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
                    break
                except Exception as exc:  # noqa: BLE001
                    if _context_limit_error(exc) and stream_attempt == 0:
                        self._compact_messages(force=True)
                        assistant_blocks = []
                        current_text = ""
                        stop_reason = "end_turn"
                        stream_attempt += 1
                        continue
                    if _context_limit_error(exc):
                        msg = (
                            "Context limit reached even after compaction. "
                            "I compacted the conversation; please retry the last request."
                        )
                        yield TextDelta(text=msg)
                        yield TurnDone(
                            stop_reason="context_limit",
                            total_usage=total_usage,
                        )
                        return
                    raise

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
                            "content": _tool_result_content(
                                block["name"], result, self._overlay_budget
                            ),
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

            # Anthropic API requires every tool_use in an assistant message to
            # have a matching tool_result in the next user message. If dispatch
            # was cancelled or otherwise terminated early, fill placeholders for
            # the unmatched tool_use ids so the conversation stays valid.
            done_ids = {b["tool_use_id"] for b in tool_result_blocks}
            for block in assistant_blocks:
                if block.get("type") != "tool_use" or block["id"] in done_ids:
                    continue
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": "ERROR: cancelled before execution",
                        "is_error": True,
                    }
                )
                yield ToolResult(
                    tool_use_id=block["id"],
                    name=block["name"],
                    output="cancelled before execution",
                    is_error=True,
                )

            if tool_result_blocks:
                self.messages.append({"role": "user", "content": tool_result_blocks})
                self._compact_messages()

        yield TurnDone(stop_reason="max_loops", total_usage=total_usage)
