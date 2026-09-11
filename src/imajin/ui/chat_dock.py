from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from qtpy.QtCore import QEvent, Qt, Signal
from qtpy.QtGui import QInputMethodEvent, QKeyEvent, QWheelEvent
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from imajin.agent.execution import get_execution_service
from imajin.agent.local_models import (
    LocalModel,
    choose_num_ctx,
    discover_ollama_models,
    estimate_prompt_tokens,
)
from imajin.agent.providers.base import Event
from imajin.agent.qt_tool_runner import MainThreadToolRunner
from imajin.agent.tool_subset import CORE_TOOL_NAMES, core_tools, missing_core_names
from imajin.ui.chat_transcript import ChatTranscript
from imajin.ui.provider_status import ProviderStatus, compute_statuses
from imajin.ui.theme import apply_dock_theme


def _codex_default_model() -> str:
    """Best-effort read of codex's own model catalog for the picker's default slug.

    codex has no "sonnet"/"opus"-style stable alias (unlike the claude-agent
    rows below) -- every slug it accepts for `-m` is itself a dated/versioned
    name that rotates over time (this project's own memory already records
    one such rotation: bare "sol" became "gpt-5.6-sol"). Reading the live
    catalog here instead of hardcoding a slug is the same reason local Ollama
    rows come from live discovery rather than a fixed name -- see the comment
    below _MODEL_CHOICES.

    ~/.codex/models_cache.json ($CODEX_HOME/models_cache.json when that's
    set -- the same resolution codex_agent.codex_available() uses for
    auth.json) is codex's own cache of the models it would offer an
    interactive picker, refreshed by `codex update` / ordinary use. This
    picks the lowest-`priority` entry marked `visibility: "list"` (codex's
    own "show this, ranked" flag) -- the same one codex itself would default
    an interactive picker to.

    Runs once at import time, so it must never raise: a missing
    CODEX_HOME/file (not logged in yet, or logged in but never run anything
    that populated the cache), malformed JSON, or an unexpected shape all
    fall through to "" so the caller can supply a verified fallback instead.
    """
    codex_home = os.environ.get("CODEX_HOME")
    cache_path = (
        Path(codex_home) if codex_home else Path.home() / ".codex"
    ) / "models_cache.json"
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        listed = [
            m
            for m in data["models"]
            if isinstance(m, dict) and m.get("visibility") == "list" and m.get("slug")
        ]
        if not listed:
            return ""
        listed.sort(key=lambda m: m.get("priority", float("inf")))
        return str(listed[0]["slug"])
    except (OSError, ValueError, KeyError, TypeError):
        return ""


# Verified against the live cache on the dev machine (2026-09-11, codex-cli
# 0.144.3): the lowest-`priority` `visibility: "list"` entry was
# "gpt-5.6-sol" -- the exact slug the codex-agent runner's own probe evidence
# verified working end-to-end (see codex_agent.py's _build_argv /
# _MODEL_REASONING_EFFORT). Used only when _codex_default_model() can't read
# the user's own cache; deliberately NOT the bare "" this slice's brief
# allows as a fallback, because CodexAgentRunner._build_argv passes
# `-m self.model` unconditionally with no empty-string case (traced in
# codex_agent.py, which this slice does not own and so cannot change) -- an
# empty model would reach codex as a literal `-m ""` on every turn instead of
# letting codex fall back to its own default.
_CODEX_FALLBACK_MODEL = "gpt-5.6-sol"

# The model field for API-backed Claude and OpenAI is a *tier* token, not a pinned
# id: the provider resolves it to the latest concrete model at connection time (see
# imajin.agent.model_catalog). Subscription entries use the CLI's own always-latest
# aliases. Codex (subscription) has no such alias -- see _codex_default_model()
# above. Local (Ollama) rows are deliberately NOT listed here -- a hardcoded
# model name would go stale the moment the user pulls a different one, so
# ChatDock._build_model_choices appends those from live discovery instead.
_MODEL_CHOICES: list[tuple[str, str, str]] = [
    ("Claude Sonnet (API, latest)", "anthropic", "sonnet"),
    ("Claude Opus (API, latest)", "anthropic", "opus"),
    ("Claude Sonnet (subscription)", "claude-agent", "sonnet"),
    ("Claude Opus (subscription)", "claude-agent", "opus"),
    ("GPT (OpenAI, latest)", "openai", "gpt"),
    ("Codex (subscription)", "codex-agent", _codex_default_model() or _CODEX_FALLBACK_MODEL),
]


def _format_context_length(n: int) -> str:
    """Render token count the way Ollama model cards do, e.g. 262144 -> "256K",
    rather than showing the raw six-digit number."""
    return f"{round(n / 1024)}K"


def _local_model_label(model: LocalModel) -> str:
    details = []
    if model.parameter_size:
        details.append(model.parameter_size)
    if model.context_length:
        details.append(_format_context_length(model.context_length))
    suffix = f" ({', '.join(details)})" if details else ""
    return f"Local: {model.name}{suffix}"


def _fallback_local_choice(settings: Any) -> tuple[str, str, str]:
    """One row to show when discover_ollama_models finds no tool-capable model
    (daemon offline, or nothing pulled yet) -- an empty picker looks broken, a
    row naming the configured fallback (or saying none is configured) reads as
    the true state. It only becomes *selectable* once compute_statuses reports
    Ollama available again (see provider_status.probe_ollama), so this can never
    hand the user a model that was never actually confirmed to exist.
    """
    model = settings.ollama_model.strip()
    if model:
        return (f"Local: {model} (unconfirmed)", "ollama", model)
    return ("Local: none configured", "ollama", "")


# What to actually DO about an unavailable backend. The subscription backends
# ship no in-app login on purpose (Imajin never handles those credentials), so
# for them the terminal is the only route — pointing the user at the API Keys
# dialog would be advice that cannot work.
_FIX_HINTS: dict[str, str] = {
    "claude-agent": "Run `claude` in a terminal and sign in.",
    "codex-agent": "Run `codex login` in a terminal.",
    "ollama": "Start Ollama, then `ollama pull` a tool-capable model.",
}
_DEFAULT_FIX_HINT = "Open Imajin → API Keys…"


def _fix_hint(kind: str) -> str:
    return _FIX_HINTS.get(kind, _DEFAULT_FIX_HINT)


def _short_label(label: str) -> str:
    short = label.replace("Claude ", "").replace(" (OpenAI)", "").replace("Local: ", "")
    if len(short) > 26:
        short = short[:24] + "…"
    return short


# A picker kind with no entry in `statuses` means whoever added it to
# _MODEL_CHOICES/_build_model_choices forgot to add a matching
# compute_statuses() branch -- that used to read as "available" here (a bare
# `.get(kind)` returning None was treated the same as a green light in half a
# dozen places below), which is exactly how a backend with no working status
# check could still land in the menu, selected by default, with no warning in
# its label. Every lookup now goes through _status_for() instead of a bare
# `self._statuses.get(kind)`, so that mistake reads as "unavailable" -- with a
# reason that says why -- everywhere at once.
_UNREGISTERED_STATUS = ProviderStatus(available=False, reason="not registered")


class _ModelPickerButton(QPushButton):
    """Pill-shaped button that opens a menu of model choices."""

    currentIndexChanged = Signal(int)  # any change, including auto-fallback
    userSelected = Signal(int)  # only a click in the menu (worth persisting)

    def __init__(
        self,
        choices: list[tuple[str, str, str]],
        statuses: dict[str, ProviderStatus] | None = None,
        preferred: tuple[str, str] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("modelBtn")
        self._choices = choices
        self._statuses: dict[str, ProviderStatus] = statuses or {}
        # Restore the last-used (provider, model) choice when it's available;
        # otherwise fall back to the first available choice.
        self._index = self._resolve_initial_index(preferred)
        self._build_menu()
        self._refresh_text()

    def _status_for(self, kind: str) -> ProviderStatus:
        return self._statuses.get(kind, _UNREGISTERED_STATUS)

    def _resolve_initial_index(self, preferred: tuple[str, str] | None) -> int:
        if preferred is not None:
            pref_kind, pref_model = preferred
            for i, (_, kind, model) in enumerate(self._choices):
                if kind == pref_kind and model == pref_model:
                    if self._status_for(kind).available:
                        return i
                    break  # preferred choice exists but is currently unavailable
        return self._first_available_index()

    def _first_available_index(self) -> int:
        for i, (_, kind, _) in enumerate(self._choices):
            if self._status_for(kind).available:
                return i
        return 0

    def _build_menu(self) -> None:
        menu = QMenu(self)
        last_kind: str | None = None
        for i, (label, kind, _) in enumerate(self._choices):
            if last_kind is not None and kind != last_kind:
                menu.addSeparator()
            status = self._status_for(kind)
            if not status.available:
                action = menu.addAction(f"{label} — {status.reason}")
                action.setEnabled(False)
                action.setToolTip(f"Unavailable: {status.reason}. {_fix_hint(kind)}")
            else:
                action = menu.addAction(label)
                action.triggered.connect(
                    lambda _checked=False, idx=i: self.setCurrentIndex(idx)
                )
            last_kind = kind
        self.setMenu(menu)

    def refresh_statuses(self, statuses: dict[str, ProviderStatus]) -> None:
        self._statuses = statuses
        # If the current selection went unavailable, switch to first available.
        kind = self._choices[self._index][1]
        if not self._status_for(kind).available:
            new_idx = self._first_available_index()
            if new_idx != self._index:
                self._index = new_idx
                self.currentIndexChanged.emit(new_idx)
        self._build_menu()
        self._refresh_text()

    def set_choices(
        self, choices: list[tuple[str, str, str]], statuses: dict[str, ProviderStatus]
    ) -> None:
        """Replace the choice list in place (e.g. newly discovered local models).

        Keeps the same (kind, model) selected when it is still in the new list
        *and* still available; otherwise falls back to the first available
        choice, same policy as refresh_statuses for a pure status change. This
        is a rebuild, not a click, so userSelected never fires -- but
        currentIndexChanged always does when the index actually moves, exactly
        once: ChatDock.invalidate_runner relies on that signal to release a
        runner that's pointed at a model no longer in the list (e.g. the user
        deleted a pulled model between turns).
        """
        start_index = self._index
        _, kind, model = self._choices[start_index]
        self._choices = choices
        self._statuses = statuses

        found = next(
            (i for i, (_, k, m) in enumerate(self._choices) if k == kind and m == model),
            None,
        )
        if found is not None and self._status_for(kind).available:
            self._index = found
        else:
            self._index = self._first_available_index()

        self._build_menu()
        self._refresh_text()
        if self._index != start_index:
            self.currentIndexChanged.emit(self._index)

    def current_status(self) -> ProviderStatus:
        kind = self._choices[self._index][1]
        return self._status_for(kind)

    def setCurrentIndex(self, idx: int) -> None:
        if idx == self._index:
            return
        self._index = idx
        self._refresh_text()
        # Only menu clicks reach setCurrentIndex, so this is a real user selection
        # to persist. The refresh_statuses auto-fallback sets _index directly and
        # emits only currentIndexChanged, so it never lands here.
        self.userSelected.emit(idx)
        self.currentIndexChanged.emit(idx)

    def currentIndex(self) -> int:
        return self._index

    def count(self) -> int:
        return len(self._choices)

    def itemText(self, idx: int) -> str:
        return self._choices[idx][0]

    def _refresh_text(self) -> None:
        label = self._choices[self._index][0]
        status = self.current_status()
        suffix = "  ▾"
        if not status.available:
            self.setText(f"{_short_label(label)} ({status.reason}){suffix}")
        else:
            self.setText(f"{_short_label(label)}{suffix}")


class _ComposerInput(QPlainTextEdit):
    submitted = Signal()

    def __init__(self, min_visible_lines: int = 2, max_visible_lines: int = 4) -> None:
        super().__init__()
        self._min_visible_lines = min_visible_lines
        self._max_visible_lines = max_visible_lines
        self._frame_padding = 14
        self._has_preedit = False
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, True)
        self.setInputMethodHints(Qt.InputMethodHint.ImhNone)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setFixedHeight(52)  # provisional; refined once the font is realized
        self.document().contentsChanged.connect(self._adjust_height)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._adjust_height()

    def _adjust_height(self) -> None:
        line_h = self.fontMetrics().lineSpacing()
        if line_h <= 0:
            return
        blocks = max(self._min_visible_lines, self.document().blockCount())
        visible = min(blocks, self._max_visible_lines)
        target = visible * line_h + self._frame_padding
        self.setFixedHeight(target)

    def inputMethodEvent(self, event: QInputMethodEvent) -> None:
        self._has_preedit = bool(event.preeditString())
        super().inputMethodEvent(event)
        if event.commitString():
            self._has_preedit = False

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ShortcutOverride and self.hasFocus():
            event.accept()
            return False
        return super().event(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        scrollbar = self.verticalScrollBar()
        if scrollbar is not None and scrollbar.maximum() > 0:
            super().wheelEvent(event)
            event.accept()
            return
        event.ignore()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._has_preedit:
                super().keyPressEvent(event)
                return
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)
            else:
                self.submitted.emit()
            return
        super().keyPressEvent(event)


def _local_system_prompt(available_tools: set[str]) -> str:
    """build_system_prompt(available_tools=...), so the model's own instructions
    never point it at a tool outside the core set it was actually given.

    Called directly rather than through a capability probe: an earlier
    inspect.signature() shim guarded against prompts.py's half of this change
    landing later, but both halves are in the tree now, and that shim failed
    OPEN -- a rename would have silently reverted the local path to the full
    33k-char prompt naming 87 uncallable tools, with no error. A TypeError on
    a rename is the better failure, and the tests call this kwarg by keyword.
    """
    from imajin.agent.prompts import build_system_prompt

    return build_system_prompt(available_tools=available_tools)


class _OllamaCoreToolsProvider:
    """Wraps a local-model Provider so its turns advertise core_tools(...)
    instead of the full registry.

    Why here, rather than on OllamaProvider itself or on AgentRunner:
    AgentRunner.turn() always calls the module-level tools_for_anthropic()
    fresh and hands the FULL list straight to provider.stream() (see
    runner.py) -- _make_provider building a plain OllamaProvider does not
    change that, since nothing about a provider's construction narrows what a
    *later* turn sends it. Trimming has to happen where stream() actually
    receives the tools, which is here -- the one seam _ensure_runner controls
    without editing runner.py or providers/ollama.py.

    Filtering is gated on "this tools list IS the full top-level registry" (by
    exact name-set), not applied to every stream() call: a specialist consult
    mid-turn (imajin.tools.specialists) fetches this SAME provider instance
    back via get_current_provider() and calls .stream() again with its own,
    disjoint, subagent-scoped tool list (SubAgent.run(), specialists/base.py).
    Unconditional filtering would run core_tools() on THAT list too -- and
    since none of a specialist's tools are in CORE_TOOL_NAMES, it would come
    back empty, silently breaking specialist consults for local models. The
    top-level list's exact name-set is captured once, at construction, and is
    the only shape of `tools` that ever gets narrowed; anything else (a
    specialist's own list) passes straight through unchanged.
    """

    def __init__(self, inner: Any, full_tool_names: frozenset[str]) -> None:
        self._inner = inner
        self._full_tool_names = full_tool_names
        self.name = inner.name
        self.model = inner.model

    def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], system: str
    ) -> Iterator[Event]:
        names = frozenset(t.get("name") for t in tools)
        if names == self._full_tool_names:
            tools = core_tools(tools)
        yield from self._inner.stream(messages, tools, system)


class ChatDock(QWidget):
    _job_updated = Signal(object)

    def __init__(self, viewer: Any, settings: Any) -> None:
        super().__init__()
        apply_dock_theme(self)
        self.viewer = viewer
        self.settings = settings
        self._runner = None
        self._worker = None
        self._provider_kind = None
        self._provider_model = None
        # Lives on the main (Qt) thread; tool calls from worker are routed
        # through this so napari Layer creation stays on the main thread.
        self._tool_runner = MainThreadToolRunner(parent=self)
        self.execution_service = get_execution_service()
        self._job_listener = lambda job: self._job_updated.emit(job)
        self._job_updated.connect(self._on_job_update)
        self.execution_service.add_listener(self._job_listener)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.transcript = ChatTranscript()
        layout.addWidget(self.transcript, stretch=1)

        composer = QFrame()
        composer.setObjectName("composer")
        composer_layout = QVBoxLayout(composer)
        composer_layout.setContentsMargins(8, 6, 8, 6)
        composer_layout.setSpacing(4)

        self.input = _ComposerInput(max_visible_lines=4)
        self.input.setPlaceholderText("Type a request…   (Shift+Enter = newline)")
        self.input.submitted.connect(self._on_send)
        composer_layout.addWidget(self.input)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(6)

        self.model_choices = self._build_model_choices()
        statuses = compute_statuses(self.settings)
        preferred = (self.settings.default_provider, self.settings.default_model)
        self.model_picker = _ModelPickerButton(
            self.model_choices, statuses=statuses, preferred=preferred
        )
        self.model_picker.currentIndexChanged.connect(self._on_model_change)
        self.model_picker.userSelected.connect(self._on_user_model_select)
        toolbar.addWidget(self.model_picker)

        toolbar.addStretch(1)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("composerTool")
        self.clear_btn.setToolTip("Reset conversation history")
        self.clear_btn.clicked.connect(self._on_clear)
        toolbar.addWidget(self.clear_btn)

        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("sendBtn")
        self.send_btn.setToolTip("Send (Enter)")
        self.send_btn.clicked.connect(self._on_send)
        toolbar.addWidget(self.send_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setToolTip("Cancel the running turn")
        self.stop_btn.clicked.connect(self._on_cancel)
        self.stop_btn.hide()
        toolbar.addWidget(self.stop_btn)

        self.cancel_btn = self.stop_btn  # backward-compat alias

        composer_layout.addLayout(toolbar)
        layout.addWidget(composer)
        self.input.setFocus(Qt.FocusReason.OtherFocusReason)

    def _release_runner(self) -> None:
        # Discard the current runner, closing it first if it holds resources (the
        # subscription runner owns a background loop + `claude` subprocess).
        runner, self._runner = self._runner, None
        close = getattr(runner, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass

    def invalidate_runner(self) -> None:
        self._release_runner()
        self._provider_kind = None
        self._provider_model = None
        # Re-probe in case API keys were just edited or Ollama just came up, and
        # rebuild the local rows so a model pulled (or un-pulled) since the dock
        # was created shows up without restarting Imajin.
        self.model_choices = self._build_model_choices()
        self.model_picker.set_choices(self.model_choices, compute_statuses(self.settings))

    def closeEvent(self, event) -> None:
        # Both teardowns must run on close: stop receiving job callbacks, then
        # release the runner/worker. (These lived in two separate closeEvent
        # overrides before — the second silently shadowed the first, so the
        # runner was never released on close.)
        self.execution_service.remove_listener(self._job_listener)
        self._release_runner()
        super().closeEvent(event)

    def _build_model_choices(self) -> list[tuple[str, str, str]]:
        """Cloud rows are static (_MODEL_CHOICES); local rows come from live
        Ollama discovery, filtered to models that actually advertise "tools" --
        Imajin always needs tool calling, so a vision- or text-only model would
        just be a picker entry that fails on the first turn. self._local_models
        is rebuilt alongside the row list so _make_provider can look up each
        local row's real LocalModel (context length, capabilities) by name; the
        row tuple itself only carries the (label, kind, model) every kind shares.
        """
        models = [
            m
            for m in discover_ollama_models(self.settings.ollama_base_url)
            if m.supports_tools
        ]
        self._local_models = {m.name: m for m in models}
        if models:
            local_rows = [(_local_model_label(m), "ollama", m.name) for m in models]
        else:
            local_rows = [_fallback_local_choice(self.settings)]
        return [*_MODEL_CHOICES, *local_rows]

    def _make_provider(self):
        from imajin.agent.providers import (
            AnthropicProvider,
            OllamaProvider,
            OpenAICompatProvider,
        )
        from imajin.tools import tools_for_anthropic

        idx = self.model_picker.currentIndex()
        _, kind, model = self.model_choices[idx]
        if kind == "anthropic":
            if not self.settings.anthropic_api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY not set. Open Imajin → API Keys… or set the env var."
                )
            return AnthropicProvider(
                api_key=self.settings.anthropic_api_key, model=model
            )
        if kind == "openai":
            if not self.settings.openai_api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not set. Open Imajin → API Keys… or set the env var."
                )
            return OpenAICompatProvider(
                api_key=self.settings.openai_api_key,
                model=model,
                base_url=self.settings.openai_base_url,
            )

        # "ollama" -- native /api/chat, never the OpenAI-compat endpoint: the
        # compat endpoint ignores num_ctx in every form tested and silently
        # truncates to Ollama's 4096-token default, which with the full 104-tool
        # prompt (~28.3K tokens) leaves the model narrating a tool call instead
        # of emitting one (see local_models.py's module docstring for the
        # measured numbers). choose_num_ctx sizes the window from the model's
        # real context_length when discovery found one; when it has to clamp
        # below what the prompt actually needs, that's real, user-visible
        # truncation, so it's announced in the transcript here rather than left
        # to surface later as a model that "forgot" its tools mid-turn.
        local_model = self._local_models.get(model)
        if local_model is None:
            # Unconfirmed fallback row (settings.ollama_model; discovery found
            # nothing) -- no known context_length to clamp to, and no confirmed
            # capabilities, so default vision off rather than trust a guess.
            local_model = LocalModel(
                name=model,
                context_length=None,
                capabilities=frozenset(),
                parameter_size=None,
                size_bytes=None,
            )
        # Advertise the 20-tool core, not the full registry: with all 107
        # tools offered, qwen3.5:4b never terminated a multi-step task (see
        # tool_subset.py for the measured before/after). Recomputing the
        # estimate from the SUBSET -- not the full list -- is what actually
        # lets num_ctx reflect the smaller prompt; estimating from the full
        # list here while only ever sending the subset (see
        # _OllamaCoreToolsProvider, wired in by _ensure_runner) would silently
        # throw the saving away.
        full_tools_spec = tools_for_anthropic()
        tools_spec = core_tools(full_tools_spec)
        missing = missing_core_names(full_tools_spec)
        if missing:
            # A core tool was renamed/removed without updating tool_subset.py:
            # local models now see fewer than 20 tools. Loud on purpose --
            # silent capability loss is worse than a visible warning.
            self._append_system(
                f"[warning] {len(missing)} core tool(s) not found in the "
                f"registry: {', '.join(missing)}. Local models will see "
                f"{len(tools_spec)} tools instead of {len(CORE_TOOL_NAMES)}."
            )
        # Says how to reach the other 87 tools, not "call get_help" -- get_help
        # is onboarding-scoped and returns a docs link, so pointing the user
        # there for a missing capability sends them somewhere that cannot answer.
        self._append_system(
            f"[info] Local models see a {len(tools_spec)}-tool core set "
            f"({len(tools_spec)} of {len(full_tools_spec)} available) so a small "
            "model decides instead of dithering; switch to a cloud model for the "
            "full set."
        )
        available_tool_names = {t["name"] for t in tools_spec}
        est = estimate_prompt_tokens(
            _local_system_prompt(available_tool_names), tools_spec, []
        )
        # floor=16384 -- was the implicit 32768 default, which is what would
        # re-inflate a legitimately smaller subset-driven estimate right back
        # up to the window a 107-tool prompt needed, throwing the saving away
        # instead of letting it show up as a smaller num_ctx / less VRAM.
        num_ctx = choose_num_ctx(local_model, est, floor=16384)
        if num_ctx < est:
            self._append_system(
                f"[warning] {model}'s context window ({num_ctx} tokens) is "
                f"smaller than the system prompt + {len(tools_spec)} tools "
                f"(~{est} tokens). Requests will be truncated and the model may "
                "stop seeing some or all of its tools."
            )
        return OllamaProvider(
            model=model,
            base_url=self.settings.ollama_base_url,
            num_ctx=num_ctx,
            supports_vision=local_model.supports_vision,
        )

    def _ensure_runner(self):
        from imajin.agent.prompts import build_system_prompt

        idx = self.model_picker.currentIndex()
        _, kind, model = self.model_choices[idx]
        if (
            self._runner is not None
            and self._provider_kind == kind
            and self._provider_model == model
        ):
            return self._runner

        def call_tool_via_jobs(tool_name: str, **kwargs: Any) -> Any:
            return self.execution_service.call_tool_blocking(
                tool_name,
                kwargs=kwargs,
                source="llm",
                driver=f"llm:{model}",
                title=tool_name,
                tool_caller=self._tool_runner.call,
            )

        if kind == "claude-agent":
            # Subscription-backed: the Claude Code agent owns its own loop, so this
            # is a ClaudeAgentRunner (not a Provider behind AgentRunner). It presents
            # the same turn()/reset()/cancel() surface, so everything downstream is
            # unchanged. No API key — it uses the local `claude` login.
            from imajin.agent.providers.claude_agent import ClaudeAgentRunner

            self._runner = ClaudeAgentRunner(
                model=model,
                system_prompt=build_system_prompt(),
                tool_caller=call_tool_via_jobs,
            )
        elif kind == "codex-agent":
            # Subscription-backed like claude-agent above: codex owns its own
            # agentic loop (it drives the `codex` CLI, which calls back into
            # Imajin's tools over an in-process MCP bridge), so this is a
            # CodexAgentRunner, not a Provider behind AgentRunner. Same
            # turn()/reset()/cancel()/close() surface, so everything
            # downstream is unchanged. No API key -- it uses whatever the
            # user set up themselves with `codex login`; see
            # imajin.agent.providers.codex_agent's module docstring for the
            # ToS/auth decision this mirrors from ClaudeAgentRunner.
            from imajin.agent.providers.codex_agent import CodexAgentRunner

            self._runner = CodexAgentRunner(
                model=model,
                system_prompt=build_system_prompt(),
                tool_caller=call_tool_via_jobs,
            )
        elif kind == "ollama":
            # Local path only (SCOPE DISCIPLINE: anthropic/openai below keep
            # seeing the full registry via plain AgentRunner). The provider
            # gets wrapped -- not AgentRunner itself -- so runner.py and
            # providers/ollama.py stay untouched; see _OllamaCoreToolsProvider
            # for why that's also the *safe* seam (a filter living on
            # AgentRunner or applied unconditionally in the provider would
            # also reach specialist consults, which reuse this same provider
            # instance with their own, disjoint tool list).
            from imajin.agent.runner import AgentRunner
            from imajin.tools import tools_for_anthropic

            full_tools_spec = tools_for_anthropic()
            subset_tools_spec = core_tools(full_tools_spec)
            available_tool_names = {t["name"] for t in subset_tools_spec}
            provider = _OllamaCoreToolsProvider(
                self._make_provider(), frozenset(t["name"] for t in full_tools_spec)
            )
            self._runner = AgentRunner(
                provider,
                _local_system_prompt(available_tool_names),
                tool_caller=call_tool_via_jobs,
            )
        else:
            from imajin.agent.runner import AgentRunner

            provider = self._make_provider()
            self._runner = AgentRunner(
                provider,
                build_system_prompt(),
                tool_caller=call_tool_via_jobs,
            )
        self._provider_kind = kind
        self._provider_model = model
        return self._runner

    def _on_model_change(self, _index: int) -> None:
        if self._runner is not None:
            self._append_system("Model changed — conversation reset.")
        self._release_runner()

    def _on_user_model_select(self, index: int) -> None:
        # Persist the user's pick so the next launch restores it. Best-effort: a
        # write failure (e.g. read-only config dir) must not break model switching.
        _, kind, model = self.model_choices[index]
        self.settings.default_provider = kind
        self.settings.default_model = model
        try:
            self.settings.save_secrets()
        except Exception:  # noqa: BLE001 - preference save is non-critical
            pass

    def _on_clear(self) -> None:
        if self._runner is not None:
            self._runner.reset()
        self.transcript.clear()

    def _on_send(self) -> None:
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.clear()

        try:
            runner = self._ensure_runner()
        except Exception as e:
            self._append_system(f"[error] {e}")
            return

        self._append_user(text)
        self._begin_assistant_turn()

        from napari.qt import thread_worker

        @thread_worker
        def _do_turn():
            yield from runner.turn(text)

        worker = _do_turn()
        worker.yielded.connect(self._on_event)
        worker.finished.connect(self._on_finished)
        worker.errored.connect(self._on_errored)
        self._worker = worker
        self._set_streaming(True)
        worker.start()

    def _on_cancel(self) -> None:
        if self._runner is not None:
            self._runner.cancel()
        self.execution_service.cancel_running(source="llm")
        if self._worker is not None:
            try:
                self._worker.quit()
            except Exception:
                pass

    def _on_finished(self) -> None:
        self._set_streaming(False)
        self._worker = None

    def _on_errored(self, exc: Exception) -> None:
        self._append_system(f"[runner error] {type(exc).__name__}: {exc}")
        self._on_finished()

    def _set_streaming(self, streaming: bool) -> None:
        self.send_btn.setVisible(not streaming)
        self.stop_btn.setVisible(streaming)

    def _on_event(self, event: Any) -> None:
        from imajin.agent.providers.base import (
            Stop,
            TextDelta,
            ToolUse,
            ToolUseStart,
        )
        from imajin.agent.runner import ToolResult, TurnDone

        if isinstance(event, TextDelta):
            self._append_text_delta(event.text)
        elif isinstance(event, ToolUseStart):
            self._append_system(f"→ {event.name}…")
        elif isinstance(event, ToolUse):
            args = json.dumps(event.input, ensure_ascii=False, default=str)
            if len(args) > 200:
                args = args[:200] + "…"
            self._append_system(f"   args: {args}")
        elif isinstance(event, ToolResult):
            tag = "ERROR" if event.is_error else "ok"
            out = repr(event.output)
            if len(out) > 240:
                out = out[:240] + "…"
            self._append_system(f"   ← {tag}: {out}")
        elif isinstance(event, Stop):
            pass
        elif isinstance(event, TurnDone):
            usage = event.total_usage
            usage_str = ""
            if usage:
                inp = usage.get("input_tokens", 0)
                out = usage.get("output_tokens", 0)
                cache_r = usage.get("cache_read_input_tokens", 0)
                usage_str = f" — tokens: in {inp} (cache_read {cache_r}), out {out}"
            self._append_system(f"[turn complete: {event.stop_reason}]{usage_str}")

    def _on_job_update(self, job: Any) -> None:
        if not self._should_show_job_progress(job):
            return
        self.transcript.upsert_system(
            f"job-progress:{job.job_id}",
            self._format_job_progress(job),
        )

    def _should_show_job_progress(self, job: Any) -> bool:
        detail = dict(getattr(job, "progress_detail", {}) or {})
        return bool(detail.get("show_in_chat"))

    def _format_job_progress(self, job: Any) -> str:
        detail = dict(getattr(job, "progress_detail", {}) or {})
        total = _as_int(detail.get("total_files"))
        completed = _as_int(detail.get("completed")) or 0
        failed = _as_int(detail.get("failed")) or 0
        skipped = _as_int(detail.get("skipped")) or 0
        processed = completed + failed + skipped
        stage = str(detail.get("stage") or getattr(job, "status", "running"))
        pct = ""
        if getattr(job, "progress", None) is not None:
            pct = f" · {float(job.progress):.0%}"

        status = getattr(job, "status", "")
        if status == "complete":
            title = "Batch complete"
        elif status == "failed":
            title = "Batch failed"
        elif status == "cancelled":
            title = "Batch cancelled"
        else:
            title = "Batch progress"

        if total:
            lines = [f"{title}: {processed}/{total} files processed{pct}"]
        else:
            lines = [f"{title}{pct}"]

        current = detail.get("current_file")
        if current and status not in {"complete", "cancelled"}:
            lines.append(f"Current: {_display_path_name(str(current))}")
        lines.append(f"Stage: {stage}")

        counts: list[str] = []
        if completed:
            counts.append(f"{completed} complete")
        if failed:
            counts.append(f"{failed} failed")
        if skipped:
            counts.append(f"{skipped} skipped")
        if total:
            remaining = max(total - processed, 0)
            counts.append(f"{remaining} remaining")
        if counts:
            lines.append("Status: " + " · ".join(counts))

        message = getattr(job, "message", None)
        if message and message not in {"Running.", "Complete.", stage}:
            lines.append(str(message))
        return "\n".join(lines)

    def _append_user(self, text: str) -> None:
        self.transcript.append_user(text)

    def _begin_assistant_turn(self) -> None:
        self.transcript.begin_assistant()

    def _append_text_delta(self, text: str) -> None:
        self.transcript.append_assistant_delta(text)

    def _append_system(self, msg: str) -> None:
        self.transcript.append_system(msg)


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _display_path_name(path: str) -> str:
    normalized = path.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    return name or Path(path).name or path
