"""End-to-end wiring proof: the 20-tool core set reaches the LOCAL provider, and
NOTHING else.

tests/test_tool_subset.py covers core_tools()/missing_core_names() in isolation --
pure functions over tool-spec dicts. That leaves the part that actually matters
untested: whether the filter is wired somewhere that has an effect. AgentRunner.turn()
rebuilds `tools_for_anthropic()` itself on every turn and hands it straight to
provider.stream(), so a filter applied at AgentRunner-construction time, or anywhere
that only shapes _make_provider's return value, would be a silent no-op -- 107 tools
would still reach the model and the termination-loop this whole change exists to fix
would still happen. These tests drive a real ChatDock._ensure_runner() and assert on
what provider.stream() is actually handed.

The mirror-image failure is worse and is why the cloud assertions below exist: if the
subsetting leaked into the anthropic/openai/claude-agent/codex-agent paths, those
backends (which handle 107 tools fine) would silently lose 87 capabilities with no
error anywhere.
"""
from __future__ import annotations

import pytest

from imajin.agent.providers.base import Stop


@pytest.fixture(autouse=True)
def _isolated_chat_dock_env(monkeypatch, tmp_path):
    """ChatDock.__init__ calls discover_ollama_models unconditionally and the picker
    persists via Settings.save_secrets(); redirect both so no test here touches a real
    Ollama daemon or the shared secrets path."""
    from imajin.config import Settings
    from imajin.ui import chat_dock as cd

    monkeypatch.setattr(cd, "discover_ollama_models", lambda base_url: [])
    monkeypatch.setattr(
        Settings, "secrets_path", classmethod(lambda cls: tmp_path / "secrets.json")
    )


class _CaptureProvider:
    """Stands in for a real Provider and records exactly what each turn sends."""

    name = "capture"

    def __init__(self, *args, **kwargs) -> None:
        self.model = kwargs.get("model", "capture-model")
        self.num_ctx = kwargs.get("num_ctx")
        self.calls: list[tuple[list[dict], str]] = []

    def stream(self, messages, tools, system):
        self.calls.append((list(tools), system))
        yield Stop(reason="end_turn", usage={})


def _dock(qtbot, viewer, **settings_kwargs):
    from imajin.config import Settings
    from imajin.ui import chat_dock as cd

    dock = cd.ChatDock(viewer=viewer, settings=Settings(**settings_kwargs))
    qtbot.addWidget(dock)
    return dock


def _select(dock, kind: str) -> None:
    idx = next(i for i, (_, k, _m) in enumerate(dock.model_choices) if k == kind)
    dock.model_picker.setCurrentIndex(idx)


def _drive_one_turn(runner) -> None:
    """Consume a full turn so provider.stream() actually fires."""
    for _ in runner.turn("Segment the GFP cells and measure them."):
        pass


def _capture_for(monkeypatch, provider_attr: str) -> list[_CaptureProvider]:
    """Patch one concrete provider class in imajin.agent.providers (where
    _make_provider imports it from) and collect every instance built."""
    import imajin.agent.providers as providers

    made: list[_CaptureProvider] = []

    def factory(*args, **kwargs):
        p = _CaptureProvider(*args, **kwargs)
        made.append(p)
        return p

    monkeypatch.setattr(providers, provider_attr, factory)
    return made


# ---------------------------------------------------------------------------
# The local path: 20 tools reach stream(), not 107
# ---------------------------------------------------------------------------


def test_ollama_turn_sends_exactly_the_core_tool_set_to_the_provider(
    qtbot, viewer, monkeypatch
):
    """The load-bearing assertion of the whole change. AgentRunner.turn() rebuilds the
    full 107-tool spec every turn, so this fails if the filter landed anywhere that
    doesn't intercept the tools on their way into stream()."""
    from imajin.agent.tool_subset import CORE_TOOL_NAMES
    from imajin.tools import tools_for_anthropic

    made = _capture_for(monkeypatch, "OllamaProvider")
    dock = _dock(qtbot, viewer, ollama_model="qwen3.5:4b")
    _select(dock, "ollama")

    runner = dock._ensure_runner()
    _drive_one_turn(runner)

    assert len(made) == 1
    sent_tools, _system = made[0].calls[0]
    assert [t["name"] for t in sent_tools] == list(CORE_TOOL_NAMES)
    assert len(sent_tools) == 20
    # Guard against the assertion above passing trivially because the registry itself
    # shrank: the full list really is much larger than what we sent.
    assert len(tools_for_anthropic()) > 100


def test_ollama_turn_sends_the_reduced_system_prompt(qtbot, viewer, monkeypatch):
    """A reduced tool set with the FULL prompt still names ~87 uncallable tools, which
    is what made both qwen models burn loops on invented calls."""
    from imajin.agent.prompts import build_system_prompt
    from imajin.agent.tool_subset import CORE_TOOL_NAMES

    made = _capture_for(monkeypatch, "OllamaProvider")
    dock = _dock(qtbot, viewer, ollama_model="qwen3.5:4b")
    _select(dock, "ollama")

    _drive_one_turn(dock._ensure_runner())

    _tools, system = made[0].calls[0]
    reduced = build_system_prompt(available_tools=set(CORE_TOOL_NAMES))
    full = build_system_prompt()
    assert system.startswith(reduced)
    assert not system.startswith(full)
    assert len(reduced) < len(full)


def test_ollama_provider_receives_a_num_ctx_sized_from_the_subset(
    qtbot, viewer, monkeypatch
):
    """num_ctx must be computed from the reduced prompt + 20 tools. Sizing it from the
    full 107 would keep requesting the 57344-token window that spills this model off
    the GPU, making the subsetting invisible where it costs VRAM."""
    made = _capture_for(monkeypatch, "OllamaProvider")
    dock = _dock(qtbot, viewer, ollama_model="qwen3.5:4b")
    _select(dock, "ollama")
    dock._ensure_runner()

    assert made[0].num_ctx is not None
    assert made[0].num_ctx < 32768


def test_specialist_consult_tools_pass_through_the_wrapper_unfiltered(
    qtbot, viewer, monkeypatch
):
    """A specialist consult reuses the SAME provider instance via
    get_current_provider() and calls stream() with its own subagent-scoped tool list.
    None of those names are in CORE_TOOL_NAMES, so filtering them would hand the
    specialist an EMPTY tool list -- silently breaking consults on local models only."""
    made = _capture_for(monkeypatch, "OllamaProvider")
    dock = _dock(qtbot, viewer, ollama_model="qwen3.5:4b")
    _select(dock, "ollama")
    runner = dock._ensure_runner()

    specialist_tools = [
        {"name": "some_subagent_only_tool", "description": "", "input_schema": {}},
        {"name": "another_specialist_tool", "description": "", "input_schema": {}},
    ]
    list(runner.provider.stream([], specialist_tools, "specialist system prompt"))

    sent_tools, _system = made[0].calls[0]
    assert [t["name"] for t in sent_tools] == [
        "some_subagent_only_tool",
        "another_specialist_tool",
    ]


# ---------------------------------------------------------------------------
# The cloud paths: still the full registry, still the byte-identical full prompt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "provider_attr", "settings_kwargs"),
    [
        ("anthropic", "AnthropicProvider", {"anthropic_api_key": "sk-test"}),
        ("openai", "OpenAICompatProvider", {"openai_api_key": "sk-test"}),
    ],
)
def test_cloud_provider_turn_still_sends_the_full_registry_and_full_prompt(
    qtbot, viewer, monkeypatch, kind, provider_attr, settings_kwargs
):
    """SCOPE DISCIPLINE: the tool reduction is a local-model workaround. Claude and
    GPT handle 107 tools fine, and quietly dropping 87 of them here would remove real
    capability with no error surfaced anywhere."""
    from imajin.agent.prompts import build_system_prompt
    from imajin.tools import tools_for_anthropic

    made = _capture_for(monkeypatch, provider_attr)
    dock = _dock(qtbot, viewer, **settings_kwargs)
    _select(dock, kind)

    _drive_one_turn(dock._ensure_runner())

    sent_tools, system = made[0].calls[0]
    full = tools_for_anthropic()
    assert len(sent_tools) == len(full)
    assert [t["name"] for t in sent_tools] == [t["name"] for t in full]
    assert system.startswith(build_system_prompt())


@pytest.mark.parametrize(
    ("kind", "module_path", "runner_attr"),
    [
        ("claude-agent", "imajin.agent.providers.claude_agent", "ClaudeAgentRunner"),
        ("codex-agent", "imajin.agent.providers.codex_agent", "CodexAgentRunner"),
    ],
)
def test_subscription_runner_gets_the_byte_identical_full_system_prompt(
    qtbot, viewer, monkeypatch, kind, module_path, runner_attr
):
    """These two own their own agent loops and bridge the registry themselves, so the
    only thing chat_dock hands them is the system prompt -- which must remain the
    unreduced one."""
    import importlib

    from imajin.agent.prompts import build_system_prompt

    module = importlib.import_module(module_path)
    captured: dict[str, str] = {}

    class _FakeRunner:
        def __init__(self, *, model, system_prompt, tool_caller, **kwargs):
            captured["system_prompt"] = system_prompt

    monkeypatch.setattr(module, runner_attr, _FakeRunner)
    dock = _dock(qtbot, viewer)
    _select(dock, kind)
    dock._ensure_runner()

    assert captured["system_prompt"] == build_system_prompt()


@pytest.mark.parametrize(
    "kind", ["claude-agent", "codex-agent"]
)
def test_subscription_runner_bridges_the_full_registry_not_the_core_set(kind):
    """Both bridge tools by re-selecting from iter_tools() rather than taking a list
    from chat_dock, so this asserts the selection itself never narrowed."""
    from imajin.tools import tools_for_anthropic
    from imajin.tools.registry import iter_tools

    bridged = [e.name for e in iter_tools() if e.subagent is None and e.llm]
    assert len(bridged) == len(tools_for_anthropic())
    assert len(bridged) > 100
