"""Single source of truth for what LLM backends exist and how to use them.

Before this module, "which backends exist" was answered independently in six
places: chat_dock._MODEL_CHOICES, chat_dock._FIX_HINTS, chat_dock._make_provider's
if/elif chain, chat_dock._ensure_runner's if/elif chain, provider_status.compute_statuses's
five branches, and cli._doctor's hardcoded kind tuple. Adding "codex-agent" meant
hand-editing (most of) those by hand, and a backend with no compute_statuses entry
used to render as AVAILABLE in the picker -- a bare ``.get(kind)`` read a missing
status as a green light rather than as an error (see chat_dock._UNREGISTERED_STATUS
for the fix that made a missing entry read as unavailable instead).

This module collapses the *dispatch* -- "does this kind exist, how do I check it,
how do I build it" -- into one table, ``BACKEND_REGISTRY``, keyed by ``kind``.
chat_dock.py and cli.py now read from it instead of hand-maintaining their own copy.
What it does NOT own (still hand-maintained, deliberately -- see the bottom of this
docstring): chat_dock._MODEL_CHOICES's actual row labels/model-tier tokens (there
can be more than one row per kind -- e.g. anthropic has a "sonnet" row and an
"opus" row -- so it isn't a per-kind fact this table can hold without inventing a
list-of-rows shape for zero behavioural benefit), and provider_status.compute_statuses's
own bare-name probe calls (see "why probes delegate" below).

Two runner SHAPES, not one -- see claude_agent.py's module docstring for the full
reasoning, summarized here. A "provider" backend (anthropic, openai, ollama) is
stateless per turn: its factory returns a bare Provider (imajin.agent.providers.base),
and imajin.agent.runner.AgentRunner owns the tool-calling loop around it, rebuilding
the tool list and calling stream() itself every turn. A "fused" backend (claude-agent,
codex-agent) owns its *own* agentic loop end to end -- it drives an external CLI that
calls back into Imajin's tools itself -- so it cannot sit behind Provider.stream() at
all; its factory returns the finished runner object directly, presenting
turn()/reset()/cancel()/close() with no AgentRunner involved. Flattening these into one
factory signature would hide a real difference in what "tools" and "system_prompt" even
mean to each (a Provider's tools are decided by AgentRunner fresh every turn; a fused
runner decides its own tools, once, at construction) -- so BackendSpec keeps two
distinct, named factory slots (``make_provider`` / ``make_runner``) rather than one
generic one, and a kind sets exactly the slot matching its own ``shape``.

Lazy-import discipline: every probe and factory function below does its own import
INSIDE its body, never at this module's top level, and always from the concrete
class's OWN defining module (never re-imported through some other module's copy).
tests/test_chat_dock_tool_scoping.py patches AnthropicProvider/OllamaProvider/
OpenAICompatProvider on ``imajin.agent.providers`` and ClaudeAgentRunner/CodexAgentRunner
on their own ``.claude_agent``/``.codex_agent`` modules, expecting exactly this
lazy-resolution timing -- a fresh ``from X import Y`` inside a function body reads
whatever Y currently is on X, patched or not, every time the function runs. Hoisting
any of these to an eager top-level import here would still work for Imajin's own run
path, but would silently stop those tests from faking a network/subprocess call --
the same trap already documented for splitting tools/*.py modules (see
memory/tool-module-split-monkeypatch.md): re-export is not patch-through.

Why probes delegate to provider_status.compute_statuses() instead of re-implementing
each check: claude-agent/codex-agent/ollama's availability comes from
subscription_available()/codex_available()/probe_ollama() -- and tests/test_provider_status.py
*and* every test in the whole suite (conftest.py's autouse ``_no_local_model_network``
/ ``_no_codex_subscription`` fixtures) patch those three names as attributes of
``imajin.ui.provider_status`` specifically, relying on ``compute_statuses`` calling them
as bare names resolved against its OWN module globals at call time. An independent probe
here -- even a lazy one -- importing straight from local_models.py / claude_agent.py /
codex_agent.py would resolve a *different*, unpatched copy and start hitting a real
Ollama daemon / real subscription CLIs in every test in the suite, not just this file's.
Routing through ``compute_statuses`` means these three probes always see whatever
provider_status.py's own bare names currently point at, patched or not -- the same
value the model picker itself would show. anthropic/openai have no such external
process to probe (just a Settings field), so their probes are direct, and
tests/test_backend_registry.py cross-checks that those two independent copies still
agree with compute_statuses's own, so the one deliberate duplication left in this file
cannot silently drift.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from imajin.agent.tool_subset import CORE_TOOL_NAMES, core_tools, missing_core_names

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class BackendContext:
    """Everything a backend's factory might need, gathered by the caller (ChatDock)
    so this module never has to import Qt or know ChatDock's own shape.

    ``local_models`` and ``append_system`` are read only by kinds that need them --
    today, just ollama, for context-window sizing and the loud-degradation warnings
    (see _make_ollama_provider). Every other factory ignores both.
    """

    settings: Any
    model: str
    tool_caller: Callable[..., Any] | None = None
    local_models: dict[str, Any] = field(default_factory=dict)
    append_system: Callable[[str], None] | None = None


@dataclass(frozen=True)
class BackendSpec:
    """One backend kind: how to check it, and how to build it.

    ``shape`` is the load-bearing field -- see this module's docstring (and
    claude_agent.py's) for why "provider" and "fused" cannot share one factory
    signature. Exactly one of ``make_provider`` / ``make_runner`` should be set,
    matching ``shape`` -- enforced in __post_init__ so a mis-wired entry fails
    loudly at import time instead of quietly at first use.

    ``wrap_for_runner``, set only for a "provider" kind whose bare Provider needs
    post-processing before AgentRunner can use it (today: just ollama's
    core-tool-set scoping -- see _wrap_ollama_for_runner), takes the provider
    ``make_provider`` returned and returns ``(possibly-wrapped provider, system
    prompt)``. ``None`` means "use the provider as-is, with the caller's own full
    system prompt" -- anthropic and openai's case.
    """

    kind: str
    shape: Literal["provider", "fused"]
    fix_hint: str | None
    make_provider: Callable[[BackendContext], Any] | None = None
    make_runner: Callable[[BackendContext], Any] | None = None
    wrap_for_runner: Callable[[Any], tuple[Any, str]] | None = None

    def __post_init__(self) -> None:
        if self.shape == "provider" and self.make_provider is None:
            raise ValueError(f"{self.kind!r}: shape='provider' requires make_provider")
        if self.shape == "fused" and self.make_runner is None:
            raise ValueError(f"{self.kind!r}: shape='fused' requires make_runner")


# Availability deliberately does NOT live here. An earlier draft of this module
# carried a `probe` per spec, but nothing in the app ever called it -- the picker
# and `imajin doctor` both read imajin.ui.provider_status.compute_statuses(), and
# the registry's copies of the "no API key" rule were a second, drifting source of
# truth for a question that already had one. Worse for testing: conftest.py's
# autouse fixtures neutralise the live probes by patching bare names INSIDE
# provider_status (probe_ollama, codex_available), so a registry-side copy would
# have quietly escaped them and put real machine state back into the suite -- the
# exact class of non-hermeticity the local-model and Codex work each had to fix.
# This module owns dispatch, shape and fix hints; provider_status owns "can it run".


# --- "provider" shape: a bare Provider that AgentRunner drives --------------


def _make_anthropic_provider(ctx: BackendContext) -> Any:
    from imajin.agent.providers import AnthropicProvider

    if not ctx.settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Open Imajin → API Keys… or set the env var."
        )
    return AnthropicProvider(api_key=ctx.settings.anthropic_api_key, model=ctx.model)


def _make_openai_provider(ctx: BackendContext) -> Any:
    from imajin.agent.providers import OpenAICompatProvider

    if not ctx.settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Open Imajin → API Keys… or set the env var."
        )
    return OpenAICompatProvider(
        api_key=ctx.settings.openai_api_key,
        model=ctx.model,
        base_url=ctx.settings.openai_base_url,
    )


def _local_system_prompt(available_tools: set[str] | frozenset[str]) -> str:
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


def _make_ollama_provider(ctx: BackendContext) -> Any:
    """Build the native-``/api/chat`` OllamaProvider for ctx.model, sized and
    warned about from ctx.local_models / ctx.append_system.

    Native /api/chat, never the OpenAI-compat endpoint: the compat endpoint
    ignores num_ctx in every form tested and silently truncates to Ollama's
    4096-token default, which with the full 104-tool prompt (~28.3K tokens)
    leaves the model narrating a tool call instead of emitting one (see
    local_models.py's module docstring for the measured numbers). choose_num_ctx
    sizes the window from the model's real context_length when discovery found
    one; when it has to clamp below what the prompt actually needs, that's real,
    user-visible truncation, so it's announced via ctx.append_system here rather
    than left to surface later as a model that "forgot" its tools mid-turn.
    """
    from imajin.agent.local_models import LocalModel, choose_num_ctx, estimate_prompt_tokens
    from imajin.agent.providers import OllamaProvider
    from imajin.tools import tools_for_anthropic

    local_model = ctx.local_models.get(ctx.model)
    if local_model is None:
        # Unconfirmed fallback row (settings.ollama_model; discovery found
        # nothing) -- no known context_length to clamp to, and no confirmed
        # capabilities, so default vision off rather than trust a guess.
        local_model = LocalModel(
            name=ctx.model,
            context_length=None,
            capabilities=frozenset(),
            parameter_size=None,
            size_bytes=None,
        )
    # Advertise the 20-tool core, not the full registry: with all 107 tools
    # offered, qwen3.5:4b never terminated a multi-step task (see tool_subset.py
    # for the measured before/after). Recomputing the estimate from the SUBSET --
    # not the full list -- is what actually lets num_ctx reflect the smaller
    # prompt; estimating from the full list here while only ever sending the
    # subset (see _wrap_ollama_for_runner, wired in by chat_dock._ensure_runner)
    # would silently throw the saving away.
    full_tools_spec = tools_for_anthropic()
    tools_spec = core_tools(full_tools_spec)
    missing = missing_core_names(full_tools_spec)
    append_system = ctx.append_system
    if missing and append_system is not None:
        # A core tool was renamed/removed without updating tool_subset.py: local
        # models now see fewer than 20 tools. Loud on purpose -- silent
        # capability loss is worse than a visible warning.
        append_system(
            f"[warning] {len(missing)} core tool(s) not found in the "
            f"registry: {', '.join(missing)}. Local models will see "
            f"{len(tools_spec)} tools instead of {len(CORE_TOOL_NAMES)}."
        )
    if append_system is not None:
        # Says how to reach the other 87 tools, not "call get_help" -- get_help
        # is onboarding-scoped and returns a docs link, so pointing the user
        # there for a missing capability sends them somewhere that cannot answer.
        append_system(
            f"[info] Local models see a {len(tools_spec)}-tool core set "
            f"({len(tools_spec)} of {len(full_tools_spec)} available) so a small "
            "model decides instead of dithering; switch to a cloud model for the "
            "full set."
        )
    available_tool_names = {t["name"] for t in tools_spec}
    est = estimate_prompt_tokens(_local_system_prompt(available_tool_names), tools_spec, [])
    # floor=16384 -- was the implicit 32768 default, which is what would
    # re-inflate a legitimately smaller subset-driven estimate right back up to
    # the window a 107-tool prompt needed, throwing the saving away instead of
    # letting it show up as a smaller num_ctx / less VRAM.
    num_ctx = choose_num_ctx(local_model, est, floor=16384)
    if num_ctx < est and append_system is not None:
        append_system(
            f"[warning] {ctx.model}'s context window ({num_ctx} tokens) is "
            f"smaller than the system prompt + {len(tools_spec)} tools "
            f"(~{est} tokens). Requests will be truncated and the model may "
            "stop seeing some or all of its tools."
        )
    return OllamaProvider(
        model=ctx.model,
        base_url=ctx.settings.ollama_base_url,
        num_ctx=num_ctx,
        supports_vision=local_model.supports_vision,
    )


class _OllamaCoreToolsProvider:
    """Wraps a local-model Provider so its turns advertise core_tools(...)
    instead of the full registry.

    Why here, rather than on OllamaProvider itself or on AgentRunner:
    AgentRunner.turn() always calls the module-level tools_for_anthropic()
    fresh and hands the FULL list straight to provider.stream() (see
    runner.py) -- building a plain OllamaProvider does not change that, since
    nothing about a provider's construction narrows what a *later* turn sends
    it. Trimming has to happen where stream() actually receives the tools,
    which is here -- the one seam chat_dock._ensure_runner controls without
    editing runner.py or providers/ollama.py.

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
    ) -> Any:
        names = frozenset(t.get("name") for t in tools)
        if names == self._full_tool_names:
            tools = core_tools(tools)
        yield from self._inner.stream(messages, tools, system)


def _wrap_ollama_for_runner(provider: Any) -> tuple[Any, str]:
    """Wrap a bare OllamaProvider for AgentRunner: core-tool-set-scoped
    Provider + the matching reduced system prompt.

    Local path only (SCOPE DISCIPLINE: anthropic/openai keep seeing the full
    registry via plain AgentRunner + the full prompt). The provider gets
    wrapped -- not AgentRunner itself -- so runner.py and providers/ollama.py
    stay untouched; see _OllamaCoreToolsProvider for why that's also the
    *safe* seam (a filter living on AgentRunner or applied unconditionally in
    the provider would also reach specialist consults, which reuse this same
    provider instance with their own, disjoint tool list).

    Computes tools_for_anthropic()/core_tools() independently of
    _make_ollama_provider's own copy (a second, known-duplicate pass -- see
    tool_subset.py's module docstring and this project's Known Target #2) --
    deliberately not deduplicated here: that is a separate, already-scoped
    slice, and folding it into this one would conflate two different changes
    in one diff.
    """
    from imajin.tools import tools_for_anthropic

    full_tools_spec = tools_for_anthropic()
    subset_tools_spec = core_tools(full_tools_spec)
    available_tool_names = {t["name"] for t in subset_tools_spec}
    wrapped = _OllamaCoreToolsProvider(provider, frozenset(t["name"] for t in full_tools_spec))
    return wrapped, _local_system_prompt(available_tool_names)


# --- "fused" shape: owns its own agentic loop end to end, no AgentRunner ----


def _make_claude_agent_runner(ctx: BackendContext) -> Any:
    """Subscription-backed: the Claude Code agent owns its own loop, so this is a
    ClaudeAgentRunner (not a Provider behind AgentRunner) -- see
    imajin.agent.providers.claude_agent's module docstring for the full reasoning.
    It presents the same turn()/reset()/cancel() surface, so everything downstream
    is unchanged. No API key -- it uses the local `claude` login.
    """
    from imajin.agent.prompts import build_system_prompt
    from imajin.agent.providers.claude_agent import ClaudeAgentRunner

    return ClaudeAgentRunner(
        model=ctx.model,
        system_prompt=build_system_prompt(),
        tool_caller=ctx.tool_caller,
    )


def _make_codex_agent_runner(ctx: BackendContext) -> Any:
    """Subscription-backed like claude-agent above: codex owns its own agentic
    loop (it drives the `codex` CLI, which calls back into Imajin's tools over
    an in-process MCP bridge), so this is a CodexAgentRunner, not a Provider
    behind AgentRunner. Same turn()/reset()/cancel()/close() surface, so
    everything downstream is unchanged. No API key -- it uses whatever the user
    set up themselves with `codex login`; see
    imajin.agent.providers.codex_agent's module docstring for the ToS/auth
    decision this mirrors from ClaudeAgentRunner.
    """
    from imajin.agent.prompts import build_system_prompt
    from imajin.agent.providers.codex_agent import CodexAgentRunner

    return CodexAgentRunner(
        model=ctx.model,
        system_prompt=build_system_prompt(),
        tool_caller=ctx.tool_caller,
    )


# --- the table itself --------------------------------------------------------
#
# Order matters only for cli.py's --doctor output (it prints in this order) --
# kept identical to the tuple it used to hardcode, so adding a row here does not
# reshuffle the doctor output of every kind already registered.

BACKEND_REGISTRY: dict[str, BackendSpec] = {
    spec.kind: spec
    for spec in (
        BackendSpec(
            kind="anthropic",
            shape="provider",
            fix_hint=None,
            make_provider=_make_anthropic_provider,
        ),
        BackendSpec(
            kind="claude-agent",
            shape="fused",
            fix_hint="Run `claude` in a terminal and sign in.",
            make_runner=_make_claude_agent_runner,
        ),
        BackendSpec(
            kind="openai",
            shape="provider",
            fix_hint=None,
            make_provider=_make_openai_provider,
        ),
        BackendSpec(
            kind="codex-agent",
            shape="fused",
            fix_hint="Run `codex login` in a terminal.",
            make_runner=_make_codex_agent_runner,
        ),
        BackendSpec(
            kind="ollama",
            shape="provider",
            fix_hint="Start Ollama, then `ollama pull` a tool-capable model.",
            make_provider=_make_ollama_provider,
            wrap_for_runner=_wrap_ollama_for_runner,
        ),
    )
}


def get_backend(kind: str) -> BackendSpec | None:
    """The BackendSpec for `kind`, or None if nothing is registered for it.

    None (never a KeyError, never a made-up default) is the contract: an
    unregistered kind must read as "cannot be built" just as unambiguously as
    it already reads as "unavailable" in the model picker (see
    chat_dock._UNREGISTERED_STATUS) -- callers decide what to do with None
    (chat_dock raises a clear RuntimeError rather than guessing a fallback
    backend), never silently substitute a different kind's factory.
    """
    return BACKEND_REGISTRY.get(kind)


def kinds() -> tuple[str, ...]:
    """Every registered kind, in registration order."""
    return tuple(BACKEND_REGISTRY)
