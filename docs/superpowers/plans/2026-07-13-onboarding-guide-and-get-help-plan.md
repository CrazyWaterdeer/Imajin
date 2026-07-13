# Onboarding guide + in-app `get_help` — Implementation plan

- **Date:** 2026-07-13
- **Spec:** [`../specs/2026-07-09-onboarding-guide-and-get-help-design.md`](../specs/2026-07-09-onboarding-guide-and-get-help-design.md)
  (Approved, Codex-reviewed 2×: gpt-5.5 then gpt-5.6-sol)
- **Branch:** `feat/onboarding-guide-and-help` (already checked out)
- **Status:** ready for Codex review of this plan
- **Author:** Jin (with Claude Code)

## Approach

Four small commits, each leaving the test suite green (`ruff check` + the
non-integration pytest subset). Docs land first as one coherent restructure (so no
commit contains a dangling cross-link), then the tool + its deterministic tests,
then the prompt block, then the contributor-guidance line. The behavioural evals
are added under the `integration` marker and are **not** part of the CI gate.

The design decisions (matching semantics, onboarding scope, stable `id`, heading
contract, total keyword ban, latest-docs URL) are fixed in the spec; this plan
pins the *concrete* `_TOPICS` data, exact file edits, the commit order, and the
test-to-requirement mapping.

## Repo facts this plan relies on (verified)

- `@tool` decorator lives in `src/imajin/tools/registry.py`. Signature knobs:
  `llm: bool = True`, `manual: bool | None` (defaults to `subagent is None`, i.e.
  `True`), `worker`, `phase`. For an assistant-only, instant tool call
  `@tool(description=..., llm=True, manual=False)`.
- Every tool call is wrapped by `record_call` (provenance) — `get_help` will be
  logged like any tool; it takes no session state so this is harmless.
- Tools register on import via `src/imajin/tools/__init__.py` (a list of
  `from imajin.tools import <mod>  # noqa`). Add `help` there.
- `get_tool(name)`, `iter_tools()`, `tools_for_anthropic()`, `call_tool()` are the
  registry accessors the tests use. `tools_for_anthropic()` emits
  `{name, description, input_schema}` for every entry with `llm=True`.
- `README.md` is 253 lines; the `## Features` block is lines 25–188 (the bulk).
- `SYSTEM_PROMPT` is a triple-quoted string in `src/imajin/agent/prompts.py`; the
  `# Bias to action` section is lines 8–21, `# Batch progress` starts line 23. The
  new help block goes **between** them (after line 21).
- Ruff `line-length = 100`. `tests/conftest.py` sets `QT_QPA_PLATFORM=offscreen`
  and an autouse session reset; a pure `get_help` test needs no fixture.
- `test_anchor.py` already exists but is about `imajin.anchor` (result-bundle
  folder anchoring) — unrelated; no name clash with the new `_github_slug`.
- No `CONTRIBUTING.md` yet; CI is `.github/workflows/ci.yml`.

## Concrete `_TOPICS` (the single source pinned here)

Order below **is** the match priority (specific → general). Every `title` matches
`^[A-Za-z0-9 ]+$`; `anchor` is `title.lower().replace(" ", "-")`. Keywords contain
no bare denylist token (`analysis`, `analyse`, `cells`, `compare`, `data`, `image`,
`channel`) — multi-word keywords that *contain* such a word are allowed (they're
specific enough not to over-match).

| # | id | title (→ anchor) | summary (one sentence) | keywords |
|---|----|------------------|------------------------|----------|
| 1 | `install_and_run` | Install and run (`install-and-run`) | Install with `uv sync` and launch the app with `uv run imajin` (run `--doctor` first to check your setup). | install, setup, set up, uv sync, run imajin, launch, doctor, get started, getting started |
| 2 | `open_your_data` | Open your data (`open-your-data`) | Open `.lsm` / `.czi` / OME-TIFF by drag-and-drop or File▸Open; channels split into layers. | open, load, open file, load file, import file, drag and drop, lsm, czi, ome-tiff |
| 3 | `tell_imajin_your_channels` | Tell Imajin your channels (`tell-imajin-your-channels`) | Assign each channel a target / counterstain / ignore role so the tools know what to analyse. | channels, channel role, counterstain, target channel, annotate role, assign role, wavelength |
| 4 | `first_analysis` | Your first analysis by chat (`your-first-analysis-by-chat`) | Type an instruction like "find the cells in Ch2 and measure their intensity" and watch the pipeline run. | measure, segment, intensity, find cells, first analysis, count cells, segmentation, measure intensity |
| 5 | `reading_results` | Reading results (`reading-results`) | Read outputs in the tables dock and figures, and save a self-contained bundle with `save_result_bundle`. | results, tables, table dock, figures, save results, result bundle, export, save_result_bundle |
| 6 | `two_ways_to_drive_it` | Two ways to drive it (`two-ways-to-drive-it`) | Drive everything by chat or by the manual button dock — identical results and provenance. | chat dock, manual dock, buttons, two ways, interface, gui |
| 7 | `if_something_goes_wrong` | If something goes wrong (`if-something-goes-wrong`) | Fixes for GPU vs CPU, provider/API key vs subscription, and running offline with Ollama. | error, not working, troubleshoot, gpu, cpu, cuda, no api key, offline, ollama, crash, fails |
| 8 | `asking_for_help` | Asking the assistant for help (`asking-the-assistant-for-help`) | Ask the assistant onboarding questions and it can link the exact guide section. | help, ask the assistant, assistant help |
| 9 | `go_deeper` | Go deeper (`go-deeper`) | Follow-on docs: the feature reference, the capabilities matrix, and the design principles. | go deeper, advanced, more docs, features, capabilities, design principles, learn more |
| 10 | `what_is_imajin` | What is Imajin (`what-is-imajin`) | Imajin is a conversational confocal-microscopy assistant with a chat dock and a manual dock. | what is imajin, what can you do, what can imajin do, overview, about |

Notes:
- `what_is_imajin` is ordered **last** on purpose: its keyword `what can you do` is
  broad, so it must lose to any more-specific match; as the final entry it only
  wins when nothing else does (and the pure `get_help()` overview is the usual
  entry point for "what can you do?").
- The guide file lists sections 1–10 in reading order (What is Imajin first);
  `_TOPICS` order differs because it encodes *priority*, not reading order. The
  bijection test compares the two as **sets**, so order divergence is fine.
- Matching also tests `id` and `title` as tokens; both are harmless extra hooks
  (users rarely type `snake_case` ids or full titles) and are covered by the
  no-collision test.

## Commit 1 — docs restructure (`docs: split README into front door + features.md + getting_started.md`)

Three file changes in one commit so every cross-link resolves at each step:

1. **`docs/getting_started.md` (new).** The 10 `##` sections in reading order:
   `What is Imajin`, `Install and run`, `Open your data`, `Tell Imajin your
   channels`, `Two ways to drive it`, `Your first analysis by chat`, `Reading
   results`, `Asking the assistant for help`, `If something goes wrong`, `Go
   deeper`. Task-oriented English prose; the first-analysis example gives the
   Korean equivalent in parentheses. "Go deeper" links (relative) to
   `features.md`, `analysis_capabilities.md`, `design_principles.md`. Top of file
   carries the **maintenance HTML comment** from the spec (edit a section → update
   its `_TOPICS` entry; renamed heading → `anchor`; reworded prose → re-check
   `summary`/`keywords`). Every heading obeys `^[A-Za-z0-9 ]+$`.
2. **`docs/features.md` (new).** The `## Features` prose from README lines 36–188,
   moved verbatim (a move, not a rewrite), under a one-line header ("Detailed
   feature reference — see also the capabilities matrix in
   `analysis_capabilities.md` and the getting-started guide") + a back-link to the
   README. Top-of-file HTML comment states the **ownership boundary**
   (`analysis_capabilities.md` = sole authority for supported combinations;
   `features.md` = workflows/benefits prose, must not assert coverage).
3. **`README.md` (slim, ~60–70 lines).** Reduce to: `# Imajin` + tagline; 2–3
   sentence What/Why; a `## Documentation` link list near the top (🚀 Getting
   started, 📖 Features, 📊 Capabilities matrix, 🖼 Gallery, 🧭 Design principles);
   `## Install`; `## Run`; condensed `## Configuration`; short `## Status`;
   `## License`. The `## Features` block is deleted (moved to `features.md`); the
   `## Stack` block folds a sentence into Install or moves to `features.md`.

**Preservation checklist (reviewer diffs old README → new README + features.md +
getting_started.md; each must survive somewhere):** configuration caveats
(subscription vs `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` vs Ollama base URL); install
constraints (NVIDIA GPU + CUDA; Python pinned to 3.12 for the PyTorch cp314 wheel
gap; `uv`); the `connectome` optional extra + degrade-to-typed-status; status
caveats (ephemeral sessions; offscreen Qt skips); every bullet of the old
`## Features` list. Verify with a `git show HEAD:README.md` diff before committing.

Gate: no code changed; existing tests unaffected. Manual checklist review.

## Commit 2 — `get_help` tool + deterministic tests (`feat(tools): add onboarding get_help tool`)

**`src/imajin/tools/help.py` (new).**
- `from __future__ import annotations`; imports `from imajin.tools.registry import tool`.
- `_REPO = "https://github.com/CrazyWaterdeer/Imajin"`, `_BRANCH = "master"`,
  `GUIDE_URL = f"{_REPO}/blob/{_BRANCH}/docs/getting_started.md"`.
- `_TOPICS`: a list of small dataclass/namedtuple records
  `Topic(id, title, anchor, summary, keywords)` exactly as tabled above.
- `_github_slug(title: str) -> str`: `title.lower().replace(" ", "-")` — correct
  only within the enforced `^[A-Za-z0-9 ]+$` charset (documented in a comment).
- `def _match(topic: str | None) -> Topic | None`: `q = (topic or "").strip()
  .casefold()`; if not `q` → `None` (overview); else return the first `Topic`
  whose `id`/`title`/any `keyword` (casefolded) is a substring of `q`.
- `@tool(description="Return a link to the getting-started guide (onboarding
  topics). Call for meta questions about using Imajin, not to run an analysis.",
  llm=True, manual=False)` on
  `def get_help(topic: str | None = None) -> dict[str, Any]:` returning the three
  shapes in the spec (overview / match / no-match), topics list carrying
  `{"id", "title", "summary"}`.
- Register: add `from imajin.tools import help  # noqa: F401, E402` to
  `src/imajin/tools/__init__.py` (alphabetical-ish with the others).

**`tests/test_tools_help.py` (new)** — the CI-gate tests, all no-LLM:
- shape (overview has non-empty topics each with `id`+`title`+`summary`;
  `get_help("")`/`"   "`/`None` → overview; `get_help("nonexistent-xyz")` →
  `matched=False`, still has topics + `guide_url`, raises nothing).
- registration (`get_tool("get_help")` resolves; appears in
  `tools_for_anthropic()`; result round-trips `json.dumps(..., default=str)`).
- intent routing table on `id` (≥1 phrase/section + `"measure"` /
  `"how do I measure intensity"`→`first_analysis`, `"install"`→`install_and_run`,
  `"open my data"`→`open_your_data`).
- every-keyword-resolves (`get_help(k)["id"] == entry.id` for every keyword).
- no cross-topic collision (no id/title/keyword is a substring of another topic's
  title or keywords).
- keyword hygiene: **no keyword casefolded equals any denylist word**
  (`analysis, analyse, cells, compare, data, image, channel`).
- section↔topic bijection (parse `##` headings of `docs/getting_started.md`,
  skipping fenced-code blocks; set-equal to `_TOPICS` by anchor).
- heading contract + anchor equality (every heading matches `^[A-Za-z0-9 ]+$`;
  slugs unique; each `Topic.anchor == _github_slug(heading)`).
- link presence (`README.md` links `docs/getting_started.md` + `docs/features.md`;
  every relative link in the guide's "Go deeper" section resolves to a real file).

Gate: `ruff check` + `pytest tests/test_tools_help.py` green, plus the full
non-integration subset unaffected.

## Commit 3 — system-prompt block (`feat(agent): intent-based help-vs-acting block`)

Insert, in `src/imajin/agent/prompts.py` after the Bias-to-action section (line 21,
before `# Batch progress`), a `# Helping vs acting` block (~8 lines) with the
spec's intent + data-context precedence (act on data even when phrased as a
question; help on orientation/"don't run anything"; "is Y possible?" → tool
list/matrix; compound → run + optional link; ambiguous → act). Mentions
`get_help(topic)` is onboarding-only.

Gate: existing action-bias / prompt tests
(`test_claude_agent_runner.py`, `test_chat_dock_phase3.py`, etc.) still pass —
run them explicitly since this touches the prompt.

## Commit 4 — contributor guidance + integration evals (`docs(contributing): ownership rule; test(help): action-bias evals`)

- Add a minimal `CONTRIBUTING.md` (or a "Contributing" section) carrying the
  `features.md` vs `analysis_capabilities.md` ownership boundary as a review
  checklist line (spec Component 2), so it is not only a file comment.
- Add the behavioural evals to the test suite under `@pytest.mark.integration`
  (opt-in, not CI gate): trace-based, pinned model; act-cases assert `get_help`
  absent + analysis calls present; help-cases assert a `get_help` call with the
  expected resolved `id`. Follow the existing integration-test pattern
  (`test_anthropic_integration.py`).

Gate: `ruff check`; the default subset unchanged (integration deselected).

## Verification (after commit 3, before commit 4)

Beyond unit tests, exercise the real surface with the `verify` skill or a quick
REPL: import `imajin.tools`, call `get_help()` and `get_help("how do I install")`
and confirm the returned dicts + URLs; confirm `get_help` shows up in
`tools_for_anthropic()`. Confirm the app still imports (`uv run imajin --doctor`
if feasible in this environment; otherwise a module import smoke check).

## Requirement → guardrail traceability

- Anchor drift → heading-contract test + anchor-equality test + bijection test.
- Keyword mis-routing → every-keyword-resolves + no-collision + total-ban tests +
  specific→general ordering.
- Action-bias regression → intent-based prompt block + trace-based integration
  evals (act vs help).
- README content loss → preservation checklist diff at commit 1.
- Summary staleness → stable `id` + maintenance HTML comment (human step; noted as
  residual, accepted at this scale).
- Link rot → explicit latest-docs decision; centralised `_REPO`/`_BRANCH`.

## Risks / rollback

- Each commit is independent and reversible; docs commit carries no code risk.
- If the prompt block regresses action-bias tests, revert commit 3 alone and
  re-tune wording — the docs + tool (commits 1–2) stand on their own.
- Keyword collisions surfaced by the no-collision test are fixed by renaming a
  keyword, never by loosening the test.

## Out of scope (unchanged from spec)

Korean guide variant; in-app rendered help panel; serving full prose offline;
scored/multi-candidate matching; build-time manifest codegen; CI HTTP link-check.
