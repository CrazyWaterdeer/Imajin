# Onboarding guide + in-app `get_help` — Design

- **Date:** 2026-07-09
- **Status:** Approved (brainstorming) → Codex-reviewed + revised → pending user spec review
- **Author:** Jin (with Claude Code)
- **External review:** Codex (gpt-5.5) read-only pass folded in — offline claim
  corrected, prompt block turned into a meta-question allow-list, routing/slug
  tests added, doc-ownership rule, centralised repo/branch constant, README
  preservation checklist.

## Summary

Give first-time users a friendly on-ramp, and let the in-app assistant point
them to it. Two coupled pieces of work:

1. **Documentation restructure.** The current `README.md` (~254 lines) is a
   dense feature catalogue, not a newcomer's entry point. Slim it into a short
   "front door", move its detailed feature prose into `docs/features.md`, and
   add a task-oriented onboarding tutorial at `docs/getting_started.md`.
2. **Agent integration.** A new `get_help(topic)` tool returns a GitHub
   deep-link (section anchor) into `docs/getting_started.md`, plus the list of
   topics. A short block in the system prompt makes the assistant aware of it,
   subordinate to the existing "bias to action" rule.

## Goals

- A newcomer can go from clone → first analysis result by reading one short,
  task-oriented guide.
- `README.md` reads as a welcoming index, not a wall of features.
- When a user asks the in-app assistant "what can you do / how do I start / how
  do I do X", it can hand them the precise guide section link and give a brief
  answer from its own prompt knowledge.
- No runtime dependency: `get_help` performs no file read and no network call —
  it returns section links **plus a one-line summary per topic**, assembled from
  a small static map. This is not full offline help: offline, the user still
  gets an orientation and a link to open later, but not the guide's full prose.

## Non-goals (YAGNI)

- No embeddings / RAG / semantic search — substring matching over ~10 topics
  (one per guide section).
- No separate docs website or GitHub wiki.
- No in-app rendered help panel/dock (possible later).
- `get_help` does **not** reproduce full guide prose offline; it returns section
  links plus a one-line summary per topic. Full prose lives only in
  `docs/getting_started.md` (rendered on GitHub).
- No changes to `docs/design_principles.md`, the gallery, or the analysis tools.

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| Guide center of gravity | Getting-started onboarding (install → load → first analysis), task-oriented, short |
| How the agent surfaces it | Prompt awareness **+** `get_help` tool |
| Language | English |
| Doc structure | Slim README front door; detail moved to `docs/` |
| Guide location | `docs/getting_started.md` (**not** under `src/`); `get_help` returns GitHub links, not bundled text |
| `features.md` vs `analysis_capabilities.md` | Keep both, cross-linked (different formats: prose vs matrix) |

## Documentation architecture

```
README.md                      slim front door (~60 lines)
  ├─ docs/getting_started.md    onboarding tutorial (get_help links here)
  ├─ docs/features.md           detailed feature prose moved out of README
  └─ (existing) docs/analysis_capabilities.md · design_principles.md · gallery/
```

### Component 1 — `README.md` (slim)

Target skeleton:

- `# Imajin` + one-line tagline
- 2–3 sentence *What / Why* (conversational confocal assistant; two
  interchangeable interfaces — chat dock and manual dock — that call the same
  tools)
- `## Documentation` links section, near the top:
  - 🚀 **Getting started** → `docs/getting_started.md`
  - 📖 **Features** → `docs/features.md`
  - 📊 **Capabilities matrix** → `docs/analysis_capabilities.md`
  - 🖼 **Figure gallery** → `docs/gallery/`
  - 🧭 **Design principles** → `docs/design_principles.md`
- `## Install` (`uv sync`, GPU/CUDA note)
- `## Run` (`uv run imajin --doctor`, `uv run imajin`)
- `## Configuration` (condensed: subscription / API key / local Ollama; link to
  the guide for detail)
- `## Status` (short) · `## License`

The current `## Features` block (colocalization, stats, QC, figures, batch,
calcium, tracing, LLM backends, provenance) is **removed** from README and
relocated to `docs/features.md`.

**Preservation checklist (no operational detail lost).** The move is
content-preserving; a reviewer diffs old README → (new README + features.md +
getting_started.md) and confirms each of these survives *somewhere*:

- Configuration caveats: subscription (no API key) vs `ANTHROPIC_API_KEY` /
  `OPENAI_API_KEY` vs local Ollama base URL.
- Install/runtime constraints: NVIDIA GPU + CUDA note; Python pinned to 3.12
  (PyTorch cp314 wheel gap); `uv` requirement.
- The `connectome` optional extra and its degrade-to-typed-status behaviour.
- Status section caveats (ephemeral sessions; offscreen Qt test skips).
- Every feature bullet in the old `## Features` list.

### Component 2 — `docs/features.md`

The detailed feature prose lifted out of the README, with a one-line header
("Detailed feature reference — see also the capabilities matrix in
`analysis_capabilities.md` and the getting-started guide") and a back-link to
the README. Content is a move, not a rewrite; no feature descriptions are lost.

**Ownership rule (governance).** To keep this from diverging from
`analysis_capabilities.md`: `analysis_capabilities.md` is **canonical for the
capability truth** (which analysis × target × tool × stat × graph combinations
exist — the matrix). `features.md` is a **narrative overview** that must not
contradict the matrix and links to it for specifics; it should stay prose-level
and avoid re-listing exhaustive parameter tables. A new capability updates the
matrix first; `features.md` gets a sentence only if the narrative changes. This
rule goes in a comment at the top of `features.md`.

### Component 3 — `docs/getting_started.md` (new)

English, task-oriented. Each `##` heading is a stable anchor that `get_help`
deep-links to. **Headings are kept ASCII and punctuation-light** (no `&`, `/`,
`:`, parentheses, or non-ASCII) so their GitHub slugs are trivial and
predictable — this is the cheapest defence against slug drift. Sections:

1. **What is Imajin** — one short paragraph.
2. **Install and run** — `uv sync`; `uv run imajin --doctor`; `uv run imajin`.
   (Note: "and", not "&", to keep the slug `install-and-run`.)
3. **Open your data** — drag-drop / File▸Open; `.lsm` / `.czi` / OME-TIFF.
4. **Tell Imajin your channels** — target vs counterstain roles; why it matters.
5. **Two ways to drive it** — chat dock ↔ manual dock; identical results and
   provenance.
6. **Your first analysis by chat** — example prompt in English with the Korean
   equivalent in parentheses, e.g. *"Find the cells in Ch2 and measure their
   intensity" (Ch2에서 세포 찾고 세기 측정)*; what happens; where output lands.
7. **Reading results** — tables dock, figures, `save_result_bundle`.
8. **Asking the assistant for help** — "what can you do?", "how do I compare two
   groups?"; mentions that the assistant can link specific guide sections.
9. **If something goes wrong** — GPU vs CPU, provider/API key vs subscription,
   offline/Ollama — brief.
10. **Go deeper** — links to `features.md`, `analysis_capabilities.md`,
    `design_principles.md`.

### Component 4 — `get_help` tool

- **File:** `src/imajin/tools/help.py`; imported in `tools/__init__.py` to
  register on package import.
- **Signature:** `get_help(topic: str | None = None) -> dict[str, Any]`
- **Registration:** `@tool(description=..., llm=True, manual=False)` — assistant
  tool, not a manual-dock form; not a worker (instant); no pipeline `phase`.
- **Static data (no file read at runtime):**
  - `_REPO = "https://github.com/CrazyWaterdeer/Imajin"` and `_BRANCH = "master"`
    are the **single source** for the repo/branch; `GUIDE_URL` is derived
    (`f"{_REPO}/blob/{_BRANCH}/docs/getting_started.md"`). Centralising these is
    the one edit needed if the repo moves or the default branch is renamed.
  - `_TOPICS`: ordered list of `(title, anchor, summary, keywords)`, one per
    `getting_started.md` section.
    - `anchor` matches GitHub's auto-generated heading slug (headings are kept
      slug-simple per Component 3).
    - `summary` is a single plain sentence — the pointer text the assistant can
      say even offline (this is tool metadata, deliberately **not** the guide's
      prose).
    - `keywords` is a **specific** synonym list so intent words absent from the
      title still resolve — e.g. "Your first analysis by chat" carries
      `["measure", "segment", "intensity", "find cells", "first analysis"]`.
      Broad, cross-cutting words (bare `"analysis"`, `"cells"`, `"compare"`) are
      **avoided** so they don't hijack an ordered first-match.
- **Behavior:**
  - `topic=None` → `{"guide_url": GUIDE_URL, "topics": [{"title", "summary"} …],
    "note": "Share guide_url, or call get_help('<topic>') for a section link."}`
  - `topic` matches (case-insensitive substring of `topic` against `title` or any
    `keyword`) → `{"matched": True, "title", "summary",
    "url": f"{GUIDE_URL}#{anchor}", "topics": […]}`. Topics are ordered
    specific → general; first match wins.
  - no match → `{"matched": False, "guide_url": GUIDE_URL, "topics": […],
    "note": "No exact section; pick one of the topics above."}`
- **Return schema:** a JSON-serialisable `dict[str, Any]`; it travels to the
  model through the runner's `_compact_tool_result` (`json.dumps(..., default=str)`)
  exactly like every other tool result. Small payload — well under the 6000-char
  compaction cap.
- **Offline / installed:** everything above is assembled from `_TOPICS` in
  memory; no file I/O, no network.

### Component 5 — system prompt block

Add a short section to `SYSTEM_PROMPT` in `agent/prompts.py` (~6 lines), placed
after the "bias to action" rules and explicitly subordinate to them. The wording
must draw the line at **meta questions about the app**, never at imperative task
requests:

> **Helping a new or stuck user.** `get_help(topic)` returns a link to the
> getting-started guide. Call it ONLY for meta questions *about Imajin itself* —
> "what can you/Imajin do?", "how do I get started?", "where do I find X?",
> "is Y possible?" — then share the section link plus a one-line answer. An
> imperative task ("measure Ch2 intensity", "find the cells", "세포 찾아",
> "compare the two groups") is NOT a help request: run the analysis, do not call
> `get_help`. When in doubt between helping and acting, act. Never answer a task
> with a link.

This inverts the risky framing ("where the action isn't obvious", dropped) into
an explicit allow-list of meta-question shapes, so a concrete task never routes
to help.

## Data flow

```
User (new): "what can you do, and how do I start?"   # meta question → help
  └─ assistant calls get_help()  → { guide_url, topics: [{title, summary}, …] }
  └─ assistant: brief orientation from topic summaries + the guide link

User: "measure Ch2 intensity"   # imperative task → NOT help
  └─ assistant runs the analysis pipeline (list_layers → segment → measure); no get_help

User (offline, meta question): still gets the one-line summaries + a link to open later.
```

## Testing (`tests/test_tools_help.py`)

Deterministic unit tests (no LLM):

- **Shape:** `get_help()` returns non-empty `topics` (each with `title` +
  `summary`) and `guide_url`. `get_help("nonexistent-xyz")` → `matched=False`,
  still returns `topics` + `guide_url`, raises nothing.
- **Registration:** `get_tool("get_help")` resolves and it appears in
  `tools_for_anthropic()`; the returned dict round-trips through
  `json.dumps(default=str)`.
- **Intent routing (not just anchors):** a table of representative phrases →
  expected section, asserting the *right* section wins under ordered first-match,
  e.g. `"measure"`/`"how do I measure intensity"` → first-analysis;
  `"install"` → install-and-run; `"what can you do"` → (None/overview or the
  what-is section, whichever we choose). Include at least one phrase per section.
- **Section ↔ topic bijection:** every `##` heading in `docs/getting_started.md`
  maps to exactly one `_TOPICS` entry and vice-versa — catches a section added
  to the guide but not the map (and a stale map entry).
- **Anchor / slug correctness:** a small `_github_slug()` helper with explicit
  fixtures for the gotchas (`"Install and run"` → `install-and-run`; a name with
  `&`/punctuation; a would-be duplicate → `-1` suffix). Then assert every
  `_TOPICS.anchor` equals the slug of its section heading in the actual doc.
- **Keyword hygiene:** assert no broad cross-cutting keyword (`analysis`,
  `cells`, `compare`, `data`) sits in more than one topic's `keywords` (guards
  first-match hijacking).
- **Link-presence guard:** `README.md` links to `docs/getting_started.md` and
  `docs/features.md`.

Behavioural evals (LLM-dependent, `@pytest.mark.integration`, not in the default
CI subset) — the guard for Codex's action-bias risk:

- With data loaded, `"measure Ch2 intensity"` and `"세포 찾아"` → the agent calls
  the analysis pipeline, **not** `get_help`.
- `"what can you do?"` / `"how do I get started?"` → the agent calls `get_help`.

These are prompt-behaviour checks; they exercise the real model, so they run
under the existing `integration` marker (manual/CI-opt-in), alongside current
action-bias tests. The deterministic tests above are the CI gate.

## Risks & edge cases

- **Anchor drift.** Mitigated three ways: headings kept slug-simple (Component
  3), a tested `_github_slug()` helper with gotcha fixtures, and the section↔topic
  bijection + anchor-equality tests. A renamed heading fails CI until `_TOPICS`
  is updated.
- **Keyword mis-routing.** Ordered first-match with broad keywords could hijack
  a query; mitigated by the keyword-hygiene test and the specific-over-broad
  ordering rule (Component 4).
- **Default branch / version in URL.** `_REPO`/`_BRANCH` are centralised (one
  edit). Links pin `master`, so an installed older checkout's `get_help` points
  at current-`master` docs, not its own version. Acceptable pre-release (the app
  is installed via `git clone` + `uv sync`, not PyPI); versioned/commit-pinned
  links are noted as future work.
- **Prompt regression (highest behavioural risk).** The help block is an
  allow-list of meta-question shapes and explicitly excludes imperative tasks;
  the integration evals assert "measure Ch2" → analysis and "what can you do" →
  help. If those evals or existing action-bias tests regress, the wording is
  wrong.
- **README content loss.** Guarded by the preservation checklist (Component 1).

## Out of scope / future

- Localised (Korean) guide variant.
- In-app help panel that renders the guide inside napari.
- Serving full section prose offline (would require bundling the doc under
  `src/` — deliberately rejected here).
