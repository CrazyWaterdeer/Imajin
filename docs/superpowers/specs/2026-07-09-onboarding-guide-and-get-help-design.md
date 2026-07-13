# Onboarding guide + in-app `get_help` — Design

- **Date:** 2026-07-09
- **Status:** Approved (brainstorming) → Codex-reviewed 2× (gpt-5.5, then gpt-5.6-sol) + revised → ready for plan
- **Author:** Jin (with Claude Code)
- **External review:** Two read-only Codex passes folded in. **Pass 1 (gpt-5.5):**
  offline claim corrected, prompt block turned into a meta-question allow-list,
  routing/slug tests added, doc-ownership rule, centralised repo/branch constant,
  README preservation checklist. **Pass 2 (gpt-5.6-sol):** action/help boundary
  made *semantic* (intent + data-context, not grammar); `get_help` scoped to
  onboarding (the compare-groups inconsistency removed); matching semantics and a
  stable topic `id` pinned; anchor/routing tests recast to enforce the heading
  *contract* rather than model GitHub; behavioural evals made trace-based +
  pinned-model; the doc-ownership rule elevated to contributor guidance; the
  master-pinned URL made an explicit "latest-docs" decision. Rejected as YAGNI:
  scored/multi-candidate matching, build-time manifest codegen, CI HTTP link-check.

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
- When a user asks the in-app assistant an *onboarding* meta-question ("what can
  you do / how do I start / where do I find X"), it can hand them the precise
  getting-started section link and a brief answer from its own prompt knowledge.
  Concrete analysis how-tos ("compare these two groups") are answered by *doing*
  the analysis (when data is loaded) or by pointing at the capabilities matrix —
  not by `get_help` (see Component 4 scope).
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
- `get_help` is **onboarding-scoped**: every topic is a `getting_started.md`
  section, giving a clean bijection. It is *not* a how-to index for analysis
  tasks — a query like "how do I compare groups" is deliberately not a `get_help`
  topic; it routes to acting (data loaded) or to `analysis_capabilities.md`.

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
`analysis_capabilities.md`, draw the boundary by *kind of claim*, not by polite
request:

- `analysis_capabilities.md` (the matrix) is the **sole authority for supported
  combinations and exhaustive coverage** — which analysis × target × tool × stat
  × graph combinations exist. Any statement of the form "Imajin supports/can do
  X" is the matrix's to make.
- `features.md` describes **workflows and benefits** (what a feature is *for*, how
  a user reaches it) in prose. It must not assert coverage or enumerate supported
  combinations; where a reader needs the authoritative list, it links to the
  matrix.

A new capability updates the matrix first; `features.md` gets a sentence only if
the *narrative* changes. This boundary lives **both** as a short comment at the
top of `features.md` **and** as a line in the contributor/review guidance
(`CONTRIBUTING`/PR checklist) — a file comment alone is too easy to miss on the
PR that introduces the contradiction.

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
8. **Asking the assistant for help** — onboarding meta-questions like "what can
   you do?", "how do I get started?", "where do I find X?"; mentions that the
   assistant can link specific guide sections. (Analysis how-tos such as comparing
   groups are handled by *running* the analysis or via the capabilities matrix,
   not by this section — this is what keeps `get_help` onboarding-scoped.)
9. **If something goes wrong** — GPU vs CPU, provider/API key vs subscription,
   offline/Ollama — brief.
10. **Go deeper** — links to `features.md`, `analysis_capabilities.md`,
    `design_principles.md`.

**Maintenance note (kept in the doc as an HTML comment).** Editing a section means
updating its `_TOPICS` entry: a *renamed* heading updates `anchor` (CI catches the
mismatch); *reworded* prose means re-reading that topic's `summary`/`keywords` for
staleness — this last step is human (no test reads meaning), which is exactly why
each topic has a stable `id` to anchor it. Adding/removing a section means
adding/removing its `_TOPICS` entry (the bijection test fails until you do).

### Component 4 — `get_help` tool

- **File:** `src/imajin/tools/help.py`; imported in `tools/__init__.py` to
  register on package import.
- **Signature:** `get_help(topic: str | None = None) -> dict[str, Any]`
- **Registration:** `@tool(description=..., llm=True, manual=False)` — assistant
  tool, not a manual-dock form; not a worker (instant); no pipeline `phase`.
- **Scope:** onboarding only — every topic is a `getting_started.md` section, so
  `_TOPICS` is a bijection with the guide. `get_help` is **not** a how-to index
  for analysis tasks; concrete analysis questions are answered by running the
  analysis (bias to action) or by the capabilities matrix. This scope is what
  makes the matching problem small and the section↔topic tests total.
- **Static data (no file read at runtime):**
  - `_REPO = "https://github.com/CrazyWaterdeer/Imajin"` and `_BRANCH = "master"`
    are the **single source** for the repo/branch; `GUIDE_URL` is derived
    (`f"{_REPO}/blob/{_BRANCH}/docs/getting_started.md"`). Centralising these is
    the one edit needed if the repo moves or the default branch is renamed.
  - `_TOPICS`: ordered list of `(id, title, anchor, summary, keywords)`, one per
    `getting_started.md` section.
    - `id` is a stable `snake_case` identifier (e.g. `install_and_run`,
      `first_analysis`) that does **not** change when a heading is reworded. It is
      the authoritative key for a topic — titles, summaries and anchors hang off
      it, and the tests key on `id`, never on prose. This is the cheap decoupling
      Codex asked for (stable IDs) without a build-time manifest.
    - `anchor` matches GitHub's auto-generated heading slug (headings are kept
      slug-simple per Component 3).
    - `summary` is a single plain sentence — the pointer text the assistant can
      say even offline (this is tool metadata, deliberately **not** the guide's
      prose).
    - `keywords` is a **specific** synonym list so intent words absent from the
      title still resolve — e.g. "Your first analysis by chat" carries
      `["measure", "segment", "intensity", "find cells", "first analysis"]`.
      Broad, cross-cutting words (bare `"analysis"`, `"cells"`, `"compare"`,
      `"data"`, …) are **banned outright** from every list (enforced by the
      keyword-hygiene test in Testing) so they cannot hijack an ordered
      first-match.
- **Matching semantics (pinned — resolves Codex "operand direction / empty
  string / tie-break" ambiguity):**
  - Normalise the query once: `q = (topic or "").strip().casefold()`. If `q` is
    empty (covers `None`, `""`, whitespace) it is the **overview** case — never a
    match against the first entry.
  - An entry matches iff its `id`, its `title`, or one of its `keywords` — each
    `casefold()`ed — occurs **as a substring of `q`** (direction fixed: needle =
    the entry's token, haystack = the user query). So `"how do i install"` hits
    the `install_and_run` keyword `"install"`; a lone stray char cannot match a
    multi-character token.
  - Entries are ordered **specific → general**; the **first** match wins
    (deterministic). Broad cross-cutting words are banned from every `keywords`
    list (see Testing → keyword hygiene), so ordering cannot be silently hijacked.
  - This stays first-match over ~10 topics — **no** scored ranking or
    multi-candidate return (rejected as over-engineering for the scope above).
- **Behavior / return:**
  - overview (`q` empty) → `{"guide_url": GUIDE_URL, "topics":
    [{"id", "title", "summary"} …], "note": "Share guide_url, or call
    get_help('<topic>') for a section link."}`
  - match → `{"matched": True, "id", "title", "summary",
    "url": f"{GUIDE_URL}#{anchor}", "topics": […]}`
  - no match → `{"matched": False, "guide_url": GUIDE_URL, "topics": […],
    "note": "No exact section; pick one of the topics above."}`
- **Return schema:** a JSON-serialisable `dict[str, Any]`; it travels to the
  model through the runner's `_compact_tool_result` (`json.dumps(..., default=str)`)
  exactly like every other tool result. Small payload — well under the 6000-char
  compaction cap.
- **Offline / installed:** everything above is assembled from `_TOPICS` in
  memory; no file I/O, no network.

### Component 5 — system prompt block

Add a short section to `SYSTEM_PROMPT` in `agent/prompts.py` (~8 lines), placed
after the "bias to action" rules and explicitly subordinate to them. The rule is
**intent- and data-context-based, not grammar-based** — a question mark does not
make something a help request, and an imperative verb does not make something a
task (Codex P0-1). Decide by what the user wants done, in this precedence:

> **Helping vs acting.** `get_help(topic)` returns a link to the getting-started
> guide (onboarding only; it is not a how-to index for analyses).
> 1. If the user wants something *done to the current data* — segment, measure,
>    count, compare, plot — **act**, even when it is phrased as a question ("how do
>    I measure Ch2 intensity?" with an image loaded → run it, don't link).
> 2. If the user asks for orientation/instructions, or explicitly says not to run
>    anything ("what can you do?", "how do I get started?", "where do I find X?",
>    "don't run anything, just show me how") → call `get_help` and reply with the
>    section link plus one line.
> 3. "Is Y possible?" → answer from your tool list / the capabilities matrix; add
>    a `get_help` link only when Y is an onboarding step.
> 4. Compound "explain **and** run" → run it; you may add one guide link.
> When genuinely ambiguous, **act** (bias to action). Never answer a concrete data
> task with only a link.

This replaces the earlier shape-based allow-list: routing now turns on *intent +
whether data is loaded*, which is the real failure axis — a question-shaped
request to act must still act, and an imperative-shaped "just show me how" must
still route to help. The ambiguous forms above are added to the evals (Testing),
since the old four examples proved only the easy boundary.

## Data flow

```
User (new): "what can you do, and how do I start?"   # orientation → help (rule 2)
  └─ assistant calls get_help()  → { guide_url, topics: [{id, title, summary}, …] }
  └─ assistant: brief orientation from topic summaries + the guide link

User: "measure Ch2 intensity"   # imperative task → act (rule 1)
  └─ assistant runs the analysis pipeline (list_layers → segment → measure); no get_help

User: "how do I measure Ch2 intensity?"  (image loaded)   # question-shaped, wants action → act (rule 1)
  └─ assistant runs list_layers → segment → measure; no get_help — the '?' does not make it help

User: "don't run anything, just show me how to get started"   # imperative-shaped, wants help → help (rule 2)
  └─ assistant calls get_help("get started") → section link + one line; runs nothing

User (offline, orientation question): still gets the one-line summaries + a link to open later.
```

## Testing (`tests/test_tools_help.py`)

Deterministic unit tests (no LLM) — **the CI gate**:

- **Shape:** `get_help()` returns non-empty `topics` (each with `id` + `title` +
  `summary`) and `guide_url`. `get_help("nonexistent-xyz")` → `matched=False`,
  still returns `topics` + `guide_url`, raises nothing. `get_help("")`,
  `get_help("   ")` and `get_help(None)` each return the overview (never a
  spurious first-entry match).
- **Registration:** `get_tool("get_help")` resolves and it appears in
  `tools_for_anthropic()`; the returned dict round-trips through
  `json.dumps(default=str)`.
- **Intent routing (table, on `id`):** representative phrases → expected topic
  `id`, asserting the *right* topic wins under ordered first-match — at least one
  phrase per section, plus `"measure"` / `"how do I measure intensity"` →
  `first_analysis`, `"install"` → `install_and_run`, `"open my data"` →
  `open_your_data`. Assertions key on `id`, not prose.
- **Every keyword resolves to its owner:** for each entry, every one of its
  `keywords` routes back to that same `id` (i.e. is not shadowed by an earlier
  entry). This is the test that actually catches first-match failures — far
  stronger than "one phrase per section" (Codex P1-5).
- **No cross-topic collision:** no topic's `id`/`title`/keyword is a substring of
  another topic's `title` or any of its `keywords` (else list order silently
  decides the winner).
- **Keyword hygiene (total ban):** a denylist of broad cross-cutting words
  (`analysis`, `analyse`, `cells`, `compare`, `data`, `image`, `channel`) must
  appear in **zero** topics' `keywords` — not merely "at most one"; the old rule
  still permitted the word once (Codex P1-5). Fails CI if any is present.
- **Section ↔ topic bijection:** parse every `##` heading in
  `docs/getting_started.md` — **ignoring headings inside fenced code blocks** —
  and assert a 1:1 map to `_TOPICS` (by `id`/anchor). Catches a section added to
  the guide but not the map, and a stale map entry.
- **Heading contract + anchor equality (enforce the contract, don't model
  GitHub — Codex P1-4):** instead of a fixture set of exotic GitHub slug cases
  (which tests a homemade model outside our own heading rules), assert the
  *contract* that makes slugs trivial: every `##` heading matches
  `^[A-Za-z0-9 ]+$` (ASCII, punctuation-light) and the derived slugs are unique.
  `_github_slug()` is then a minimal `lower().replace(" ", "-")`, **provably
  correct within that restricted charset**; assert every `_TOPICS.anchor` equals
  the slug of its real heading. (If a heading ever needs punctuation, add an
  explicit HTML `<a>` anchor and test against that — noted, not needed now.)
- **Link-presence guard:** `README.md` links to `docs/getting_started.md` and
  `docs/features.md`, **and** every relative link in the guide's "Go deeper"
  section resolves to a file that exists on disk (not just a matching string).

Behavioural evals (LLM-dependent, `@pytest.mark.integration`, opt-in) — the guard
for the action-bias risk, made reproducible + trace-based per Codex P1-6:

- **Fixtures & determinism:** run against a pinned provider/model (recorded in the
  test), with a fixed loaded-data fixture for the "act" cases and no data for the
  pure-orientation cases. Every eval asserts on the **captured tool-call trace**,
  not on the model's prose.
- **Act cases (assert `get_help` absent):** `"measure Ch2 intensity"`,
  `"how do I measure Ch2 intensity?"` (data loaded) and `"세포 찾아"` → the trace
  contains the analysis calls (at least a `segment` then a measure/count) and
  **no** `get_help` anywhere.
- **Help cases (assert argument + section):** `"what can you do?"`,
  `"how do I get started?"` and `"don't run anything, just show me how to start"`
  → the trace contains a `get_help` call whose resolved `id` is the expected one,
  and no analysis calls fire.
- **Policy:** these run under the existing `integration` marker (manual/CI-opt-in)
  alongside current action-bias tests, with a documented pass-rate/retry allowance
  for LLM nondeterminism. The deterministic tests above remain the CI gate.

## Risks & edge cases

- **Anchor drift.** Mitigated by keeping headings slug-simple (Component 3), the
  enforced heading-charset contract + a minimal provably-correct `_github_slug()`,
  and the section↔topic bijection + anchor-equality tests. A renamed heading fails
  CI until `_TOPICS` is updated.
- **Keyword mis-routing.** Ordered first-match could hijack a query; mitigated by
  the *total* broad-word ban, the every-keyword-resolves and no-collision tests,
  and the specific-over-broad ordering rule (Component 4).
- **Silent summary/keyword staleness (Codex P2-7).** The bijection/anchor tests
  catch *structural* drift (a renamed or added heading) but **not** a `summary`
  gone quietly stale relative to its section's prose — no test reads meaning.
  Accepted mitigation at ~10 topics: the stable `id` couples each entry to its
  section, and the guide's maintenance note requires re-reading a topic's
  `summary`/`keywords` when its section changes. A build-time manifest generated
  from doc metadata was considered and **rejected** as over-engineering here.
- **Default branch / version in URL — explicit decision, not just a risk
  (Codex P2-9).** `_REPO`/`_BRANCH` are centralised so links repoint in one edit.
  We **choose "latest-docs" semantics**: `get_help` links always point at
  `master`'s guide, which is correct for the current install story (git `clone` +
  `uv sync`, updated by `git pull`) where "latest docs" is "what you can update
  to". Known limitation: an un-pulled older checkout may see newer instructions,
  and a renamed/deleted `master` would break shipped links. Commit-/tag-pinned
  links plus a stable landing URL are the planned upgrade once the app ships
  versioned (PyPI). No CI HTTP link-check is added (external-network dependency in
  CI is undesirable).
- **Prompt regression (highest behavioural risk).** The help block routes on
  *intent + data-context* (not grammar) and stays subordinate to bias-to-action;
  the trace-based evals assert question-shaped "how do I measure Ch2" → analysis
  and imperative-shaped "just show me how" → help. If those evals or the existing
  action-bias tests regress, the wording is wrong.
- **README content loss.** Guarded by the preservation checklist (Component 1).

## Out of scope / future

- Localised (Korean) guide variant.
- In-app help panel that renders the guide inside napari.
- Serving full section prose offline (would require bundling the doc under
  `src/` — deliberately rejected here).
