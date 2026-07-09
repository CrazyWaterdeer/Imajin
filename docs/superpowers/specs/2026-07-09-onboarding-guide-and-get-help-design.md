# Onboarding guide + in-app `get_help` — Design

- **Date:** 2026-07-09
- **Status:** Approved (brainstorming) → pending spec review
- **Author:** Jin (with Claude Code)

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
- Works when installed and offline: `get_help` performs no runtime file read
  and no network call — it returns links assembled from a small static map.

## Non-goals (YAGNI)

- No embeddings / RAG / semantic search — substring matching over ~8 topics.
- No separate docs website or GitHub wiki.
- No in-app rendered help panel/dock (possible later).
- `get_help` does **not** quote full guide prose offline; it returns links, and
  the assistant explains briefly from its existing system-prompt knowledge.
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
relocated verbatim-ish to `docs/features.md`.

### Component 2 — `docs/features.md`

The detailed feature prose lifted out of the README, with a one-line header
("Detailed feature reference — see also the capabilities matrix in
`analysis_capabilities.md` and the getting-started guide") and a back-link to
the README. Content is a move, not a rewrite; no feature descriptions are lost.

### Component 3 — `docs/getting_started.md` (new)

English, task-oriented. Each `##` heading is a stable anchor that `get_help`
deep-links to. Sections:

1. **What is Imajin** — one short paragraph.
2. **Install & run** — `uv sync`; `uv run imajin --doctor`; `uv run imajin`.
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
  - `GUIDE_URL = "https://github.com/CrazyWaterdeer/Imajin/blob/master/docs/getting_started.md"`
  - `_TOPICS`: ordered list of `(title, anchor, keywords)`, one per
    `getting_started.md` section. `anchor` matches GitHub's auto-generated
    heading slug. `keywords` is a small list of synonyms so intent words that
    aren't in the title still resolve — e.g. the "Your first analysis by chat"
    section carries `["measure", "segment", "analyze", "intensity", "cells",
    "first"]`, so `get_help("measure")` lands there.
- **Behavior:**
  - `topic=None` → `{"guide_url": GUIDE_URL, "topics": [title, …],
    "note": "Share guide_url, or call get_help('<topic>') for a section link."}`
  - `topic` matches (case-insensitive substring of `topic` against, or against
    any of, the `title` and `keywords`) → `{"matched": True, "title": …,
    "url": f"{GUIDE_URL}#{anchor}", "topics": [title, …]}`. First matching
    section in order wins.
  - no match → `{"matched": False, "guide_url": GUIDE_URL, "topics": [title, …],
    "note": "No exact section; pick one of the topics above."}`
- **Offline / installed:** returns only links assembled from `_TOPICS`; no file
  I/O, no network. Result is small (well under the runner's 6000-char
  tool-result compaction cap).

### Component 5 — system prompt block

Add a short section to `SYSTEM_PROMPT` in `agent/prompts.py` (~5 lines), placed
after the "bias to action" rules and explicitly subordinate to them:

> **Helping a new or stuck user.** Imajin ships a getting-started guide. When
> the user is clearly new, asks what Imajin or you can do, how to get started,
> or how to do a task where the right action isn't obvious, call
> `get_help(topic)` and share the section link it returns, plus a brief answer
> from your own knowledge. Don't paste a whole guide, and never let this stall
> action: if the intent is a concrete analysis, just run it.

## Data flow

```
User: "how do I measure intensity?"
  └─ assistant calls get_help("measure")   # matches the "first analysis" keywords
       └─ returns { url: ".../getting_started.md#your-first-analysis-by-chat", topics: [...] }
  └─ assistant replies: brief how-to (from system prompt) + the section link
User (offline): same, minus the ability to open the link now.
```

## Testing (`tests/test_tools_help.py`)

- `get_help()` returns a non-empty `topics` list and `guide_url`.
- `get_help("measure")` (or another section keyword) → `matched=True` and a
  `url` containing `#`.
- `get_help("nonexistent-xyz")` → `matched=False`, still returns `topics` and
  `guide_url`, raises nothing.
- Tool is registered: `get_tool("get_help")` resolves and it appears in
  `tools_for_anthropic()`.
- **Anchor-sync test:** parse the `##`/`###` headings of `docs/getting_started.md`,
  compute GitHub slugs, and assert every `anchor` in `_TOPICS` exists among
  them. Guards drift between the tool's map and the actual document. (Runs in
  the repo tree, where `docs/` is present.)
- **Link-presence guard (light):** assert `README.md` links to
  `docs/getting_started.md` and `docs/features.md`.

## Risks & edge cases

- **Anchor drift.** GitHub slug rules (lowercase; punctuation stripped; spaces →
  `-`) must be mirrored by the test's slugger and by `_TOPICS`. The anchor-sync
  test is the guard; if a heading is renamed, the test fails until `_TOPICS` is
  updated.
- **Default branch in URL.** `GUIDE_URL` pins `blob/master`. If the default
  branch is ever renamed, update the constant (one place).
- **Prompt regression.** The help block must not induce the assistant to ask
  "would you like help?" instead of acting; the wording subordinates it to
  bias-to-action, and existing action-bias tests still apply.
- **README content loss.** `features.md` is a move; a reviewer should confirm no
  feature bullet is dropped in the relocation.

## Out of scope / future

- Localised (Korean) guide variant.
- In-app help panel that renders the guide inside napari.
- Serving full section prose offline (would require bundling the doc under
  `src/` — deliberately rejected here).
