"""In-app onboarding help.

`get_help` is an assistant-only, instant tool that points a new or stuck user at the
getting-started guide. It performs **no file read and no network call**: everything is
assembled from the static `_TOPICS` map below, which is a bijection with the `##`
sections of ``docs/getting_started.md``. It is deliberately *onboarding-scoped* — it is
not a how-to index for analyses (a concrete analysis request should run the pipeline,
per the "Helping vs acting" rule in the system prompt).

Maintenance: if you rename/add/remove a guide section, update `_TOPICS` here. The tests
in ``tests/test_tools_help.py`` enforce the section<->topic bijection, the heading/anchor
contract, keyword hygiene, and routing.
"""

from __future__ import annotations

from typing import Any, NamedTuple  # `Any` must be importable (get_type_hints on the

# tool signature resolves the stringized ``-> dict[str, Any]`` annotation).
from imajin.tools.registry import tool

# Single source for the docs location. `GUIDE_URL` is derived; if the repo moves or the
# default branch is renamed, this is the only edit. Links use "latest-docs" semantics
# (always point at the branch below) — correct for the git-clone + `uv sync` install.
_REPO = "https://github.com/CrazyWaterdeer/Imajin"
_BRANCH = "master"
GUIDE_URL = f"{_REPO}/blob/{_BRANCH}/docs/getting_started.md"


def _github_slug(title: str) -> str:
    """GitHub heading slug — correct only within the enforced ``^[A-Za-z0-9 ]+$`` charset
    that ``docs/getting_started.md`` headings obey (lowercase, spaces -> hyphens)."""
    return title.lower().replace(" ", "-")


class Topic(NamedTuple):
    id: str
    title: str
    anchor: str
    summary: str
    keywords: tuple[str, ...]


def _topic(id: str, title: str, summary: str, keywords: tuple[str, ...]) -> Topic:
    # anchor is derived from the title so it cannot drift from the heading slug.
    return Topic(id=id, title=title, anchor=_github_slug(title), summary=summary, keywords=keywords)


# Ordered by MATCH PRIORITY (specific -> general), not reading order: `what_is_imajin`
# is last so its broad phrases lose to any more specific match. The guide file lists the
# same sections in reading order; the bijection test compares the two as sets.
_TOPICS: tuple[Topic, ...] = (
    _topic(
        "install_and_run",
        "Install and run",
        "Install with `uv sync` and launch with `uv run imajin` (run `--doctor` first).",
        ("install", "setup", "set up", "uv sync", "run imajin", "launch", "doctor",
         "get started", "getting started"),
    ),
    _topic(
        "open_your_data",
        "Open your data",
        "Open .lsm / .czi / OME-TIFF by drag-and-drop or File Open; channels split into layers.",
        ("open", "load", "open file", "load file", "import file", "drag and drop",
         "lsm", "czi", "ome-tiff"),
    ),
    _topic(
        "tell_imajin_your_channels",
        "Tell Imajin your channels",
        "Assign each channel a target / counterstain / ignore role so the tools know what to use.",
        ("channels", "channel role", "counterstain", "target channel", "annotate role",
         "assign role", "wavelength"),
    ),
    _topic(
        "first_analysis",
        "Your first analysis by chat",
        'Type an instruction like "find the cells in Ch2 and measure their intensity".',
        ("measure", "segment", "intensity", "find cells", "first analysis", "count cells",
         "segmentation", "measure intensity"),
    ),
    _topic(
        "reading_results",
        "Reading results",
        "Read outputs in the tables dock and figures, and save a bundle with `save_result_bundle`.",
        ("results", "tables", "table dock", "figures", "save results", "result bundle",
         "export", "save_result_bundle"),
    ),
    _topic(
        "two_ways_to_drive_it",
        "Two ways to drive it",
        "Drive everything by chat or the manual button dock; identical results and provenance.",
        ("chat dock", "manual dock", "buttons", "two ways", "interface", "gui"),
    ),
    _topic(
        "if_something_goes_wrong",
        "If something goes wrong",
        "Fixes for GPU vs CPU, provider/API key vs subscription, and running offline with Ollama.",
        ("error", "not working", "troubleshoot", "gpu", "cpu", "cuda", "no api key",
         "offline", "ollama", "crash", "fails"),
    ),
    _topic(
        "asking_for_help",
        "Asking the assistant for help",
        "Ask the assistant onboarding questions and it can link the exact guide section.",
        ("help", "ask the assistant", "assistant help"),
    ),
    _topic(
        "go_deeper",
        "Go deeper",
        "Follow-on docs: the feature reference, the capabilities matrix, and design principles.",
        ("go deeper", "advanced", "more docs", "features", "capabilities",
         "design principles", "learn more"),
    ),
    _topic(
        "what_is_imajin",
        "What is Imajin",
        "Imajin is a conversational confocal-microscopy assistant with a chat dock and manual dock.",
        ("what is imajin", "what can you do", "what can imajin do", "overview", "about"),
    ),
)


def _match(topic: str | None) -> Topic | None:
    """First topic whose id/title/keyword (casefolded) is a substring of the query.

    Direction is fixed: the entry token is the needle, the user query is the haystack. An
    empty/whitespace/None query is the overview case (returns None, not a first-entry match).
    """
    q = (topic or "").strip().casefold()
    if not q:
        return None
    for t in _TOPICS:
        tokens = (t.id, t.title, *t.keywords)
        if any(tok.casefold() in q for tok in tokens):
            return t
    return None


def _topics_payload() -> list[dict[str, str]]:
    return [{"id": t.id, "title": t.title, "summary": t.summary} for t in _TOPICS]


@tool(
    description=(
        "Return a link to the getting-started guide plus its onboarding topics. Call for "
        "meta questions about using Imajin (what can you do / how do I start / where do I "
        "find X), NOT to run an analysis. Optional `topic` returns a specific section link."
    ),
    llm=True,
    manual=False,
)
def get_help(topic: str | None = None) -> dict[str, Any]:
    """Point a new or stuck user at the onboarding guide (no file/network access)."""
    topics = _topics_payload()
    if not (topic or "").strip():
        return {
            "guide_url": GUIDE_URL,
            "topics": topics,
            "note": "Share guide_url, or call get_help('<topic>') for a section link.",
        }
    match = _match(topic)
    if match is None:
        return {
            "matched": False,
            "guide_url": GUIDE_URL,
            "topics": topics,
            "note": "No exact section; pick one of the topics above.",
        }
    return {
        "matched": True,
        "id": match.id,
        "title": match.title,
        "summary": match.summary,
        "url": f"{GUIDE_URL}#{match.anchor}",
        "topics": topics,
    }
