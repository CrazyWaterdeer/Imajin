<!--
Maintenance note for contributors. Each `##` heading below is a stable anchor that
the in-app `get_help` tool (src/imajin/tools/help.py, the `_TOPICS` map) deep-links
to. When you edit this guide:
  - Renamed a heading?  Update that topic's `anchor` in `_TOPICS` (CI's anchor test
    fails on a mismatch). Keep headings ASCII and punctuation-light (match
    ^[A-Za-z0-9 ]+$) so their GitHub slug stays trivial.
  - Reworded a section?  Re-read that topic's `summary`/`keywords` for staleness —
    no test reads meaning, which is why each topic has a stable `id`.
  - Added / removed a section?  Add / remove its `_TOPICS` entry (the section<->topic
    bijection test fails until you do).
-->

# Getting started with Imajin

A short, task-oriented walkthrough: from a fresh clone to your first analysis
result. For the full feature reference see [features.md](features.md); for the
exact analysis × target × tool × statistics × graph combinations see the
[capabilities matrix](analysis_capabilities.md).

## What is Imajin

Imajin is a conversational confocal-microscopy assistant. It loads your imaging
data into [napari](https://napari.org) and runs the routine analysis pipeline —
load, preprocess, segment, measure, visualize, write methods — through two
interchangeable interfaces: a **chat dock** (an LLM agent) and a **manual button
dock** (offline forms). Both drive the exact same tools, so a chat command and a
button click produce identical results and identical provenance.

## Install and run

You need [uv](https://docs.astral.sh/uv/) and, for the segmentation and tracking
paths, an NVIDIA GPU with a recent CUDA driver.

```bash
git clone https://github.com/CrazyWaterdeer/Imajin.git
cd Imajin
uv sync
```

Then check your environment and launch the app:

```bash
uv run imajin --doctor   # smoke test: imports, CUDA, GPU renderer, provider keys
uv run imajin            # launch napari + chat dock + manual dock
```

If `--doctor` flags a missing GPU or provider key, see
[If something goes wrong](#if-something-goes-wrong) — Imajin still runs on CPU and
without an API key.

## Open your data

Drag a file onto the napari window, or use **File ▸ Open**. Imajin reads Zeiss
`.lsm` and `.czi` plus OME-TIFF. Multi-channel images split into one layer per
channel, named from the instrument metadata when it is present. Large stacks load
into RAM for responsive Z-browsing, with an automatic disk-backed fallback when
memory is tight.

## Tell Imajin your channels

Before analysing, give each channel a **role**: *target* (the thing you want to
segment and measure), *counterstain* (a reference, e.g. a nuclear stain), or
*ignore*. Roles are inferred from metadata wavelengths where possible and you can
override them. This matters because target channels are the default for
segmentation, intensity measurement, size, and time-course analysis — the tools
read the roles, never the file name.

## Two ways to drive it

Everything Imajin can do is available two ways, and they are interchangeable:

- the **manual button dock** — magicgui forms, LLM-free, offline, deterministic;
- the **chat dock** — type instructions in natural language and an LLM agent calls
  the tools for you.

Because both call the same `tools/*.py` functions, you can start in chat and finish
with a button (or vice versa) and get the same results and the same provenance log.

## Your first analysis by chat

Load an image, set your channel roles, then type an instruction into the chat dock,
for example:

> *"Find the cells in Ch2 and measure their intensity"*
> (Ch2에서 세포 찾고 세기 측정)

The assistant inspects the layers, segments the target channel (Cellpose-SAM),
measures per-object intensity, and shows you the result table — all in one go,
without stopping to ask what to do next. You can keep going in the same style:
*"now compare the two groups"*, *"plot it"*, *"save the results"*.

## Reading results

Measurements land in the **tables dock** (layer-linked, sortable, filterable).
Figures open as image layers. To capture a run, ask the assistant to save it — or
call `save_result_bundle` — and Imajin writes a self-contained bundle (tables,
figures, methods) right next to your input data.

## Asking the assistant for help

You can ask the assistant onboarding questions in plain language — *"what can you
do?"*, *"how do I get started?"*, *"where do I find the capabilities?"* — and it can
point you at the exact section of this guide. (For a concrete analysis, just ask it
to do the analysis; it will run the pipeline rather than hand you a link.)

## If something goes wrong

- **No GPU / CUDA errors.** Segmentation and tracking prefer a GPU but fall back to
  CPU (slower). `uv run imajin --doctor` tells you what was detected.
- **No API key.** You do not need one: use the Claude **subscription** path (the
  logged-in `claude` CLI) or a fully local **Ollama** endpoint. API keys
  (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) are only one of several options.
- **Fully offline.** Point the OpenAI-compatible provider at
  `http://localhost:11434/v1` (Ollama) from the settings dock — no key, no network.
- **Nothing happens on launch.** Re-run `uv run imajin --doctor` and check the
  reported provider availability and GPU renderer.

## Go deeper

- [features.md](features.md) — the detailed feature reference.
- [analysis_capabilities.md](analysis_capabilities.md) — the capabilities matrix
  (analysis × target × tools × statistics × graph) and a statistics-selection guide.
- [design_principles.md](design_principles.md) — the enduring design principles
  (metadata vs meaning, no filename parsing, channel roles, data models).
