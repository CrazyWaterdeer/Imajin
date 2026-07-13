# Imajin

Conversational confocal-microscopy assistant. Load your imaging data into
[napari](https://napari.org) and run the routine analysis pipeline — load,
preprocess, segment, measure, visualize, write methods — by chatting with an LLM
agent **or** clicking manual buttons. Both drive the same tools, so a chat command
and a button click produce identical results and identical provenance.

## Documentation

- 🚀 **[Getting started](docs/getting_started.md)** — clone → first analysis result.
- 📖 **[Features](docs/features.md)** — the detailed feature reference.
- 📊 **[Capabilities matrix](docs/analysis_capabilities.md)** — analysis × target ×
  tools × statistics × graph, plus a statistics-selection guide.
- 🖼 **[Figure gallery](docs/gallery/)** — every plot type at default styling.
- 🧭 **[Design principles](docs/design_principles.md)** — metadata vs meaning, no
  filename parsing, channel roles, data models.

## Why

Confocal analysis today is split across Zen, Fiji/ImageJ, and ad-hoc Python.
Imajin bundles the routine pipeline into one app where you either drive things
manually or say *"find the cells in this z-stack and measure channel 2 intensity"*
and watch it happen — offered through two interchangeable interfaces:

- a **manual button dock** (magicgui forms — LLM-free, offline, deterministic);
- an **LLM chat dock** — Claude through a **Pro/Max subscription** (no API key, via
  the Claude Agent SDK) or an **Anthropic API key**, plus any OpenAI-compatible
  endpoint (ChatGPT, Ollama, vLLM, LM Studio).

## Install

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU + recent CUDA driver
for the segmentation/tracking paths. Python is pinned to **3.12** (PyTorch has no
`cp314` CUDA wheels yet — see [features.md](docs/features.md#stack)).

```bash
git clone https://github.com/CrazyWaterdeer/Imajin.git
cd Imajin
uv sync
```

## Run

```bash
# environment smoke test (imports, CUDA, GPU renderer, provider keys)
uv run imajin --doctor

# launch the GUI (napari + chat dock + manual dock)
uv run imajin
```

## Configuration

The chat dock's model picker shows each backend's live availability; use whichever
you have credentials for. You do **not** need an API key.

- **Claude via subscription (no API key).** With the
  [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI installed and
  logged in (`claude` on your `PATH` with a Pro/Max login, or a
  `CLAUDE_CODE_OAUTH_TOKEN`), the "Claude … (subscription)" entries just work —
  Imajin drives them through the Claude Agent SDK.
- **Claude / OpenAI via API key.** Read from environment variables (or the in-app
  settings dock): `ANTHROPIC_API_KEY` for Claude, `OPENAI_API_KEY` for OpenAI and
  Anthropic-compatible backends.
- **Fully local.** Install Ollama and point the OpenAI-compatible provider at
  `http://localhost:11434/v1` from the settings dock — no key required.

See the [getting-started guide](docs/getting_started.md#if-something-goes-wrong)
for the GPU-vs-CPU and offline paths.

## Status

Core workflows are implemented (file loading → preprocessing → segmentation →
measurement/time-course → colocalization → 3D views → reporting), including
folder-batch recipes that emit self-contained result bundles next to the input
data. Sessions are ephemeral; reproducibility — and resuming a half-finished batch
— comes from bundle metadata and recipe import rather than project files. Offscreen
Qt tests skip OpenGL screenshot/animation paths; heavy model/API paths remain
marked `slow` or `integration`.

## License

MIT.
