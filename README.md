# imajin

GUI-based AI agent for confocal microscopy analysis. Loads Zeiss `.lsm` (and `.czi`,
`.ome.tif`), runs cell segmentation (Cellpose-SAM), intensity / colocalization /
tracing analyses, and is driven by either a manual button dock or an LLM chat dock
(Claude default, ChatGPT or local Ollama optional).

## Quick start

```bash
# install (CPU-only path)
uv sync

# install with CUDA (recommended for Cellpose/Stardist speed)
uv sync --extra cuda

# verify environment
uv run imajin --doctor

# launch GUI
uv run imajin
```

## Configuration

Set provider keys via environment:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...   # optional, also used for any OpenAI-compatible endpoint
```

Local LLM via Ollama: leave keys blank, set `base_url=http://localhost:11434/v1` in
the settings dock.

## Status

In active development. See plan at
`~/.claude/plans/ultraplan-cannot-launch-remote-humble-seahorse.md`.
