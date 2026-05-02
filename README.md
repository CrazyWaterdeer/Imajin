# imajin

Conversational confocal microscopy assistant. Loads Zeiss `.lsm` / `.czi` /
OME-TIFF data into [napari](https://napari.org), runs the standard analysis stack
(Cellpose-SAM segmentation, intensity measurement, colocalization, 3D rendering,
skeleton-based morphology, cell tracking, methods writeup), and exposes every
operation through **two interchangeable interfaces**:

- a **manual button dock** (magicgui forms — LLM-free, offline, deterministic)
- an **LLM chat dock** (Claude by default; any OpenAI-compatible endpoint —
  ChatGPT, Ollama, vLLM, LM Studio — also works)

Both drivers call the same `tools/*.py` functions, so a chat command and a
button click produce identical results and identical provenance entries.

## Why

Confocal analysis today is split across Zen, Fiji/ImageJ, and ad-hoc Python.
imajin bundles the routine pipeline — load, preprocess, segment, measure,
visualize, write methods — into one app where you can either drive things
manually or say "이 z-stack에서 세포 찾고 채널2 강도 측정해줘" and watch it
happen.

## Features

- **File loading**: LSM (tifffile + `CZ_LSMINFO`), CZI (bioio-czi), OME-TIFF.
  LSM / TIFF / OME-TIFF are loaded into RAM by default for responsive Z-stack
  browsing, with automatic disk-backed memmap fallback when available RAM is
  too low; CZI remains lazy via bioio/dask. Multi-channel images split into
  per-channel layers with names from instrument metadata when present.
  Drag-and-drop registered through `npe2`.
- **Channel annotation**: simple target / counterstain / ignore roles, with
  canonical green, red, UV, and IR/far-red channel colors inferred from file
  metadata wavelengths when available, with manual annotations as overrides.
  Target channels are the default for cell segmentation, intensity measurement,
  size, and time course analysis.
- **Preprocessing**: rolling-ball background subtraction, percentile auto-
  contrast, Gaussian denoise. All scikit-image; per-channel.
- **Segmentation**: Cellpose-SAM (`cpsam` generalist model) with 2D / 3D
  toggle and GPU acceleration. Caches model weights between calls.
- **Measurement**: scikit-image `regionprops_table` per Labels layer with
  per-channel intensity columns, manual-edit-aware refresh, pandas
  `query`-style filter, group-by summary, and ROI intensity-over-time tables
  for live imaging / time-series data. Tables persist in a session registry
  and surface in a layer-linked Qt table dock.
- **Colocalization**: Manders M1/M2 (Otsu / zero / scalar threshold modes)
  and Pearson correlation, both mask-aware.
- **3D + visualization**: `set_view`, `set_colormap`, `screenshot`,
  `max_projection`, `orthogonal_views`, `animate_z_rotation` (mp4 / gif).
- **Experiment annotations**: samples, replicates, files, and layer groups can
  be annotated as control / treatment / genotype / condition groups for
  report generation and future batch summaries.
- **Cell tracking**: `track_cells` via [btrack](https://github.com/quantumjot/btrack)
  on T-axis Labels.
- **Neural morphology**: available as an isolated advanced module
  (skeletonization, branch metrics). Connectome / NBLAST hooks are stubbed
  pending a target organism / dataset and are not part of the default cell
  analysis workflow.
- **LLM-driven analysis**: provider abstraction with prompt caching for
  Anthropic and a translation layer for any OpenAI-compatible `/v1` endpoint.
  Streaming chat and tool-use are non-blocking via napari's `thread_worker`.
- **Specialist sub-agents**: `consult_neural_tracer` and
  `consult_methods_writer` route domain-specific questions to focused
  sub-agents with their own prompts and (for the tracer) their own tool sets.
- **Provenance**: every tool call lands in a per-session JSONL log with
  inputs, outputs, duration, and driver (`manual` vs `llm:<model>`). Used
  by `generate_methods` to render a deterministic Methods paragraph for
  papers, or by `consult_methods_writer` for an LLM-polished version.

## Stack

`napari ≥ 0.7` + `PyQt6`, `magicgui`, `tifffile`, `bioio` + `bioio-czi`,
`dask`, `cellpose ≥ 4`, `scikit-image`, `skan`, `btrack`, `anthropic`,
`openai`, `pydantic v2`, `torch + torchvision` (CUDA cu128 via custom uv
index). Python pinned to **3.12** because PyTorch has no `cp314` CUDA
wheels yet (PyTorch issue #169929).

## Install

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU + recent CUDA
driver for the segmentation/tracking paths.

```bash
git clone https://github.com/CrazyWaterdeer/Imajin.git
cd Imajin
uv sync
```

## Run

```bash
# environment smoke test (CUDA, imports, model download)
uv run imajin --doctor

# launch the GUI (napari + chat dock + manual dock)
uv run imajin
```

## Configuration

LLM provider keys are read from environment variables (or the in-app
settings dock):

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # Claude (default)
export OPENAI_API_KEY=sk-...               # OpenAI / Anthropic-compat backends
```

For a fully local stack, install Ollama and point the OpenAI-compatible
provider at `http://localhost:11434/v1` from the settings dock — no key
required.

## Status

Core single-file workflows are implemented (file loading → preprocessing →
cell segmentation → measurement/time-course measurement → colocalization →
3D views → reporting). Workflow templates, true folder-batch processing, and
connectome-backed neural identification are deferred. Offscreen Qt tests skip
OpenGL screenshot/animation paths; heavy model/API paths remain marked as
`slow` or `integration`.

## License

MIT.
