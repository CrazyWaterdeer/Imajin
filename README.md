# imajin

Conversational confocal microscopy assistant. Loads Zeiss `.lsm` / `.czi` /
OME-TIFF data into [napari](https://napari.org), runs the standard analysis stack
(Cellpose-SAM segmentation, intensity measurement, colocalization, 3D rendering,
skeleton-based morphology, cell tracking, methods writeup), and exposes every
operation through **two interchangeable interfaces**:

- a **manual button dock** (magicgui forms — LLM-free, offline, deterministic)
- an **LLM chat dock** — Claude through a **Pro/Max subscription** (no API key,
  via the Claude Agent SDK) or an **Anthropic API key**, plus any
  OpenAI-compatible endpoint (ChatGPT, Ollama, vLLM, LM Studio)

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
  toggle and GPU acceleration. Caches model weights between calls. To restrict
  detection to a hand-drawn region, draw a polygon / rectangle / ellipse on a
  Shapes layer, run `boundary_mask_from_shapes(shapes_layer, reference_layer)`
  (rasterised via napari, transform-aware, broadcast across Z for stacks), then
  pass the result as `boundary_mask=` to `segment_target_objects` /
  `auto_segment_target` / `segment_3d_cells_auto`.
- **Measurement**: scikit-image `regionprops_table` per Labels layer with
  per-channel intensity columns, manual-edit-aware refresh, pandas
  `query`-style filter, group-by summary, and ROI intensity-over-time tables
  for live imaging / time-series data. Tables persist in a session registry
  and surface in a layer-linked Qt table dock.
- **Colocalization**: Manders M1/M2 (Otsu / zero / scalar threshold modes)
  and Pearson correlation, both mask-aware.
- **Quality control & ROI review**: QC metrics for label layers, measurement
  tables, and timecourses (`compute_segmentation_qc`, `compute_measurement_qc`,
  `compute_timecourse_qc`), pass / warning / fail status (`mark_qc_status`),
  label outlines and jump-to-object navigation — all surfaced in a QC dock. Raw
  segmentation can be curated by hand in the interactive ROI review dock.
- **Statistics**: publication-oriented descriptive summaries (`describe_table`),
  group comparison with conservative defaults (`compare_groups`), and time-course
  normalization plus response-feature extraction (`normalize_timecourse`,
  `extract_timecourse_features`).
- **Publication figures**: styled matplotlib group-distribution, time-course,
  and scatter plots (`plot_group_distribution`, `plot_timecourse`,
  `plot_scatter`) alongside the calcium ΔF/F0 heatmap.
- **3D + visualization**: `set_view`, `set_colormap`, `screenshot`,
  `max_projection`, `orthogonal_views`, `animate_z_rotation` (mp4 / gif).
- **Experiment annotations**: samples, replicates, files, and layer groups can
  be annotated as control / treatment / genotype / condition groups for
  report generation and group summaries.
- **Batch, recipes & reporting**: register a folder of files, validate
  acquisition metadata up front (`validate_analysis_metadata`), capture a
  reusable analysis recipe (`create_analysis_recipe`) and replay it across
  annotated samples, aggregate per-object measurements to sample / group level
  (`summarize_experiment`), and track progress (`get_batch_progress`). Results
  export as self-contained bundles (`save_result_bundle`) next to the input
  data; `generate_report` / `generate_experiment_report` render session- and
  experiment-level HTML or markdown.
- **Cell tracking**: `track_cells` via [btrack](https://github.com/quantumjot/btrack)
  on T-axis Labels.
- **Calcium imaging (v1)**: rolling-percentile ΔF/F0, honest defocus +
  lateral-motion gating with per-cell coverage / longest-run / missing-fraction
  reporting (`assess_calcium_timecourse`), a ΔF/F0 raster heatmap, and a synthetic
  ground-truth validation harness. Plus **v2a** confidence-gated sparse motion
  correction (`correct_calcium_motion`) — landmark tracking + ROI relocation with
  neighbour-deformation interpolation — producing a corrected ΔF/F0 table; and
  **v2b** dense piecewise-affine warp (`stabilize_calcium_dense`) for dense sheets,
  hull-bounded and gated by density/triangle/strain/fold checks.
- **Neural morphology**: an isolated advanced module — skeletonization, branch
  metrics, Sholl analysis, SWC/CSV export, and local **morphometric neuron-type
  classification / similarity search** against a labelled reference library you
  build from your own traces (registration-free, offline). Spatial NBLAST and
  external connectome lookup (neuPrint/FlyWire) are an opt-in Tier-2 backend
  (`uv sync --extra connectome`) that is not yet wired up; mouse connectomes
  (MICrONS/Allen) are out of scope. Not part of the default cell workflow.
- **LLM-driven analysis**: three interchangeable chat backends behind one dock —
  (1) **Claude via subscription**, where the Claude Agent SDK drives your
  logged-in `claude` CLI (no API key) and imajin's own tools are bridged in as an
  in-process MCP server; (2) **Claude via Anthropic API key**, direct with prompt
  caching; and (3) any **OpenAI-compatible `/v1` endpoint** (OpenAI, Ollama,
  vLLM, LM Studio) through a translation layer. The API-backed Claude / OpenAI
  entries resolve to the **latest model** for a tier (`sonnet` / `opus` / `gpt`)
  at connection time, so new releases need no code change. Streaming chat and
  tool-use are non-blocking via napari's `thread_worker`.
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
`openai`, `claude-agent-sdk` (subscription path), `pydantic v2`,
`torch + torchvision` (CUDA cu128 via custom uv index). Python pinned to
**3.12** because PyTorch has no `cp314` CUDA wheels yet (PyTorch issue #169929).

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
# environment smoke test (imports, CUDA, GPU renderer, provider keys)
uv run imajin --doctor

# launch the GUI (napari + chat dock + manual dock)
uv run imajin
```

## Configuration

The chat dock's model picker shows each backend's live availability; use
whichever you have credentials for.

**Claude via subscription (no API key).** With the
[Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI installed and
logged in (`claude` on your `PATH` with a Pro/Max login, or a
`CLAUDE_CODE_OAUTH_TOKEN`), the "Claude … (subscription)" entries work with no
further setup — imajin drives them through the Claude Agent SDK.

**Claude / OpenAI via API key.** Read from environment variables (or the in-app
settings dock):

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # Claude (Anthropic API)
export OPENAI_API_KEY=sk-...               # OpenAI / Anthropic-compat backends
```

**Fully local.** Install Ollama and point the OpenAI-compatible provider at
`http://localhost:11434/v1` from the settings dock — no key required.

## Status

Core workflows are implemented (file loading → preprocessing → segmentation →
measurement/time-course measurement → colocalization → 3D views → reporting),
including folder-batch recipes that emit self-contained result bundles next to
the input data. Sessions are ephemeral; reproducibility comes from bundle
metadata and recipe import rather than project files. Offscreen Qt tests skip
OpenGL screenshot/animation paths; heavy model/API paths remain marked as
`slow` or `integration`.

## License

MIT.
