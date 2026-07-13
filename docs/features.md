<!--
Docs ownership boundary (see CONTRIBUTING.md). analysis_capabilities.md (the
capabilities matrix) is the SOLE authority for the exhaustive set of supported
combinations (analysis × target × tool × statistics × graph). This file is a
NARRATIVE reference: it describes what each feature is for and how you reach it,
and it must NOT present itself as the authoritative exhaustive list — where a
reader needs the full supported set, link to the matrix. A new capability updates
the matrix first; this file gets a sentence only if the narrative changes.
-->

# Features

Detailed feature reference for Imajin. See also the capabilities matrix in
[analysis_capabilities.md](analysis_capabilities.md) (the authoritative list of
supported analysis × target × tool × statistics × graph combinations) and the
task-oriented [getting-started guide](getting_started.md). Back to the
[README](../README.md).

> A user-facing **capabilities matrix** — analysis type × target × tools ×
> statistics × graph options, plus a statistics-selection guide and typical
> workflows — is in [docs/analysis_capabilities.md](analysis_capabilities.md).
> A **figure gallery** (every plot type at default styling) is in
> [docs/gallery/](gallery/) — it renders on GitHub; the interactive
> [index.html](gallery/index.html) opens the same thing in a browser.
> The enduring **design principles** (metadata vs meaning, no filename parsing,
> channel roles, data models) are in [docs/design_principles.md](design_principles.md).

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
  for live imaging / time-series data. `measure_projected_intensity` projects
  the z-stack first (mean by default — the standard for intensity comparison)
  and measures 2D ROIs on the result. Tables persist in a session registry
  and surface in a layer-linked Qt table dock.
- **Colocalization**: Manders M1/M2 (Otsu / zero / scalar threshold modes)
  and Pearson correlation, both mask-aware.
- **Channel-as-mask (inside vs outside a domain)**: use one segmented channel to
  scope another. `mask_logic` does boolean set-ops on mask / label layers
  (`not` / `and` / `or` / `subtract`), so "outside the green domain" is
  `subtract(specimen, green)` or `not(green, within_layer=specimen)`.
  `partition_inside_outside(region_layer, within_layer)` builds a single
  inside/outside Labels map — bounded by a required specimen mask, with an optional
  `boundary_buffer_um` guard band around the ambiguous domain edge — that feeds
  `measure_intensity([signal])` for inside/outside signal in one call (rows carry a
  `region` column). Typical recipe: `segment_intensity_regions("green")` →
  `partition_inside_outside(green_regions, specimen)` → `measure_intensity(partition,
  ["red"])`. For a **per-object** question instead ("are individual red cells brighter
  inside green?"), `classify_labels_by_mask(cells, green, within_layer=specimen)` tags
  each segmented cell inside/outside by its overlap with the domain (writing a `region`
  per cell), then `filter_table` to inside/outside and `compare_groups(group_col=
  "region")`. Two comparison paths, with honest statistics: the **domain-level** inside
  vs outside is *paired* within a sample — compare per sample as `log2(inside/outside)`
  and test the contrast across replicates with `compare_groups(..., test="wilcoxon")`
  (paired signed-rank; `compare_groups` also warns against pseudoreplication on clustered
  per-cell rows). When replicates are analysed one file at a time, `combine_tables`
  concatenates their per-file tables into one — tagging each source with a `sample_name`
  (replicate id) — so the paired test spans replicates and the combined table exports as a
  single CSV. A CSV merged outside the app comes back in via `import_table` (the counterpart
  to `export_table`), so externally-combined data still plots and tests with the same tools.
  Combining per-file tables leaves one sparse intensity column per file; `coalesce_columns`
  collapses them into a single value column, `map_column` assigns a user-confirmed `group` from
  `sample_name`, and `select_representative_rows` keeps each sample's main region (largest object)
  before `compare_groups` — the whole pool-and-compare path stays in-app, no external scripting.
  **Per-cell** independence only holds for a single image / genuinely
  independent units. Classify on one channel and measure a *different* one — using the
  masking channel as the outcome is circular.
- **Statistics (paired)**: `compare_groups` supports independent Welch / Mann-Whitney /
  ANOVA / Kruskal and **paired** Wilcoxon signed-rank / paired-t (`test="wilcoxon"` /
  `"paired_t"`) with signed rank-biserial / Cohen's dz effect sizes, for within-sample
  designs like inside-vs-outside a domain. `test="auto"` is assumption-aware — it chooses
  parametric vs non-parametric from a **Shapiro-Wilk normality check** (consistently across 2 and
  3+ groups: Welch/ANOVA when normal, Mann-Whitney/Kruskal when not), warns when a group has n<3,
  flags a likely **paired design** (same sample in two groups) so you don't run an independent test
  on within-subject data, and reports the rationale in `test_selection`. For 3+ groups it adds a
  **multiplicity-corrected post-hoc** (Games-Howell for ANOVA, Dunn's + Holm for Kruskal) under
  `posthoc`, so which pairs differ is answered without uncorrected multiple comparisons. When aggregating objects to a per-sample value it
  **area-weights by default** (`weight_col="auto"` → total signal / total area, using the
  regionprops `area` column when present), so many small debris objects don't skew the sample
  mean one-vote-each; `plot_group_distribution` matches, and both report `weighted_by`. Pass
  `weight_col=None` for a plain per-object mean (e.g. per-cell designs). *(For the full list of
  tests, effect sizes, and when each applies, see the [statistics-selection guide in the
  capabilities matrix](analysis_capabilities.md).)*
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
  `plot_scatter`) alongside the calcium ΔF/F0 heatmap, plus multi-channel RGB
  composites (`export_channel_composite_png`) with per-channel max / mean
  projection, role-aware colormaps (counterstain → gray), and a scale bar.
  `plot_group_distribution` takes a `kind` (`box` / `bar` / `violin` / `dots` —
  points + mean±SEM, best for small n), draws **paired connecting lines**
  (`paired=True`) for within-subject designs, and **multiplicity-corrected
  post-hoc significance brackets** for 3+ groups, with palette / y-limits /
  log-scale / point styling controls. `plot_grouped_bars` renders a two-factor
  grouped ("paired") bar chart — control-vs-treated bars clustered per
  condition, with per-condition significance. Both accept a **condition matrix**
  (`condition_matrix=`) that draws filled ● / open ○ factor circles beneath the
  bars in place of long compound tick labels. All figures share a
  **colorblind-safe palette** (a de-emphasised slate-grey control) and Noto
  Sans / Serif fonts, and **auto-format axis / tick / legend labels to
  scientific typography** — raw column names never appear: underscores become
  Title Case, units move into parentheses (`area_um2` → Area (µm²)), Greek and
  special symbols are restored (`dff` → ΔF/F₀), and dimensionless intensity axes
  fall back to (A.U.). *(The full plot catalogue at default styling is the
  [figure gallery](gallery/).)*
- **3D + visualization**: `set_view`, `set_colormap`, `screenshot`,
  `max_projection`, `average_projection`, `orthogonal_views`,
  `animate_z_rotation` (mp4 / gif).
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
  experiment-level HTML or markdown. A half-finished batch resumes from its own
  output: `plan_resume` locates the prior bundle and diffs analysed vs pending
  files, then `open_result_bundle` re-imports its recipe and appends only the
  pending ones. For manual one-file-at-a-time review, `advance_to_file` steps to
  the next file while freeing each finished file's layers from memory.
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
  logged-in `claude` CLI (no API key) and Imajin's own tools are bridged in as an
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
