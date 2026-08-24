# Imajin Analysis Capabilities Matrix (User Guide)

A reference for **what** Imajin does (analysis type), **where** it acts (target),
and **how** (tools), plus which **statistics** and **graphs** attach to each
analysis. The same tools run whether you ask in natural language from the chat
dock or click a button in the manual dock.

> Notation: `tool_name` is the tool function the chat agent calls. "Target"
> means the napari layer / channel / table the analysis actually consumes.

## Capability matrix

| Analysis type | Target | Main tools (how) | Statistics | Graphs |
|---|---|---|---|---|
| File load · metadata | `.lsm` / `.czi` / OME-TIFF | `load_file`, `reload_file`, `advance_to_file` (sequential unload) | — | — |
| Channel annotation · interpretation | image layer | `annotate_channel`, `resolve_target_channel`, `detect_counterstain_channel` | — | — |
| Preprocessing | target channel | `rolling_ball_background`, `auto_contrast`, `gaussian_denoise`, `deconvolve` (Richardson-Lucy, theoretical/Gaussian PSF) | — | — |
| Cell / object segmentation | target channel | `segment_target_objects`, `auto_segment_target`, `segment_3d_cells_auto`, `cellpose_sam`, `analyze_target_cells` (one-shot segment + measure) | — | QC overlay (`compute_segmentation_qc`) |
| Expression-domain segmentation | channel intensity | `segment_intensity_regions`, `segment_expression_domain` | — | — |
| ROI-restricted segmentation | Shapes + reference layer | `boundary_mask_from_shapes` → pass as `boundary_mask=` | — | — |
| **Spot / puncta detection** | target channel (puncta, FISH, vesicles) | `detect_spots` (blob LoG/DoG, µm-scale, 2D-projection/3D, boundary-aware → Points + table), `compute_spots_qc` | `describe_table`, `compare_groups` | `plot_group_distribution` |
| **Object spatial relationships** | Points/Labels × Labels | `assign_objects_to_parents` (spots-per-cell), `measure_distance_to_reference` (distance to surface, µm), `nearest_neighbor_distances` | `compare_groups` | `plot_group_distribution`, `plot_scatter` |
| **Intensity measurement (per object)** | labels × channel | `measure_intensity`, `measure_projected_intensity` (measure after projection), `refresh_measurement` | `describe_table`, `compare_groups` | `plot_group_distribution` |
| **Timecourse intensity (ROI over time)** | ROI labels × T | `measure_intensity_over_time`, `extract_timepoint` | `normalize_timecourse`, `extract_timecourse_features` | `plot_timecourse` |
| Colocalization | channel pair · object pair | `manders_coefficients` (M1/M2), `pearson_correlation`, `costes_threshold` (auto threshold), `costes_significance` (randomization p), `object_colocalization` (proximity vs mask-constrained null) | Costes p-value | `plot_scatter` |
| **inside / outside domain** | channel mask | `mask_logic`, `partition_inside_outside`, `classify_labels_by_mask` | `compare_groups` (**paired** wilcoxon) | `plot_group_distribution` (`paired=True`) |
| Calcium imaging | ROI × T movie | `assess_calcium_timecourse` (ΔF/F0 + gating), `correct_calcium_motion`, `stabilize_calcium_dense` | — | `plot_dff_heatmap` |
| Cell tracking | T-axis labels | `track_cells` (btrack) | — | — |
| Neural morphology (advanced, separate workflow) | skeleton / trace | `enhance_neural_processes`, `skeletonize`, `prune_skeleton`, `compute_sholl_analysis`, `extract_branch_metrics`, `classify_neuron_type`, `find_similar_neurons`, `export_neural_trace` | morphometrics | Sholl, etc. |
| Filament tracer (connectivity + shape) | skeleton / mask | `propose_filament_bridges` (gap-bridge, evidence-gated + QC), `build_rooted_tree` (rooted directed tree / valid SWC), `measure_filament_diameter` (EDT diameter profile), `compute_tree_topology` (branch order, Strahler, path-to-soma) | topology metrics | — |
| **Table wrangling · pooling** | session tables | `combine_tables` (merge per-file tables + `sample_name` tag), `import_table` (external CSV → session), `coalesce_columns` (merge sparse intensity columns), `map_column` (assign group), `select_representative_rows` (main region only), `filter_table`, `summarize_table`, `export_table` | — | — |
| **Statistics** | measurement table | `describe_table`, `compare_groups`, `summarize_experiment` | (core) | `plot_group_distribution` |
| QC · ROI review | labels / table / timecourse | `compute_segmentation_qc`, `compute_measurement_qc`, `compute_timecourse_qc`, `mark_qc_status`, `review_target_roi`, `jump_to_object` | — | label outlines |
| Batch · experiment | file group | `register_files`, `annotate_samples`, `create_analysis_recipe`, `run_recipe_on_samples`, `get_batch_progress`, `plan_resume` → `open_result_bundle` (resume) | `summarize_experiment` | experiment report |
| Reporting | session / experiment | `save_result_bundle` (outputs into one folder), `generate_report`, `generate_experiment_report`, `generate_methods` | — | — |
| Visualization · 3D | layer | `set_view`, `set_colormap`, `max_projection`, `average_projection`, `orthogonal_views`, `animate_z_rotation`, `export_channel_composite_png`, `screenshot` | — | — |

## Graph options in detail

### `plot_group_distribution` — group comparison (the most options)
| Option | Values | Description |
|---|---|---|
| `kind` | `box` (default) / `bar` / `violin` / `dots` | `dots` = all points + mean±SEM crossbar (**the standard for small n**); `bar` = mean + SEM bars |
| `paired` | `True` / `False` | **per-sample connecting lines** when the same `sample_name` appears in two groups (inside/outside, before/after) |
| `show_posthoc` | `True` (default) | auto-draws **corrected post-hoc significance brackets** (Games-Howell / Dunn+Holm) for 3+ groups |
| `weight_col` | `auto` (default) / `None` / column name | **area-weighting** when aggregating objects → sample (total/total when area present). `None` = unweighted |
| `level` | `auto` / `sample` / `object` | sample-level aggregation vs. object-level |
| `palette` | `["#..","#.."]` | group colors |
| `ymin` / `ymax` / `log_y` / `zero_baseline` | | y-axis range · log · start at 0 |
| `point_size` / `jitter` / `show_points` | | point size · jitter width · show points |
| `show_n` / `show_stats` / `stats_test` | | n labels · significance display · test type |
| `format` / `title` / `ylabel` / `width` / `height` / `dpi` | | svg (default) / pdf / png, size |

### Other plots
- **`plot_grouped_bars`** — two-factor (condition × treatment) grouped/"paired" bars. `condition_col`
  forms the x-axis clusters and `group_col` (control/treated) places bars side by side within each
  (color-coded). Sample-level mean±SEM + points, a **circle legend** at the bottom, and for 2 groups
  **per-condition control-vs-treated significance** is annotated automatically. Use it to see "does
  the treatment effect depend on condition."
- **`plot_timecourse`** — mean line + band (`interval`: `sem`/`ci95`/`none`), individual traces (`show_individual`, `max_individual_traces`).
- **`plot_scatter`** — scatter of two numeric columns, group color (`group_col`), log (`log10`), regression line (`fit_line`).
- **`plot_dff_heatmap`** — calcium ΔF/F0 raster heatmap.
- **`export_channel_composite_png`** — multi-channel RGB composite (per-channel max/mean projection, role-aware colormaps, scale bar).

## Figure labeling rules (scientific labeling)

**Raw column names never appear in figures.** Axis, tick, legend, and colorbar labels are
automatically cleaned into scientific typography (`_pretty_label`). Rules:

- **remove underscores + Title Case**: `mean_intensity` → **Mean Intensity**, `outside_green` → **Outside Green**.
- **units in parentheses**: `area_um2` → **Area (µm²)**, `time_s` → **Time (s)**, `volume_um3` → **Volume (µm³)**.
- **Greek / special symbols**: `dff` → **ΔF/F₀**, `f0` → **F₀**, `log10` → **log₁₀**.
- **abbreviations / markers / channels stay uppercase**: `GFP` · `ROI` · `SEM` · `Ch2-T2` · `mCherry` are kept as-is (Title Case won't break them).
- statistics abbreviations: `sem`→SEM, `sd`→SD, `ci`→CI.
- **unknown unit → (A.U.)**: when a measurement axis has no known unit and is not dimensionless
  (count/ratio, etc.), **(A.U.)** is appended automatically. Example: `mean_intensity` → **Mean
  Intensity (A.U.)** (fluorescence intensity is in arbitrary units).

### Condition matrix
A molecular-biology notation that draws **per-factor circles beneath** the bar/box graph to indicate
each column's (bar's) condition. `condition_matrix={"Treatment":[false,true,…], "Genotype":[…]}` —
give each factor an on/off value per column and each row is drawn with **filled circles (●) for
positive, open circles (○) for negative** (factor name on the left). It replaces long compound tick
labels (WT+treated, etc.) with a clean circle grid. Supported by both `plot_group_distribution`
(columns = groups) and `plot_grouped_bars` (columns = condition clusters).

| Input (column) | Label |
|---|---|
| `mean_intensity` | Mean Intensity |
| `mean_intensity_GFP` | Mean Intensity GFP |
| `area_um2` | Area (µm²) |
| `dff` | ΔF/F₀ |
| `time_s` | Time (s) |

**Follow the same rules when you supply labels yourself**: `title` · `xlabel` · `ylabel`, when given,
are used verbatim — so use Title Case, no underscores, units in parentheses ("Mean Intensity
(a.u.)"), standard Greek symbols (ΔF/F₀), uppercase abbreviations. Fonts are Noto Sans (default) /
Noto Serif (`font="serif"`).

## Statistics selection guide

`compare_groups` is the core. `test="auto"` (default) **checks the assumptions for you**.

- **parametric vs non-parametric**: Shapiro-Wilk normality test → **Welch** (2 groups) / **ANOVA** (3+) when normal, **Mann-Whitney** / **Kruskal** when not. Consistent philosophy across 2 and 3+ groups.
- **paired design**: when the same unit is measured in two conditions (inside/outside, before/after, matched), **specify `test="wilcoxon"` (or `"paired_t"`) directly**. auto only warns when it detects a paired structure; it does not switch automatically (whether a design is paired is an experimental-design claim).
- **area-weighting**: when pooling per-object intensity into a sample value, the default is **area-weighting** (`weight_col="auto"`, total signal / total area when area present), so small debris does not get one vote per object. Use `weight_col=None` when each object is an independent observation (per-cell).
- **post-hoc for 3+ groups**: in addition to the omnibus, **multiplicity-corrected pairwise tests** are returned under `posthoc` (ANOVA→Games-Howell, Kruskal→Dunn's+Holm). Do not run uncorrected pairwise tests yourself.
- **caution**: always check the result's `warnings` / `test_selection` — small-n (n<3), non-normality, and **pseudoreplication** (treating cells as independent samples) warnings live there.

## Representative workflows

**① Single-image cell measurement**
`load_file` → (if needed `resolve_target_channel`) → `segment_target_objects` → `measure_intensity` → `describe_table` / `plot_group_distribution`.

**② Multi-file group comparison (pooling)**
Segment and measure per file → `combine_tables` (sample_name tagging) → `map_column` (sample→group) → `coalesce_columns` (merge intensity columns) → (optional) `select_representative_rows` (main region only) → `compare_groups` → `plot_group_distribution`.
A CSV combined outside the app comes back in via `import_table` and continues identically.

**③ inside / outside domain**
`segment_intensity_regions("green")` → `partition_inside_outside(green, specimen)` → `measure_intensity(partition, ["red"])` → `compare_groups(group_col="region", test="wilcoxon")` → `plot_group_distribution(paired=True)`.

**④ Timecourse / calcium**
`measure_intensity_over_time(ROI, movie)` → `normalize_timecourse` / `extract_timecourse_features` → `plot_timecourse`. For calcium, `assess_calcium_timecourse` → `plot_dff_heatmap`.

**⑤ Sequential multi-file (collected into one folder)**
Call `start_analysis(<name>)` once → analyze per file (`analyze_target_cells` appends to the open bundle, as do `save_result_bundle`, figures, stats and QC) → `finalize_analysis` at the end; do not call it between files. Each file's measurements land in `tables/<file>.csv` and `tables/combined.csv` is rebuilt across all of them, so the folder stands alone as a result set. Without `start_analysis` each file gets its own folder.

---
*This document is based on the tools actually registered in the code (110). Update it whenever new tools/options are added.*
