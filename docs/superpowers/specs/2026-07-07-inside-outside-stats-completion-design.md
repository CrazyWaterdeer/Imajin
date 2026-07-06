# Inside/Outside Statistics Completion — Design

Status: design (revised after one Codex review; ready to plan)
Date: 2026-07-07
Builds on: `2026-07-07-channel-mask-inside-outside-design.md` (mask_logic /
partition_inside_outside / measure_intensity `region` column, all shipped).

## Problem

The channel-as-mask feature shipped the *measurement* half of "compare a signal inside vs
outside a domain," but left two explicitly-deferred pieces of the *statistics* half:

1. **The domain-level comparison is paired.** `partition_inside_outside` + `measure_intensity`
   give one `inside` row and one `outside` row per image. Across a batch these accumulate with
   `sample_name`, but the inside/outside values from one image are **paired** (same specimen),
   so `compare_groups`' independent Welch/Mann-Whitney is the wrong test — there is no paired /
   signed-rank / one-sample mode today (`test ∈ {ttest, welch, mannwhitney, anova, kruskal}`,
   `stats.py:431`).

2. **There is no per-object (per-cell) inside/outside comparison.** The partition is
   *domain-level* — every green sub-region collapses into one "inside." A common question is
   instead "are individual red **cells** brighter when they sit inside the green domain?" That
   needs each segmented cell classified inside/outside by its overlap with the domain. For a
   **single image** (or genuinely independent units) those cells feed the existing two-group
   `compare_groups`; across specimens they are *nested* and must be aggregated per sample and
   compared paired (see Statistics below) — the classifier produces the per-cell `region` either
   way, but nothing produces the classification today.

Both were named as follow-ups in the prior spec. They complete the statistical story and both
compose with the shipped `label_names → region` column mechanism.

## Goal

Two small additions, each reusing shipped infrastructure, with the statistics done honestly:

- **`classify_labels_by_mask`** — per-object overlap classification. Writes a per-cell
  classification and (to drive `measure_intensity`) a `label_names = {cell_label:
  "inside"|"outside"|"excluded"}` onto the cells layer, so `measure_intensity(cells, [signal])`
  emits a per-cell `region` column. Returns the classification table as the primary artifact.
- **`compare_groups` paired mode** — `test="wilcoxon"` (signed-rank) and `test="paired_t"`,
  pairing two groups within each `sample_name`; plus a **pseudoreplication warning** when an
  independent test is run over clustered object-level rows.

```
per-cell (single image / independent units):
  segment red cells → classify_labels_by_mask(red_cells, green, within=specimen)
  → measure_intensity(red_cells, ["red"])   [region per cell]
  → filter_table(region in {inside,outside}) → compare_groups(group_col="region")   # independent

per-sample domain-level (paired, the batch answer):
  (batch) inside/outside rows tagged sample_name + region
  → compare_groups(value_col=..., group_col="region", test="wilcoxon")              # paired
```

## Feature A — `classify_labels_by_mask`

```python
@tool(phase="7", worker=True)
def classify_labels_by_mask(
    labels_layer: str,                 # segmented objects (each object a unique label)
    region_layer: str,                 # the domain mask (e.g. segmented green); >0 = foreground
    overlap_threshold: float = 0.5,    # object is "inside" when >= this fraction of its voxels
                                       # lie in the region
    within_layer: str | None = None,   # optional specimen bound
    within_threshold: float = 0.5,     # object is "excluded" when < this fraction lies in within
    inside_name: str = "inside",
    outside_name: str = "outside",
    write_label_names: bool = True,     # stamp label_names on the cells layer to drive measure
    broadcast_2d_to_3d: bool = True,
    table_name: str | None = None,
) -> dict[str, Any]: ...
```

- **Overlap per object (vectorised, no per-object loop):** with `region_bool` aligned to the
  cells shape (shared `_align`/`_foreground` from `masks.py`),
  `inside = np.bincount(labels[region_bool].ravel(), minlength=n)` and
  `total = np.bincount(labels.ravel(), minlength=n)`; `overlap = inside/total` per label (0 =
  background, dropped). Verified exact on a 3-cell probe (0.0 / 0.5 / 1.0).
- **Classification:** `overlap >= overlap_threshold → inside_name` else `outside_name`; if
  `within_layer` given and an object's within-overlap `< within_threshold → "excluded"`.
- **Circularity guard (Codex):** if `region_layer` equals `labels_layer`'s source channel, or a
  caller later measures the very channel that defined the domain, the comparison is tautological.
  The tool can't see the downstream measured channel, so this is a **documented warning** (and a
  note in the return): classify on one channel, measure a *different* one.
- **Wiring to measurement (guarded in-place write, Codex #4):** when `write_label_names`,
  set `label_names = {label: class}` on the cells layer via `call_on_main`+`get_layer`, but
  **preserve and return** any prior `label_names` (`previous_label_names`), **warn on overwrite**,
  and also stamp provenance under a dedicated key
  `metadata["classification"] = {"region": mapping, "region_layer", "overlap_threshold",
  "within_layer", "within_threshold", "tool": "classify_labels_by_mask"}` so `label_names` isn't
  the sole (overloaded) record. `write_label_names=False` skips the mutation entirely (table-only).
- **Returns / table:** the stored classification table (`label`, `overlap_fraction`,
  `within_fraction` when applicable, `region`) is the primary artifact, plus `{ok, labels_layer,
  table_name, counts: {inside, outside, excluded}, n_objects, overlap_threshold, within_threshold,
  broadcast_z, previous_label_names, warnings}`. `excluded` objects are a **visible** class — the
  per-cell workflow must `filter_table` to `{inside, outside}` before a two-group `compare_groups`
  (documented; otherwise compare sees three groups → Kruskal).
- **Shape/axes:** `mask_logic` guards (2D YX / 3D ZYX; region aligned to the cells shape; a 2D→3D
  broadcast is allowed by default but **loudly warned** with `broadcast_z=True` — a real 3D domain
  should not be silently extruded).

## Feature B — `compare_groups` paired mode + pseudoreplication warning

Extend `test` to `Literal[..., "wilcoxon", "paired_t"]` (independent modes unchanged;
`test="auto"` never selects paired — pairing can't be inferred).

- **Pairing (safe aggregation, Codex #2):** paired modes require the pairing key `sample_col`.
  Reuse `_analysis_frame` at `level in {auto, sample}` → the **already-aggregated** sample-level
  frame (one row per `sample × group` via its groupby mean/median). Pivot
  `index=sample_col, columns=group_col, values=analysis_value`; **assert one row per cell** and
  **error on any duplicate** `sample × group` (never silently take `first`). Require exactly two
  groups; drop samples missing either group as incomplete pairs, **reporting the dropped count and
  `n_pairs`**; error if `sample_col` absent or `< 2` complete pairs.
- **Tests:** `wilcoxon → scipy.stats.wilcoxon(a, b)` (paired signed-rank); `paired_t →
  scipy.stats.ttest_rel(a, b)`, on the aligned per-sample vectors `a`(group A), `b`(group B).
- **Effect size (unambiguous, Codex #3):** compute **signed rank-biserial** directly from the
  differences `d = b - a`: rank `|d|` over nonzero diffs, `R+`/`R-` = summed ranks of positive/
  negative diffs, `r_rb = (R+ - R-) / (R+ + R-)`. For `paired_t`, `cohens_dz = mean(d)/std(d,
  ddof=1)`. Report `group_a`/`group_b` order and `mean/median_difference_b_minus_a` explicitly,
  plus a paired bootstrap CI of the mean paired difference.
- **Pseudoreplication warning (Codex #1):** in the existing independent path, when `data_level`
  resolves to `object` **and** `sample_col` is present with `> 1` object per sample (clustering),
  append a warning: object-level rows are nested within samples; prefer sample-level aggregation +
  paired mode for cross-specimen inference. (Cheap; steers users off pseudoreplication.)
- CSV/`register_stats_rows`/`result_table`/return shape unchanged (the return already surfaces
  `test` + `p_value` from `result_df.loc[0]`; the paired row is a 1-row df with its own columns).

## Statistics — what is valid

- **Per-cell independent** (`compare_groups` object-level): valid for a **single image** /
  exploratory analysis, or when the cell is genuinely the experimental unit. **Not** valid as
  confirmatory inference across specimens (pseudoreplication) — hence the warning + docs.
- **Domain-level inside/outside**: correctly **paired** by sample → paired mode.
- **Circularity**: never use the measured signal (or a biologically downstream one) to define the
  mask/classification and then test it as the outcome — documented in both tools.

## Non-goals (v1)

- Directional alternatives (`greater`/`less`) for the paired tests — two-sided only for now.
- A "straddling"/boundary third class beyond the `overlap_threshold` split (plus `excluded`).
- Paired tests for `> 2` groups (repeated-measures ANOVA / Friedman).
- Mixed-effects / permutation models for nested per-cell data (the warning points at it; not built).
- New plots (existing `plot_group_distribution` consumes the `region`-tagged table as-is).

## Files touched

| File | Change |
| --- | --- |
| `src/imajin/tools/masks.py` | `classify_labels_by_mask` + headless core `_classify_overlap(labels, region_bool, threshold, within_bool, within_threshold, names)` → (class-by-label dict, counts) |
| `src/imajin/tools/stats.py` | `compare_groups`: `wilcoxon`/`paired_t` via a `_paired_compare(...)` helper (safe pivot, drop-incomplete, signed rank-biserial / dz); pseudoreplication warning in the independent path |
| `tests/test_tools_masks.py` | `_classify_overlap` unit tests + tool tests incl. classify→measure→(filter)→compare per-cell path, guarded overwrite, excluded class |
| `tests/test_tools_stats.py` | paired wilcoxon/paired_t vs scipy; duplicate `sample×group` → error; incomplete-pair drop + warn + `n_pairs`; `<2` pairs / absent `sample_col` → error; `auto` still independent; pseudoreplication warning fires |

No ManualDock work (`*_layer` dropdowns; `test` scalar). No `measure_intensity` change.

## Test plan

- **`_classify_overlap` (pure):** objects with 0 % / 50 % / 100 % overlap at `threshold=0.5` →
  `outside/inside/inside` (exactly-at-threshold is inside); `within` marks an off-specimen object
  `excluded` at its own threshold; counts correct; empty labels → empty result (no raise).
- **`classify_labels_by_mask` (viewer):** red cells some inside a green blob, some out → the cells
  layer gains `label_names` + `metadata["classification"]`; a **pre-existing** `label_names` is
  preserved in `previous_label_names` and overwrite is warned; `measure_intensity(cells,["red"])`
  has a per-cell `region`; `filter_table` to inside/outside → `compare_groups(group_col="region")`
  gives `welch_ttest` in the expected direction; 2D→3D broadcast warns; excluded-by-`within` case.
- **paired `compare_groups`:** table with `sample_name`(N) × `region`{inside,outside} × value →
  `wilcoxon`/`paired_t` p-values match direct `scipy.stats.wilcoxon` / `ttest_rel` on aligned
  vectors; a sample missing one region is dropped, warned, `n_pairs` reflects it; duplicate
  `sample×region` rows at object→sample aggregation are collapsed by the mean (no `first`); a
  hand-built duplicate at sample level → error; `<2` pairs → error; missing `sample_col` → error;
  `test="auto"` still Welch; object-level clustered table → pseudoreplication warning present.

## Changelog — design → revised (accepted Codex findings)

- **#1 (pseudoreplication):** reframed per-cell as single-image/independent-units only; added a
  clustering warning to the independent path; batch guidance points to paired mode.
- **#2 (unsafe pivot):** pivot the already-aggregated sample-level frame; error on duplicate
  `sample × group` instead of `aggfunc="first"`.
- **#3 (effect size):** signed rank-biserial from `R+`/`R-` (not scipy's `W`); explicit group
  order and `diff = b - a`; Cohen's dz for paired_t.
- **#4 (in-place mutation):** preserve+return prior `label_names`, warn on overwrite, add a
  dedicated `metadata["classification"]` provenance key, and a `write_label_names=False` escape.
- **Overlap:** separate `within_threshold`; store thresholds in metadata; `excluded` kept visible
  with a documented `filter_table` step before two-group compare; broadcast loudly warned.
- **Circularity:** documented in both tools (classify one channel, measure another).
