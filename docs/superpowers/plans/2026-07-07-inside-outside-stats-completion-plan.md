# Inside/Outside Statistics Completion — Implementation Plan

Status: plan (revised after one Codex review; ready to implement)
Date: 2026-07-07
Spec: `docs/superpowers/specs/2026-07-07-inside-outside-stats-completion-design.md`

Branch `feat/inside-outside-stats` off `master`. Commit-by-commit; each commit runs its own
tests + the affected suites as a gate. Merge `--no-ff` + push when green.

Delivers: (A) `classify_labels_by_mask` (per-object inside/outside classification), and (B) a
paired mode (`wilcoxon` / `paired_t`) plus a pseudoreplication warning in `compare_groups` —
completing the inside/outside statistical story on top of the shipped channel-as-mask tools.

## Cross-cutting

- Reuse `masks.py` cores `_foreground` / `_align` (axes guard, 2D→3D broadcast + `broadcast_z`).
- Statistics live in `stats.py`; classification in `masks.py`. No `measure_intensity` change
  (already emits `region` from `label_names`). No ManualDock change (`*_layer` dropdowns; scalars).
- Headless-safe: numpy/scipy/skimage top-level ok; napari only via `snapshot_layer` /
  `get_layer` / `add_labels_from_worker` on `call_on_main`.

## Commit order

### Commit 1 — `compare_groups` paired mode + pseudoreplication warning

- `src/imajin/tools/stats.py`:
  - Extend `test: Literal[..., "wilcoxon", "paired_t"]`.
  - `_signed_rank_biserial(d) -> float`: `nz=d[d!=0]; ranks=scipy.stats.rankdata(abs(nz));
    Rp=ranks[nz>0].sum(); Rn=ranks[nz<0].sum(); return (Rp-Rn)/(Rp+Rn)` (0.0 when all zero).
  - `_paired_compare(analysis, group_col, analysis_col, sample_col, *, test, reference_group,
    n_bootstrap, seed) -> (rows, group_counts, warnings)`:
    1. `sample_col not in analysis.columns` → `ValueError` ("paired tests need sample-level data
       with a {sample_col!r} column; run at level='sample' or add sample_name").
    2. duplicate guard: `analysis.duplicated([sample_col, group_col]).any()` → `ValueError`
       (never silently aggregate with `first`); the sample-level `_analysis_frame` already yields
       one row per `sample × group`, so this only fires on malformed input.
    3. `pivot = analysis.pivot(index=sample_col, columns=group_col, values=analysis_col)`.
       Order the two columns: sorted, `reference_group` first if given → `group_a, group_b`.
       Require exactly two groups else `ValueError`.
    4. `pair = pivot[[group_a, group_b]].dropna()`; `n_pairs=len(pair)`;
       `n_dropped = len(pivot) - n_pairs`; warn the dropped count; `n_pairs < 2` → `ValueError`.
    5. `a = pair[group_a].to_numpy(); b = pair[group_b].to_numpy(); d = b - a`.
       `wilcoxon`: `stat,p = scipy.stats.wilcoxon(a, b)` (guard all-zero `d` → `stat=0,p=1`,
       warn); effect `_signed_rank_biserial(d)`, `test_name="wilcoxon_signed_rank"`.
       `paired_t`: `stat,p = scipy.stats.ttest_rel(a, b)`; `cohens_dz = mean(d)/std(d,ddof=1)`
       (guard zero std), `test_name="paired_t"`.
    6. bootstrap CI of `mean(d)` (resample `d`, reuse the seed/n_bootstrap style).
    7. row: `test, requested_test, data_level="sample", value_col, group_col, group_a, group_b,
       n_pairs, n_dropped_incomplete, mean_a, mean_b, mean/median_difference_b_minus_a,
       mean_difference_ci95_low/high, statistic, p_value, effect_size, effect_name`.
  - In `compare_groups`, right after `_analysis_frame(...)`:
    - if `test in {"wilcoxon","paired_t"}`: `rows, group_counts, pw = _paired_compare(analysis,
      group_col, analysis_col, sample_col, ...)`; `warnings += pw`; build `result_df`; skip the
      independent branch.
    - else (independent, unchanged) — add the **pseudoreplication warning**: when
      `data_level == "object"` and `sample_col in df.columns` and some sample has `>1` row
      (`df.groupby(sample_col).size().gt(1).any()`), append: object rows are nested within
      samples; prefer sample-level aggregation + a paired test for cross-specimen inference.
  - `result_table` / `register_stats_rows` / return shape unchanged (paired row is a 1-row df).
- Tests (`tests/test_tools_stats.py`):
  - `wilcoxon` / `paired_t` on an N-sample inside/outside table match direct
    `scipy.stats.wilcoxon` / `ttest_rel` on the aligned vectors (p and statistic).
  - a sample missing one region → dropped, warned, `n_pairs` reflects it.
  - a hand-built duplicate `sample × region` sample-level row → `ValueError`.
  - `< 2` complete pairs → `ValueError`; missing `sample_col` (object-only table) + paired →
    `ValueError`.
  - `test="auto"` on the same table still runs Welch (independent) — no behavior change.
  - object-level clustered table (sample_name, >1 obj/sample, level="object") → warning present.
  - signed rank-biserial sign matches the direction of `b - a` on a monotone example.

**Gate:** `pytest tests/test_tools_stats.py -q`.

### Commit 2 — `classify_labels_by_mask` (per-object classification)

- `src/imajin/tools/masks.py`:
  - `_classify_overlap(labels, region_bool, *, overlap_threshold, within_bool=None,
    within_threshold=0.5, inside_name, outside_name, excluded_name="excluded") ->
    (mapping, overlap, within_frac, counts)` (headless):
    `n=int(labels.max())+1`; `total=np.bincount(labels.ravel(), minlength=n).astype(float)`;
    `ins=np.bincount(labels[region_bool].ravel(), minlength=n).astype(float)`;
    `overlap=np.divide(ins,total,out=zeros,where=total>0)`; `within_frac` likewise if given.
    For each `lbl` with `total>0`: `excluded` if within given and `within_frac<within_threshold`,
    elif `overlap>=overlap_threshold` → inside, else outside. Return dicts + counts.
  - `_stamp_classification(labels_layer, mapping, provenance) -> prev` (main-thread; run via
    `call_on_main`): `L=get_layer(labels_layer); prev=dict(L.metadata.get("label_names") or {});
    L.metadata["label_names"]=mapping; L.metadata["classification"]=provenance; return prev`.
  - `@tool(phase="7", worker=True) def classify_labels_by_mask(labels_layer, region_layer,
    overlap_threshold=0.5, within_layer=None, within_threshold=0.5, inside_name="inside",
    outside_name="outside", write_label_names=True, broadcast_2d_to_3d=True, table_name=None)`:
    1. `_load_mask_layer(labels_layer)` (axes guard); `labels = data.astype(int32)`.
    2. `_load_mask_layer(region_layer)` → `_align(labels.shape, _foreground(region), ...)`
       (+`broadcast_z`); same for `within_layer` if given.
    3. `mapping, overlap, wfrac, counts = _classify_overlap(...)`.
    4. build a DataFrame (`label`, `overlap_fraction`, [`within_fraction`], `region`) →
       `put_table(table_name or f"{labels_layer}_classification", df, spec={tool,...})`.
    5. if `write_label_names`: `prev = call_on_main(_stamp_classification, labels_layer, mapping,
       provenance)`; if `prev` non-empty and differs → warn "overwrote existing label_names".
    6. warnings: broadcast_z (loud), circularity note (classify one channel, measure another),
       any `excluded` count reminder to `filter_table` before a two-group compare.
    7. return `{ok, labels_layer, table_name, counts, n_objects, overlap_threshold,
       within_threshold, broadcast_z, previous_label_names: prev, warnings}`.
- `src/imajin/tools/masks.py` imports: add `get_layer` (from `imajin.session`), `put_table`
  (from `imajin.session`), `pandas as pd`.
- Tests (`tests/test_tools_masks.py`):
  - **pure `_classify_overlap`:** three objects 0 %/50 %/100 % at `threshold=0.5` →
    outside/inside/inside; `within` marks an off-specimen object excluded; counts right; empty
    labels → empty mapping.
  - **tool (viewer):** red cells (a labels layer) some in a green blob, some out →
    `classify_labels_by_mask` writes `label_names` + `metadata["classification"]`;
    `measure_intensity(cells,["red"])` has per-cell `region`; `filter_table` in/out →
    `compare_groups(group_col="region")` returns `welch_ttest`, expected direction.
  - **guarded overwrite:** a pre-existing `label_names` on the cells layer → returned in
    `previous_label_names`, overwrite warned.
  - **excluded via `within`;** `write_label_names=False` leaves the layer metadata untouched
    (table still written); 2D region → 3D cells broadcast warns.

**Gate:** `pytest tests/test_tools_masks.py -q` then the full suite `pytest -q`.

### Commit 3 — docs + verify

- `README.md`: extend the channel-as-mask bullet with the two comparison paths — per-cell
  (`classify_labels_by_mask` → filter → independent `compare_groups`, single-image/exploratory)
  and per-sample domain-level (`compare_groups(..., test="wilcoxon")`, paired) — with the
  pseudoreplication + circularity caveats in one line.
- Verify (beyond unit tests): a headless end-to-end — segment red cells, classify by a green
  domain, measure, filter, `compare_groups` (independent) for one image; and a synthetic N-sample
  inside/outside table through `compare_groups(test="wilcoxon")` — print p-values and directions.

**Gate:** none (docs).

## Risks / mitigations

- **scipy `wilcoxon` on all-zero differences raises** → guard (all `d==0` → stat 0, p 1, warn).
- **pivot silently averaging duplicates** → explicit `duplicated` guard → error before pivot.
- **`_analysis_frame` returning object level for paired** (sample_col absent) → paired path errors
  with a clear message; independent path adds the pseudoreplication warning instead.
- **In-place `label_names` overwrite** → prev preserved+returned, warned, provenance under a
  separate `classification` key, and a `write_label_names=False` escape.
- **`bincount` with a huge max label** (sparse label ids) allocates `max+1` — acceptable for
  segmentation label ranges; documented, not optimised in v1.

## Out of scope (per spec)

Directional paired alternatives; a straddling/boundary class; `>2`-group paired tests;
mixed-effects/permutation models for nested data; new plots.

## Revisions from plan review (accepted Codex findings)

Concrete deltas folded into the commits above:

- **#4 skip background:** classify the label range **1..max** — never `lbl == 0` (background has
  `total > 0` too). `_align` already guarantees `region_bool.shape == labels.shape`.
- **#1 paired ordering / direction:** columns ordered as `group_a = reference_group` (baseline)
  when given else the lexically-first, `group_b` the other; run `wilcoxon(b, a)` / `ttest_rel(b, a)`
  so the statistic sign matches `d = b - a`. `reference_group` given but absent from the two
  groups → `ValueError`.
- **#2 non-destructive stamp:** `_stamp_classification` writes `{**prev, **mapping}` (preserve
  prior `label_names` keys not re-classified) and warns when a preserved key's value changes.
- **#6 finite filter (not just NaN):** after the pivot+`dropna`, additionally keep only
  `np.isfinite(a) & np.isfinite(b)` pairs (drops `inf`), counted into `n_dropped`; then the
  all-zero-`d` short-circuit. (Verified: `scipy.stats.wilcoxon(a, a)` does **not** raise — it
  returns NaN + a RuntimeWarning — so the all-zero case must be special-cased to `stat=0, p=1`.)
- **#8 `cohens_dz` guard:** `std(d, ddof=1) == 0` with non-zero mean → `np.nan` + warning
  (undefined standardized effect); all-zero `d` → the `p=1` short-circuit.
- **#9 pseudoreplication precision:** warn only when `data_level == "object"` **and**
  `sample_col in df.columns` **and** `df.groupby([sample_col, group_col]).size().gt(1).any()`
  (real clustering: multiple objects per `sample × group`), not merely multiple rows per sample.
- **#5 huge-label guard:** if `labels.max()` is implausibly large (e.g. `> 5_000_000` and
  `>> n_unique`), error suggesting `relabel_sequential` rather than allocating a giant `bincount`.
- **#3/#16 group_counts:** for paired mode, `group_counts = {group_a: n_pairs, group_b: n_pairs}`
  (complete-pair counts); documented in the result.
- **#11 threading:** `call_on_main` is synchronous (returns the callee's value — `coloc`/`measure`
  already rely on this), so the returned `prev` and the subsequent `measure_intensity` do not race.

### Gate changes

- **Commit 2 gate:** `pytest tests/test_tools_masks.py tests/test_tools_measure.py
  tests/test_tools_stats.py -q` (the classify→measure→filter→compare integration path) then the
  **full suite**.
- **Commit 3 gate:** the two smokes are **encoded as tests** (the per-cell recipe test in Commit 2
  and a paired-vs-scipy test in Commit 1), and Commit 3 runs the full suite (no "gate: none").

### Added tests (coverage gaps #7/#10/#14/#15/#16/#18)

- signed rank-biserial: assert exact `Rp`/`Rn` and value on a known example (tie-aware via
  `rankdata`), not just sign; a ties case and a one-nonzero-diff case.
- paired bootstrap resamples `d` (not `a`,`b` independently): deterministic CI with a fixed seed
  brackets `mean(d)`.
- duplicate boundary: a raw **object-level** table with many rows per `sample × region` is valid
  (aggregates fine); only a malformed **already-sample-level** duplicate `sample × group` errors.
- `within` denominator is total object pixels: an object partly in-region and partly off-specimen
  classifies by object-pixel fractions.
- circularity: same-channel classify-then-measure runs (allowed) but the tool returns the
  exploratory-only warning.
