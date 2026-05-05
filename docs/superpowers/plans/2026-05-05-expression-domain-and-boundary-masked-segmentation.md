# Expression Domain and Boundary-Masked Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generic two-tier expression-quantification pipeline (permissive Tier-1 domain + boundary-constrained Tier-2 cells) so reporter analyses can capture both baseline expression and active subset, while keeping all existing single-tier flows unchanged.

**Architecture:** Two new public tools (`segment_expression_domain`, `detect_counterstain_channel`), one parameter added to `segment_target_objects` (`boundary_mask`), one branching extension to `analyze_target_cells`, and two optional fields on `AnalysisRecipe` (`cell_diameter_um`, `domain`). No new modules. Tier-1 uses a noise-floor threshold computed from the dark percentile of the image, with no background subtraction (cluster-safe). Tier-2 reuses the existing `segment_target_objects` pipeline, intersecting its output with the Tier-1 mask. Counterstain detection is a session-level lookup over channel annotations + a marker→nuclear table.

**Tech Stack:** Python 3.14, pandas, numpy, scipy.ndimage, scikit-image, napari (existing project tooling). Tests use pytest with the existing `viewer` fixture from `tests/conftest.py`.

**Spec:** `docs/superpowers/specs/2026-05-05-expression-domain-and-boundary-masked-segmentation-design.md`

---

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `src/imajin/tools/segment.py` | All segmentation primitives. Add `_threshold_noise_floor`, `_intersect_labels_with_mask`, `segment_expression_domain`. Modify `segment_target_objects` (add `boundary_mask`) and `_write_segmentation_qc_png` (optional secondary outline). | Modified |
| `src/imajin/tools/channels.py` | Counterstain detection lives next to existing channel-annotation tools. Add `_NUCLEAR_MARKERS` table and `detect_counterstain_channel` tool. | Modified |
| `src/imajin/tools/workflows.py` | Workflow orchestration. Add `_derive_size_params` and the two-tier branch inside `analyze_target_cells`. | Modified |
| `src/imajin/agent/state.py` | Recipe dataclass and persistence. Add `cell_diameter_um` and `domain` fields to `AnalysisRecipe`; thread them through `put_recipe`. | Modified |
| `src/imajin/tools/experiment.py` | `create_analysis_recipe` tool wraps `put_recipe`. Pass new fields through. | Modified |
| `src/imajin/project.py` | Project save/load already serializes recipes via `asdict`; no schema change needed if dataclass round-trips cleanly. Verify in tests. | Possibly modified (only if tests fail) |
| `tests/test_tools_segment.py` | Unit tests for `_threshold_noise_floor`, `_intersect_labels_with_mask`, `segment_expression_domain`, `boundary_mask` regression and behaviour, QC PNG secondary outline. | Modified |
| `tests/test_tools_channels.py` | Unit tests for `detect_counterstain_channel` resolution paths and marker lookup. | Modified |
| `tests/test_phase2_workflow.py` | End-to-end two-tier `analyze_target_cells` test on a synthetic fixture. | Modified |
| `tests/test_project_persistence.py` | Recipe round-trip with `cell_diameter_um` and `domain`. | Modified |

---

## Tasks

### Task 1: `_threshold_noise_floor` helper

Computes a noise-floor threshold from the lowest `dark_percentile`% of finite pixels: `median + k_mad * 1.4826 * MAD`. Pure function, used by `segment_expression_domain` in Task 4.

**Files:**
- Modify: `src/imajin/tools/segment.py`
- Test: `tests/test_tools_segment.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tools_segment.py`:

```python
def test_threshold_noise_floor_returns_value_above_dark_region() -> None:
    rng = np.random.default_rng(42)
    img = np.zeros((200, 200), dtype=np.float32)
    img[:, :100] = rng.normal(10.0, 1.0, (200, 100))
    img[:, 100:] = 100.0

    t = segment._threshold_noise_floor(img, k_mad=5.0, dark_percentile=10.0)

    assert 10.0 < t < 25.0, f"threshold {t} should sit above dark median + a few sigma"
    assert t < 100.0, "threshold must stay below the bright region"


def test_threshold_noise_floor_handles_constant_image() -> None:
    img = np.full((50, 50), 7.0, dtype=np.float32)
    t = segment._threshold_noise_floor(img, k_mad=5.0, dark_percentile=10.0)
    assert t == pytest.approx(7.0)


def test_threshold_noise_floor_ignores_non_finite() -> None:
    img = np.full((50, 50), np.nan, dtype=np.float32)
    img[:25, :25] = 5.0
    t = segment._threshold_noise_floor(img, k_mad=3.0, dark_percentile=20.0)
    assert np.isfinite(t)
    assert t >= 5.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tools_segment.py::test_threshold_noise_floor_returns_value_above_dark_region -v`

Expected: FAIL with `AttributeError: module 'imajin.tools.segment' has no attribute '_threshold_noise_floor'`.

- [ ] **Step 3: Add the helper to `segment.py`**

Insert after `_robust_background_sigma` (around line 383 in `src/imajin/tools/segment.py`):

```python
def _threshold_noise_floor(
    image: np.ndarray,
    *,
    k_mad: float,
    dark_percentile: float,
) -> float:
    finite = np.asarray(image[np.isfinite(image)], dtype=np.float32)
    if finite.size == 0:
        return 0.0
    if float(finite.max()) <= float(finite.min()):
        return float(finite.min())
    cutoff = float(np.percentile(finite, dark_percentile))
    dark = finite[finite <= cutoff]
    if dark.size == 0:
        dark = finite
    med = float(np.median(dark))
    mad = float(np.median(np.abs(dark - med)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0.0:
        return med
    return med + float(k_mad) * sigma
```

- [ ] **Step 4: Run all three tests to verify they pass**

Run: `uv run pytest tests/test_tools_segment.py -k threshold_noise_floor -v`

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/segment.py tests/test_tools_segment.py
git commit -m "feat(segment): add _threshold_noise_floor helper for noise-based thresholding"
```

---

### Task 2: `_intersect_labels_with_mask` helper

Sets labels outside a binary mask to 0. Used by `segment_target_objects` boundary intersection in Task 6.

**Files:**
- Modify: `src/imajin/tools/segment.py`
- Test: `tests/test_tools_segment.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tools_segment.py`:

```python
def test_intersect_labels_with_mask_zeros_outside() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[1:4, 1:4] = 1
    labels[6:9, 6:9] = 2

    mask = np.zeros_like(labels, dtype=bool)
    mask[0:5, 0:5] = True

    out = segment._intersect_labels_with_mask(labels, mask)

    assert (out == 1).sum() == (labels == 1).sum()
    assert (out == 2).sum() == 0


def test_intersect_labels_with_mask_renumbers_when_requested() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[1:3, 1:3] = 5
    labels[1:3, 4:6] = 9

    mask = np.ones_like(labels, dtype=bool)

    out = segment._intersect_labels_with_mask(labels, mask, renumber=True)

    unique = sorted(np.unique(out).tolist())
    assert unique == [0, 1, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tools_segment.py::test_intersect_labels_with_mask_zeros_outside -v`

Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Add the helper to `segment.py`**

Insert near `_remove_small_labeled_objects` (around line 322):

```python
def _intersect_labels_with_mask(
    labels: np.ndarray,
    mask: np.ndarray,
    *,
    renumber: bool = False,
) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32)
    binary = np.asarray(mask, dtype=bool)
    if arr.shape != binary.shape:
        raise ValueError(
            f"labels shape {arr.shape} does not match mask shape {binary.shape}"
        )
    out = np.where(binary, arr, 0).astype(np.int32, copy=False)
    if not renumber:
        return out
    unique = np.unique(out)
    unique = unique[unique > 0]
    if unique.size == 0:
        return out
    remap = np.zeros(int(unique.max()) + 1, dtype=np.int32)
    remap[unique] = np.arange(1, unique.size + 1, dtype=np.int32)
    return remap[out]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_tools_segment.py -k intersect_labels -v`

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/segment.py tests/test_tools_segment.py
git commit -m "feat(segment): add _intersect_labels_with_mask helper"
```

---

### Task 3: `detect_counterstain_channel` + nuclear marker lookup

Resolves which loaded layer is the counterstain and whether its marker is nuclear. Reads the existing `_CHANNELS` annotation store via `list_channel_annotations()`. Adds the marker→nuclear lookup table.

**Files:**
- Modify: `src/imajin/tools/channels.py`
- Test: `tests/test_tools_channels.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_tools_channels.py`:

```python
def test_detect_counterstain_returns_annotated_topro(viewer) -> None:
    import numpy as np
    from imajin.tools import channels as channels_tools
    from imajin.agent.state import put_channel_annotation

    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="ch3_633")
    put_channel_annotation(
        layer_name="ch3_633",
        role="counterstain",
        marker="topro",
    )

    result = channels_tools.detect_counterstain_channel()

    assert result["counterstain_layer"] == "ch3_633"
    assert result["counterstain_marker"] == "topro"
    assert result["is_nuclear"] is True
    assert result["confidence"] == "annotated"
    assert result["needs_user_confirmation"] is False


def test_detect_counterstain_marks_non_nuclear_phalloidin(viewer) -> None:
    import numpy as np
    from imajin.tools import channels as channels_tools
    from imajin.agent.state import put_channel_annotation

    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="ch_actin")
    put_channel_annotation(
        layer_name="ch_actin",
        role="counterstain",
        marker="phalloidin",
    )

    result = channels_tools.detect_counterstain_channel()
    assert result["counterstain_marker"] == "phalloidin"
    assert result["is_nuclear"] is False


def test_detect_counterstain_infers_from_633_layer_name(viewer) -> None:
    import numpy as np
    from imajin.tools import channels as channels_tools

    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="Ch3-633")
    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="Ch1-488")

    result = channels_tools.detect_counterstain_channel()

    assert result["counterstain_layer"] == "Ch3-633"
    assert result["confidence"] == "inferred"
    assert result["needs_user_confirmation"] is True
    assert result["is_nuclear"] is None
    assert "Ch3-633" in result["candidate_layers"]


def test_detect_counterstain_returns_none_when_absent(viewer) -> None:
    import numpy as np
    from imajin.tools import channels as channels_tools

    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="Ch1-488")

    result = channels_tools.detect_counterstain_channel()
    assert result["counterstain_layer"] is None
    assert result["confidence"] == "none"
    assert result["needs_user_confirmation"] is False
    assert result["candidate_layers"] == []


def test_detect_counterstain_filters_by_sample_layers(viewer) -> None:
    import numpy as np
    from imajin.tools import channels as channels_tools
    from imajin.agent.state import put_channel_annotation, put_sample

    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="A_topro")
    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="B_topro")
    put_channel_annotation(layer_name="A_topro", role="counterstain", marker="topro")
    put_channel_annotation(layer_name="B_topro", role="counterstain", marker="topro")

    put_sample(sample_name="sampleA", layers=["A_topro"])

    result = channels_tools.detect_counterstain_channel(sample_name="sampleA")
    assert result["counterstain_layer"] == "A_topro"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools_channels.py -k detect_counterstain -v`

Expected: All fail with `AttributeError: module 'imajin.tools.channels' has no attribute 'detect_counterstain_channel'`.

- [ ] **Step 3: Implement in `channels.py`**

Add at the top of `src/imajin/tools/channels.py` after the existing imports:

```python
_NUCLEAR_MARKERS: dict[str, bool] = {
    "topro": True,
    "to-pro": True,
    "to pro": True,
    "to-pro-3": True,
    "topro-3": True,
    "topro3": True,
    "dapi": True,
    "hoechst": True,
    "draq5": True,
    "nc82": False,
    "bruchpilot": False,
    "phalloidin": False,
}


def _normalize_marker(value: str | None) -> str | None:
    if not value:
        return None
    return value.strip().lower().replace("_", "-")


def _marker_is_nuclear(marker: str | None) -> bool | None:
    norm = _normalize_marker(marker)
    if norm is None:
        return None
    return _NUCLEAR_MARKERS.get(norm)


def _layer_name_suggests_far_red(layer_name: str) -> bool:
    text = layer_name.lower().replace("_", " ")
    keywords = ("633", "640", "647", "far red", "farred")
    return any(k in text for k in keywords)
```

Then append the new `@tool` definition at the bottom:

```python
@tool(
    name="detect_counterstain_channel",
    description="Identify the counterstain channel for the current sample (or all "
    "loaded layers if no sample given). Resolution priority: layers annotated as "
    "role=counterstain first; otherwise layers whose name suggests a far-red "
    "(633/640/647) wavelength. Returns confidence and whether the marker is "
    "nuclear (TOPRO/DAPI/Hoechst). Used by expression-domain workflows to decide "
    "whether to intersect the reporter mask with a structural counterstain.",
    phase="1.5",
)
def detect_counterstain_channel(
    sample_name: str | None = None,
) -> dict[str, Any]:
    from imajin.agent.state import (
        get_sample,
        list_channel_annotations,
        viewer_or_none,
    )

    sample_layer_names: set[str] | None = None
    if sample_name is not None:
        sample = get_sample(sample_name)
        sample_layer_names = {str(n) for n in (sample.layers or [])}

    annotations = list_channel_annotations()
    annotated_counterstain = [
        entry
        for entry in annotations
        if entry.get("role") == "counterstain"
        and (
            sample_layer_names is None
            or entry.get("layer_name") in sample_layer_names
        )
    ]
    if annotated_counterstain:
        first = annotated_counterstain[0]
        marker = first.get("marker")
        return {
            "counterstain_layer": first.get("layer_name"),
            "counterstain_marker": _normalize_marker(marker),
            "is_nuclear": _marker_is_nuclear(marker),
            "confidence": "annotated",
            "needs_user_confirmation": False,
            "candidate_layers": [
                entry.get("layer_name") for entry in annotated_counterstain
            ],
        }

    viewer = viewer_or_none()
    candidates: list[str] = []
    if viewer is not None:
        for layer in viewer.layers:
            name = str(layer.name)
            if sample_layer_names is not None and name not in sample_layer_names:
                continue
            if _layer_name_suggests_far_red(name):
                candidates.append(name)

    if candidates:
        return {
            "counterstain_layer": candidates[0],
            "counterstain_marker": None,
            "is_nuclear": None,
            "confidence": "inferred",
            "needs_user_confirmation": True,
            "candidate_layers": candidates,
        }

    return {
        "counterstain_layer": None,
        "counterstain_marker": None,
        "is_nuclear": None,
        "confidence": "none",
        "needs_user_confirmation": False,
        "candidate_layers": [],
    }
```

- [ ] **Step 4: Run all detect tests to verify they pass**

Run: `uv run pytest tests/test_tools_channels.py -k detect_counterstain -v`

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/channels.py tests/test_tools_channels.py
git commit -m "feat(channels): add detect_counterstain_channel tool with marker lookup"
```

---

### Task 4: `segment_expression_domain` core (no counterstain)

Permissive thresholding tool. Cluster-safe (no background subtraction). This task implements the reporter-only branch; counterstain intersection is added in Task 5.

**Files:**
- Modify: `src/imajin/tools/segment.py`
- Test: `tests/test_tools_segment.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_tools_segment.py`:

```python
def test_segment_expression_domain_captures_dim_and_bright_regions(viewer) -> None:
    rng = np.random.default_rng(0)
    img = np.zeros((200, 200), dtype=np.float32)
    img[:, :] = rng.normal(5.0, 1.0, img.shape)
    img[40:80, 40:80] += 30.0
    img[100:160, 100:160] += 8.0

    viewer.add_image(img, name="reporter")

    res = segment.segment_expression_domain(
        "reporter",
        k_mad=5.0,
        dark_percentile=10.0,
        min_area_um2=1.0,
    )

    assert res["empty_mask"] is False
    assert res["n_components"] >= 2
    labels = np.asarray(viewer.layers[res["labels_layer"]].data)
    assert labels[60, 60] > 0, "bright region must be inside domain"
    assert labels[130, 130] > 0, "dim region must also be inside domain"
    assert labels[10, 190] == 0, "background must be excluded"


def test_segment_expression_domain_empty_mask_for_pure_noise(viewer) -> None:
    rng = np.random.default_rng(1)
    img = rng.normal(5.0, 1.0, (100, 100)).astype(np.float32)
    viewer.add_image(img, name="noise_only")

    res = segment.segment_expression_domain(
        "noise_only",
        k_mad=20.0,
        dark_percentile=10.0,
    )

    assert res["empty_mask"] is True
    assert res["n_components"] == 0


def test_segment_expression_domain_labels_layer_naming(viewer) -> None:
    img = np.zeros((50, 50), dtype=np.float32)
    img[10:40, 10:40] = 100.0
    viewer.add_image(img, name="my_reporter")

    res = segment.segment_expression_domain("my_reporter", k_mad=3.0)

    assert res["labels_layer"] == "my_reporter_domain"
    assert "my_reporter_domain" in [L.name for L in viewer.layers]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_tools_segment.py -k segment_expression_domain -v`

Expected: All fail with `AttributeError`.

- [ ] **Step 3: Implement in `segment.py`**

Add this `@tool` definition at the bottom of `src/imajin/tools/segment.py` (after the existing `segment_target_objects`):

```python
@tool(
    description="Segment a permissive expression domain on a reporter channel using "
    "a noise-floor threshold (median + k*MAD of the dark percentile). No background "
    "subtraction, so cluster interiors are preserved. Use as Tier 1 of two-tier "
    "expression analyses where baseline expression must be captured alongside "
    "active sub-objects. Returns one or a few large connected-component labels.",
    phase="2",
    worker=True,
)
def segment_expression_domain(
    image_layer: str,
    threshold_strategy: str = "noise_floor",
    k_mad: float = 5.0,
    dark_percentile: float = 10.0,
    counterstain_layer: str | None = None,
    counterstain_dilation_um: float = 0.0,
    is_nuclear: bool | None = None,
    min_area_um2: float = 5.0,
    dilation_um: float = 0.0,
    save_qc_png: bool = True,
    qc_png_path: str | None = None,
) -> dict[str, Any]:
    if threshold_strategy != "noise_floor":
        raise ValueError(
            f"threshold_strategy must be 'noise_floor' (got {threshold_strategy!r})"
        )

    L = call_on_main(snapshot_layer, image_layer)
    data = L.data
    data = np.asarray(data.compute() if hasattr(data, "compute") else data)
    raw = np.asarray(data, dtype=np.float32)

    axes = _layer_axes_for_seg(L, raw.ndim)
    if "T" in axes:
        raise ValueError(
            f"segment_expression_domain refuses to run on a time-series layer "
            f"({axes}, shape {raw.shape})."
        )
    if raw.ndim < 2 or raw.ndim > 3:
        raise ValueError(
            f"segment_expression_domain expects 2D (YX) or 3D (ZYX), got {raw.shape}."
        )

    spacing = _voxel_spacing(tuple(L.scale), raw.ndim)
    threshold = _threshold_noise_floor(
        raw, k_mad=k_mad, dark_percentile=dark_percentile
    )
    binary = np.isfinite(raw) & (raw > threshold)

    counterstain_used = False
    counterstain_warnings: list[str] = []
    if counterstain_layer is not None:
        if not is_nuclear:
            counterstain_warnings.append(
                "counterstain marker is non-nuclear or unknown; reporter-only "
                "domain used"
            )
        else:
            cs_layer = call_on_main(snapshot_layer, counterstain_layer)
            cs_data = cs_layer.data
            cs_data = np.asarray(
                cs_data.compute() if hasattr(cs_data, "compute") else cs_data,
                dtype=np.float32,
            )
            if cs_data.shape != raw.shape:
                counterstain_warnings.append(
                    f"counterstain shape {cs_data.shape} differs from reporter "
                    f"shape {raw.shape}; counterstain ignored"
                )
            else:
                from skimage import filters as _filters
                cs_finite = cs_data[np.isfinite(cs_data)]
                if cs_finite.size and float(cs_finite.max()) > float(cs_finite.min()):
                    cs_threshold = float(_filters.threshold_otsu(cs_finite))
                    cs_binary = np.isfinite(cs_data) & (cs_data > cs_threshold)
                    if counterstain_dilation_um > 0 and spacing is not None:
                        cs_binary = _dilate_binary_um(
                            cs_binary,
                            spacing=spacing,
                            radius_um=counterstain_dilation_um,
                        )
                    binary = binary & cs_binary
                    counterstain_used = True
                else:
                    counterstain_warnings.append(
                        "counterstain has no usable signal; reporter-only domain used"
                    )

    if min_area_um2 > 0 and spacing is not None:
        physical_min_size = _min_size_from_physical(
            min_size=None,
            min_volume_um3=None,
            min_area_um2=min_area_um2,
            spacing=spacing,
            ndim=raw.ndim,
        )
        if physical_min_size:
            binary = _remove_small_binary_objects(binary, physical_min_size)

    if dilation_um > 0 and spacing is not None:
        binary = _dilate_binary_um(binary, spacing=spacing, radius_um=dilation_um)

    if not np.any(binary):
        empty = np.zeros(raw.shape, dtype=np.int32)
        out_name = f"{L.name}_domain"
        layer = call_on_main(
            add_labels_from_worker,
            empty,
            name=out_name,
            scale=tuple(L.scale),
            metadata={
                "source_layer": L.name,
                "segmentation_method": "expression_domain",
                "noise_floor_threshold": float(threshold),
                "k_mad": float(k_mad),
                "dark_percentile": float(dark_percentile),
                "counterstain_used": counterstain_used,
                "counterstain_warnings": counterstain_warnings,
                "empty_mask": True,
            },
        )
        return {
            "labels_layer": layer.name,
            "n_components": 0,
            "domain_area_um2": 0.0,
            "noise_floor_threshold": float(threshold),
            "counterstain_used": counterstain_used,
            "counterstain_warnings": counterstain_warnings,
            "qc_png_path": None,
            "qc_png_error": None,
            "qc_png_skipped_reason": "empty mask",
            "empty_mask": True,
        }

    from skimage import measure as _measure

    labels = _measure.label(binary, connectivity=1).astype(np.int32)
    n_components = int(labels.max())
    if spacing is not None and len(spacing) >= 2:
        if raw.ndim == 3:
            voxel_area = float(spacing[1] * spacing[2])
        else:
            voxel_area = float(spacing[0] * spacing[1])
        domain_area_um2 = float(np.count_nonzero(labels)) * voxel_area
    else:
        domain_area_um2 = float(np.count_nonzero(labels))

    out_name = f"{L.name}_domain"
    layer = call_on_main(
        add_labels_from_worker,
        labels,
        name=out_name,
        scale=tuple(L.scale),
        metadata={
            "source_layer": L.name,
            "segmentation_method": "expression_domain",
            "noise_floor_threshold": float(threshold),
            "k_mad": float(k_mad),
            "dark_percentile": float(dark_percentile),
            "counterstain_used": counterstain_used,
            "counterstain_warnings": counterstain_warnings,
            "n_components": n_components,
            "domain_area_um2": domain_area_um2,
            "empty_mask": False,
        },
    )

    saved_qc_png: str | None = None
    qc_png_error: str | None = None
    qc_png_skipped_reason: str | None = None
    if save_qc_png:
        try:
            out_path = (
                normalize_user_path(qc_png_path).resolve()
                if qc_png_path
                else _default_qc_png_path(layer.name)
            )
            saved_qc_png, qc_png_skipped_reason = _save_qc_png(
                raw,
                labels,
                out_path,
                labels_layer=layer.name,
                source_layer=L.name,
                method="expression_domain",
                force=qc_png_path is not None,
            )
            if saved_qc_png:
                try:
                    layer.metadata["qc_png_path"] = saved_qc_png
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001
            qc_png_error = f"{type(exc).__name__}: {exc}"

    return {
        "labels_layer": layer.name,
        "n_components": n_components,
        "domain_area_um2": domain_area_um2,
        "noise_floor_threshold": float(threshold),
        "counterstain_used": counterstain_used,
        "counterstain_warnings": counterstain_warnings,
        "qc_png_path": saved_qc_png,
        "qc_png_error": qc_png_error,
        "qc_png_skipped_reason": qc_png_skipped_reason,
        "empty_mask": False,
    }
```

Add the `_dilate_binary_um` helper near the other helpers (around line 332):

```python
def _dilate_binary_um(
    binary: np.ndarray,
    *,
    spacing: tuple[float, ...],
    radius_um: float,
) -> np.ndarray:
    from scipy import ndimage as ndi

    if radius_um <= 0:
        return binary
    pixel_radius_per_axis: list[int] = []
    for sp in spacing[-binary.ndim:]:
        pr = max(1, int(round(float(radius_um) / float(sp))))
        pixel_radius_per_axis.append(pr)
    structure = np.ones(
        tuple(2 * r + 1 for r in pixel_radius_per_axis), dtype=bool
    )
    return ndi.binary_dilation(binary, structure=structure)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tools_segment.py -k segment_expression_domain -v`

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/segment.py tests/test_tools_segment.py
git commit -m "feat(segment): add segment_expression_domain Tier-1 primitive"
```

---

### Task 5: Counterstain intersection branch test

The intersection logic is already implemented in Task 4. This task adds an explicit test that verifies the counterstain branch behaves as specified, including the non-nuclear warning path.

**Files:**
- Modify: `tests/test_tools_segment.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_tools_segment.py`:

```python
def test_segment_expression_domain_intersects_with_nuclear_counterstain(viewer) -> None:
    rng = np.random.default_rng(2)
    reporter = np.zeros((200, 200), dtype=np.float32)
    reporter[:, :] = rng.normal(5.0, 1.0, reporter.shape)
    reporter[20:180, 20:180] += 30.0

    counterstain = np.zeros_like(reporter)
    counterstain[40:80, 40:80] = 200.0
    counterstain[100:140, 100:140] = 200.0

    viewer.add_image(reporter, name="reporter")
    viewer.add_image(counterstain, name="topro")

    res = segment.segment_expression_domain(
        "reporter",
        k_mad=5.0,
        counterstain_layer="topro",
        is_nuclear=True,
        counterstain_dilation_um=0.0,
        min_area_um2=0.0,
    )

    labels = np.asarray(viewer.layers[res["labels_layer"]].data)
    assert res["counterstain_used"] is True
    assert res["counterstain_warnings"] == []
    assert labels[60, 60] > 0
    assert labels[120, 120] > 0
    assert labels[90, 90] == 0, "between nuclei must be excluded by counterstain"


def test_segment_expression_domain_skips_non_nuclear_counterstain(viewer) -> None:
    reporter = np.zeros((100, 100), dtype=np.float32)
    reporter[20:80, 20:80] = 50.0
    counterstain = np.zeros_like(reporter)
    counterstain[10:90, 10:90] = 200.0

    viewer.add_image(reporter, name="rep2")
    viewer.add_image(counterstain, name="actin")

    res = segment.segment_expression_domain(
        "rep2",
        k_mad=3.0,
        counterstain_layer="actin",
        is_nuclear=False,
        min_area_um2=0.0,
    )

    assert res["counterstain_used"] is False
    assert any(
        "non-nuclear" in w for w in res["counterstain_warnings"]
    )
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_tools_segment.py -k counterstain -v`

Expected: 2 passed (logic was implemented in Task 4).

- [ ] **Step 3: Commit**

```bash
git add tests/test_tools_segment.py
git commit -m "test(segment): cover counterstain branches of segment_expression_domain"
```

---

### Task 6: Add `boundary_mask` parameter to `segment_target_objects`

Adds an optional labels-layer name. When set, the final labels are intersected with the binarized boundary mask.

**Files:**
- Modify: `src/imajin/tools/segment.py:905-1116`
- Test: `tests/test_tools_segment.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_tools_segment.py`:

```python
def test_segment_target_objects_boundary_mask_keeps_only_inside(viewer) -> None:
    img = np.zeros((100, 100), dtype=np.float32)
    img[10:30, 10:30] = 200.0
    img[60:80, 60:80] = 200.0
    viewer.add_image(img, name="img")

    boundary = np.zeros((100, 100), dtype=np.int32)
    boundary[0:50, 0:50] = 1
    viewer.add_labels(boundary, name="boundary")

    res = segment.segment_target_objects(
        "img",
        boundary_mask="boundary",
        save_qc_png=False,
    )

    labels = np.asarray(viewer.layers[res["labels_layer"]].data)
    assert (labels[10:30, 10:30] > 0).any(), "object inside boundary kept"
    assert (labels[60:80, 60:80] == 0).all(), "object outside boundary dropped"


def test_segment_target_objects_default_unchanged(viewer) -> None:
    img = np.zeros((100, 100), dtype=np.float32)
    img[40:60, 40:60] = 200.0
    viewer.add_image(img, name="reg")

    res = segment.segment_target_objects("reg", save_qc_png=False)

    assert res["n_objects"] >= 1
    labels = np.asarray(viewer.layers[res["labels_layer"]].data)
    assert (labels[40:60, 40:60] > 0).any()
```

- [ ] **Step 2: Run tests to confirm boundary test fails, default still passes**

Run: `uv run pytest tests/test_tools_segment.py -k "segment_target_objects_boundary_mask or segment_target_objects_default_unchanged" -v`

Expected: `boundary_mask_keeps_only_inside` fails with `TypeError: unexpected keyword argument 'boundary_mask'`. The default-unchanged test passes (regression baseline).

- [ ] **Step 3: Add `boundary_mask` parameter and intersection logic**

In `src/imajin/tools/segment.py`, modify the `segment_target_objects` signature (around line 905) to add `boundary_mask`:

```python
def segment_target_objects(
    image_layer: str,
    background_radius: int = 48,
    background_method: str = "opening",
    background_percentile: float = 20.0,
    threshold_method: str = "auto",
    threshold_percentile: float = 99.0,
    min_snr: float = 2.0,
    high_snr: float = 4.0,
    min_size: int | None = None,
    min_area_um2: float | None = None,
    min_volume_um3: float | None = None,
    smoothing_sigma: float = 1.0,
    fill_holes: bool = True,
    split_touching: bool = False,
    min_distance: int = 20,
    min_distance_um: float | None = None,
    save_qc_png: bool = True,
    qc_png_path: str | None = None,
    boundary_mask: str | None = None,
) -> dict[str, Any]:
```

Then, just before the `out_name = f"{L.name}_objects"` line (around line 1014), insert the intersection step:

```python
    if boundary_mask is not None:
        boundary_layer_snapshot = call_on_main(snapshot_layer, boundary_mask)
        boundary_data = boundary_layer_snapshot.data
        boundary_data = np.asarray(
            boundary_data.compute() if hasattr(boundary_data, "compute") else boundary_data
        )
        if boundary_data.shape != masks.shape:
            raise ValueError(
                f"boundary_mask shape {boundary_data.shape} does not match "
                f"target image shape {masks.shape}"
            )
        masks = _intersect_labels_with_mask(
            masks, boundary_data > 0, renumber=True
        )
        qc = _label_qc(masks)
        signal_qc, qc_warnings = _target_object_qc(
            raw,
            corrected_for_threshold,
            masks,
            noise_sigma=noise_sigma,
        )
```

Also add `boundary_mask` to the metadata dict returned in the `add_labels_from_worker` call (around line 1023) and to the final return value (around line 1083). Insert into the metadata dict:

```python
            "boundary_mask": boundary_mask,
```

And in the final return dict (around line 1100), insert:

```python
        "boundary_mask": boundary_mask,
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_tools_segment.py -k "segment_target_objects_boundary_mask or segment_target_objects_default_unchanged" -v`

Expected: 2 passed.

- [ ] **Step 5: Run full segment test file to verify no regression**

Run: `uv run pytest tests/test_tools_segment.py -v --no-header -q`

Expected: All previous tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/imajin/tools/segment.py tests/test_tools_segment.py
git commit -m "feat(segment): add boundary_mask parameter to segment_target_objects"
```

---

### Task 7: QC PNG secondary outline

Extend `_write_segmentation_qc_png` and `_save_qc_png` to optionally render a second outline (e.g., domain boundary) in dashed cyan, leaving the primary mask boundaries in orange.

**Files:**
- Modify: `src/imajin/tools/segment.py:203-260`
- Test: `tests/test_tools_segment.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_tools_segment.py`:

```python
def test_qc_png_renders_secondary_outline(tmp_path) -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    image[20:30, 20:30] = 100.0
    primary = np.zeros((64, 64), dtype=np.int32)
    primary[22:28, 22:28] = 1
    secondary = np.zeros((64, 64), dtype=np.int32)
    secondary[15:35, 15:35] = 1

    out_path = tmp_path / "two_tier_qc.png"
    segment._write_segmentation_qc_png(
        image,
        primary,
        out_path,
        secondary_outline_mask=secondary,
    )

    assert out_path.exists()
    rgb = np.asarray(Image.open(out_path))
    cyan_pixels = (
        (rgb[..., 1] > 150)
        & (rgb[..., 2] > 150)
        & (rgb[..., 0] < 100)
    )
    assert cyan_pixels.any(), "secondary outline should render in cyan"
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_tools_segment.py::test_qc_png_renders_secondary_outline -v`

Expected: FAIL with `TypeError: unexpected keyword argument 'secondary_outline_mask'`.

- [ ] **Step 3: Update `_write_segmentation_qc_png`**

Replace the body of `_write_segmentation_qc_png` in `src/imajin/tools/segment.py` (around line 203) with:

```python
def _write_segmentation_qc_png(
    image: np.ndarray,
    masks: np.ndarray,
    path: Path,
    *,
    secondary_outline_mask: np.ndarray | None = None,
) -> None:
    from PIL import Image
    from skimage.segmentation import find_boundaries

    image_plane, mask_plane = _project_for_qc(image, masks)
    base = _normalize_uint8(image_plane)
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    labels = np.asarray(mask_plane, dtype=np.int64)
    if labels.size and int(labels.max()) > 0:
        rng = np.random.default_rng(12345)
        colors = rng.integers(
            32,
            256,
            size=(int(labels.max()) + 1, 3),
            dtype=np.uint8,
        ).astype(np.float32)
        colors[0] = 0
        mask = labels > 0
        alpha = 0.38
        rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * colors[labels[mask]]
    boundaries = find_boundaries(mask_plane, mode="outer")
    rgb[boundaries] = np.asarray([255, 64, 0], dtype=np.uint8)

    if secondary_outline_mask is not None:
        _, secondary_plane = _project_for_qc(image, secondary_outline_mask)
        secondary_boundaries = find_boundaries(secondary_plane, mode="outer")
        rgb[secondary_boundaries] = np.asarray([0, 200, 220], dtype=np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/test_tools_segment.py::test_qc_png_renders_secondary_outline -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/segment.py tests/test_tools_segment.py
git commit -m "feat(segment): add secondary_outline_mask to QC PNG renderer"
```

---

### Task 8: `AnalysisRecipe` schema + `put_recipe` extension

Add `cell_diameter_um` and `domain` fields to the dataclass and let `put_recipe` accept them.

**Files:**
- Modify: `src/imajin/agent/state.py:115-156`
- Test: `tests/test_project_persistence.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_project_persistence.py`:

```python
def test_recipe_round_trip_with_cell_diameter_and_domain(tmp_path) -> None:
    from imajin.agent.state import (
        get_recipe,
        put_recipe,
        reset_recipes,
    )
    from imajin.project import Project, create_project

    create_project(tmp_path / "proj")
    reset_recipes()
    put_recipe(
        name="r1",
        target_channel="green",
        cell_diameter_um=15.0,
        domain={
            "strategy": "noise_floor",
            "k_mad": 5.0,
            "dark_percentile": 10.0,
            "counterstain_layer": None,
            "counterstain_dilation_um": 0.0,
            "min_area_um2": None,
            "dilation_um": 0.0,
        },
    )

    Project.current().save()
    reset_recipes()
    Project.current().load()

    rec = get_recipe("r1")
    assert rec.cell_diameter_um == pytest.approx(15.0)
    assert rec.domain is not None
    assert rec.domain["strategy"] == "noise_floor"
    assert rec.domain["k_mad"] == pytest.approx(5.0)


def test_recipe_round_trip_without_domain_defaults_none(tmp_path) -> None:
    from imajin.agent.state import (
        get_recipe,
        put_recipe,
        reset_recipes,
    )
    from imajin.project import Project, create_project

    create_project(tmp_path / "proj2")
    reset_recipes()
    put_recipe(name="rNo", target_channel="green")

    Project.current().save()
    reset_recipes()
    Project.current().load()

    rec = get_recipe("rNo")
    assert rec.cell_diameter_um is None
    assert rec.domain is None
```

(Add `import pytest` at top if not present.)

- [ ] **Step 2: Run tests to confirm failure**

Run: `uv run pytest tests/test_project_persistence.py -k "recipe_round_trip" -v`

Expected: FAIL with `TypeError: put_recipe() got an unexpected keyword argument 'cell_diameter_um'` or `AttributeError`.

- [ ] **Step 3: Extend `AnalysisRecipe` dataclass and `put_recipe`**

In `src/imajin/agent/state.py`, modify the `AnalysisRecipe` dataclass (lines 115-126) to:

```python
@dataclass
class AnalysisRecipe:
    recipe_id: str
    name: str
    target_channel: str | None = None
    preprocessing: list[dict[str, Any]] = field(default_factory=list)
    segmentation: dict[str, Any] = field(default_factory=dict)
    measurement: dict[str, Any] = field(default_factory=dict)
    timecourse: dict[str, Any] | None = None
    colocalization: list[tuple[str, str]] = field(default_factory=list)
    notes: str | None = None
    cell_diameter_um: float | None = None
    domain: dict[str, Any] | None = None
```

Modify `put_recipe` (around line 131) to accept and store the new fields:

```python
def put_recipe(
    name: str,
    target_channel: str | None = None,
    preprocessing: list[dict[str, Any]] | None = None,
    segmentation: dict[str, Any] | None = None,
    measurement: dict[str, Any] | None = None,
    timecourse: dict[str, Any] | None = None,
    colocalization: list[tuple[str, str]] | None = None,
    notes: str | None = None,
    cell_diameter_um: float | None = None,
    domain: dict[str, Any] | None = None,
) -> str:
    name = name.strip()
    if not name:
        raise ValueError("recipe name must not be empty")
    _RECIPES[name] = AnalysisRecipe(
        recipe_id=name,
        name=name,
        target_channel=target_channel,
        preprocessing=list(preprocessing or []),
        segmentation=dict(segmentation or {}),
        measurement=dict(measurement or {}),
        timecourse=dict(timecourse) if timecourse else None,
        colocalization=list(colocalization or []),
        notes=notes,
        cell_diameter_um=cell_diameter_um,
        domain=dict(domain) if domain else None,
    )
    _autosave_project("recipe_saved")
    return name
```

Update the project-load branch (around line 783) where recipes are reconstructed. Find:

```python
            target_channel=rec.get("target_channel"),
            preprocessing=list(rec.get("preprocessing") or []),
```

After those lines, add:

```python
            cell_diameter_um=rec.get("cell_diameter_um"),
            domain=dict(rec["domain"]) if rec.get("domain") else None,
```

(Place these inside the same `put_recipe` call kwargs.)

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_project_persistence.py -k "recipe_round_trip" -v`

Expected: 2 passed.

- [ ] **Step 5: Run all persistence tests to ensure no regression**

Run: `uv run pytest tests/test_project_persistence.py -v`

Expected: All passed.

- [ ] **Step 6: Commit**

```bash
git add src/imajin/agent/state.py tests/test_project_persistence.py
git commit -m "feat(state): add cell_diameter_um and domain to AnalysisRecipe"
```

---

### Task 9: `create_analysis_recipe` tool accepts new fields

Thread the new fields through the public tool that wraps `put_recipe`.

**Files:**
- Modify: `src/imajin/tools/experiment.py:478-512`
- Test: `tests/test_phase3_experiment.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_phase3_experiment.py`:

```python
def test_create_analysis_recipe_passes_through_domain(viewer) -> None:
    from imajin.agent.state import get_recipe, reset_recipes
    from imajin.tools.experiment import create_analysis_recipe

    reset_recipes()
    create_analysis_recipe(
        name="calexa_recipe",
        target_channel="green",
        cell_diameter_um=15.0,
        domain={"strategy": "noise_floor", "k_mad": 5.0},
    )

    rec = get_recipe("calexa_recipe")
    assert rec.cell_diameter_um == 15.0
    assert rec.domain == {"strategy": "noise_floor", "k_mad": 5.0}
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_phase3_experiment.py::test_create_analysis_recipe_passes_through_domain -v`

Expected: FAIL.

- [ ] **Step 3: Update `create_analysis_recipe`**

In `src/imajin/tools/experiment.py` (around line 478), find the existing function definition and add the new parameters. Replace the function with:

```python
@tool(
    description="Create or replace a reusable analysis recipe. Captures target "
    "channel, preprocessing steps, segmentation params, and measurement settings. "
    "Optional cell_diameter_um drives Tier-2 size derivation; optional domain "
    "block enables two-tier expression-domain analysis.",
    phase="3",
)
def create_analysis_recipe(
    name: str,
    target_channel: str | None = None,
    preprocessing: list[dict[str, Any]] | None = None,
    segmentation: dict[str, Any] | None = None,
    measurement: dict[str, Any] | None = None,
    timecourse: dict[str, Any] | None = None,
    colocalization: list[tuple[str, str]] | None = None,
    notes: str | None = None,
    cell_diameter_um: float | None = None,
    domain: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from imajin.agent.state import put_recipe

    recipe_id = put_recipe(
        name=name,
        target_channel=target_channel,
        preprocessing=preprocessing,
        segmentation=segmentation,
        measurement=measurement,
        timecourse=timecourse,
        colocalization=colocalization,
        notes=notes,
        cell_diameter_um=cell_diameter_um,
        domain=domain,
    )
    return {"recipe_id": recipe_id, "name": name}
```

If the original `create_analysis_recipe` had a different return shape, preserve it by adding the new fields rather than rewriting. Read the original at `src/imajin/tools/experiment.py:478-512` first; the function above mirrors the simple `{"recipe_id", "name"}` shape. If different, only add the `cell_diameter_um` and `domain` arguments to the existing signature and `put_recipe` call.

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/test_phase3_experiment.py::test_create_analysis_recipe_passes_through_domain -v`

Expected: PASS.

- [ ] **Step 5: Run all experiment tests for regression**

Run: `uv run pytest tests/test_phase3_experiment.py -v`

Expected: All passed.

- [ ] **Step 6: Commit**

```bash
git add src/imajin/tools/experiment.py tests/test_phase3_experiment.py
git commit -m "feat(experiment): thread cell_diameter_um and domain through create_analysis_recipe"
```

---

### Task 10: `_derive_size_params` helper

Translates a single user-facing `cell_diameter_um` into derived numbers used by Tier-2 size and watershed defaults.

**Files:**
- Modify: `src/imajin/tools/workflows.py`
- Test: `tests/test_phase2_workflow.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_phase2_workflow.py`:

```python
def test_derive_size_params_with_diameter() -> None:
    from imajin.tools.workflows import _derive_size_params

    out = _derive_size_params(cell_diameter_um=15.0, voxel_spacing=(0.5, 0.5))

    assert out["min_distance_um"] == pytest.approx(15.0 * 0.7)
    assert out["min_area_um2"] == pytest.approx(np.pi * (15.0 / 4) ** 2)
    assert out["cellpose_diameter_px"] == pytest.approx(15.0 / 0.5)


def test_derive_size_params_returns_empty_when_diameter_none() -> None:
    from imajin.tools.workflows import _derive_size_params

    out = _derive_size_params(cell_diameter_um=None, voxel_spacing=(0.5, 0.5))
    assert out == {}


def test_derive_size_params_handles_missing_voxel() -> None:
    from imajin.tools.workflows import _derive_size_params

    out = _derive_size_params(cell_diameter_um=10.0, voxel_spacing=None)
    assert out["min_distance_um"] == pytest.approx(7.0)
    assert "cellpose_diameter_px" not in out
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_phase2_workflow.py -k derive_size_params -v`

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add helper to `workflows.py`**

Insert near the top of `src/imajin/tools/workflows.py` (after the existing helpers):

```python
def _derive_size_params(
    cell_diameter_um: float | None,
    voxel_spacing: tuple[float, ...] | None,
) -> dict[str, float]:
    if cell_diameter_um is None or cell_diameter_um <= 0:
        return {}
    out: dict[str, float] = {
        "min_distance_um": float(cell_diameter_um) * 0.7,
        "min_area_um2": float(np.pi * (cell_diameter_um / 4.0) ** 2),
    }
    if voxel_spacing is not None:
        xy = voxel_spacing[-1]
        if xy and xy > 0:
            out["cellpose_diameter_px"] = float(cell_diameter_um) / float(xy)
    return out
```

Confirm `import numpy as np` is present at the top of `src/imajin/tools/workflows.py`. If absent, add it.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_phase2_workflow.py -k derive_size_params -v`

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/workflows.py tests/test_phase2_workflow.py
git commit -m "feat(workflows): add _derive_size_params helper"
```

---

### Task 11: Two-tier branch in `analyze_target_cells`

Add `domain_strategy`, `domain_options`, `counterstain_layer`, `cell_diameter_um` parameters. When `domain_strategy` is set, run Tier-1 + Tier-2 + concat measurements with `tier` column.

**Files:**
- Modify: `src/imajin/tools/workflows.py:352-562`
- Test: `tests/test_phase2_workflow.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_phase2_workflow.py`:

```python
def test_analyze_target_cells_two_tier_produces_long_format(viewer) -> None:
    from imajin.tools.workflows import analyze_target_cells

    rng = np.random.default_rng(0)
    img = np.zeros((200, 200), dtype=np.float32)
    img[:, :] = rng.normal(5.0, 1.0, img.shape)
    img[40:80, 40:80] += 60.0
    img[120:160, 120:160] += 12.0

    viewer.add_image(img, name="reporter", scale=(0.5, 0.5))

    res = analyze_target_cells(
        target="reporter",
        domain_strategy="noise_floor",
        domain_options={"k_mad": 5.0, "dark_percentile": 10.0, "min_area_um2": 1.0},
        cell_diameter_um=10.0,
    )

    assert res["ok"] is True
    assert res["domain_layer"].endswith("_domain")
    assert res["n_domain_components"] >= 2
    assert "tier_table_name" in res

    from imajin.agent.state import get_table
    table = get_table(res["tier_table_name"])
    assert "tier" in table.columns
    assert set(table["tier"].unique()) == {"domain", "cells"}
    domain_rows = table[table["tier"] == "domain"]
    cell_rows = table[table["tier"] == "cells"]
    assert len(domain_rows) >= 2
    assert len(cell_rows) >= 1


def test_analyze_target_cells_single_tier_unchanged(viewer) -> None:
    from imajin.tools.workflows import analyze_target_cells

    img = np.zeros((100, 100), dtype=np.float32)
    img[40:60, 40:60] = 200.0
    viewer.add_image(img, name="single_tier", scale=(0.5, 0.5))

    res = analyze_target_cells(target="single_tier")

    assert res["ok"] is True
    assert "domain_layer" not in res or res.get("domain_layer") is None
    assert "tier_table_name" not in res or res.get("tier_table_name") is None
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_phase2_workflow.py -k "two_tier or single_tier_unchanged" -v`

Expected: `two_tier_produces_long_format` fails with `TypeError: unexpected keyword argument 'domain_strategy'`. `single_tier_unchanged` passes (regression baseline).

- [ ] **Step 3: Add the parameters and the two-tier branch**

In `src/imajin/tools/workflows.py`, modify `analyze_target_cells` (around line 364) to accept the new parameters. Update the signature to:

```python
def analyze_target_cells(
    target: str | None = None,
    do_3D: bool | None = None,
    diameter: float | None = None,
    preprocess: str | None = None,
    segmentation_method: str = "target_objects",
    segmentation_options: dict[str, Any] | None = None,
    domain_strategy: str | None = None,
    domain_options: dict[str, Any] | None = None,
    counterstain_layer: str | None = None,
    cell_diameter_um: float | None = None,
) -> dict[str, Any]:
```

Just before the existing `return { "ok": True, ... }` block at the end of the function (around line 534), insert:

```python
    if domain_strategy is not None:
        if domain_strategy != "noise_floor":
            raise ValueError(
                f"domain_strategy must be 'noise_floor' (got {domain_strategy!r})"
            )

        from imajin.tools import channels as _channels
        from imajin.tools.segment import segment_expression_domain
        import pandas as _pd

        cs_layer = counterstain_layer
        cs_is_nuclear: bool | None = None
        if cs_layer is None:
            cs_info = _channels.detect_counterstain_channel()
            if cs_info["confidence"] == "annotated":
                cs_layer = cs_info["counterstain_layer"]
                cs_is_nuclear = cs_info["is_nuclear"]

        derived = _derive_size_params(cell_diameter_um, voxel)
        d_options = dict(domain_options or {})
        if cs_layer:
            d_options.setdefault("counterstain_layer", cs_layer)
            d_options.setdefault("is_nuclear", cs_is_nuclear)
        if "min_area_um2" not in d_options and "min_area_um2" in derived:
            d_options["min_area_um2"] = derived["min_area_um2"]

        domain_result = segment_expression_domain(
            image_layer=target_layer,
            **_filtered_kwargs(segment_expression_domain, d_options),
        )
        domain_layer = domain_result["labels_layer"]

        from imajin.tools import measure as _measure

        domain_measure = _measure.measure_intensity(
            labels_layer=domain_layer,
            image_layers=[seg_input_layer],
        )
        domain_table_name = domain_measure["table_name"]
        cells_table_name = measure_result["table_name"]

        from imajin.agent.state import get_table, put_table

        domain_df = get_table(domain_table_name).copy()
        cells_df = get_table(cells_table_name).copy()
        domain_df["tier"] = "domain"
        cells_df["tier"] = "cells"
        combined = _pd.concat([domain_df, cells_df], ignore_index=True, sort=False)

        tier_table_name = put_table(
            f"{target_layer}_two_tier",
            combined,
            spec={
                "tool": "analyze_target_cells",
                "mode": "two_tier",
                "target_channel": target_layer,
                "domain_layer": domain_layer,
                "cells_layer": seg_result["labels_layer"],
            },
        )

        return {
            "ok": True,
            "target_channel": target_layer,
            "target_source": resolution.source,
            "preprocess": pre_step,
            "preprocessed_layer": pre_record["new_layer"] if pre_record else None,
            "segmentation_method": method,
            "analysis_dim": "3d" if use_3d else "2d",
            "labels_layer": seg_result["labels_layer"],
            "cells_layer": seg_result["labels_layer"],
            "domain_layer": domain_layer,
            "n_domain_components": domain_result["n_components"],
            "domain_area_um2": domain_result["domain_area_um2"],
            "n_cells": int(seg_result.get("n_objects", 0)),
            "tier_table_name": tier_table_name,
            "table_name": measure_result["table_name"],
            "table_columns": measure_result["columns"],
            "qc_png_path": seg_result.get("qc_png_path"),
            "qc_png_error": seg_result.get("qc_png_error"),
            "qc_png_skipped_reason": seg_result.get("qc_png_skipped_reason"),
            "voxel_scale": voxel,
            "warnings": warnings + list(domain_result.get("counterstain_warnings", [])),
        }
```

The existing single-tier `return` block stays unchanged below — it executes when `domain_strategy is None`.

Also update Tier-2 to receive the boundary mask. Find the segmentation block (around line 450-468) where `segment_target_objects` is called inside the `target_objects` branch. Replace the call to pass `boundary_mask` from `domain_layer` (computed only when `domain_strategy` is set). Insert the boundary-mask handling by capturing `domain_layer` before segmentation:

To support this cleanly, restructure: move the Tier-1 domain segmentation *before* the Tier-2 segmentation when `domain_strategy` is set. Insert the following block **after** `snapshot = call_on_main(snapshot_layer, seg_input_layer)` (around line 410) and **before** `seg_options = dict(segmentation_options or {})` (around line 446):

```python
    pre_computed_domain_layer: str | None = None
    if domain_strategy is not None:
        # Compute Tier-1 domain first so it can constrain Tier-2.
        from imajin.tools import channels as _channels_pre
        from imajin.tools.segment import segment_expression_domain as _seg_dom
        cs_layer_pre = counterstain_layer
        cs_is_nuclear_pre: bool | None = None
        if cs_layer_pre is None:
            cs_info_pre = _channels_pre.detect_counterstain_channel()
            if cs_info_pre["confidence"] == "annotated":
                cs_layer_pre = cs_info_pre["counterstain_layer"]
                cs_is_nuclear_pre = cs_info_pre["is_nuclear"]
        d_opts = dict(domain_options or {})
        if cs_layer_pre:
            d_opts.setdefault("counterstain_layer", cs_layer_pre)
            d_opts.setdefault("is_nuclear", cs_is_nuclear_pre)
        derived_pre = _derive_size_params(cell_diameter_um, _voxel_spacing(tuple(snapshot.scale), getattr(snapshot.data, 'ndim', 2)))
        if "min_area_um2" not in d_opts and "min_area_um2" in derived_pre:
            d_opts["min_area_um2"] = derived_pre["min_area_um2"]
        domain_pre = _seg_dom(
            image_layer=target_layer,
            **_filtered_kwargs(_seg_dom, d_opts),
        )
        pre_computed_domain_layer = domain_pre["labels_layer"]
```

(Add `from imajin.tools.segment import _voxel_spacing` at top if not already imported.)

Then in the `seg_options` dict, before `segment_target_objects(...)` is called for the `target_objects` branch, add:

```python
    if pre_computed_domain_layer is not None:
        seg_options.setdefault("boundary_mask", pre_computed_domain_layer)
```

Also: update the late branch (the domain_result re-computation) to *reuse* `pre_computed_domain_layer` rather than re-segmenting. Replace `domain_result = segment_expression_domain(...)` and `domain_layer = domain_result["labels_layer"]` in the late branch with:

```python
        from imajin.agent.state import get_layer as _get_layer
        domain_layer = pre_computed_domain_layer
        domain_layer_md = dict(getattr(_get_layer(domain_layer), "metadata", {}) or {})
        domain_result = {
            "labels_layer": domain_layer,
            "n_components": int(domain_layer_md.get("n_components", 0)),
            "domain_area_um2": float(domain_layer_md.get("domain_area_um2", 0.0)),
            "counterstain_warnings": list(domain_layer_md.get("counterstain_warnings", [])),
        }
```

This avoids running domain segmentation twice.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_phase2_workflow.py -k "two_tier or single_tier_unchanged" -v`

Expected: 2 passed.

- [ ] **Step 5: Run full workflow test file for regression**

Run: `uv run pytest tests/test_phase2_workflow.py -v`

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/imajin/tools/workflows.py tests/test_phase2_workflow.py
git commit -m "feat(workflows): add two-tier branch to analyze_target_cells"
```

---

### Task 12: Domain QC PNG uses secondary outline

Wire the Tier-2 QC PNG to render the domain boundary as the secondary outline. This makes the user immediately see in QC where the domain (Tier-1) boundary lies relative to the active cells (Tier-2).

**Files:**
- Modify: `src/imajin/tools/segment.py` (`_save_qc_png` and `segment_target_objects` QC call site)
- Test: `tests/test_tools_segment.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_tools_segment.py`:

```python
def test_segment_target_objects_qc_includes_boundary_outline(viewer, tmp_path) -> None:
    img = np.zeros((100, 100), dtype=np.float32)
    img[30:50, 30:50] = 200.0
    viewer.add_image(img, name="rep_q")

    boundary = np.zeros((100, 100), dtype=np.int32)
    boundary[20:60, 20:60] = 1
    viewer.add_labels(boundary, name="b_q")

    out_path = tmp_path / "two_tier_qc.png"
    res = segment.segment_target_objects(
        "rep_q",
        boundary_mask="b_q",
        save_qc_png=True,
        qc_png_path=str(out_path),
    )

    assert res["qc_png_path"] is not None
    rgb = np.asarray(Image.open(out_path))
    cyan_pixels = (rgb[..., 1] > 150) & (rgb[..., 2] > 150) & (rgb[..., 0] < 100)
    assert cyan_pixels.any(), "domain outline must appear in cyan in Tier-2 QC PNG"
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_tools_segment.py::test_segment_target_objects_qc_includes_boundary_outline -v`

Expected: FAIL because no secondary outline is being passed yet.

- [ ] **Step 3: Update `_save_qc_png` signature and `segment_target_objects` call**

In `src/imajin/tools/segment.py`, update `_save_qc_png` (around line 233) to accept and forward the secondary mask:

```python
def _save_qc_png(
    image: np.ndarray,
    masks: np.ndarray,
    path: Path,
    *,
    labels_layer: str,
    source_layer: str,
    method: str,
    force: bool = False,
    secondary_outline_mask: np.ndarray | None = None,
) -> tuple[str | None, str | None]:
    if not force:
        reason = _small_default_qc_skip_reason(image, masks)
        if reason:
            return None, reason
    _write_segmentation_qc_png(
        image,
        masks,
        path,
        secondary_outline_mask=secondary_outline_mask,
    )
    try:
        record_result(
            "segmentation_qc_png",
            path,
            {
                "labels_layer": labels_layer,
                "source_layer": source_layer,
                "method": method,
            },
        )
    except Exception:
        pass
    return str(path), None
```

In `segment_target_objects`, just before the `_save_qc_png(...)` call (around line 1061), build the secondary mask if `boundary_mask` is set:

```python
    secondary_mask_array: np.ndarray | None = None
    if boundary_mask is not None:
        bm_snapshot = call_on_main(snapshot_layer, boundary_mask)
        bm_data = bm_snapshot.data
        bm_data = np.asarray(
            bm_data.compute() if hasattr(bm_data, "compute") else bm_data
        )
        secondary_mask_array = (bm_data > 0).astype(np.int32)
```

Then update the `_save_qc_png` call to pass it through:

```python
            saved_qc_png, qc_png_skipped_reason = _save_qc_png(
                raw,
                masks,
                out_path,
                labels_layer=layer.name,
                source_layer=L.name,
                method="target_objects",
                force=qc_png_path is not None,
                secondary_outline_mask=secondary_mask_array,
            )
```

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/test_tools_segment.py::test_segment_target_objects_qc_includes_boundary_outline -v`

Expected: PASS.

- [ ] **Step 5: Run full segment tests for regression**

Run: `uv run pytest tests/test_tools_segment.py -v --no-header`

Expected: All passed.

- [ ] **Step 6: Commit**

```bash
git add src/imajin/tools/segment.py tests/test_tools_segment.py
git commit -m "feat(segment): show boundary mask outline in Tier-2 QC PNG"
```

---

### Task 13: End-to-end synthetic two-tier integration test

A complete fixture exercising domain + boundary + counterstain in one workflow call, verifying that cluster periphery is recovered relative to the single-tier baseline.

**Files:**
- Modify: `tests/test_phase2_workflow.py`

- [ ] **Step 1: Write the integration test**

Append to `tests/test_phase2_workflow.py`:

```python
def test_two_tier_recovers_cluster_periphery_vs_single_tier(viewer) -> None:
    from imajin.tools.workflows import analyze_target_cells

    rng = np.random.default_rng(7)
    h, w = 256, 256
    img = rng.normal(5.0, 1.0, (h, w)).astype(np.float32)

    yy, xx = np.mgrid[0:h, 0:w]
    cluster_mask = ((yy - 60) ** 2 + (xx - 60) ** 2) < 40 ** 2
    halo_intensity = np.maximum(
        0,
        40.0 - 0.6 * np.sqrt((yy - 60) ** 2 + (xx - 60) ** 2),
    )
    img += halo_intensity
    img[cluster_mask] = 250.0  # saturated core

    viewer.add_image(img, name="reporter_long", scale=(0.5, 0.5))

    single = analyze_target_cells(target="reporter_long")
    assert single["ok"] is True
    single_labels = np.asarray(viewer.layers[single["labels_layer"]].data)
    single_area = int((single_labels > 0).sum())

    # Add the same image again so the workflow operates on an independent copy
    viewer.add_image(img, name="reporter_long_two", scale=(0.5, 0.5))
    two_tier = analyze_target_cells(
        target="reporter_long_two",
        domain_strategy="noise_floor",
        domain_options={"k_mad": 5.0, "min_area_um2": 1.0},
        cell_diameter_um=10.0,
    )
    assert two_tier["ok"] is True
    cell_labels = np.asarray(viewer.layers[two_tier["cells_layer"]].data)
    cell_area = int((cell_labels > 0).sum())

    assert cell_area > single_area, (
        f"two-tier active-cell area ({cell_area}) should exceed single-tier "
        f"area ({single_area}) when the cluster has a soft halo"
    )
    domain_labels = np.asarray(viewer.layers[two_tier["domain_layer"]].data)
    assert (domain_labels > 0).sum() > cell_area, (
        "domain mask must be at least as large as active-cell mask"
    )
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_phase2_workflow.py::test_two_tier_recovers_cluster_periphery_vs_single_tier -v`

Expected: PASS. If it fails because `cell_area <= single_area`, this is a real signal — the boundary-grow flow is not actually recovering periphery, indicating a bug in Task 11 wiring (likely the `boundary_mask` plumbing).

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest tests/ -v --no-header`

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_phase2_workflow.py
git commit -m "test(workflows): integration test verifies cluster periphery recovery"
```

---

## Self-Review

(Run after the plan is written, before user execution.)

**1. Spec coverage:**
- Tier 1 (`segment_expression_domain`) — Tasks 1, 4, 5
- Tier 2 (`boundary_mask` on `segment_target_objects`) — Tasks 2, 6
- Counterstain detection — Task 3
- QC PNG secondary outline — Tasks 7, 12
- Recipe schema (`cell_diameter_um`, `domain`) — Tasks 8, 9
- `_derive_size_params` — Task 10
- `analyze_target_cells` two-tier branch — Task 11
- End-to-end periphery-recovery test — Task 13

All spec sections covered.

**2. Placeholder scan:** No TBDs, no "implement appropriate", no "similar to Task N", no missing code.

**3. Type consistency:**
- `_threshold_noise_floor(image, *, k_mad, dark_percentile) -> float` — used consistently.
- `_intersect_labels_with_mask(labels, mask, *, renumber=False) -> np.ndarray` — used in Task 6.
- `segment_expression_domain` returns `labels_layer`, `n_components`, `domain_area_um2`, `noise_floor_threshold`, `counterstain_used`, `counterstain_warnings`, `qc_png_path`, `empty_mask` — Task 11 reads these fields by the same names.
- `detect_counterstain_channel` returns dict with `counterstain_layer`, `is_nuclear`, `confidence`, `needs_user_confirmation`, `candidate_layers` — Task 11 reads `confidence`, `counterstain_layer`, `is_nuclear`.
- `_derive_size_params` returns dict with `min_distance_um`, `min_area_um2`, optional `cellpose_diameter_px` — Task 11 consumes `min_area_um2`.

All names match across tasks.

---

## Execution Notes

- Run `uv run pytest <path>` consistently (project rule from `~/CLAUDE.md`).
- Each task is an independent commit. Avoid amending or batching commits.
- If Task 11 wiring causes the integration test in Task 13 to fail, debug `boundary_mask` plumbing first (likely culprit: `seg_options.setdefault("boundary_mask", ...)` not reaching `segment_target_objects`).
- The `viewer` fixture in `tests/conftest.py` already auto-resets channel/sample annotations between tests.
