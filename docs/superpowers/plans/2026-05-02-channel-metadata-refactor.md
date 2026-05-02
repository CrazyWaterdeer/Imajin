# Channel/Sample Metadata Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Imajin a clean, principled separation between *acquisition metadata* (physical/instrumental facts auto-extracted from files) and *user annotations* (biological meaning that only the user can confirm), and make every loader, tool, report, and prompt honor that line.

**Architecture:**
- Introduce a real `ChannelMetadata` dataclass that all loaders (LSM/OME/CZI/TIFF) populate with as much physical detail as the file exposes (wavelengths, dye, detector, gain, pinhole, etc.). `Dataset.channel_metadata` becomes `list[ChannelMetadata]`. When the data crosses into napari (per-layer dict) or into reports/JSON, it is serialized via `to_dict()`.
- Replace `ChannelEntry` with a `ChannelAnnotation` dataclass whose default role is `"unknown"` and that holds *only* user-confirmed biology (role, marker, biological target, notes). `SampleEntry` becomes `SampleAnnotation` with a free-form `extra` dict for unstructured user terms (tissue, genotype, …).
- Add a `suggest_role()` helper that produces a non-binding `role_suggestion` from marker/dye names only — never from filename tokens, never from "IR" or "green" alone. Reports surface suggestions as suggestions, not facts.
- Reorder `resolve_channel` priority to: exact layer name → user annotation → file metadata (color/dye/wavelength/name) → layer-name substring. Ambiguity raises a clear error listing matches.

**Tech Stack:** Python 3.14 / `uv run --project /home/jin/py314`, dataclasses, `tifffile` (LSM, OME), `bioio` (CZI), `pytest`, napari layer metadata (dict-shaped), Pydantic-based tool registry.

---

## File Structure

| Path | Status | Responsibility |
| --- | --- | --- |
| `src/imajin/io/channel_metadata.py` | modify | `ChannelMetadata` dataclass, `wavelength_nm`, `color_from_*`, `suggest_role`, `build_channel_info`, `pad_channel_metadata` |
| `src/imajin/io/dataset.py` | modify | `Dataset` with typed `channel_metadata: list[ChannelMetadata]` and acquisition fields (`objective`, `scan_mode`, properties `n_timepoints`, `n_z`, `is_time_series`) |
| `src/imajin/io/lsm.py` | modify | Extract richer per-channel fields (dye, detector, filter, gain, pinhole, laser power) and dataset-level fields (objective, scan mode) |
| `src/imajin/io/ome.py` | modify | Extract Channel @Fluorophore, OME @Objective, dimension order; populate `ChannelMetadata` |
| `src/imajin/io/czi.py` | modify | Best-effort `ChannelMetadata` from `bioio` metadata; degrade to `name`-only when unavailable |
| `src/imajin/io/napari_reader.py` | modify | Serialize `ChannelMetadata` to dicts when writing layer metadata; persist `role_suggestion` per channel |
| `src/imajin/agent/state.py` | modify | `ChannelAnnotation` (default role "unknown"), `SampleAnnotation` with `extra`, refactored `resolve_layer_name` priority order |
| `src/imajin/tools/channels.py` | modify | `annotate_channel` default role "unknown"; drop required `color`; pass through annotation rename |
| `src/imajin/tools/experiment.py` | modify | `annotate_sample` accepts `extra` dict |
| `src/imajin/tools/files.py` | modify | `load_file` returns serialized channel metadata + dataset-level acquisition fields |
| `src/imajin/tools/report.py` | modify | New "Acquisition" section, "Channel Annotations" stays, "Channel Suggestions" rendered as suggestions; never assert unconfirmed roles |
| `src/imajin/agent/prompts.py` | modify | Update channel-annotation pipeline + add anti-filename-parsing rule + anti-IR-counterstain rule |
| `tests/test_io_channel_metadata.py` | modify | Cover dataclass, suggest_role, ome wavelengths |
| `tests/test_io_lsm.py` | modify | Cover dye/detector/gain/pinhole/objective + suggest_role from marker name + far-red has no role suggestion |
| `tests/test_io_loader.py` | modify | Dataset acquisition fields and properties |
| `tests/test_napari_reader.py` | modify | Channel metadata round-trip including `role_suggestion` |
| `tests/test_tools_channels.py` | modify | `unknown` default; new resolve priority; ambiguity error wording |
| `tests/test_tools_experiment.py` | modify | `extra` dict pass-through |
| `tests/test_tools_report.py` | modify | Acquisition section rendered; suggestions are suggestions, not facts |
| `tests/conftest.py` | modify | `tiny_ome_tiff` already exists; no change unless a new fixture is needed by tests |

Each task below is *self-contained*: tests in the same task as the code, with explicit run commands and commit at the end. Use the project python: `uv run --project /home/jin/py314 ...`.

---

## Task 1: ChannelMetadata Dataclass + Serialization Helpers

**Files:**
- Modify: `src/imajin/io/channel_metadata.py`
- Modify: `tests/test_io_channel_metadata.py`

- [ ] **Step 1: Write the failing dataclass tests**

Append to `tests/test_io_channel_metadata.py`:

```python
from imajin.io.channel_metadata import (
    ChannelMetadata,
    build_channel_info,
    pad_channel_metadata,
)


def test_channel_metadata_to_dict_drops_none_fields() -> None:
    md = ChannelMetadata(
        index=0,
        name="GCaMP",
        color="green",
        excitation_wavelength_nm=488.0,
    )
    d = md.to_dict()

    assert d["index"] == 0
    assert d["name"] == "GCaMP"
    assert d["color"] == "green"
    assert d["excitation_wavelength_nm"] == 488.0
    assert "emission_wavelength_nm" not in d
    assert "dye_name" not in d


def test_build_channel_info_returns_dataclass() -> None:
    md = build_channel_info(name="GCaMP", excitation=488)
    assert isinstance(md, ChannelMetadata)
    assert md.name == "GCaMP"
    assert md.color == "green"
    assert md.excitation_wavelength_nm == 488.0


def test_pad_channel_metadata_fills_missing_with_dataclass() -> None:
    existing = [ChannelMetadata(index=0, name="DAPI", color="uv")]
    padded = pad_channel_metadata(existing, n_channels=3, names=["a", "b", "c"])

    assert len(padded) == 3
    assert all(isinstance(m, ChannelMetadata) for m in padded)
    assert padded[0].name == "DAPI"
    assert padded[1].name == "b"
    assert padded[2].name == "c"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project /home/jin/py314 pytest tests/test_io_channel_metadata.py -q`
Expected: FAIL — `ChannelMetadata` not importable; `build_channel_info` returns `dict`.

- [ ] **Step 3: Implement the dataclass and update helpers**

Replace the contents of `src/imajin/io/channel_metadata.py` with:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ChannelColor = Literal["green", "red", "uv", "ir"]
ChannelRoleSuggestion = Literal["target", "counterstain", "ignore"]


@dataclass
class ChannelMetadata:
    index: int = 0
    name: str | None = None
    display_name: str | None = None
    color: ChannelColor | None = None

    excitation_wavelength_nm: float | None = None
    emission_wavelength_nm: float | None = None
    emission_range_nm: tuple[float, float] | None = None

    dye_name: str | None = None
    detector_name: str | None = None
    filter_name: str | None = None
    laser_power: float | None = None
    detector_gain: float | None = None
    pinhole_diameter: float | None = None

    role_suggestion: ChannelRoleSuggestion | None = None
    role_suggestion_reason: str | None = None

    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        out: dict[str, Any] = {}
        for k, v in d.items():
            if k == "raw":
                if v:
                    out[k] = dict(v)
                continue
            if v is None:
                continue
            if isinstance(v, tuple):
                out[k] = list(v)
            else:
                out[k] = v
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChannelMetadata":
        allowed = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in allowed and k != "raw"}
        if "emission_range_nm" in kwargs and isinstance(kwargs["emission_range_nm"], list):
            kwargs["emission_range_nm"] = tuple(kwargs["emission_range_nm"])
        raw = data.get("raw") or {}
        return cls(raw=dict(raw), **kwargs)


_NAME_COLOR_ALIASES: dict[str, str] = {
    "gfp": "green",
    "fitc": "green",
    "gcamp": "green",
    "488": "green",
    "rfp": "red",
    "dsred": "red",
    "mcherry": "red",
    "tritc": "red",
    "cy3": "red",
    "561": "red",
    "568": "red",
    "594": "red",
    "dapi": "uv",
    "hoechst": "uv",
    "405": "uv",
    "cy5": "ir",
    "alexa647": "ir",
    "alexa633": "ir",
    "farred": "ir",
    "far red": "ir",
    "633": "ir",
    "640": "ir",
    "647": "ir",
}


def _norm(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def wavelength_nm(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if val <= 0:
        return None
    if val < 1e-3:
        return val * 1e9
    if val < 10:
        return val * 1000
    return val


def color_from_wavelengths(
    excitation_nm: float | None = None,
    emission_nm: float | None = None,
) -> ChannelColor | None:
    if emission_nm is not None:
        if emission_nm < 500:
            return "uv"
        if emission_nm < 570:
            return "green"
        if emission_nm < 650:
            return "red"
        return "ir"

    if excitation_nm is not None:
        if excitation_nm <= 430:
            return "uv"
        if excitation_nm < 520:
            return "green"
        if excitation_nm < 600:
            return "red"
        return "ir"

    return None


def color_from_name(name: str | None) -> ChannelColor | None:
    if not name:
        return None
    norm = _norm(name)
    compact = norm.replace(" ", "")
    for alias, color in _NAME_COLOR_ALIASES.items():
        if alias in norm or alias in compact:
            return color
    return None


_TARGET_TOKENS = ("gfp", "gcamp", "calexa", "rfp", "mcherry", "reporter", "yfp", "venus", "td tomato", "tdtomato")
_COUNTERSTAIN_TOKENS = ("dapi", "hoechst", "topro", "to pro", "to-pro", "phalloidin")


def suggest_role(
    *,
    dye_name: str | None = None,
    marker: str | None = None,
    name: str | None = None,
) -> tuple[ChannelRoleSuggestion | None, str | None]:
    """Return (suggestion, reason) only when the marker/dye name strongly implies a role.

    Filename, color alone, and IR/far red do NOT trigger a suggestion.
    """
    candidate = " ".join(filter(None, [dye_name, marker, name])).lower()
    if not candidate:
        return None, None
    for token in _TARGET_TOKENS:
        if token in candidate:
            return "target", f"marker/dye contains {token!r}"
    for token in _COUNTERSTAIN_TOKENS:
        if token in candidate:
            return "counterstain", f"marker/dye contains {token!r}"
    return None, None


def build_channel_info(
    *,
    index: int = 0,
    name: str | None = None,
    excitation: Any = None,
    emission: Any = None,
    dye_name: str | None = None,
    detector_name: str | None = None,
    filter_name: str | None = None,
    laser_power: Any = None,
    detector_gain: Any = None,
    pinhole_diameter: Any = None,
    emission_range_nm: tuple[float, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> ChannelMetadata:
    ex_nm = wavelength_nm(excitation)
    em_nm = wavelength_nm(emission)
    color = color_from_wavelengths(ex_nm, em_nm) or color_from_name(dye_name) or color_from_name(name)

    def _f(v: Any) -> float | None:
        try:
            return float(v) if v is not None and v != "" else None
        except (TypeError, ValueError):
            return None

    md = ChannelMetadata(
        index=index,
        name=str(name) if name is not None else None,
        color=color,
        excitation_wavelength_nm=ex_nm,
        emission_wavelength_nm=em_nm,
        emission_range_nm=emission_range_nm,
        dye_name=dye_name,
        detector_name=detector_name,
        filter_name=filter_name,
        laser_power=_f(laser_power),
        detector_gain=_f(detector_gain),
        pinhole_diameter=_f(pinhole_diameter),
        raw=dict(extra or {}),
    )
    suggestion, reason = suggest_role(dye_name=dye_name, marker=None, name=name)
    md.role_suggestion = suggestion
    md.role_suggestion_reason = reason
    return md


def pad_channel_metadata(
    channel_metadata: list[ChannelMetadata] | list[dict[str, Any]],
    n_channels: int,
    names: list[str] | None = None,
) -> list[ChannelMetadata]:
    out: list[ChannelMetadata] = []
    for i, m in enumerate(list(channel_metadata)[:n_channels]):
        if isinstance(m, ChannelMetadata):
            md = m
        else:
            md = ChannelMetadata.from_dict(m)
        if md.index == 0 and i != 0:
            md.index = i
        out.append(md)
    names = names or []
    while len(out) < n_channels:
        i = len(out)
        nm = names[i] if i < len(names) else None
        out.append(build_channel_info(index=i, name=nm))
    for i, md in enumerate(out):
        if md.index != i:
            md.index = i
    return out


def serialize_channel_metadata(
    channel_metadata: list[ChannelMetadata],
) -> list[dict[str, Any]]:
    return [m.to_dict() for m in channel_metadata]
```

- [ ] **Step 4: Update existing dict-shape assertions**

Edit the original assertion in `tests/test_io_channel_metadata.py::test_parse_ome_xml_extracts_channel_wavelengths` so it accesses dataclass attributes. Replace the bottom three lines with:

```python
    assert [m.color for m in metadata] == ["uv", "green", "ir"]
    assert metadata[0].emission_wavelength_nm == pytest.approx(460)
    assert metadata[1].excitation_wavelength_nm == pytest.approx(488)
```

(The test imports OME parsing — that loader will be updated in Task 5; for now this test will fail at the dataclass attribute level once OME is updated. Mark with `pytest.mark.xfail(reason="OME loader updated in Task 5")` until Task 5.)

Actually, do not xfail. Instead defer this assertion update until Task 5. For Task 1 only the *new* dataclass-focused tests run.

- [ ] **Step 5: Run tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_io_channel_metadata.py::test_channel_metadata_to_dict_drops_none_fields tests/test_io_channel_metadata.py::test_build_channel_info_returns_dataclass tests/test_io_channel_metadata.py::test_pad_channel_metadata_fills_missing_with_dataclass -v`
Expected: PASS (3/3).

Then: `uv run --project /home/jin/py314 pytest tests/test_io_channel_metadata.py -q`
Expected: 5 passed (the wavelength + color helpers still work; the OME-XML test will error because the OME loader has not been updated yet — keep that test passing by leaving its dict-style assertions and adding a temporary `m if isinstance(m, dict) else m.to_dict()` shim in `_parse_ome_xml` only if needed; otherwise wait for Task 5).

If the OME-XML test breaks here, **defer fixing it to Task 5** and instead skip it locally with `pytest -k 'not parse_ome_xml'` for this commit only.

- [ ] **Step 6: Commit**

```bash
git add src/imajin/io/channel_metadata.py tests/test_io_channel_metadata.py
git commit -m "feat(io): introduce ChannelMetadata dataclass with role_suggestion + helpers"
```

---

## Task 2: ChannelAnnotation + SampleAnnotation Cleanup

**Files:**
- Modify: `src/imajin/agent/state.py`
- Modify: `tests/test_tools_channels.py`
- Modify: `tests/test_tools_experiment.py`

- [ ] **Step 1: Write failing tests for the new defaults and `extra`**

Append to `tests/test_tools_experiment.py`:

```python
def test_annotate_sample_records_extra_dict() -> None:
    state.reset_samples()
    res = experiment.annotate_sample(
        sample_name="m1",
        group="control",
        extra={"tissue": "midgut", "sex": "F", "genotype": "w; UAS-GFP"},
    )

    assert res["extra"]["tissue"] == "midgut"
    assert res["extra"]["genotype"] == "w; UAS-GFP"
    [stored] = experiment.list_sample_annotations()
    assert stored["extra"]["sex"] == "F"
    state.reset_samples()
```

Append to `tests/test_tools_channels.py`:

```python
def test_annotate_channel_default_role_is_unknown(viewer) -> None:
    import numpy as np
    viewer.add_image(np.zeros((4, 4), dtype=np.uint16), name="ch_default")

    res = channels.annotate_channel(layer="ch_default", marker="something")

    assert res["role"] == "unknown"
    [entry] = channels.list_channel_annotations_tool()
    assert entry["role"] == "unknown"


def test_annotate_channel_accepts_explicit_unknown(viewer) -> None:
    import numpy as np
    viewer.add_image(np.zeros((4, 4), dtype=np.uint16), name="ch_explicit")

    res = channels.annotate_channel(layer="ch_explicit", role="unknown")
    assert res["role"] == "unknown"
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run --project /home/jin/py314 pytest tests/test_tools_experiment.py::test_annotate_sample_records_extra_dict tests/test_tools_channels.py::test_annotate_channel_default_role_is_unknown tests/test_tools_channels.py::test_annotate_channel_accepts_explicit_unknown -v`
Expected: FAIL (`extra` not accepted; default role is `"target"`; `"unknown"` rejected by `canonical_channel_role`).

- [ ] **Step 3: Update `state.py` — rename + extend types**

Edit `src/imajin/agent/state.py`:

1. Rename `SampleEntry` → `SampleAnnotation` (keep an alias `SampleEntry = SampleAnnotation` for one release if needed). Add `extra: dict[str, str]` field.
2. Rename `ChannelEntry` → `ChannelAnnotation` (alias retained). Default role becomes `"unknown"`.
3. Extend `_CHANNEL_ROLE_ALIASES` so `"unknown"` resolves to `"unknown"`. Update `canonical_channel_role` to allow the four-element set `{"target", "counterstain", "ignore", "unknown"}` and to default to `"unknown"` when input is empty.
4. Update `put_sample` to accept `extra: dict[str, str] | None = None` and store it.

Concrete diff (apply via Edit):

Replace block defining `SampleEntry`:

```python
@dataclass
class SampleAnnotation:
    sample_name: str
    group: str | None = None
    layers: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    notes: str | None = None
    extra: dict[str, str] = field(default_factory=dict)


SampleEntry = SampleAnnotation  # backwards-compatible alias
```

Replace `_CHANNEL_ROLE_ALIASES`:

```python
_CHANNEL_ROLE_ALIASES: dict[str, str] = {
    "target": "target",
    "primary": "target",
    "main": "target",
    "reporter": "target",
    "counterstain": "counterstain",
    "counter": "counterstain",
    "reference": "counterstain",
    "anatomy": "counterstain",
    "ignore": "ignore",
    "exclude": "ignore",
    "unused": "ignore",
    "unknown": "unknown",
    "unspecified": "unknown",
    "tbd": "unknown",
}
```

Replace `ChannelEntry`:

```python
@dataclass
class ChannelAnnotation:
    layer_name: str
    role: str = "unknown"
    color: str | None = None  # optional override; metadata is the primary source
    marker: str | None = None
    biological_target: str | None = None
    notes: str | None = None


ChannelEntry = ChannelAnnotation  # backwards-compatible alias
```

Update `canonical_channel_role`:

```python
def canonical_channel_role(value: str | None) -> str:
    if not value:
        return "unknown"
    norm = _normalize_text(value)
    role = _CHANNEL_ROLE_ALIASES.get(norm, norm)
    if role not in {"target", "counterstain", "ignore", "unknown"}:
        raise ValueError(
            "role must be target, counterstain, ignore, or unknown"
        )
    return role
```

Update `put_sample` signature and storage:

```python
def put_sample(
    sample_name: str,
    group: str | None = None,
    layers: list[str] | None = None,
    files: list[str] | None = None,
    notes: str | None = None,
    extra: dict[str, str] | None = None,
) -> str:
    sample_name = sample_name.strip()
    if not sample_name:
        raise ValueError("sample_name must not be empty")
    cleaned_group = group.strip() if isinstance(group, str) and group.strip() else None
    _SAMPLES[sample_name] = SampleAnnotation(
        sample_name=sample_name,
        group=cleaned_group,
        layers=list(layers or []),
        files=list(files or []),
        notes=notes,
        extra={str(k): str(v) for k, v in (extra or {}).items()},
    )
    return sample_name
```

Update `_CHANNELS` typing reference in the file: replace `dict[str, ChannelEntry]` with `dict[str, ChannelAnnotation]`. Update `put_channel_annotation` to write `ChannelAnnotation` and accept `role` defaulting to `"unknown"`.

- [ ] **Step 4: Update tools to forward `extra` and the new default**

Edit `src/imajin/tools/experiment.py::annotate_sample` so it accepts `extra: dict[str, str] | None = None` and passes it to `put_sample`. The returned dict should include `extra`.

Replace the function body with:

```python
@tool(...)
def annotate_sample(
    sample_name: str,
    group: str | None = None,
    layers: list[str] | None = None,
    files: list[str] | None = None,
    notes: str | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, Any]:
    normalized_files = [str(Path(f).expanduser()) for f in (files or [])]
    name = put_sample(
        sample_name=sample_name,
        group=group,
        layers=list(layers or []),
        files=normalized_files,
        notes=notes,
        extra=extra,
    )
    return {
        "sample_name": name,
        "group": group,
        "layers": list(layers or []),
        "files": normalized_files,
        "notes": notes,
        "extra": dict(extra or {}),
    }
```

Edit `src/imajin/tools/channels.py::annotate_channel` so the default role is `"unknown"` (`role: str = "unknown"`) and update the description to "default role is `unknown` until the user confirms".

- [ ] **Step 5: Run tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_tools_experiment.py tests/test_tools_channels.py -q`
Expected: PASS for the new tests; existing ones still pass — `test_annotate_sample_records_group_metadata` already provides `group="control"` so the optional change is fine, but its expected dict no longer matches because we added an `extra` key. Update it: append `"extra": {}` to the expected dict in that test.

Update `tests/test_tools_experiment.py::test_annotate_sample_records_group_metadata` so the asserted samples list contains `"extra": {}` per entry.

- [ ] **Step 6: Commit**

```bash
git add src/imajin/agent/state.py src/imajin/tools/experiment.py src/imajin/tools/channels.py tests/test_tools_experiment.py tests/test_tools_channels.py
git commit -m "feat(state): ChannelAnnotation/SampleAnnotation dataclasses; default role unknown; sample extra dict"
```

---

## Task 3: Role Suggestion Wiring + Sanity Tests

**Files:**
- Modify: `src/imajin/io/channel_metadata.py` (already has `suggest_role`; expose import)
- Modify: `tests/test_io_channel_metadata.py`

- [ ] **Step 1: Add suggest_role tests**

Append to `tests/test_io_channel_metadata.py`:

```python
from imajin.io.channel_metadata import suggest_role


def test_suggest_role_target_for_gfp_marker() -> None:
    role, reason = suggest_role(marker="GFP nuclei reporter")
    assert role == "target"
    assert "gfp" in (reason or "").lower()


def test_suggest_role_target_for_gcamp_dye() -> None:
    role, reason = suggest_role(dye_name="GCaMP6s")
    assert role == "target"


def test_suggest_role_counterstain_for_dapi() -> None:
    role, reason = suggest_role(dye_name="DAPI")
    assert role == "counterstain"


def test_suggest_role_counterstain_for_phalloidin() -> None:
    role, reason = suggest_role(marker="phalloidin-AF633")
    assert role == "counterstain"


def test_suggest_role_none_for_far_red_alone() -> None:
    role, reason = suggest_role(name="far red")
    assert role is None
    assert reason is None


def test_suggest_role_none_for_ir_color_alone() -> None:
    role, reason = suggest_role(name="ir")
    assert role is None


def test_suggest_role_none_for_filename_token() -> None:
    role, reason = suggest_role(name="control_R3")
    assert role is None
```

- [ ] **Step 2: Run tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_io_channel_metadata.py -k suggest_role -v`
Expected: PASS (the helper was already implemented in Task 1; this commit only adds tests).

- [ ] **Step 3: Commit**

```bash
git add tests/test_io_channel_metadata.py
git commit -m "test(io): cover suggest_role for marker/dye-based suggestions"
```

---

## Task 4: Dataset Acquisition-Level Fields

**Files:**
- Modify: `src/imajin/io/dataset.py`
- Modify: `tests/test_io_loader.py`

- [ ] **Step 1: Write failing dataset properties test**

Append to `tests/test_io_loader.py`:

```python
def test_dataset_exposes_n_timepoints_and_n_z(tiny_ome_tiff: Path) -> None:
    ds = load_dataset(tiny_ome_tiff)
    # tiny_ome_tiff is CZYX with C=3, Z=5
    assert ds.n_z == 5
    assert ds.n_timepoints == 1
    assert ds.is_time_series is False


def test_dataset_objective_and_scan_mode_default_to_none() -> None:
    from imajin.io.dataset import Dataset
    import numpy as np
    ds = Dataset(data=np.zeros((1, 8, 8), dtype=np.uint16), axes="ZYX")
    assert ds.objective is None
    assert ds.scan_mode is None
```

(The typed-`channel_metadata` round-trip is covered by Task 5 once OME extraction is wired through, and by Task 8 for the napari path. No standalone test is needed here.)

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_io_loader.py -k "n_timepoints or objective_and_scan" -v`
Expected: FAIL (`n_timepoints`, `n_z`, `objective`, `scan_mode` not on Dataset).

- [ ] **Step 3: Extend Dataset**

Replace `src/imajin/io/dataset.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from imajin.io.channel_metadata import ChannelMetadata


@dataclass
class Dataset:
    data: Any
    axes: str
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0)
    channel_names: list[str] = field(default_factory=list)
    channel_metadata: list[ChannelMetadata] = field(default_factory=list)
    objective: str | None = None
    scan_mode: str | None = None
    source_path: Path | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        if "C" in self.axes:
            return int(self.data.shape[self.axes.index("C")])
        return 1

    @property
    def n_z(self) -> int:
        if "Z" in self.axes:
            return int(self.data.shape[self.axes.index("Z")])
        return 1

    @property
    def n_timepoints(self) -> int:
        if "T" in self.axes:
            return int(self.data.shape[self.axes.index("T")])
        return 1

    @property
    def is_3d(self) -> bool:
        return "Z" in self.axes

    @property
    def is_time_series(self) -> bool:
        return self.n_timepoints > 1
```

- [ ] **Step 4: Re-run tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_io_loader.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/io/dataset.py tests/test_io_loader.py
git commit -m "feat(io): Dataset gains objective/scan_mode + n_timepoints/n_z properties"
```

---

## Task 5: OME Loader → ChannelMetadata Dataclass

**Files:**
- Modify: `src/imajin/io/ome.py`
- Modify: `tests/test_io_channel_metadata.py`

- [ ] **Step 1: Update the OME-XML test to assert dataclass attributes**

Edit `tests/test_io_channel_metadata.py::test_parse_ome_xml_extracts_channel_wavelengths` so the bottom three assertions become:

```python
    assert [m.color for m in metadata] == ["uv", "green", "ir"]
    assert metadata[0].emission_wavelength_nm == pytest.approx(460)
    assert metadata[1].excitation_wavelength_nm == pytest.approx(488)
    assert metadata[0].name == "DAPI"
```

- [ ] **Step 2: Run, verify it fails**

Run: `uv run --project /home/jin/py314 pytest tests/test_io_channel_metadata.py::test_parse_ome_xml_extracts_channel_wavelengths -v`
Expected: FAIL — current OME loader returns dicts.

- [ ] **Step 3: Rewrite `_parse_ome_xml` to populate ChannelMetadata + objective**

Replace `_parse_ome_xml` and `load_ome` in `src/imajin/io/ome.py`:

```python
def _parse_ome_xml(
    xml: str,
) -> tuple[
    tuple[float, float, float],
    list[str],
    list[ChannelMetadata],
    str | None,
]:
    voxel = (1.0, 1.0, 1.0)
    channels: list[str] = []
    channel_metadata: list[ChannelMetadata] = []
    objective: str | None = None
    if not xml:
        return voxel, channels, channel_metadata, objective
    try:
        root = ET.fromstring(xml)
        instrument = root.find(".//ome:Instrument/ome:Objective", _OME_NS) or root.find(
            ".//Instrument/Objective"
        )
        if instrument is not None:
            objective = (
                instrument.get("Model")
                or instrument.get("ID")
                or instrument.get("Manufacturer")
            )
        pixels = root.find(".//ome:Pixels", _OME_NS) or root.find(".//Pixels")
        if pixels is not None:
            voxel = (
                float(pixels.get("PhysicalSizeZ", 1.0)),
                float(pixels.get("PhysicalSizeY", 1.0)),
                float(pixels.get("PhysicalSizeX", 1.0)),
            )
            channel_elems = pixels.findall(
                ".//ome:Channel", _OME_NS
            ) or pixels.findall(".//Channel")
            for i, ch in enumerate(channel_elems):
                name = ch.get("Name") or ch.get("ID") or f"ch{i}"
                channels.append(name)
                md = build_channel_info(
                    index=i,
                    name=name,
                    excitation=ch.get("ExcitationWavelength"),
                    emission=ch.get("EmissionWavelength"),
                    dye_name=ch.get("Fluor") or ch.get("Fluorophore"),
                    detector_name=ch.get("DetectorSettings"),
                    extra={
                        k: v
                        for k, v in {
                            "excitation_wavelength_unit": ch.get(
                                "ExcitationWavelengthUnit"
                            ),
                            "emission_wavelength_unit": ch.get(
                                "EmissionWavelengthUnit"
                            ),
                        }.items()
                        if v is not None
                    },
                )
                channel_metadata.append(md)
    except ET.ParseError:
        pass
    return voxel, channels, channel_metadata, objective
```

Update `load_ome` so that `_parse_ome_xml` returns four values and `Dataset(...)` includes `objective=...`.

```python
    voxel_size, channel_names, channel_metadata, objective = _parse_ome_xml(ome_xml)

    return Dataset(
        data=data,
        axes=axes,
        voxel_size=voxel_size,
        channel_names=channel_names,
        channel_metadata=channel_metadata,
        objective=objective,
        source_path=p,
        raw_metadata={
            "ome_xml": ome_xml,
            "load_mode": load_mode,
            "estimated_nbytes": estimated_nbytes,
            "available_memory_bytes": available_bytes,
        },
    )
```

- [ ] **Step 4: Run all OME-related tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_io_channel_metadata.py tests/test_io_loader.py tests/test_napari_reader.py -q`
Expected: PASS (the napari layer test already iterates `len(... channel_metadata)` so dicts vs dataclasses don't matter here yet — Task 8 will adjust serialization).

- [ ] **Step 5: Commit**

```bash
git add src/imajin/io/ome.py tests/test_io_channel_metadata.py
git commit -m "feat(io.ome): populate ChannelMetadata dataclass + extract objective"
```

---

## Task 6: LSM Loader — Per-Channel Acquisition Detail

**Files:**
- Modify: `src/imajin/io/lsm.py`
- Modify: `tests/test_io_lsm.py`

- [ ] **Step 1: Write failing tests for richer LSM extraction**

Append to `tests/test_io_lsm.py`:

```python
from imajin.io.channel_metadata import ChannelMetadata
from imajin.io.lsm import _channel_metadata, _objective, _scan_mode


def test_channel_metadata_extracts_dye_detector_gain_pinhole() -> None:
    meta = {
        "ScanInformation": {
            "Tracks": [
                {
                    "IlluminationChannels": [
                        {"Wavelength": 488e-9, "Power": 2.5}
                    ],
                    "DataChannels": [
                        {
                            "Name": "Ch1-T1",
                            "DyeName": "Alexa Fluor 488",
                            "DetectorGain": 750,
                            "PinholeDiameter": 36.262,
                            "DetectorName": "PMT1",
                        }
                    ],
                },
                {
                    "IlluminationChannels": [
                        {"Wavelength": 639e-9, "Power": 1.0}
                    ],
                    "DataChannels": [
                        {
                            "Name": "Ch2-T3",
                            "DyeName": "Alexa Fluor 633",
                            "DetectorGain": 850,
                            "PinholeDiameter": 48.586,
                            "DetectorName": "PMT2",
                            "FilterName": "LP 640",
                        }
                    ],
                },
            ]
        }
    }

    metadata = _channel_metadata(meta)

    assert all(isinstance(m, ChannelMetadata) for m in metadata)
    assert metadata[0].dye_name == "Alexa Fluor 488"
    assert metadata[0].detector_name == "PMT1"
    assert metadata[0].detector_gain == 750
    assert metadata[0].pinhole_diameter == pytest.approx(36.262)
    assert metadata[0].excitation_wavelength_nm == pytest.approx(488)
    assert metadata[0].laser_power == pytest.approx(2.5)
    assert metadata[0].color == "green"

    assert metadata[1].dye_name == "Alexa Fluor 633"
    assert metadata[1].filter_name == "LP 640"
    assert metadata[1].excitation_wavelength_nm == pytest.approx(639)
    assert metadata[1].color == "ir"
    # IR/far red alone must NOT trigger a counterstain suggestion
    assert metadata[1].role_suggestion is None


def test_channel_metadata_suggest_role_uses_dye_for_gfp() -> None:
    meta = {
        "ScanInformation": {
            "Tracks": [
                {
                    "DataChannels": [
                        {"Name": "Ch1", "DyeName": "GFP"},
                    ],
                }
            ]
        }
    }

    [md] = _channel_metadata(meta)

    assert md.role_suggestion == "target"
    assert "gfp" in (md.role_suggestion_reason or "").lower()


def test_objective_and_scan_mode_extraction() -> None:
    meta = {
        "ScanInformation": {
            "Objective": "EC Plan-Neofluar 20x/0.50 M27",
            "ScanMode": "Stack",
        }
    }

    assert _objective(meta) == "EC Plan-Neofluar 20x/0.50 M27"
    assert _scan_mode(meta) == "Stack"
```

- [ ] **Step 2: Run, verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_io_lsm.py -q`
Expected: FAIL — `_objective`, `_scan_mode` not defined; per-channel dye/detector/gain not extracted.

- [ ] **Step 3: Extend `_channel_metadata` and add objective/scan_mode helpers**

In `src/imajin/io/lsm.py`:

1. Replace `_channel_metadata` so it builds `ChannelMetadata` and pulls dye/detector/filter/gain/pinhole/laser power.
2. Add `_objective(meta)` and `_scan_mode(meta)` helpers that read from `ScanInformation`.
3. Update `load_lsm` to set `objective=` and `scan_mode=` on `Dataset`.

Replace the block from `def _channel_metadata` through the end of `load_lsm` with:

```python
def _channel_metadata(lsm_meta: dict[str, Any]) -> list[ChannelMetadata]:
    names = _channel_names(lsm_meta)
    out: list[ChannelMetadata] = []

    si = lsm_meta.get("ScanInformation", {})
    index = 0
    if isinstance(si, dict):
        tracks = si.get("Tracks", [])
        for track in tracks if isinstance(tracks, list) else []:
            if not isinstance(track, dict):
                continue
            illumination = track.get("IlluminationChannels", [])
            illum_records: list[dict[str, Any]] = [
                il for il in illumination if isinstance(il, dict)
            ] if isinstance(illumination, list) else []
            data_channels = track.get("DataChannels", [])
            for i, ch in enumerate(
                data_channels if isinstance(data_channels, list) else []
            ):
                if not isinstance(ch, dict):
                    continue
                il = illum_records[i] if i < len(illum_records) else {}
                name = (
                    _first_present(ch, "Name", "ChannelName")
                    or (names[index] if index < len(names) else None)
                )
                dye = _first_present(ch, "DyeName", "Fluor", "Fluorophore")
                excitation = _first_present(
                    ch,
                    "ExcitationWavelength",
                    "LaserWavelength",
                    "IlluminationWavelength",
                )
                if excitation is None and il:
                    excitation = _first_present(
                        il, "Wavelength", "LaserWavelength", "ExcitationWavelength"
                    )
                emission = _first_present(
                    ch,
                    "EmissionWavelength",
                    "DetectionWavelength",
                    "AcquisitionWavelength",
                )
                md = build_channel_info(
                    index=index,
                    name=str(name) if name is not None else None,
                    excitation=excitation,
                    emission=emission,
                    dye_name=str(dye) if dye is not None else None,
                    detector_name=_first_present(ch, "DetectorName", "DetectorChannelName"),
                    filter_name=_first_present(ch, "FilterName", "EmissionFilter"),
                    laser_power=_first_present(il, "Power", "LaserPower"),
                    detector_gain=_first_present(ch, "DetectorGain", "Gain"),
                    pinhole_diameter=_first_present(ch, "PinholeDiameter", "Pinhole"),
                )
                out.append(md)
                index += 1

    if not out:
        out = [
            build_channel_info(index=i, name=name)
            for i, name in enumerate(names)
        ]
    return out


def _objective(lsm_meta: dict[str, Any]) -> str | None:
    si = lsm_meta.get("ScanInformation", {})
    if isinstance(si, dict):
        for key in ("Objective", "ObjectiveName", "ObjectiveID"):
            v = si.get(key)
            if v:
                return str(v)
    return None


def _scan_mode(lsm_meta: dict[str, Any]) -> str | None:
    si = lsm_meta.get("ScanInformation", {})
    if isinstance(si, dict):
        v = si.get("ScanMode") or si.get("Mode")
        if v:
            return str(v)
    return None
```

Then update the `return Dataset(...)` block in `load_lsm` to include:

```python
    return Dataset(
        data=data,
        axes=axes,
        voxel_size=_voxel_size_um(lsm_meta),
        channel_names=_channel_names(lsm_meta),
        channel_metadata=_channel_metadata(lsm_meta),
        objective=_objective(lsm_meta),
        scan_mode=_scan_mode(lsm_meta),
        source_path=p,
        raw_metadata={
            "lsm": dict(lsm_meta),
            "load_mode": load_mode,
            "estimated_nbytes": estimated_nbytes,
            "available_memory_bytes": available_bytes,
        },
    )
```

Update the existing `test_channel_metadata_from_lsm_scan_wavelengths` so it accesses dataclass attrs:

```python
    assert [m.name for m in metadata] == ["GCaMP", "mCherry", "Cy5"]
    assert [m.color for m in metadata] == ["green", "red", "ir"]
    assert metadata[0].excitation_wavelength_nm == pytest.approx(488)
```

- [ ] **Step 4: Run tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_io_lsm.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/io/lsm.py tests/test_io_lsm.py
git commit -m "feat(io.lsm): extract dye/detector/gain/pinhole/laser/objective/scan_mode into ChannelMetadata"
```

---

## Task 7: CZI Loader — Best-Effort ChannelMetadata

**Files:**
- Modify: `src/imajin/io/czi.py`

- [ ] **Step 1: Update CZI to populate ChannelMetadata dataclass**

Replace `load_czi` body so it returns dataclass-typed `channel_metadata` and tolerates missing wavelength data:

```python
def load_czi(path: Path | str) -> Dataset:
    from bioio import BioImage

    p = Path(path)
    img = BioImage(str(p))
    data = img.dask_data

    ps = img.physical_pixel_sizes
    voxel_size = (
        float(ps.Z or 1.0),
        float(ps.Y or 1.0),
        float(ps.X or 1.0),
    )

    channel_names = list(img.channel_names) if img.channel_names else []
    channel_metadata: list[ChannelMetadata] = [
        build_channel_info(index=i, name=name)
        for i, name in enumerate(channel_names)
    ]

    raw: dict[str, Any] = {}
    try:
        meta = img.metadata
        raw["czi"] = (
            dict(meta) if hasattr(meta, "items") else {"_repr": repr(meta)[:1000]}
        )
        parsed = _channel_metadata_from_xml(meta)
        if parsed:
            channel_metadata = parsed
    except Exception:
        raw["czi"] = {}

    return Dataset(
        data=data,
        axes="TCZYX",
        voxel_size=voxel_size,
        channel_names=channel_names,
        channel_metadata=channel_metadata,
        source_path=p,
        raw_metadata=raw,
    )
```

Also update `_channel_metadata_from_xml` to call `build_channel_info(index=...)` so each entry is a `ChannelMetadata` instance:

```python
def _channel_metadata_from_xml(metadata: Any) -> list[ChannelMetadata]:
    root = metadata
    if isinstance(metadata, str):
        try:
            root = ET.fromstring(metadata)
        except ET.ParseError:
            return []
    if not hasattr(root, "iter"):
        return []

    channels: list[ChannelMetadata] = []
    for elem in root.iter():
        if _local_name(str(elem.tag)).lower() != "channel":
            continue
        attrs = getattr(elem, "attrib", {}) or {}
        name = attrs.get("Name") or attrs.get("ShortName") or attrs.get("Id")
        excitation = (
            attrs.get("ExcitationWavelength")
            or attrs.get("Excitation")
            or attrs.get("LaserWavelength")
        )
        emission = (
            attrs.get("EmissionWavelength")
            or attrs.get("Emission")
            or attrs.get("DetectionWavelength")
        )
        dye = attrs.get("Fluor") or attrs.get("Fluorophore") or attrs.get("DyeName")
        channels.append(
            build_channel_info(
                index=len(channels),
                name=name,
                excitation=excitation,
                emission=emission,
                dye_name=dye,
            )
        )
    return channels
```

(There is no automated CZI test fixture in this repo. CZI extraction stays best-effort and is exercised manually. Existing tests must still pass: run `uv run --project /home/jin/py314 pytest -q -m "not slow and not integration"` and confirm no regressions.)

- [ ] **Step 2: Run full fast test suite**

Run: `uv run --project /home/jin/py314 pytest -q -m "not slow and not integration"`
Expected: PASS (CZI is not exercised; OME/LSM/napari paths must remain green).

- [ ] **Step 3: Commit**

```bash
git add src/imajin/io/czi.py
git commit -m "feat(io.czi): emit ChannelMetadata dataclass + propagate dye_name when present"
```

---

## Task 8: Napari Reader Serialization + role_suggestion Round-Trip

**Files:**
- Modify: `src/imajin/io/napari_reader.py`
- Modify: `tests/test_napari_reader.py`

- [ ] **Step 1: Write failing test asserting serialized dicts and role_suggestion**

Append to `tests/test_napari_reader.py`:

```python
import numpy as np
import tifffile

from imajin.io.channel_metadata import ChannelMetadata
from imajin.io.napari_reader import _to_layer
from imajin.io.dataset import Dataset


def test_layer_metadata_serializes_channel_metadata_to_dicts(tiny_ome_tiff: Path) -> None:
    layers = _do_read(str(tiny_ome_tiff))
    _, kwargs, _ = layers[0]
    cm = kwargs["metadata"]["channel_metadata"]
    assert all(isinstance(m, dict) for m in cm)
    # ChannelMetadata.to_dict drops None fields, so 'index' and 'name' are present
    assert cm[0]["index"] == 0
    assert cm[0]["name"] == "DAPI"


def test_layer_metadata_preserves_role_suggestion() -> None:
    ds = Dataset(
        data=np.zeros((2, 8, 8), dtype=np.uint16),
        axes="CYX",
        channel_names=["GFP", "Cy5"],
        channel_metadata=[
            ChannelMetadata(index=0, name="GFP", color="green",
                            role_suggestion="target",
                            role_suggestion_reason="marker contains 'gfp'"),
            ChannelMetadata(index=1, name="Cy5", color="ir"),
        ],
    )
    _, kwargs, _ = _to_layer(ds)
    cm = kwargs["metadata"]["channel_metadata"]
    assert cm[0]["role_suggestion"] == "target"
    assert "gfp" in cm[0]["role_suggestion_reason"].lower()
    assert "role_suggestion" not in cm[1]
```

- [ ] **Step 2: Run, verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_napari_reader.py -q`
Expected: FAIL.

- [ ] **Step 3: Update `_to_layer` to serialize via `to_dict`**

Edit `src/imajin/io/napari_reader.py` so the channel-metadata branch goes through `serialize_channel_metadata`:

```python
from imajin.io.channel_metadata import (
    ChannelMetadata,
    pad_channel_metadata,
    serialize_channel_metadata,
)
...
        metadata["channel_names"] = list(kwargs["name"])
        padded = pad_channel_metadata(
            list(getattr(ds, "channel_metadata", []) or []),
            n_channels=n_ch,
            names=list(kwargs["name"]),
        )
        metadata["channel_metadata"] = serialize_channel_metadata(padded)
    else:
        kwargs["name"] = base
        kwargs["scale"] = tuple(scale_per_axis.get(a, 1.0) for a in ds.axes)
        if getattr(ds, "channel_metadata", None):
            metadata["channel_metadata"] = serialize_channel_metadata(
                [
                    m if isinstance(m, ChannelMetadata) else ChannelMetadata.from_dict(m)
                    for m in ds.channel_metadata
                ]
            )
```

Also propagate dataset-level acquisition fields onto the layer metadata:

```python
    metadata = {"voxel_size_um": ds.voxel_size, "axes": ds.axes}
    if ds.objective:
        metadata["objective"] = ds.objective
    if ds.scan_mode:
        metadata["scan_mode"] = ds.scan_mode
    metadata["n_z"] = ds.n_z
    metadata["n_timepoints"] = ds.n_timepoints
```

- [ ] **Step 4: Run tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_napari_reader.py tests/test_tools_channels.py -q`
Expected: PASS — `test_resolve_channel_uses_file_wavelength_metadata` already keeps dict-shaped `channel_metadata`, so it still resolves correctly.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/io/napari_reader.py tests/test_napari_reader.py
git commit -m "feat(io.napari): serialize ChannelMetadata to dicts; propagate role_suggestion + acquisition fields"
```

---

## Task 9: Refactor `resolve_channel` Priority Order

**Files:**
- Modify: `src/imajin/agent/state.py`
- Modify: `tests/test_tools_channels.py`

- [ ] **Step 1: Write failing tests that pin priority order**

Append to `tests/test_tools_channels.py`:

```python
def test_resolve_priority_user_annotation_overrides_metadata(viewer) -> None:
    """If file metadata says ch0 is green and the user has annotated ch1 as the green
    target, resolve('green') must return ch1 (annotation wins over metadata)."""
    import numpy as np
    md0 = {
        "channel_names": ["ch0", "ch1"],
        "channel_metadata": [
            {"index": 0, "name": "GFP", "color": "green"},
            {"index": 1, "name": "Cy5", "color": "ir"},
        ],
    }
    viewer.add_image(np.zeros((4, 4), dtype=np.uint16), name="ch0", metadata=md0)
    viewer.add_image(np.zeros((4, 4), dtype=np.uint16), name="ch1", metadata=md0)

    channels.annotate_channel(layer="ch1", role="target", color="green", marker="GFP")
    assert channels.resolve_channel("green")["layer"] == "ch1"


def test_resolve_priority_metadata_dye_name(viewer) -> None:
    import numpy as np
    md = {
        "channel_names": ["a", "b"],
        "channel_metadata": [
            {"index": 0, "name": "ch0", "dye_name": "Alexa Fluor 488"},
            {"index": 1, "name": "ch1", "dye_name": "Alexa Fluor 633"},
        ],
    }
    viewer.add_image(np.zeros((4, 4), dtype=np.uint16), name="a", metadata=md)
    viewer.add_image(np.zeros((4, 4), dtype=np.uint16), name="b", metadata=md)

    assert channels.resolve_channel("Alexa Fluor 633")["layer"] == "b"


def test_resolve_priority_layer_name_substring_last(viewer) -> None:
    """When neither annotation nor metadata color matches, fall through to substring."""
    import numpy as np
    viewer.add_image(np.zeros((4, 4), dtype=np.uint16), name="midgut_acquisition_001")

    assert channels.resolve_channel("acquisition")["layer"] == "midgut_acquisition_001"


def test_resolve_ambiguous_color_lists_matches(viewer) -> None:
    import numpy as np
    md = {
        "channel_names": ["a", "b"],
        "channel_metadata": [
            {"index": 0, "name": "ch0", "color": "green"},
            {"index": 1, "name": "ch1", "color": "green"},
        ],
    }
    viewer.add_image(np.zeros((4, 4), dtype=np.uint16), name="a", metadata=md)
    viewer.add_image(np.zeros((4, 4), dtype=np.uint16), name="b", metadata=md)

    with pytest.raises(KeyError) as exc:
        channels.resolve_channel("green")
    assert "ambiguous" in str(exc.value).lower()
    assert "a" in str(exc.value) and "b" in str(exc.value)
```

- [ ] **Step 2: Run, verify failures**

Run: `uv run --project /home/jin/py314 pytest tests/test_tools_channels.py -q`
Expected: most pass; the new ones may fail because (a) the existing logic does not lift annotation above metadata when both exist and (b) `dye_name` is not consulted when resolving.

- [ ] **Step 3: Refactor `resolve_layer_name`**

Replace `resolve_layer_name` and helpers in `src/imajin/agent/state.py`:

```python
def _layer_metadata_text_full(layer: Any) -> str:
    md = getattr(layer, "metadata", {}) or {}
    bits = [getattr(layer, "name", "")]
    if isinstance(md, dict):
        for key in ("channel_name", "marker", "color", "wavelength", "excitation"):
            if md.get(key) is not None:
                bits.append(str(md[key]))
        info = _layer_channel_metadata(layer)
        for key in (
            "name",
            "marker",
            "color",
            "dye_name",
            "detector_name",
            "filter_name",
            "excitation_wavelength_nm",
            "emission_wavelength_nm",
        ):
            if info.get(key) is not None:
                bits.append(str(info[key]))
    return " ".join(bits)


def resolve_layer_name(query: str) -> str:
    viewer = get_viewer()

    # Priority 1: exact layer name match
    try:
        viewer.layers[query]
        return query
    except KeyError:
        pass

    q_norm = _normalize_text(query)
    q_color = canonical_channel_color(query)

    # Priority 2: user annotation
    annotation_matches: list[str] = []
    for entry in _CHANNELS.values():
        values = [
            entry.layer_name,
            entry.color or "",
            entry.marker or "",
            entry.role,
            entry.biological_target or "",
        ]
        norm_values = {_normalize_text(v) for v in values if v}
        if q_norm and q_norm in norm_values:
            annotation_matches.append(entry.layer_name)
        elif q_color is not None and entry.color == q_color:
            annotation_matches.append(entry.layer_name)
    annotation_unique = list(dict.fromkeys(annotation_matches))
    if len(annotation_unique) == 1:
        return annotation_unique[0]
    if len(annotation_unique) > 1:
        raise KeyError(
            f"Layer alias {query!r} is ambiguous between user-annotated layers "
            f"{annotation_unique}. Use a full layer name."
        )

    # Priority 3: file metadata (color, dye, name, wavelength)
    metadata_matches: list[str] = []
    for layer in viewer.layers:
        info = _layer_channel_metadata(layer)
        if not info:
            continue
        color = canonical_channel_color(info.get("color"))
        if q_color is not None and color == q_color:
            metadata_matches.append(layer.name)
            continue
        for key in ("name", "dye_name", "marker"):
            v = info.get(key)
            if isinstance(v, str) and q_norm and q_norm in _normalize_text(v):
                metadata_matches.append(layer.name)
                break
    metadata_unique = list(dict.fromkeys(metadata_matches))
    if len(metadata_unique) == 1:
        return metadata_unique[0]
    if len(metadata_unique) > 1:
        raise KeyError(
            f"Layer alias {query!r} is ambiguous between file-metadata channels "
            f"{metadata_unique}. Annotate one as the target or use a full layer name."
        )

    # Priority 4: layer-name substring
    substring_matches = [
        layer.name
        for layer in viewer.layers
        if q_norm and q_norm in _normalize_text(_layer_metadata_text_full(layer))
    ]
    substring_unique = list(dict.fromkeys(substring_matches))
    if len(substring_unique) == 1:
        return substring_unique[0]
    if len(substring_unique) > 1:
        raise KeyError(
            f"Layer alias {query!r} is ambiguous. Matches: {substring_unique}. "
            "Use a full layer name or annotate channels more specifically."
        )

    return query
```

(The original `_layer_metadata_text` can be removed — `_layer_metadata_text_full` replaces it.)

- [ ] **Step 4: Run tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_tools_channels.py -q`
Expected: PASS (all priority ordering tests + the existing ones, after Task 2's default-role change).

- [ ] **Step 5: Commit**

```bash
git add src/imajin/agent/state.py tests/test_tools_channels.py
git commit -m "feat(state): resolve_channel priority — exact > annotation > metadata > substring"
```

---

## Task 10: `load_file` + `list_channel_metadata` Tool

**Files:**
- Modify: `src/imajin/tools/files.py`
- Modify: `src/imajin/tools/channels.py`
- Modify: `tests/test_tools_files.py` (existing)
- Modify: `tests/test_tools_channels.py`

- [ ] **Step 1: Write failing tests for the new tool + tool output**

Append to `tests/test_tools_channels.py`:

```python
def test_list_channel_metadata_returns_per_layer_dicts(viewer) -> None:
    import numpy as np
    md = {
        "channel_names": ["a", "b"],
        "channel_metadata": [
            {"index": 0, "name": "GFP", "color": "green"},
            {"index": 1, "name": "Cy5", "color": "ir"},
        ],
    }
    viewer.add_image(np.zeros((4, 4), dtype=np.uint16), name="a", metadata=md)
    viewer.add_image(np.zeros((4, 4), dtype=np.uint16), name="b", metadata=md)

    res = channels.list_channel_metadata_tool()

    assert {item["layer"] for item in res} == {"a", "b"}
    by_layer = {item["layer"]: item for item in res}
    assert by_layer["a"]["color"] == "green"
    assert by_layer["b"]["color"] == "ir"
    # No role_suggestion was provided in metadata, so it must NOT appear as a fact
    assert "role_suggestion" not in by_layer["a"]
```

Append to `tests/test_tools_files.py` (read it first to find a good insertion point):

```python
def test_load_file_returns_objective_and_n_z(tiny_ome_tiff, viewer) -> None:
    res = files.load_file(str(tiny_ome_tiff))
    assert res["axes"] == "CZYX"
    assert res["n_z"] == 5
    assert res["n_timepoints"] == 1
```

- [ ] **Step 2: Run, verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_tools_files.py::test_load_file_returns_objective_and_n_z tests/test_tools_channels.py::test_list_channel_metadata_returns_per_layer_dicts -v`
Expected: FAIL.

- [ ] **Step 3: Add `list_channel_metadata` tool + extend `load_file`**

In `src/imajin/tools/channels.py`, add:

```python
@tool(
    name="list_channel_metadata",
    description="List acquisition metadata for each napari layer (color, wavelengths, "
    "dye, detector, gain, pinhole, role_suggestion). These are physical/instrument "
    "facts derived from the file. They are NOT user-confirmed roles — for that, see "
    "list_channel_annotations.",
    phase="1.5",
)
def list_channel_metadata_tool() -> list[dict[str, Any]]:
    from imajin.agent.state import get_viewer, _layer_channel_metadata

    viewer = get_viewer()
    out: list[dict[str, Any]] = []
    for layer in viewer.layers:
        info = _layer_channel_metadata(layer)
        if not info:
            continue
        item = {"layer": getattr(layer, "name", "")}
        item.update(info)
        out.append(item)
    return out
```

In `src/imajin/tools/files.py::load_file`, return acquisition-level fields. Replace the return block with:

```python
    return {
        "path": str(Path(path).resolve()),
        "axes": ds.axes,
        "shape": tuple(int(s) for s in ds.data.shape),
        "voxel_size_um": tuple(ds.voxel_size),
        "channel_names": list(ds.channel_names),
        "channel_metadata": [
            m.to_dict() if hasattr(m, "to_dict") else dict(m)
            for m in (getattr(ds, "channel_metadata", []) or [])
        ],
        "objective": ds.objective,
        "scan_mode": ds.scan_mode,
        "n_z": ds.n_z,
        "n_timepoints": ds.n_timepoints,
        "is_time_series": ds.is_time_series,
        "layer_names": [L.name for L in layers],
        "load_mode": ds.raw_metadata.get("load_mode"),
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_tools_files.py tests/test_tools_channels.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/files.py src/imajin/tools/channels.py tests/test_tools_files.py tests/test_tools_channels.py
git commit -m "feat(tools): list_channel_metadata + load_file returns objective/n_z/n_timepoints"
```

---

## Task 11: Reports — Acquisition Section + Suggestions Are Suggestions

**Files:**
- Modify: `src/imajin/tools/report.py`
- Modify: `tests/test_tools_report.py`

- [ ] **Step 1: Write failing report tests**

Append to `tests/test_tools_report.py`:

```python
def test_generate_report_separates_acquisition_from_annotations(
    fake_session, tmp_path, viewer
) -> None:
    import numpy as np
    from imajin.agent import state

    md = {
        "channel_names": ["GFP", "Cy5"],
        "channel_metadata": [
            {
                "index": 0,
                "name": "GFP",
                "color": "green",
                "excitation_wavelength_nm": 488,
                "dye_name": "Alexa Fluor 488",
                "role_suggestion": "target",
                "role_suggestion_reason": "marker contains 'gfp'",
            },
            {
                "index": 1,
                "name": "Cy5",
                "color": "ir",
                "excitation_wavelength_nm": 639,
                "dye_name": "Alexa Fluor 633",
            },
        ],
        "objective": "EC Plan-Neofluar 20x/0.50 M27",
        "n_z": 67,
        "n_timepoints": 1,
    }
    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="GFP", metadata=md)
    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="Cy5", metadata=md)

    state.put_channel_annotation(
        "GFP", role="target", color="green", marker="GFP", biological_target="VNC neurons"
    )

    out = tmp_path / "report.md"
    report.generate_report(str(out), format="md")
    body = out.read_text(encoding="utf-8")

    assert "Acquisition" in body
    assert "EC Plan-Neofluar 20x/0.50 M27" in body
    assert "488" in body and "639" in body
    # User annotation appears as a fact under its own section
    assert "Channel Annotations" in body
    assert "GFP" in body and "VNC neurons" in body
    # Cy5 has NO user annotation and NO role suggestion — must NOT be claimed as a counterstain
    assert "counterstain" not in body.lower() or "suggestion" in body.lower()
    # Suggestion for GFP must be marked as suggestion, not stated as a confirmed role
    if "Suggested" in body or "suggestion" in body.lower():
        assert "marker contains" in body or "gfp" in body.lower()


def test_report_does_not_invent_groups_from_filename(fake_session, tmp_path) -> None:
    out = tmp_path / "report.md"
    report.generate_report(str(out), format="md")
    body = out.read_text(encoding="utf-8")

    # No samples were annotated; the report must not invent control/treatment labels
    assert "control" not in body.lower()
    assert "treatment" not in body.lower()
```

- [ ] **Step 2: Run, verify failure**

Run: `uv run --project /home/jin/py314 pytest tests/test_tools_report.py -q`
Expected: FAIL — there is no acquisition section yet.

- [ ] **Step 3: Add `_render_acquisition_markdown` + suggestion-as-suggestion rendering**

In `src/imajin/tools/report.py`:

1. Add a helper that pulls per-layer metadata via `viewer_or_none()` and renders an "Acquisition" section. If the viewer is unavailable, render an empty string (so the report works in headless test sessions).

```python
def _render_acquisition_markdown() -> str:
    from imajin.agent.state import viewer_or_none

    viewer = viewer_or_none()
    if viewer is None or not list(viewer.layers):
        return ""

    seen_files: set[str] = set()
    file_blocks: list[str] = []
    suggestions: list[str] = []
    for layer in viewer.layers:
        md = getattr(layer, "metadata", {}) or {}
        if not isinstance(md, dict):
            continue
        axes = md.get("axes")
        n_z = md.get("n_z")
        n_t = md.get("n_timepoints")
        objective = md.get("objective")
        scan_mode = md.get("scan_mode")
        cm_list = md.get("channel_metadata") or []
        key = (str(axes), str(objective), str(n_z), str(n_t))
        key_repr = repr(key)
        if key_repr in seen_files:
            continue
        seen_files.add(key_repr)

        bits: list[str] = []
        if axes:
            bits.append(f"axes={axes}")
        if n_z is not None:
            bits.append(f"z={n_z}")
        if n_t is not None:
            bits.append(f"t={n_t}")
        if objective:
            bits.append(f"objective={objective}")
        if scan_mode:
            bits.append(f"scan_mode={scan_mode}")
        file_block_lines = [f"- {', '.join(bits)}"] if bits else []
        for ch in cm_list:
            if not isinstance(ch, dict):
                continue
            ch_bits: list[str] = []
            for key in ("name", "color", "excitation_wavelength_nm",
                        "emission_wavelength_nm", "dye_name", "detector_name",
                        "filter_name", "detector_gain", "pinhole_diameter"):
                if ch.get(key) not in (None, ""):
                    ch_bits.append(f"{key}={ch[key]}")
            if ch_bits:
                file_block_lines.append(f"  - {' '.join(ch_bits)}")
            sug = ch.get("role_suggestion")
            if sug:
                reason = ch.get("role_suggestion_reason") or "marker/dye match"
                suggestions.append(
                    f"- {ch.get('name', '?')}: suggested role **{sug}** ({reason}). "
                    f"Confirm via `annotate_channel`."
                )
        if file_block_lines:
            file_blocks.append("\n".join(file_block_lines))

    if not file_blocks and not suggestions:
        return ""

    parts = ["## Acquisition", ""]
    if file_blocks:
        parts.extend(file_blocks)
        parts.append("")
    if suggestions:
        parts.append("### Channel Suggestions (not confirmed)")
        parts.append("")
        parts.extend(suggestions)
        parts.append("")
    return "\n".join(parts)
```

2. Update `_render_channels_markdown` so it never invents a role and so it labels the section "Channel Annotations (user-confirmed)". Replace its body with:

```python
def _render_channels_markdown(channels: list[dict[str, Any]]) -> str:
    if not channels:
        return ""
    lines = ["## Channel Annotations (user-confirmed)", ""]
    for channel in channels:
        layer = channel.get("layer_name", "?")
        role = channel.get("role", "unknown")
        color = channel.get("color") or "unspecified"
        marker = channel.get("marker") or "unspecified"
        target = channel.get("biological_target")
        suffix = f", target={target}" if target else ""
        lines.append(f"- **{layer}**: {role}, {color}, marker={marker}{suffix}")
    lines.append("")
    return "\n".join(lines)
```

3. In `generate_report`, render acquisition before channels and include it in both `.md` and `.html` outputs:

```python
    methods = _render_methods_markdown(records)
    acquisition_md = _render_acquisition_markdown()
    samples = list_samples()
    samples_md = _render_samples_markdown(samples)
    channels = list_channel_annotations()
    channels_md = _render_channels_markdown(channels)
    ...
    if format == "md":
        extra = ""
        if acquisition_md:
            extra += "\n" + acquisition_md
        if samples_md:
            extra += "\n" + samples_md
        if channels_md:
            extra += "\n" + channels_md
        out.write_text(methods + extra, encoding="utf-8")
    else:
        out.write_text(
            _render_report_html(records, methods, samples_md, channels_md, acquisition_md),
            encoding="utf-8",
        )
```

Update `_render_report_html` signature to accept `acquisition_md=""` and inject `<pre>{escape(acquisition_md)}</pre>` if non-empty.

- [ ] **Step 4: Run tests**

Run: `uv run --project /home/jin/py314 pytest tests/test_tools_report.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/imajin/tools/report.py tests/test_tools_report.py
git commit -m "feat(report): Acquisition section + suggestions rendered as suggestions; never claim unconfirmed roles"
```

---

## Task 12: Update Agent System Prompt

**Files:**
- Modify: `src/imajin/agent/prompts.py`

- [ ] **Step 1: Replace the "channel annotation" pipeline section + add filename-rule**

Edit `src/imajin/agent/prompts.py::SYSTEM_PROMPT`. Replace the block beginning `Pipeline "channel annotation"` with:

```
Pipeline "channel annotation" — triggered by "green channel", "red channel",
"UV channel", "IR channel", "far red", "primary", "counterstain", "target":
  step 1: when the user names a color, prefer `resolve_channel`. If exactly one
          layer matches by user annotation OR file metadata, use it. If multiple
          match, ask one focused question. The role default is `unknown` until
          the user confirms — never assume `target` or `counterstain` from
          color alone.
  step 2: when the user names roles ("green is target, far red is phalloidin"),
          call `annotate_channel` for each statement. Annotation overrides any
          metadata-derived role suggestion.
  step 3: when the file's `channel_metadata` carries a `role_suggestion`, treat
          it as a *suggestion only*. Mention it in user-facing summaries with
          words like "suggested" or "likely", and prompt the user to confirm
          via `annotate_channel`. Never write it into reports as a fact.

Pipeline "sample grouping" — triggered by user statements such as
"control 1/2/3", "treatment", "이 파일은 control":
  step 1: invoke annotate_sample for each replicate or sample mapping the user
          gives. Free-form labels (tissue, genotype, sex, region, replicate)
          go into the `extra` dict.
  step 2: invoke list_sample_annotations when you need to confirm the design.
  step 3: do NOT parse the filename to invent groups. Filenames like
          "J41 + 1234 vF midgut R3 1" are free-form text. Only structure them
          into group/tissue/genotype after the user explicitly tells you the
          mapping.
```

Append a new "Metadata vs annotation" section near the existing "Conventions" section:

```
# Metadata vs annotation — DO NOT confuse the two

- File `channel_metadata` (color, wavelength, dye, detector, gain, pinhole) is
  acquisition-level fact. Use it freely.
- `role_suggestion` on `channel_metadata` is a *suggestion*; treat it as a
  prompt to confirm with the user, not as an established role.
- "IR" / "far red" alone is a color, not a role. Never assume a far-red
  channel is the counterstain unless the user says so or its dye/marker is
  DAPI/Hoechst/TOPRO/phalloidin.
- `channel_annotation` (target/counterstain/ignore/unknown) is set ONLY by
  the user via `annotate_channel`. Default is `unknown`.
- Reports must distinguish acquisition facts (Acquisition section) from user
  annotations (Channel Annotations section) — do not promote a suggestion to
  a fact in a report.
```

(No tests for prompt text — it's a string; correctness is verified by review.)

- [ ] **Step 2: Run a smoke compile + the prompts-importing test path**

Run: `uv run --project /home/jin/py314 python -c "from imajin.agent.prompts import build_system_prompt; print(build_system_prompt()[:200])"`
Expected: prints the start of the system prompt without error.

Run: `uv run --project /home/jin/py314 pytest tests/test_runner.py -q 2>&1 | head -50`
Expected: existing runner tests still pass.

- [ ] **Step 3: Commit**

```bash
git add src/imajin/agent/prompts.py
git commit -m "docs(prompt): metadata vs annotation rule + role default unknown + no filename parsing"
```

---

## Task 13: Final Verification

**Files:**
- (Read-only checks)

- [ ] **Step 1: Full lint/compile**

Run: `uv run --project /home/jin/py314 python -m compileall -q src tests`
Expected: no errors.

- [ ] **Step 2: Full fast pytest run**

Run: `uv run --project /home/jin/py314 pytest -q -m "not slow and not integration"`
Expected: all green. Investigate any failure before declaring complete.

- [ ] **Step 3: Doctor**

Run: `uv run --project /home/jin/py314 imajin --doctor`
Expected: returns 0 and lists registered tools including `annotate_channel`, `list_channel_annotations`, `list_channel_metadata`, `resolve_channel`, `annotate_sample`, `list_sample_annotations`.

- [ ] **Step 4: Hand-test on the real LSM file (optional but recommended)**

Run: `uv run --project /home/jin/py314 python -c "
from imajin.io import load_dataset
ds = load_dataset('250828_263_myr.GFP_Male VNC_2.lsm')
print('axes:', ds.axes, 'shape:', ds.data.shape)
print('voxel_size:', ds.voxel_size)
print('objective:', ds.objective)
print('scan_mode:', ds.scan_mode)
for m in ds.channel_metadata:
    print(m.to_dict())
"`
Expected: prints
- axes ZCYX, shape (67, 2, 1024, 1024)
- voxel ≈ (1.913, 0.391, 0.391)
- objective EC Plan-Neofluar 20x/0.50 M27
- channel 0 with name Ch1, color green, dye Alexa Fluor 488, gain 750, pinhole 36.262
- channel 1 with name Ch2, color ir, dye Alexa Fluor 633, gain 850, pinhole 48.586, filter LP 640, role_suggestion None

- [ ] **Step 5: Self-review against spec checklist**

Walk the user's spec one more time and tick:

- [ ] ChannelMetadata dataclass with all fields → Task 1
- [ ] ChannelAnnotation default unknown, role union → Task 2
- [ ] SampleAnnotation with extra dict, no filename parsing → Task 2 + Task 12
- [ ] LSM extracts dye/detector/filter/gain/pinhole/laser → Task 6
- [ ] LSM extracts objective/scan_mode → Task 6
- [ ] OME extracts ex/em/objective → Task 5
- [ ] CZI best-effort → Task 7
- [ ] Plain TIFF stays mostly blank → unchanged (existing tiny_ome_tiff fixture covers it)
- [ ] Color rules match spec → covered in `color_from_wavelengths` (Task 1)
- [ ] Role suggestion rules — GFP/GCaMP/CaLexA → target; DAPI/Hoechst/TOPRO/phalloidin → counterstain; IR/green alone → none → Task 1 + Task 3
- [ ] annotate_channel — no required color, default role unknown → Task 2
- [ ] list_channel_metadata tool → Task 10
- [ ] list_channel_annotations tool → existing
- [ ] resolve_channel priority — exact > annotation > metadata > substring → Task 9
- [ ] Default agent behavior — no filename inference, no IR=counterstain → Task 12
- [ ] Reports — acquisition vs annotation vs methods → Task 11
- [ ] Tests — every category in the spec → Tasks 1, 3, 4, 5, 6, 8, 9, 10, 11
- [ ] Final commands — compileall / pytest / doctor → Task 13

- [ ] **Step 6: Commit any final tidy**

If `pytest` exposes a small fix needed, make it, run again, and commit:

```bash
git add -A
git commit -m "chore: final tidy after channel-metadata refactor"
```

---

## Notes for the Engineer

- All paths are relative to `/home/jin/Imajin`.
- All Python invocations go through `uv run --project /home/jin/py314 ...`. Do not call `python`, `pip`, or `python3` directly.
- The `viewer` fixture in `tests/conftest.py` is a `_FakeViewer` under `QT_QPA_PLATFORM=offscreen`. It supports `add_image`, `add_labels`, dictionary-style `layers[name]`, and a per-layer `metadata` dict.
- Backward-compatible aliases (`SampleEntry`, `ChannelEntry`) are kept so any tests/imports that referenced them continue to work; remove the aliases after one release.
- `pad_channel_metadata` and `serialize_channel_metadata` are the bridge between the typed `ChannelMetadata` flowing through loaders and the dict-shaped layer metadata expected by napari and existing agent state code.
- Reports must NEVER make biological claims that are not user-confirmed. If you find yourself adding text like "control" or "Cy5 was the counterstain" in a code path that hasn't read `_SAMPLES`/`_CHANNELS`, you've gone too far.

