from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

_VIEWER: Any | None = None


@dataclass
class TableEntry:
    df: pd.DataFrame
    spec: dict[str, Any] = field(default_factory=dict)


_TABLES: dict[str, TableEntry] = {}


def _slugify(name: str) -> str:
    base = name.rsplit(".", 1)[0]  # strip extension only on the rightmost dot
    base = re.sub(r"[^A-Za-z0-9_]+", "_", base).strip("_")
    return base or "file"


@dataclass
class FileRecord:
    file_id: str
    path: str
    original_name: str
    file_type: str | None = None
    metadata_summary: dict[str, Any] = field(default_factory=dict)
    load_status: str = "unloaded"  # "unloaded" | "loaded" | "failed"
    notes: str | None = None


_FILES: dict[str, FileRecord] = {}


def put_file(
    path: str,
    original_name: str,
    file_type: str | None = None,
    metadata_summary: dict[str, Any] | None = None,
    notes: str | None = None,
    load_status: str = "unloaded",
) -> str:
    base = _slugify(original_name)
    file_id = base
    n = 2
    while file_id in _FILES:
        file_id = f"{base}_{n}"
        n += 1
    _FILES[file_id] = FileRecord(
        file_id=file_id,
        path=path,
        original_name=original_name,
        file_type=file_type,
        metadata_summary=dict(metadata_summary or {}),
        notes=notes,
        load_status=load_status,
    )
    return file_id


def get_file(file_id: str) -> FileRecord:
    if file_id not in _FILES:
        raise KeyError(
            f"File id {file_id!r} not found. Available: {list(_FILES)}"
        )
    return _FILES[file_id]


def list_files() -> list[dict[str, Any]]:
    return [asdict(rec) for rec in _FILES.values()]


def update_file_status(file_id: str, status: str, notes: str | None = None) -> None:
    rec = get_file(file_id)
    rec.load_status = status
    if notes is not None:
        rec.notes = notes


def reset_files() -> None:
    _FILES.clear()


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


_RECIPES: dict[str, AnalysisRecipe] = {}


def put_recipe(
    name: str,
    target_channel: str | None = None,
    preprocessing: list[dict[str, Any]] | None = None,
    segmentation: dict[str, Any] | None = None,
    measurement: dict[str, Any] | None = None,
    timecourse: dict[str, Any] | None = None,
    colocalization: list[tuple[str, str]] | None = None,
    notes: str | None = None,
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
    )
    return name


def get_recipe(name: str) -> AnalysisRecipe:
    if name not in _RECIPES:
        raise KeyError(f"Recipe {name!r} not found. Available: {list(_RECIPES)}")
    return _RECIPES[name]


def list_recipes() -> list[dict[str, Any]]:
    return [asdict(r) for r in _RECIPES.values()]


def reset_recipes() -> None:
    _RECIPES.clear()


@dataclass
class AnalysisRun:
    run_id: str
    sample_id: str
    file_id: str
    recipe_id: str
    status: str = "pending"  # "pending" | "running" | "complete" | "failed"
    table_names: list[str] = field(default_factory=list)
    layer_names: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


_RUNS: dict[str, AnalysisRun] = {}
_RUN_COUNTER: list[int] = [0]


def put_run(
    sample_id: str,
    file_id: str,
    recipe_id: str,
    status: str = "pending",
    table_names: list[str] | None = None,
    layer_names: list[str] | None = None,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
    run_id: str | None = None,
) -> str:
    if run_id is None:
        _RUN_COUNTER[0] += 1
        run_id = f"run_{_RUN_COUNTER[0]:04d}"
    _RUNS[run_id] = AnalysisRun(
        run_id=run_id,
        sample_id=sample_id,
        file_id=file_id,
        recipe_id=recipe_id,
        status=status,
        table_names=list(table_names or []),
        layer_names=list(layer_names or []),
        summary=dict(summary or {}),
        error=error,
    )
    return run_id


def get_run(run_id: str) -> AnalysisRun:
    if run_id not in _RUNS:
        raise KeyError(f"Run {run_id!r} not found. Available: {list(_RUNS)}")
    return _RUNS[run_id]


def list_runs() -> list[dict[str, Any]]:
    return [asdict(r) for r in _RUNS.values()]


def reset_runs() -> None:
    _RUNS.clear()
    _RUN_COUNTER[0] = 0


@dataclass
class SampleAnnotation:
    sample_name: str
    sample_id: str = ""  # defaults to sample_name in put_sample
    group: str | None = None
    layers: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    file_ids: list[str] = field(default_factory=list)
    notes: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# Backward-compat alias so external imports `from state import SampleEntry` keep working.
SampleEntry = SampleAnnotation


_SAMPLES: dict[str, SampleAnnotation] = {}


_CHANNEL_COLOR_ALIASES: dict[str, str] = {
    "green": "green",
    "gfp": "green",
    "fitc": "green",
    "gcamp": "green",
    "488": "green",
    "red": "red",
    "rfp": "red",
    "dsred": "red",
    "mcherry": "red",
    "tritc": "red",
    "cy3": "red",
    "561": "red",
    "568": "red",
    "594": "red",
    "uv": "uv",
    "ultraviolet": "uv",
    "blue": "uv",
    "dapi": "uv",
    "hoechst": "uv",
    "405": "uv",
    "ir": "ir",
    "infrared": "ir",
    "farred": "ir",
    "far red": "ir",
    "far-red": "ir",
    "cy5": "ir",
    "alexa647": "ir",
    "633": "ir",
    "640": "ir",
    "647": "ir",
}

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
}


@dataclass
class ChannelEntry:
    layer_name: str
    role: str = "target"
    color: str | None = None
    marker: str | None = None
    biological_target: str | None = None
    notes: str | None = None


_CHANNELS: dict[str, ChannelEntry] = {}


def _normalize_text(value: str) -> str:
    return " ".join(
        value.lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("channel", " ")
        .replace("ch.", "ch ")
        .split()
    )


def canonical_channel_color(value: str | None) -> str | None:
    if not value:
        return None
    norm = _normalize_text(value)
    compact = norm.replace(" ", "")
    if norm in _CHANNEL_COLOR_ALIASES:
        return _CHANNEL_COLOR_ALIASES[norm]
    if compact in _CHANNEL_COLOR_ALIASES:
        return _CHANNEL_COLOR_ALIASES[compact]
    for alias, color in _CHANNEL_COLOR_ALIASES.items():
        if alias in norm or alias in compact:
            return color
    return None


def canonical_channel_role(value: str | None) -> str:
    if not value:
        return "target"
    norm = _normalize_text(value)
    role = _CHANNEL_ROLE_ALIASES.get(norm, norm)
    if role not in {"target", "counterstain", "ignore"}:
        raise ValueError("role must be target, counterstain, or ignore")
    return role


def set_viewer(v: Any) -> None:
    global _VIEWER
    _VIEWER = v


def get_viewer() -> Any:
    if _VIEWER is None:
        raise RuntimeError(
            "No napari viewer registered. Call set_viewer(viewer) at startup, "
            "or pass arrays directly when calling tools from a script."
        )
    return _VIEWER


def viewer_or_none() -> Any | None:
    return _VIEWER


def get_layer(name: str) -> Any:
    viewer = get_viewer()
    try:
        return viewer.layers[name]
    except KeyError as e:
        resolved = resolve_layer_name(name)
        if resolved != name:
            return viewer.layers[resolved]
        names = [L.name for L in viewer.layers]
        raise KeyError(f"Layer {name!r} not found. Available: {names}") from e


def _layer_names() -> list[str]:
    viewer = get_viewer()
    return [L.name for L in viewer.layers]


def _channel_index_for_layer(layer: Any, md: dict[str, Any]) -> int | None:
    name = getattr(layer, "name", "")
    names = md.get("channel_names")
    if isinstance(names, list):
        for i, channel_name in enumerate(names):
            if str(channel_name) == name:
                return i
    norm = _normalize_text(name)
    for token in norm.split():
        if token.startswith("ch") and token[2:].isdigit():
            return int(token[2:])
        if token.isdigit():
            return int(token)
    return None


def _layer_channel_metadata(layer: Any) -> dict[str, Any]:
    md = getattr(layer, "metadata", {}) or {}
    if not isinstance(md, dict):
        return {}

    channel_metadata = md.get("channel_metadata")
    idx = _channel_index_for_layer(layer, md)
    if isinstance(channel_metadata, list) and idx is not None and idx < len(channel_metadata):
        item = channel_metadata[idx]
        if isinstance(item, dict):
            return item

    keys = {
        "channel_name",
        "marker",
        "color",
        "wavelength",
        "excitation",
        "emission",
        "excitation_wavelength_nm",
        "emission_wavelength_nm",
    }
    return {k: md[k] for k in keys if k in md}


def _layer_channel_color(layer: Any) -> str | None:
    info = _layer_channel_metadata(layer)
    color = info.get("color")
    if isinstance(color, str):
        canonical = canonical_channel_color(color)
        if canonical:
            return canonical
    for key in ("name", "channel_name", "marker"):
        value = info.get(key)
        if isinstance(value, str):
            canonical = canonical_channel_color(value)
            if canonical:
                return canonical
    return canonical_channel_color(_layer_metadata_text(layer))


def _layer_metadata_text(layer: Any) -> str:
    md = getattr(layer, "metadata", {}) or {}
    bits = [getattr(layer, "name", "")]
    if isinstance(md, dict):
        for key in ("channel_name", "marker", "color", "wavelength", "excitation"):
            if key in md and md[key] is not None:
                bits.append(str(md[key]))
        channel_info = _layer_channel_metadata(layer)
        for key in (
            "name",
            "marker",
            "color",
            "excitation_wavelength_nm",
            "emission_wavelength_nm",
        ):
            if key in channel_info and channel_info[key] is not None:
                bits.append(str(channel_info[key]))
    return " ".join(bits)


def resolve_layer_name(query: str) -> str:
    viewer = get_viewer()
    try:
        viewer.layers[query]
        return query
    except KeyError:
        pass

    q_norm = _normalize_text(query)
    q_color = canonical_channel_color(query)
    matches: list[str] = []

    for entry in _CHANNELS.values():
        values = [
            entry.layer_name,
            entry.color or "",
            entry.marker or "",
            entry.role,
            entry.biological_target or "",
        ]
        if q_norm in {_normalize_text(v) for v in values if v}:
            matches.append(entry.layer_name)
        elif q_color is not None and entry.color == q_color:
            matches.append(entry.layer_name)

    if not matches and q_color is not None:
        for layer in viewer.layers:
            inferred = _layer_channel_color(layer)
            if inferred == q_color:
                matches.append(layer.name)

    if not matches:
        for layer in viewer.layers:
            if q_norm and q_norm in _normalize_text(_layer_metadata_text(layer)):
                matches.append(layer.name)

    unique = list(dict.fromkeys(matches))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        raise KeyError(
            f"Layer alias {query!r} is ambiguous. Matches: {unique}. "
            "Use a full layer name or annotate channels more specifically."
        )
    return query


def get_table(name: str) -> pd.DataFrame:
    if name not in _TABLES:
        raise KeyError(f"Table {name!r} not found. Available: {list(_TABLES)}")
    return _TABLES[name].df


def get_table_entry(name: str) -> TableEntry:
    if name not in _TABLES:
        raise KeyError(f"Table {name!r} not found. Available: {list(_TABLES)}")
    return _TABLES[name]


def put_table(
    name: str, df: pd.DataFrame, spec: dict[str, Any] | None = None
) -> str:
    base = name
    i = 1
    while name in _TABLES:
        name = f"{base}_{i}"
        i += 1
    _TABLES[name] = TableEntry(df=df, spec=dict(spec or {}))
    _emit_tables_changed()
    return name


def update_table(name: str, df: pd.DataFrame) -> None:
    if name not in _TABLES:
        raise KeyError(f"Table {name!r} not found")
    _TABLES[name].df = df
    _emit_tables_changed()


def list_tables() -> list[str]:
    return list(_TABLES)


def reset_tables() -> None:
    _TABLES.clear()
    _emit_tables_changed()


def attach_sample_columns_to_table(
    table_name: str,
    sample_id: str,
    sample_name: str,
    group: str | None,
    file_id: str | None,
    source_file: str | None,
    source_layer: str | None,
) -> None:
    """Add identifier columns required by the Phase-3 spec onto an existing table.
    No-op if the table doesn't exist or already has these columns."""
    if table_name not in _TABLES:
        return
    df = _TABLES[table_name].df
    additions = {
        "sample_id": sample_id,
        "sample_name": sample_name,
        "group": group,
        "file_id": file_id,
        "source_file": source_file,
        "source_layer": source_layer,
    }
    for col, value in additions.items():
        if col not in df.columns:
            df[col] = value
    _TABLES[table_name].df = df
    _emit_tables_changed()


def put_sample(
    sample_name: str,
    group: str | None = None,
    layers: list[str] | None = None,
    files: list[str] | None = None,
    file_ids: list[str] | None = None,
    notes: str | None = None,
    sample_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    sample_name = sample_name.strip()
    if not sample_name:
        raise ValueError("sample_name must not be empty")
    if group is not None:
        group = group.strip() or None
    sid = (sample_id or "").strip() or sample_name
    _SAMPLES[sample_name] = SampleAnnotation(
        sample_name=sample_name,
        sample_id=sid,
        group=group,
        layers=list(layers or []),
        files=list(files or []),
        file_ids=list(file_ids or []),
        notes=notes,
        extra=dict(extra or {}),
    )
    return sample_name


def list_samples() -> list[dict[str, Any]]:
    return [asdict(entry) for entry in _SAMPLES.values()]


def get_sample(sample_name: str) -> SampleAnnotation:
    if sample_name not in _SAMPLES:
        raise KeyError(f"Sample {sample_name!r} not found. Available: {list(_SAMPLES)}")
    return _SAMPLES[sample_name]


def reset_samples() -> None:
    _SAMPLES.clear()


def put_channel_annotation(
    layer_name: str,
    role: str = "target",
    color: str | None = None,
    marker: str | None = None,
    biological_target: str | None = None,
    notes: str | None = None,
) -> str:
    resolved_layer = resolve_layer_name(layer_name)
    if resolved_layer not in _layer_names():
        raise KeyError(f"Layer {layer_name!r} not found. Available: {_layer_names()}")
    canonical_role = canonical_channel_role(role)
    canonical_color = canonical_channel_color(color)
    if color and canonical_color is None:
        raise ValueError(
            "color must be one of green, red, uv, or ir/far red "
            f"(got {color!r})"
        )
    _CHANNELS[resolved_layer] = ChannelEntry(
        layer_name=resolved_layer,
        role=canonical_role,
        color=canonical_color,
        marker=marker,
        biological_target=biological_target,
        notes=notes,
    )
    return resolved_layer


def list_channel_annotations() -> list[dict[str, Any]]:
    return [asdict(entry) for entry in _CHANNELS.values()]


def reset_channel_annotations() -> None:
    _CHANNELS.clear()


@dataclass
class ChannelResolution:
    """Result of resolving a target channel for the analysis workflow."""

    layer: str
    source: str  # explicit, annotation, phrase, inference
    color: str | None = None
    candidates: list[str] = field(default_factory=list)
    note: str | None = None


class AmbiguousChannelError(KeyError):
    """Raised when no single target channel can be resolved without user input."""

    def __init__(self, message: str, candidates: list[str]):
        super().__init__(message)
        self.candidates = list(candidates)


def _confirmed_target_layers() -> list[str]:
    return [entry.layer_name for entry in _CHANNELS.values() if entry.role == "target"]


def _layer_kind(layer: Any) -> str:
    return getattr(layer, "kind", type(layer).__name__.lower())


def _image_layer_names() -> list[str]:
    """Image layers in the viewer (skip labels/tracks/etc.). Excludes annotated
    counterstain/ignore layers — those should never be auto-picked as targets."""
    viewer = viewer_or_none()
    if viewer is None:
        return []
    skip_roles = {"counterstain", "ignore"}
    excluded = {
        entry.layer_name
        for entry in _CHANNELS.values()
        if entry.role in skip_roles
    }
    out: list[str] = []
    for layer in viewer.layers:
        if _layer_kind(layer) != "image":
            continue
        if layer.name in excluded:
            continue
        out.append(layer.name)
    return out


def resolve_target_channel(query: str | None = None) -> ChannelResolution:
    """Resolve a target channel for analysis workflows.

    Resolution policy (Phase-2 spec):

    1. Explicit user-specified layer name → exact match wins.
    2. User-confirmed ``target`` annotation → if exactly one, use it.
    3. User phrase (e.g. ``green``, ``GFP``) → resolved through metadata.
    4. Strong single inference → if exactly one image layer is present.
    5. Ambiguous → raise ``AmbiguousChannelError`` with candidates so the agent
       can ask the user one focused question.

    Counterstain / ignore annotated channels are never auto-selected.
    """

    viewer = get_viewer()

    if query:
        try:
            viewer.layers[query]
            return ChannelResolution(layer=query, source="explicit")
        except KeyError:
            pass

        try:
            resolved = resolve_layer_name(query)
        except KeyError as e:
            raise AmbiguousChannelError(str(e), _image_layer_names()) from e

        if resolved != query and resolved in _layer_names():
            entry = _CHANNELS.get(resolved)
            if entry and entry.role in {"counterstain", "ignore"}:
                raise AmbiguousChannelError(
                    f"Layer {resolved!r} is annotated as {entry.role!r}; refusing "
                    "to auto-use it as the target channel. Confirm a target "
                    "annotation or pass the layer name explicitly.",
                    _image_layer_names(),
                )
            color = canonical_channel_color(query)
            source = "annotation" if entry and entry.role == "target" else "phrase"
            return ChannelResolution(layer=resolved, source=source, color=color)

        raise AmbiguousChannelError(
            f"Could not resolve channel {query!r}. "
            f"Available image layers: {_image_layer_names()}",
            _image_layer_names(),
        )

    confirmed = _confirmed_target_layers()
    if len(confirmed) == 1:
        return ChannelResolution(
            layer=confirmed[0],
            source="annotation",
            color=_CHANNELS[confirmed[0]].color,
        )
    if len(confirmed) > 1:
        raise AmbiguousChannelError(
            f"Multiple target channels are annotated: {confirmed}. "
            "Pick one explicitly.",
            confirmed,
        )

    image_layers = _image_layer_names()
    if len(image_layers) == 1:
        return ChannelResolution(
            layer=image_layers[0],
            source="inference",
            note="single image layer in viewer; assumed target",
        )

    raise AmbiguousChannelError(
        f"No confirmed target channel and {len(image_layers)} candidate image "
        f"layers ({image_layers}); ask the user which channel is the target.",
        image_layers,
    )


_TABLE_LISTENERS: list[Any] = []


def on_tables_changed(callback: Any) -> None:
    if callback not in _TABLE_LISTENERS:
        _TABLE_LISTENERS.append(callback)


def _emit_tables_changed() -> None:
    for cb in list(_TABLE_LISTENERS):
        try:
            cb()
        except Exception:
            pass
