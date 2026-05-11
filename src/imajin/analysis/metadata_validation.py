from __future__ import annotations

from pathlib import Path
from typing import Any

from imajin.io.metadata import read_metadata_summary

INTENSITY_SETTINGS: tuple[str, ...] = (
    "laser_intensity",
    "detector_gain",
    "bit_depth",
    "pinhole_size",
)

SETTING_LABELS: dict[str, str] = {
    "laser_intensity": "laser intensity",
    "detector_gain": "detector gain",
    "bit_depth": "color bit depth",
    "pinhole_size": "pinhole size",
}

_INTENSITY_TERMS = {
    "intensity",
    "mean_intensity",
    "max_intensity",
    "min_intensity",
    "integrated_intensity",
    "sum_intensity",
    "fluorescence",
    "signal",
    "calexa",
    "gcamp",
}
_AREA_ONLY_TERMS = {
    "area",
    "area_um2",
    "volume",
    "volume_um3",
    "size",
    "morphology",
    "centroid",
    "count",
    "n_objects",
}


def _norm(value: Any) -> str:
    return " ".join(str(value).lower().replace("_", " ").replace("-", " ").split())


def _compact(value: Any) -> str:
    return _norm(value).replace(" ", "")


def _canonical_color(value: Any) -> str | None:
    text = _compact(value)
    if not text:
        return None
    aliases = {
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
        "blue": "uv",
        "dapi": "uv",
        "hoechst": "uv",
        "405": "uv",
        "ir": "ir",
        "farred": "ir",
        "infrared": "ir",
        "cy5": "ir",
        "alexa647": "ir",
        "633": "ir",
        "640": "ir",
        "647": "ir",
    }
    if text in aliases:
        return aliases[text]
    for alias, color in aliases.items():
        if alias in text:
            return color
    return None


def required_settings_for_analysis(
    *,
    analysis_kind: str = "auto",
    measurement: dict[str, Any] | None = None,
) -> tuple[str, tuple[str, ...]]:
    kind = str(analysis_kind or "auto").strip().lower().replace("-", "_")
    if kind in {"intensity", "fluorescence", "timecourse", "time_course", "calexa"}:
        return "intensity", INTENSITY_SETTINGS
    if kind in {"area", "size", "morphology", "count", "shape"}:
        return kind, ()
    if kind != "auto":
        raise ValueError(
            "analysis_kind must be auto, intensity, area, size, morphology, or count"
        )

    props = measurement.get("properties") if isinstance(measurement, dict) else None
    if props:
        prop_text = {_compact(item) for item in props}
        if any(any(term.replace("_", "") in prop for term in _INTENSITY_TERMS) for prop in prop_text):
            return "intensity", INTENSITY_SETTINGS
        if prop_text and all(
            any(term.replace("_", "") in prop for term in _AREA_ONLY_TERMS)
            for prop in prop_text
        ):
            return "area", ()
    return "intensity", INTENSITY_SETTINGS


def _channel_text(name: str, info: dict[str, Any]) -> str:
    parts: list[str] = [name]
    for key in (
        "name",
        "channel_name",
        "dye_name",
        "marker",
        "color",
        "display_color_name",
        "excitation_wavelength_nm",
        "emission_wavelength_nm",
        "laser_wavelength_nm",
    ):
        value = info.get(key)
        if value is not None:
            parts.append(str(value))
    return " ".join(parts)


def _resolve_channel(
    summary: dict[str, Any],
    target_channel: str | None,
) -> tuple[int | None, dict[str, Any] | None, str]:
    channels = list(summary.get("channel_metadata") or [])
    names = [str(n) for n in (summary.get("channel_names") or [])]
    n_channels = int(summary.get("n_channels") or len(channels) or len(names) or 1)
    while len(channels) < n_channels:
        channels.append({})
    while len(names) < n_channels:
        names.append(str(channels[len(names)].get("name") or f"ch{len(names)}"))

    if target_channel is None:
        if n_channels == 1:
            return 0, channels[0], "single_channel"
        return None, None, "target channel is unspecified for a multi-channel file"

    query = _norm(target_channel)
    query_compact = query.replace(" ", "")
    query_color = _canonical_color(target_channel)

    exact = [
        idx
        for idx, name in enumerate(names)
        if query_compact and query_compact == _compact(name)
    ]
    if len(exact) == 1:
        idx = exact[0]
        return idx, channels[idx], "channel_name"

    color_matches: list[int] = []
    if query_color is not None:
        for idx, info in enumerate(channels):
            color = _canonical_color(info.get("color")) or _canonical_color(
                info.get("display_color_name")
            )
            if color == query_color:
                color_matches.append(idx)
        if len(color_matches) == 1:
            idx = color_matches[0]
            return idx, channels[idx], "channel_color"

    text_matches = [
        idx
        for idx, (name, info) in enumerate(zip(names, channels))
        if query and query in _norm(_channel_text(name, info))
    ]
    if len(text_matches) == 1:
        idx = text_matches[0]
        return idx, channels[idx], "metadata_text"

    matches = list(dict.fromkeys(exact + color_matches + text_matches))
    if len(matches) > 1:
        return None, None, f"target channel {target_channel!r} is ambiguous"
    if n_channels == 1:
        return 0, channels[0], "single_channel_fallback"
    return None, None, f"target channel {target_channel!r} was not found in metadata"


def _value_key(value: Any) -> tuple[str, Any]:
    if value is None or value == "":
        return ("missing", None)
    if isinstance(value, (int, float)):
        return ("number", round(float(value), 9))
    return ("text", _norm(value))


def _setting_value(info: dict[str, Any], setting: str) -> Any:
    if setting == "bit_depth":
        return info.get("bit_depth", info.get("color_bit_depth"))
    return info.get(setting)


def _load_summary(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    summary = record.get("metadata_summary")
    if isinstance(summary, dict) and summary and "metadata_error" not in summary:
        return dict(summary), None
    path = record.get("path")
    if not path:
        return None, "missing path"
    try:
        return read_metadata_summary(str(path)), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def validate_acquisition_metadata(
    records: list[dict[str, Any]],
    *,
    target_channel: str | None = None,
    analysis_kind: str = "auto",
    measurement: dict[str, Any] | None = None,
    strict_missing: bool = False,
) -> dict[str, Any]:
    resolved_kind, required_settings = required_settings_for_analysis(
        analysis_kind=analysis_kind,
        measurement=measurement,
    )
    warnings: list[str] = []
    channels: list[dict[str, Any]] = []
    metadata_errors: list[dict[str, str]] = []

    if not records:
        return {
            "ok": False,
            "status": "fail",
            "analysis_kind": resolved_kind,
            "settings_checked": list(required_settings),
            "error": "no files were provided for metadata validation",
            "warnings": [],
            "mismatches": [],
            "missing_settings": [],
            "channels": [],
            "metadata_only": True,
        }

    if not required_settings:
        return {
            "ok": True,
            "status": "pass",
            "analysis_kind": resolved_kind,
            "settings_checked": [],
            "warnings": [],
            "mismatches": [],
            "missing_settings": [],
            "channels": [],
            "metadata_only": True,
        }

    for record in records:
        path = str(record.get("path") or "")
        per_record_target = record.get("target_channel") or target_channel
        summary, error = _load_summary(record)
        if error is not None or summary is None:
            metadata_errors.append({"path": path, "error": error or "unknown error"})
            continue
        idx, channel_info, reason = _resolve_channel(summary, per_record_target)
        if idx is None or channel_info is None:
            warnings.append(f"{Path(path).name}: {reason}")
            continue
        channels.append(
            {
                "path": path,
                "channel_index": idx,
                "channel_name": (
                    (summary.get("channel_names") or [None] * (idx + 1))[idx]
                    if idx < len(summary.get("channel_names") or [])
                    else channel_info.get("name")
                ),
                "target_resolution": reason,
                "settings": {
                    setting: _setting_value(channel_info, setting)
                    for setting in required_settings
                },
            }
        )

    mismatches: list[dict[str, Any]] = []
    missing_settings: list[dict[str, Any]] = []
    for setting in required_settings:
        known: dict[tuple[str, Any], list[str]] = {}
        missing: list[str] = []
        for channel in channels:
            value = channel["settings"].get(setting)
            if value is None or value == "":
                missing.append(channel["path"])
                continue
            known.setdefault(_value_key(value), []).append(channel["path"])
        if len(known) > 1:
            values = [
                {
                    "value": value,
                    "paths": paths,
                }
                for (_kind, value), paths in known.items()
            ]
            mismatches.append(
                {
                    "setting": setting,
                    "label": SETTING_LABELS.get(setting, setting),
                    "values": values,
                }
            )
        if missing or not known:
            missing_settings.append(
                {
                    "setting": setting,
                    "label": SETTING_LABELS.get(setting, setting),
                    "paths": missing,
                    "all_missing": not known,
                }
            )

    if metadata_errors:
        warnings.extend(
            f"{Path(item['path']).name}: metadata could not be read ({item['error']})"
            for item in metadata_errors
        )

    fail_for_missing = strict_missing and (missing_settings or metadata_errors or warnings)
    status = "pass"
    if mismatches or fail_for_missing:
        status = "fail"
    elif missing_settings or metadata_errors or warnings:
        status = "warning"

    error = None
    if mismatches:
        labels = ", ".join(item["label"] for item in mismatches)
        error = f"acquisition metadata mismatch for target channel: {labels}"
    elif fail_for_missing:
        error = "acquisition metadata is incomplete; target settings cannot be verified"

    return {
        "ok": status != "fail",
        "status": status,
        "analysis_kind": resolved_kind,
        "settings_checked": list(required_settings),
        "setting_labels": {k: SETTING_LABELS.get(k, k) for k in required_settings},
        "n_files": len(records),
        "n_channels_checked": len(channels),
        "warnings": warnings,
        "mismatches": mismatches,
        "missing_settings": missing_settings,
        "metadata_errors": metadata_errors,
        "channels": channels,
        "metadata_only": True,
        "error": error,
    }
