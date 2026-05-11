from __future__ import annotations

import re
from typing import Any


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
    "farred": "ir",
    "far red": "ir",
    "633": "ir",
    "640": "ir",
    "647": "ir",
}

_SETTING_ALIASES: dict[str, tuple[str, ...]] = {
    "laser_intensity": (
        "laser_intensity",
        "laser power",
        "laser_power",
        "laserpower",
        "power",
        "transmission",
        "laser transmission",
        "laser_transmission",
        "attenuation",
    ),
    "detector_gain": (
        "detector_gain",
        "detector gain",
        "detectorgain",
        "gain",
        "master gain",
        "master_gain",
        "amplifier gain",
        "amplifier_gain",
        "voltage",
        "detector voltage",
        "detector_voltage",
    ),
    "pinhole_size": (
        "pinhole_size",
        "pinhole size",
        "pinholesize",
        "pinhole",
        "pinhole diameter",
        "pinhole_diameter",
        "pinholesizeairy",
        "pinhole airy",
    ),
    "bit_depth": (
        "bit_depth",
        "bit depth",
        "bitsperpixel",
        "bits per pixel",
        "bits_per_pixel",
        "significantbits",
        "significant_bits",
        "validbits",
        "valid_bits",
        "colorbit",
        "color bit",
    ),
}


def _norm(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def _compact_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _coerce_setting_value(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        val = float(value)
        return int(val) if val.is_integer() else val
    text = str(value).strip()
    if not text:
        return None
    try:
        val = float(text)
        return int(val) if val.is_integer() else val
    except ValueError:
        pass
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if match:
        try:
            val = float(match.group(0))
            return int(val) if val.is_integer() else val
        except ValueError:
            pass
    return text


def _lookup_alias(mapping: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    if not isinstance(mapping, dict):
        return None
    compact = {_compact_key(k): v for k, v in mapping.items()}
    for alias in aliases:
        key = _compact_key(alias)
        if key in compact:
            return compact[key]
    return None


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


def bit_depth_from_dtype(dtype: Any) -> int | None:
    if dtype is None or dtype == "":
        return None
    try:
        import numpy as np

        return int(np.dtype(dtype).itemsize * 8)
    except Exception:
        return None


def acquisition_settings_from_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    for field, aliases in _SETTING_ALIASES.items():
        value = _coerce_setting_value(_lookup_alias(mapping, aliases))
        if value is not None:
            settings[field] = value
            if field == "bit_depth":
                settings["color_bit_depth"] = value
    return settings


def laser_settings_from_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    settings = acquisition_settings_from_mapping(mapping)
    wavelength = _lookup_alias(
        mapping,
        (
            "laser_wavelength",
            "laser wavelength",
            "laserwavelength",
            "wavelength",
            "excitation_wavelength",
            "excitation wavelength",
        ),
    )
    wavelength_value = wavelength_nm(wavelength)
    if wavelength_value is not None:
        settings["laser_wavelength_nm"] = float(wavelength_value)
    return settings


def apply_dtype_bit_depth(
    channel_metadata: list[dict[str, Any]],
    dtype: Any,
) -> list[dict[str, Any]]:
    bit_depth = bit_depth_from_dtype(dtype)
    if bit_depth is None:
        return channel_metadata
    for item in channel_metadata:
        item.setdefault("bit_depth", bit_depth)
        item.setdefault("color_bit_depth", bit_depth)
    return channel_metadata


def color_from_wavelengths(
    excitation_nm: float | None = None,
    emission_nm: float | None = None,
) -> str | None:
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


def color_from_name(name: str | None) -> str | None:
    if not name:
        return None
    norm = _norm(name)
    compact = norm.replace(" ", "")
    for alias, color in _NAME_COLOR_ALIASES.items():
        if alias in norm or alias in compact:
            return color
    return None


def display_color_name(rgb: tuple[float, float, float]) -> str | None:
    r, g, b = rgb
    candidates = {
        "gray": (1.0, 1.0, 1.0),
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "cyan": (0.0, 1.0, 1.0),
        "magenta": (1.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
    }
    for name, target in candidates.items():
        if all(abs(value - expected) < 1 / 255 for value, expected in zip(rgb, target)):
            return name
    return None


def rgb_from_8bit_triplet(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        r, g, b = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None
    if max(r, g, b) > 1.0:
        r, g, b = r / 255.0, g / 255.0, b / 255.0
    rgb = tuple(max(0.0, min(1.0, channel)) for channel in (r, g, b))
    if not any(rgb):
        return None
    return rgb


def build_channel_info(
    *,
    name: str | None = None,
    excitation: Any = None,
    emission: Any = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ex_nm = wavelength_nm(excitation)
    em_nm = wavelength_nm(emission)
    color = color_from_wavelengths(ex_nm, em_nm) or color_from_name(name)
    info: dict[str, Any] = dict(extra or {})
    if name:
        info["name"] = str(name)
    if ex_nm is not None:
        info["excitation_wavelength_nm"] = float(ex_nm)
    if em_nm is not None:
        info["emission_wavelength_nm"] = float(em_nm)
    if color:
        info["color"] = color
    return info


def pad_channel_metadata(
    channel_metadata: list[dict[str, Any]],
    n_channels: int,
    names: list[str] | None = None,
) -> list[dict[str, Any]]:
    out = [dict(m) for m in channel_metadata[:n_channels]]
    names = names or []
    while len(out) < n_channels:
        i = len(out)
        name = names[i] if i < len(names) else None
        out.append(build_channel_info(name=name))
    return out
