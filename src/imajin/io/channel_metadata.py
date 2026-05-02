from __future__ import annotations

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
