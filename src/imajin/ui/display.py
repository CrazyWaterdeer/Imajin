"""Detect monitor resolution before Qt loads, to set QT_SCALE_FACTOR.

Must run before any Qt import, since QT_SCALE_FACTOR is read at QApplication
construction. All detection is best-effort — return None on any failure and
let the caller fall back to 1.0x.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess


_PS_HEIGHT_CMD = (
    "Add-Type -AssemblyName System.Windows.Forms; "
    "[System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Height"
)


def _is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _powershell_height() -> int | None:
    ps = shutil.which("powershell.exe")
    if not ps:
        return None
    try:
        out = subprocess.run(
            [ps, "-NoProfile", "-Command", _PS_HEIGHT_CMD],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    try:
        return int(out.stdout.strip())
    except (ValueError, AttributeError):
        return None


def _xrandr_height() -> int | None:
    xr = shutil.which("xrandr")
    if not xr:
        return None
    try:
        out = subprocess.run(
            [xr, "--current"], capture_output=True, text=True, timeout=2
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    # Look for a line like "   1920x1080     59.96*+  60.00"
    # The starred mode on the first connected output is the current one.
    heights: list[int] = []
    for line in out.stdout.splitlines():
        m = re.search(r"\b(\d{3,5})x(\d{3,5})\b.*\*", line)
        if m:
            heights.append(int(m.group(2)))
    return max(heights) if heights else None


def detect_primary_monitor_height() -> int | None:
    """Return the primary monitor's pixel height, or None if undetectable."""
    if _is_wsl():
        return _powershell_height()
    return _xrandr_height()


def height_to_scale(height: int) -> float:
    """Map a monitor height (px) to a sensible UI scale factor."""
    if height >= 2160:
        return 2.0
    if height >= 1440:
        return 1.5
    if height >= 1200:
        return 1.25
    return 1.0


def resolve_ui_scale(setting: str) -> float:
    """Turn a settings.ui_scale value into a concrete float.

    `"auto"` (or unparseable) → detect monitor; fall back to 1.0 on failure.
    Numeric strings (`"1.0"`, `"1.5"`, etc.) → that value, clamped to [0.5, 4.0].
    Caller is responsible for honoring an existing QT_SCALE_FACTOR env override
    *before* calling this function.
    """
    s = (setting or "").strip().lower()
    if s and s != "auto":
        try:
            v = float(s)
        except ValueError:
            v = None
        if v is not None and 0.5 <= v <= 4.0:
            return v
    h = detect_primary_monitor_height()
    if h is None:
        return 1.0
    return height_to_scale(h)
