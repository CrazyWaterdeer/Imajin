from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from imajin.ui import display


@pytest.mark.parametrize(
    "height,expected",
    [
        (1080, 1.0),
        (1199, 1.0),
        (1200, 1.25),
        (1439, 1.25),
        (1440, 1.5),
        (2159, 1.5),
        (2160, 2.0),
        (4320, 2.0),
    ],
)
def test_height_to_scale_thresholds(height: int, expected: float) -> None:
    assert display.height_to_scale(height) == expected


def test_resolve_explicit_numeric_string_bypasses_detection() -> None:
    with patch.object(display, "detect_primary_monitor_height") as detect:
        assert display.resolve_ui_scale("1.5") == 1.5
        assert display.resolve_ui_scale("1.0") == 1.0
        detect.assert_not_called()


def test_resolve_clamps_invalid_numeric_to_auto() -> None:
    with patch.object(display, "detect_primary_monitor_height", return_value=1080):
        # Out-of-range strings fall through to detection.
        assert display.resolve_ui_scale("9.0") == 1.0
        assert display.resolve_ui_scale("-1") == 1.0


def test_resolve_auto_uses_detection() -> None:
    with patch.object(display, "detect_primary_monitor_height", return_value=1440):
        assert display.resolve_ui_scale("auto") == 1.5
    with patch.object(display, "detect_primary_monitor_height", return_value=1080):
        assert display.resolve_ui_scale("auto") == 1.0


def test_resolve_falls_back_when_detection_fails() -> None:
    with patch.object(display, "detect_primary_monitor_height", return_value=None):
        assert display.resolve_ui_scale("auto") == 1.0
        assert display.resolve_ui_scale("") == 1.0
        assert display.resolve_ui_scale("not-a-number") == 1.0


def test_powershell_height_parses_stdout() -> None:
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="1440\n", stderr="")
    with (
        patch.object(display.shutil, "which", return_value="/mnt/c/.../powershell.exe"),
        patch.object(display.subprocess, "run", return_value=fake),
    ):
        assert display._powershell_height() == 1440


def test_powershell_height_returns_none_on_timeout() -> None:
    with (
        patch.object(display.shutil, "which", return_value="/x/powershell.exe"),
        patch.object(
            display.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd="ps", timeout=3),
        ),
    ):
        assert display._powershell_height() is None


def test_powershell_height_returns_none_when_missing() -> None:
    with patch.object(display.shutil, "which", return_value=None):
        assert display._powershell_height() is None


def test_xrandr_height_picks_max_starred_mode() -> None:
    sample = (
        "Screen 0: minimum 320 x 200, current 3840 x 2160, maximum 16384 x 16384\n"
        "DP-1 connected primary 2560x1440+0+0 (normal left inverted right) 600mm x 340mm\n"
        "   2560x1440     59.95*+  74.97\n"
        "   1920x1080     60.00\n"
    )
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=sample, stderr="")
    with (
        patch.object(display.shutil, "which", return_value="/usr/bin/xrandr"),
        patch.object(display.subprocess, "run", return_value=fake),
    ):
        assert display._xrandr_height() == 1440
