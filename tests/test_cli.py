from __future__ import annotations

import subprocess
import sys


def test_cli_doctor_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "imajin.cli", "--doctor"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "imajin doctor" in result.stdout


def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "imajin.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "--doctor" in result.stdout
