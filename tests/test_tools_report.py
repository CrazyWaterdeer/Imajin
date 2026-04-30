from __future__ import annotations

import json
from pathlib import Path

import pytest

from imajin.tools import report


def _write_provenance_log(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "session.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


@pytest.fixture
def fake_session(tmp_path: Path, monkeypatch):
    records = [
        {
            "timestamp": "2026-04-30T10:00:00+00:00",
            "tool": "load_file",
            "inputs": {"path": "/data/sample.lsm"},
            "output_summary": {"layers": ["img_ch0", "img_ch1"]},
            "duration_s": 1.5,
            "ok": True,
            "driver": "manual",
        },
        {
            "timestamp": "2026-04-30T10:01:00+00:00",
            "tool": "list_layers",
            "inputs": {},
            "output_summary": [],
            "duration_s": 0.01,
            "ok": True,
            "driver": "manual",
        },
        {
            "timestamp": "2026-04-30T10:02:00+00:00",
            "tool": "rolling_ball_background",
            "inputs": {"layer": "img_ch0", "radius": 25},
            "output_summary": {"new_layer": "img_ch0_bg"},
            "duration_s": 4.2,
            "ok": True,
            "driver": "llm:claude-sonnet-4-6",
        },
        {
            "timestamp": "2026-04-30T10:03:00+00:00",
            "tool": "cellpose_sam",
            "inputs": {
                "image_layer": "img_ch0_bg",
                "channel": 0,
                "do_3D": True,
                "model": "cpsam",
            },
            "output_summary": {"labels_layer": "img_ch0_bg_masks", "n_cells": 42},
            "duration_s": 22.8,
            "ok": True,
            "driver": "llm:claude-sonnet-4-6",
        },
        {
            "timestamp": "2026-04-30T10:04:00+00:00",
            "tool": "measure_intensity",
            "inputs": {
                "labels_layer": "img_ch0_bg_masks",
                "image_layer": "img_ch1",
                "channels": [1],
                "properties": ["area", "mean_intensity"],
            },
            "output_summary": {"table_name": "measurements", "n_rows": 42},
            "duration_s": 0.7,
            "ok": True,
            "driver": "llm:claude-sonnet-4-6",
        },
        {
            "timestamp": "2026-04-30T10:05:00+00:00",
            "tool": "screenshot",
            "inputs": {"path": None},
            "output_summary": {"path": None},
            "duration_s": 0.3,
            "ok": True,
            "driver": "llm:claude-sonnet-4-6",
        },
        {
            "timestamp": "2026-04-30T10:06:00+00:00",
            "tool": "manders_coefficients",
            "inputs": {"image_a": "img_ch0", "image_b": "img_ch1"},
            "output_summary": {"M1": 0.81, "M2": 0.66},
            "duration_s": 0.4,
            "ok": False,
            "driver": "llm:claude-sonnet-4-6",
        },
    ]
    log_path = _write_provenance_log(tmp_path, records)

    from imajin.agent import provenance

    monkeypatch.setattr(provenance, "_LOG_PATH", log_path)
    monkeypatch.setattr(provenance, "_CURRENT_SESSION_ID", "fake_session_id")
    return records


def test_generate_methods_includes_real_steps(fake_session) -> None:
    res = report.generate_methods()
    md = res["markdown"]
    assert "Methods" in md
    assert "Cellpose-SAM" in md
    assert "rolling-ball" in md
    assert "regionprops_table" in md
    # Skipped tools should NOT appear
    assert "list_layers" not in md
    assert "screenshot" not in md
    # Failed tool (manders) should NOT appear
    assert "Manders" not in md
    assert "Pearson" not in md
    assert res["n_records"] == len(fake_session)


def test_generate_methods_empty_session(tmp_path, monkeypatch) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    from imajin.agent import provenance

    monkeypatch.setattr(provenance, "_LOG_PATH", empty)
    res = report.generate_methods()
    assert "No analytical operations" in res["markdown"]


def test_generate_report_html(fake_session, tmp_path) -> None:
    out = tmp_path / "report.html"
    res = report.generate_report(str(out), format="html")
    assert out.exists()
    body = out.read_text(encoding="utf-8")
    assert "<html>" in body and "</html>" in body
    assert "cellpose_sam" in body  # operations table shows raw tool names
    assert "Methods" in body  # methods section embedded
    assert res["format"] == "html"


def test_generate_report_md(fake_session, tmp_path) -> None:
    out = tmp_path / "report.md"
    res = report.generate_report(str(out), format="md")
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("## Methods")
    assert res["format"] == "md"


def test_generate_report_rejects_bad_format(fake_session, tmp_path) -> None:
    with pytest.raises(ValueError, match="format must be"):
        report.generate_report(str(tmp_path / "x.pdf"), format="pdf")
