from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from imajin.config import Settings, ensure_dirs


@dataclass
class CallRecord:
    timestamp: str
    tool: str
    inputs: dict[str, Any]
    output_summary: Any
    duration_s: float
    ok: bool
    driver: str


_CURRENT_SESSION_ID: str | None = None
_CURRENT_DRIVER: str = "direct"
_LOG_PATH: Path | None = None


def start_session(driver: str = "direct", settings: Settings | None = None) -> str:
    global _CURRENT_SESSION_ID, _CURRENT_DRIVER, _LOG_PATH
    settings = settings or Settings.from_env()
    ensure_dirs(settings)
    _CURRENT_SESSION_ID = uuid.uuid4().hex[:12]
    _CURRENT_DRIVER = driver
    _LOG_PATH = settings.sessions_dir / f"{_CURRENT_SESSION_ID}.jsonl"
    return _CURRENT_SESSION_ID


def set_driver(driver: str) -> None:
    global _CURRENT_DRIVER
    _CURRENT_DRIVER = driver


def _summarize(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _summarize(v) for k, v in list(value.items())[:20]}
    if isinstance(value, (list, tuple)):
        return [_summarize(v) for v in list(value)[:20]]
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {
            "_kind": "array",
            "shape": list(shape),
            "dtype": str(getattr(value, "dtype", "?")),
        }
    return f"<{type(value).__name__}>"


def record_call(
    tool: str, inputs: dict[str, Any], output: Any, duration_s: float, ok: bool
) -> None:
    global _LOG_PATH
    if _LOG_PATH is None:
        start_session(driver=_CURRENT_DRIVER)
    record = CallRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        tool=tool,
        inputs=_summarize(inputs),
        output_summary=_summarize(output),
        duration_s=duration_s,
        ok=ok,
        driver=_CURRENT_DRIVER,
    )
    assert _LOG_PATH is not None
    line = json.dumps(asdict(record), default=str) + "\n"
    try:
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        # Provenance is important, but it should never make an analysis fail
        # because the platform data directory is unwritable (common in tests,
        # sandboxes, and some managed workstations). Fall back to /tmp.
        fallback_dir = Path(tempfile.gettempdir()) / "imajin" / "sessions"
        try:
            fallback_dir.mkdir(parents=True, exist_ok=True)
            fallback_path = fallback_dir / _LOG_PATH.name
            with fallback_path.open("a", encoding="utf-8") as f:
                f.write(line)
            _LOG_PATH = fallback_path
        except OSError:
            pass


def current_session_path() -> Path | None:
    return _LOG_PATH


def current_session_id() -> str | None:
    return _CURRENT_SESSION_ID


def read_session(
    session_id: str | None = None, settings: Settings | None = None
) -> list[dict[str, Any]]:
    """Load all CallRecords from a session jsonl. If session_id is None, uses the
    current session. Returns the records in chronological order."""
    if session_id is None:
        path = _LOG_PATH
    else:
        settings = settings or Settings.from_env()
        path = settings.sessions_dir / f"{session_id}.jsonl"
    if path is None or not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
