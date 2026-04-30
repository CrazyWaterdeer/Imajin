from __future__ import annotations

import json
from typing import Any

from imajin.agent.providers.base import Provider, Stop, TextDelta


REPORT_WRITER_PROMPT = """You are a scientific writing specialist embedded in a
confocal microscopy analysis app. The main agent calls you to produce ready-to-paste
prose: paper Methods sections, slide-deck bullets, or step-by-step protocols.

You have NO tools. The main agent has already collected the provenance log and table
summaries; they are passed to you in the user message. Your job is to synthesize them
into clean prose in the requested style.

Style conventions
- "paper": one Methods paragraph (3-6 sentences), past-tense passive voice, cite
  software with versions in parentheses (Cellpose-SAM, scikit-image, btrack, skan,
  napari). Mention parameter values when they were non-default.
- "slide": 3-6 short bullets, present tense, no jargon redundancy.
- "protocol": numbered steps, imperative voice, with concrete parameter values.

Quality bar
- Do not invent steps that are not in the provenance log.
- Do not list steps that failed (ok=false in the log).
- Convert tool names into their underlying technique names (e.g., cellpose_sam →
  "Cellpose-SAM segmentation"; rolling_ball_background → "rolling-ball background
  subtraction"). Do not mention internal tool names.
- Bilingual: Korean prompt → Korean output; English → English. Keep software/library
  names in English regardless.
- Be concise. Output only the requested prose — no preamble, no markdown header
  unless the style requires it.
"""


def consult_report_writer_via_provider(
    provider: Provider,
    session_records: list[dict[str, Any]],
    style: str = "paper",
    extra_context: str | None = None,
) -> str:
    if style not in ("paper", "slide", "protocol"):
        raise ValueError(f"style must be 'paper', 'slide', or 'protocol'; got {style!r}")

    pipeline = [r for r in session_records if r.get("ok", True)]

    payload = {
        "style": style,
        "n_operations": len(pipeline),
        "operations": [
            {
                "tool": r["tool"],
                "inputs": r.get("inputs", {}),
                "duration_s": round(float(r.get("duration_s", 0.0)), 2),
            }
            for r in pipeline
        ],
    }

    user_text = (
        f"Produce a '{style}' style write-up from the provenance log below. "
        "Use only operations that actually ran (the log already excludes failures).\n\n"
        f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"
    )
    if extra_context:
        user_text += f"\n\nAdditional context from the user:\n{extra_context}"

    messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
    out = ""
    for event in provider.stream(messages, [], REPORT_WRITER_PROMPT):
        if isinstance(event, TextDelta):
            out += event.text
        elif isinstance(event, Stop):
            break
    return out.strip()
