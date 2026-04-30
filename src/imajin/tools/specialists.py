from __future__ import annotations

from typing import Any

from imajin.tools.registry import tool


@tool(
    description="Consult the neural morphology specialist. Use this when the user asks "
    "about neuron shape / branching structure / dendrite or axon morphology / "
    "skeleton metrics. Pass the user's question and (optionally) the target Labels or "
    "binary Image layer name. The specialist runs its own focused tool loop and "
    "returns a structured summary plus the tool calls it made. Currently supports "
    "skeletonization and branch metrics; NBLAST + connectome lookups are stubbed.",
    phase="6",
)
def consult_neural_tracer(
    question: str,
    target_layer: str | None = None,
    max_loops: int = 8,
) -> dict[str, Any]:
    from imajin.agent.specialists.base import get_current_provider
    from imajin.agent.specialists.neural_tracer import (
        consult_neural_tracer_via_provider,
        result_to_dict,
    )

    provider = get_current_provider()
    result = consult_neural_tracer_via_provider(
        provider, question, target_layer=target_layer, max_loops=max_loops
    )
    return result_to_dict(result)


@tool(
    description="Consult the scientific writing specialist to produce ready-to-paste "
    "prose from the session provenance log. style='paper' for a Methods paragraph, "
    "'slide' for bullet points, 'protocol' for numbered steps. The specialist has no "
    "tools and never re-runs analyses — it just synthesizes what was already done.",
    phase="7",
)
def consult_methods_writer(
    style: str = "paper",
    session_id: str | None = None,
    extra_context: str | None = None,
) -> dict[str, Any]:
    from imajin.agent import provenance
    from imajin.agent.specialists.base import get_current_provider
    from imajin.agent.specialists.report_writer import (
        consult_report_writer_via_provider,
    )

    provider = get_current_provider()
    records = provenance.read_session(session_id)
    text = consult_report_writer_via_provider(
        provider, records, style=style, extra_context=extra_context
    )
    return {
        "style": style,
        "session_id": session_id or provenance.current_session_id(),
        "n_records": len(records),
        "markdown": text,
    }
