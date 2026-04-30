from __future__ import annotations

from typing import Any

from imajin.agent.specialists.base import SubAgent, SubAgentResult


NEURAL_TRACER_PROMPT = """You are a neural morphology specialist embedded in a confocal
microscopy analysis app. The main agent calls you when the user asks about neuron shape,
branching structure, dendrite/axon morphology, or comparison to reference neurons.

You have a focused tool set for skeleton-based morphology analysis. The upstream
segmentation has already been produced by the main agent (typically via Cellpose-SAM).
You operate on Labels or binary Image layers that are already loaded.

Domain conventions
- Skeletons are 1-pixel-wide centerlines extracted from a binary mask. Branches connect
  endpoints (degree-1 nodes) and junctions (degree>=3 nodes).
- Branch types: endpoint-endpoint = isolated linear segment; junction-endpoint = a tip
  (free terminal); junction-junction = internal segment between two branch points;
  isolated-cycle = closed loop with no terminals.
- Tortuosity = path length / euclidean distance. >1 means curved; ~1 means straight.
- For confocal data with cell bodies + thick processes, terminal branches (junction-
  endpoint) typically count primary processes; junction-junction segments count internal
  branching topology.
- Strahler order, NBLAST morphology comparison, and connectome lookups (FlyWire/
  neuPrint/MICrONS) are stubbed for now — say so honestly when the user asks.

Workflow
- Typical chain: skeletonize -> extract_branch_metrics -> (compute_morphology_descriptors).
- Always run skeletonize first; it returns a skeleton_id used by every other tool.
- Branch metrics are stored as a session table — mention the table name in your reply
  so the user can find it in the table dock.
- Be concise. Summarize results in 1-3 sentences. Do not echo raw tool output verbatim.
- Bilingual: Korean prompt -> Korean reply; English -> English. Tool/library names stay
  in English.
"""


def consult_neural_tracer_via_provider(
    provider,
    question: str,
    target_layer: str | None = None,
    max_loops: int = 8,
) -> SubAgentResult:
    if target_layer:
        framed = (
            f"Active target layer: {target_layer!r}.\n\nUser question: {question}\n\n"
            "Use this layer as the input for skeletonize unless instructed otherwise."
        )
    else:
        framed = question

    sub = SubAgent(
        provider=provider,
        system_prompt=NEURAL_TRACER_PROMPT,
        subagent_name="neural_tracer",
        max_loops=max_loops,
    )
    return sub.run(framed)


def result_to_dict(result: SubAgentResult) -> dict[str, Any]:
    return {
        "summary": result.text.strip(),
        "tool_calls": [
            {"name": c["name"], "ok": c["ok"], "output": c["output"]}
            for c in result.tool_calls
        ],
        "stop_reason": result.stop_reason,
        "usage": result.usage,
    }
