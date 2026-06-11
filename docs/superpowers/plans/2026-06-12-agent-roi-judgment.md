# Agent-driven ROI judgment — implementation plan

Date: 2026-06-12
Status: draft (pre-review)

## Goal

Replace hard-coded one-shot ROI selection with a **confidence-tiered hybrid** so
target-object segmentation self-corrects on clear cases, lets the agent *see* and
judge ambiguous cases, and escalates to the existing user-review path only when
needed.

User decisions locked (2026-06-12):
- **Auto-correct = separate opt-in tool.** `segment_target_objects` stays
  single-shot; a new tool runs the deterministic loop. (Keeps existing behaviour
  and tests intact; easy to reason about.)
- **Vision attachment = only when ambiguous.** The deterministic ROI score gates
  whether the overlay image is attached to the agent, so tokens are spent only
  where judgment is actually needed.

Review fixes folded in (2026-06-12, after critical re-read):
- H1 single-shot needs its own `confidence_from_score` (selection_confidence can
  never return "high" for one candidate → would attach an image every time).
- H2 the target path must score on the **corrected** metrics the tool already
  computed, never re-call `rank_segmentation_labels` (which scores on raw).
- H3 `correct_roi` must also surface `qc_png_path` + `roi_confidence` (else the
  agent is blind right after the correction it just made).
- H4 runner must look up `vision_hint` defensively (injected tool_caller may route
  names absent from `_REGISTRY`).
- M1 vision scope + auto3d gets `roi_confidence` for free; M2 split background from
  thresholding in the loop; M3 drop the `context` flag (just never wire the loop to
  expression domains).

## Architecture: confidence ladder

```
segment_target_objects → roi_score / roi_confidence (deterministic)
  high / clear     → accept as-is (or auto_segment_target loop already converged)
  low / medium     → attach QC overlay to tool_result → agent judges, may correct_roi
  still uncertain  → review_target_roi → user marks add/remove (already built)
```

This mirrors existing idioms: `selection_confidence` (auto3d) and the connectome
status ladder. No new conceptual machinery.

## What already exists (reuse, do not rebuild)

- ROI quality scorer: `rank_segmentation_labels` — `analysis/segmentation_auto3d.py:513`
  (penalises too-wide via `mask_fraction`, too-narrow via `top_bright_outside_fraction`,
  weak `separation_snr`, degenerate shapes). Confidence tiers: `selection_confidence:615`.
- Directional QC: `target_object_qc` — `analysis/segmentation.py:343`
  (`mask_fraction>0.5` → background included; `top_bright_outside>0.25` → signal missed;
  separation `< noise_sigma` → uncertain).
- Re-run-with-new-params tool: `correct_roi` — `tools/segment.py:1181`
  (accepts `min_snr`, `high_snr`, `auto_mask_hyperbright`, `threshold_clip_percentile`,
  `background_radius`, `smoothing_sigma`, `min_size`).
- Headless QC overlay: `_write_segmentation_qc_png` — `tools/_segmentation_outputs.py:125`
  (raw MIP + colour label fill + orange boundaries; viewer-free; 3D via `_project_for_qc:100`).
- Image encoders (currently dead code): `vision.py:10,53`.
- User-review path (fully wired incl. batch): `review_target_roi:1286`,
  `analysis/interactive_roi.py`, `agent/review_checkpoint.py`.

## The one real gap

`runner.py:441-447` always attaches **text-only** tool results
(`_compact_tool_result`). `vision_hint=True` on the 4 segmentation tools is never
consumed. The agent never sees the segmentation overlay.

---

## Phase 0 — shared scorer + pure compute (foundation for both tracks)

Both tracks need a deterministic confidence on every target-object segmentation.

- **C0.1** Extract a reusable **pure** scorer `score_roi_quality(metrics, *, ndim)
  -> float` from the inline body of `rank_segmentation_labels`
  (`segmentation_auto3d.py:541-612`). It consumes a metrics dict; it does **not**
  compute QC itself. Keep `rank_segmentation_labels` as a thin caller (it keeps
  computing metrics on raw, so auto3d stays byte-identical — characterise with
  existing tests first). **H2:** the target path passes the tool's
  *corrected*-image `signal_qc` (`segment.py:784`) into the scorer; it must never
  call `rank_segmentation_labels` directly (that scores on raw, where background
  offset corrupts `top_bright_outside_fraction` and inside/outside separation).
  No `context` flag (M3) — high coverage is judged the same; the guardrail is simply
  that the loop is never wired to expression domains.
- **C0.2** New single-shot confidence `confidence_from_score(score, metrics)
  -> "high"|"medium"|"low"` (**H1** — `selection_confidence` is for *lists* of
  candidates and can never return "high" for one, so reusing it would attach an
  image on every clean segmentation). Proposed: "low" if score < 55 or a critical
  warning (zero objects, region-level merge) is present; "high" if score ≥ 75 and no
  critical warning; "medium" otherwise. Thresholds are heuristic — flagged for
  empirical tuning.
- **C0.3** Extract the pure compute of `segment_target_objects` (`segment.py:687-803`)
  into `analysis/target_segmentation.py`, split so the loop can reuse the expensive
  invariant step (**M2**):
  - `prepare_corrected(raw, spacing, *, background_radius, background_method,
    background_percentile, smoothing_sigma) -> corrected_for_threshold`
  - `threshold_and_label(corrected, raw, spacing, *, threshold params, min_snr,
    high_snr, min_size, ...) -> (labels, threshold, noise_sigma, qc_dict)`
  - thin `segment_target_array(...)` composing both for the single-shot tool.
  No viewer access. The tool keeps the `snapshot_layer` (670) /
  `add_labels_from_worker` (806) wrapping.
- **C0.4** `segment_target_objects` returns two new fields: `roi_score` (float) and
  `roi_confidence` (from C0.2, on corrected metrics). Additive only — existing keys
  unchanged (verified: no exact-key assertions in `test_tools_segment.py`).

Tests: scorer parity vs current auto3d scores (raw metrics path unchanged);
`confidence_from_score` returns "high" on a clean blob, "low" on empty/over-wide;
`segment_target_array` matches the tool's labels on a synthetic stack; new fields
present and sensible.

## Phase A — Track 2: wire the agent's eyes (ambiguous-only)

- **C A.1** New helper `agent/vision.py::overlay_image_block(qc_png_path, max_px=512)`:
  load saved QC PNG, downscale longest side to ≤512, return Anthropic image block.
  (Reads the already-saved overlay; falls back to `None` if missing — headless safe.)
- **C A.2** `runner.py` success branch: after `result = tool_caller(...)`, look up
  the entry **defensively** (**H4**) — `_REGISTRY.get(name)`, default
  `vision_hint=False` — because an injected `tool_caller` (`runner.py:438`) may route
  names absent from the registry and `get_tool` raises `KeyError` (would crash the
  whole turn). If `vision_hint` **and** `result.get("roi_confidence") in
  {"low","medium"}` **and** `result.get("qc_png_path")`, build `content` as a list
  `[{type:text, ...}, image_block]`. Otherwise unchanged (text only). The gate is
  precise now that C0.2 yields a real "high" tier (without H1, every clean
  segmentation would be "medium" and attach an image).
- **C A.5** (**H3**) Extend `correct_roi` (`segment.py:1263-1274`) to also return
  `qc_png_path` and `roi_confidence` from its internal `segment_target_objects` run,
  so the vision gate fires on the corrected result — the moment the agent most needs
  to see it. Currently it returns neither.
- **C A.3** Token-cost guard in compaction: in `_message_chars` (`runner.py:194`),
  count image blocks as a fixed nominal cost (not base64 length) so one overlay does
  not evict recent text context; rely on existing summarisation to drop aged images.
- **C A.4** Provider handling:
  - Anthropic (`anthropic.py:51`): works natively, no change.
  - OpenAI-compat (`openai_compat.py:142-145`): **degrade now** — keep dropping the
    image (text-only). Add a `TODO` + a follow-up note to emit a trailing
    `image_url` user message later. (Anthropic is the primary provider.)

Tests: runner attaches an image block iff vision_hint + low/medium confidence +
qc_png_path (monkeypatched tool result); high-confidence → text only; OpenAI
translation drops the image without error.

## Phase B — Track 1: deterministic auto-correct tool (opt-in)

- **C B.1** Pure policy `analysis/target_segmentation.py::next_correction(qc, params)
  -> dict | None` implementing the table below; returns `None` when no improving
  move is available.
- **C B.2** Pure loop `auto_correct_target(raw, spacing, params, *, max_iters=3,
  accept_score=...)` → best `(labels, params, score, history)`, scoring each try with
  `score_roi_quality` on **corrected** metrics (H2). In-memory only; no layer churn.
  **M2:** reuse `prepare_corrected` across iterations; only recompute it when a
  correction changes a background param (`background_radius`/`smoothing_sigma`).
  Track tried param-tuples and keep best-so-far (no-regression); `max_iters` bounds
  termination so threshold oscillation cannot loop forever.
- **C B.3** New tool `auto_segment_target(image_layer, ...)` (`worker=True`,
  `vision_hint=True`): one `snapshot_layer`, run `auto_correct_target`, add the single
  best labels layer, save QC PNG, return labels + `roi_score`/`roi_confidence` +
  `correction_history`. `segment_target_objects` untouched.

Correction policy (all params already on `correct_roi`):

| Symptom (QC) | Diagnosis | Move |
|---|---|---|
| `mask_fraction>0.45` & low `top_bright_outside` | too wide | `min_snr` +1.0; then enable `auto_mask_hyperbright` / `clip_percentile` |
| `top_bright_outside>0.25` | too narrow | `min_snr` −0.5; `high_snr` −1.0 |
| `separation_snr<1` | background-dominated | `background_radius` ↑ / `smoothing_sigma` ↑; else low-confidence |
| `n_objects==0` | threshold too high | `min_snr` large ↓ / percentile fallback |

Loop is **target-objects only** (never expression domains).

Tests (all headless): synthetic "too wide" stack converges to lower mask_fraction;
"too narrow" recovers bright outside signal; loop is idempotent at a good starting
point (no-regression keeps best); bounded by `max_iters`.

## Phase C — integration + prompt

- **C C.1** Prompt guidance (`agent/prompts.py`): when `roi_confidence` is low/medium
  or an overlay is attached, judge the image and either `correct_roi` (named fix) or
  `review_target_roi` (escalate). **Role division (M1):** `auto_segment_target` is the
  deterministic, no-LLM path for batch/headless and "make it accurate hands-off";
  the agent-vision + `correct_roi` path is for interactive single-sample judgment.
  (Currently the prompt says nothing about `correct_roi` or QC iteration.)
- **C C.2** PROJECT_PLAN note under Phase 5/Architecture: ROI judgment ladder shipped;
  Core Analysis Layer gained `target_segmentation.segment_target_array`.

## Risks / non-goals

- **Deepest residual risk:** the score cannot distinguish "too wide" from a
  *legitimately* high-coverage target (dense tissue where target+ cells truly fill
  >50%). The deterministic loop could erode real signal by raising `min_snr` on such
  a sample. Guardrails: it is **opt-in**, surfaces `correction_history`, and the
  user-review path is the escape hatch. Document the usage caveat:
  `auto_segment_target` is for sparse/punctate targets; dense confluent targets
  should use single-shot + review. (Same lesson as the NBLAST hidden-prerequisite:
  a heuristic that looks complete can hide a domain assumption.)
- QC thresholds (0.45/0.25/…) and the confidence cutoffs (75/55) are hand-tuned;
  expose loop knobs but keep defaults; flag for empirical tuning on real data.
- Vision adds per-call tokens → gated to low/medium confidence only (locked decision);
  C0.2's real "high" tier is what makes that gate effective.
- OpenAI-compat vision deferred (degrade to text). Not a regression — it has no eyes
  today either.
- Non-goal: changing Cellpose/intensity paths; changing single-shot
  `segment_target_objects` output for existing callers (additive fields only);
  batch_runner integration of `auto_segment_target` (follow-up).

## Resolved at review (2026-06-12)

- `accept_score` / `max_iters`: 3 iters; accept when `confidence_from_score` (C0.2,
  **not** `selection_confidence`) returns "high" (score ≥ 75, no critical warning).
- `auto_segment_target` exposes `boundary_mask` passthrough (mirrors
  `segment_target_objects`).
- `segment_3d_cells_auto` emits `roi_confidence` too (free — it already ranks
  candidates), so the vision gate can fire there as well (M1).
