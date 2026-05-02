# Phase 4 Spec: Unified Execution and Workflow UI

## Goal

Unify how analysis work is executed and presented in the UI.

By this phase, the app will have many capabilities:

- metadata/channel handling
- target-channel cell analysis
- time-course ROI measurement
- experiment/batch workflows
- reporting
- optional advanced neural tools

Without a unified execution layer and workflow-oriented UI, the app will become
hard to use and prone to UI freezes. Phase 4 should make long-running work
safe, visible, cancellable, and consistent whether started by the manual UI or
the LLM agent.

## Non-Goals

- Do not redesign the entire napari viewer.
- Do not replace napari's layer model.
- Do not add new biological analysis algorithms in this phase.
- Do not implement a full workflow engine with branching DAGs unless the simple
  job/recipe model is insufficient.

## Current Problems To Solve

### Manual and LLM Execution Differ

The LLM path can run some tools in a worker and marshal napari operations to the
main thread. The manual dock can still call tool functions synchronously through
magicgui. This means:

- manual runs can freeze the UI
- cancellation behavior differs
- provenance driver handling differs
- errors and progress are not shown consistently

### Flat Tool List Does Not Scale

As tools grow, a flat manual tool picker becomes too large and too technical.
Users need workflow panels, not a raw registry dump.

### Long Jobs Need Visibility

Cellpose, large z-stack measurement, time-course analysis, and batch processing
can take long enough that the user needs:

- current job status
- progress if available
- cancel button
- error details
- completed result links

## Core Principle

All analysis work should pass through one execution service.

It should not matter whether the job started from:

- LLM chat
- manual workflow panel
- batch recipe runner
- future keyboard/menu action

The execution service owns:

- validation
- worker/main-thread dispatch
- progress
- cancellation
- provenance driver
- result/error capture
- job history

## Data Models

### JobStatus

```python
JobStatus = Literal[
    "queued",
    "running",
    "cancel_requested",
    "cancelled",
    "complete",
    "failed",
]
```

### Job

```python
@dataclass
class Job:
    job_id: str
    title: str
    source: Literal["manual", "llm", "batch", "system"]
    tool_name: str | None = None
    workflow_name: str | None = None
    status: JobStatus = "queued"
    progress: float | None = None
    message: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    result: Any | None = None
    error: str | None = None
    provenance_session_id: str | None = None
```

### CancellationToken

Long-running tools should receive or access a cancellation token.

Required behavior:

- cancellation should be cooperative
- tools should check token between expensive steps
- if a tool cannot cancel immediately, UI should show `cancel_requested`
- cancellation should not corrupt tables/layers

### ProgressEvent

```python
@dataclass
class ProgressEvent:
    job_id: str
    progress: float | None = None
    message: str | None = None
```

Progress may be unknown for some tools. Unknown progress should still show a
running state.

## Execution Service

Create a `ToolExecutionService` or equivalent.

Responsibilities:

1. Accept tool/workflow requests.
2. Validate inputs using tool registry schema.
3. Create a `Job`.
4. Set provenance driver.
5. Decide whether compute runs on worker or main thread.
6. Provide safe main-thread calls for napari operations.
7. Capture result/error.
8. Emit job state updates.
9. Store job history.

Suggested API:

```python
class ToolExecutionService:
    def submit_tool(
        self,
        name: str,
        kwargs: dict[str, Any],
        source: Literal["manual", "llm", "batch", "system"],
        title: str | None = None,
    ) -> Job:
        ...

    def submit_workflow(
        self,
        workflow_name: str,
        callable_: Callable[..., Any],
        kwargs: dict[str, Any],
        source: Literal["manual", "llm", "batch", "system"],
        title: str | None = None,
    ) -> Job:
        ...

    def cancel(self, job_id: str) -> None:
        ...

    def list_jobs(self) -> list[Job]:
        ...
```

## Threading Policy

### Main Thread Only

Operations that touch napari/Qt objects must run on the main thread:

- get layer object
- add image/labels/tracks layer
- update layer properties
- update Qt widgets
- update table dock model

### Worker Thread

Expensive pure computation should run off the main thread:

- Cellpose inference
- background subtraction
- denoising
- regionprops measurement
- colocalization
- skeletonization
- tracking
- batch iteration control where possible

### Snapshot Pattern

Use the existing pattern:

1. main thread: snapshot layer data and metadata
2. worker thread: compute
3. main thread: add result layer/table

This pattern should be standardized and documented in code.

## Provenance Policy

Every job should have a clear driver:

- `manual`
- `llm:<model>`
- `batch`
- `system`

If a high-level workflow calls multiple tools, provenance should either:

- record each low-level tool call, plus a workflow summary, or
- record a structured workflow record containing sub-steps

Do not lose detail needed for report generation.

## Job UI

Add a job/status dock or panel.

Minimum UI:

- current running jobs
- recently completed jobs
- status icon/text
- progress bar if progress known
- cancel button
- error details expandable
- result summary

The job dock should be useful for both manual and LLM-triggered jobs.

## Workflow-Oriented Manual UI

Replace or supplement the flat tool list with workflow panels.

Suggested panels:

### Load / Metadata

- open/register files
- show metadata summary
- show channel metadata
- show warnings

### Channels

- list channels
- show metadata-inferred color
- set role: target/counterstain/ignore/unknown
- set marker/notes

### Segment & Measure

- target channel selector
- segmentation mode 2D/3D/auto
- diameter
- run segmentation
- run measurement
- run combined target-cell analysis

### Time Course

- target movie selector
- ROI labels selector
- extract reference frame
- measure ROI intensity over time
- show/export time-course table

### Experiment / Batch

- registered files
- sample/group table
- recipe selection
- run recipe
- show batch job status

### Report

- generate methods
- generate single-sample report
- generate experiment report
- export tables

The old raw tool picker can remain as an "Advanced Tools" panel.

## LLM Integration

The LLM runner should call tools through the same execution service.

Important:

- The LLM turn may need to wait for a job result before continuing.
- For short jobs, blocking until result is acceptable.
- For long batch jobs, the LLM should start the job and report that it is
  running, then summarize when complete or when the user asks.

Suggested policy:

- normal tool calls: wait for completion and return result to LLM
- long batch workflows: return job id and status immediately, unless explicitly
  requested to wait

## Cancellation Behavior

When user clicks stop/cancel:

- current LLM streaming should stop
- current cancellable job should receive cancellation request
- UI should show cancellation requested
- final status should be complete/failed/cancelled, not stuck running

If a third-party library call cannot be interrupted, the app should:

- mark cancel requested
- ignore result if user cancelled before completion
- avoid adding unwanted layers/tables after cancellation when possible

## Error Handling

Errors should be displayed in three layers:

1. User-friendly message
2. Technical detail expandable
3. Provenance/job record

Examples:

- ambiguous channel
- no target channel
- segmentation returned zero objects
- unsupported axes
- out of memory fallback
- missing model/GPU

The app should avoid raw stack traces in normal UI unless expanded.

## Tests

### Execution Service

- submit a successful short tool
- submit a failing tool
- job status transitions correctly
- result captured
- error captured
- provenance driver set correctly

### Threading

- main-thread-only function is invoked on main thread
- worker-enabled tool runs off main thread
- napari layer add is marshalled to main thread

### Cancellation

- queued job can be cancelled
- running cooperative job can be cancelled
- cancelled job does not add result layer/table
- LLM stop resets runner state

### Manual UI

- manual action submits job, not direct synchronous call
- result appears in job history
- error appears in job UI

### LLM Path

- LLM tool call uses execution service
- tool result returned to runner for short jobs
- long job returns job id/status

### Workflow Panels

- target channel selector lists annotated target/unknown channels
- segment & measure panel can submit combined workflow
- time-course panel can submit ROI measurement

## Acceptance Criteria

- Manual and LLM paths use the same execution service.
- Long-running tools no longer block the UI when launched manually.
- Every launched analysis appears as a job.
- Jobs can show running/complete/failed/cancelled states.
- Cancellation works for cooperative tools.
- napari/Qt object access remains main-thread safe.
- Existing fast tests pass.

## Suggested Implementation Order

1. Introduce `Job`, `JobStatus`, and `ToolExecutionService`.
2. Route LLM tool calls through the service.
3. Route manual dock calls through the service.
4. Add basic job dock/status UI.
5. Add cooperative cancellation token support.
6. Standardize snapshot/compute/add-result helper functions.
7. Add workflow panels incrementally.
8. Move flat tool picker to Advanced Tools.
9. Add tests for execution, cancellation, and UI submission.

## Open Questions

- Should job history persist across app restarts?
- Should batch jobs run serially only, or allow limited parallelism?
- Should Cellpose cancellation be best-effort only?
- Should long LLM-triggered jobs block the LLM turn or return job id
  immediately?
- How much of this phase should happen before the first workflow-panel UI?
