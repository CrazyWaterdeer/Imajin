# Phase 8 Spec: Distribution, Environment, and User Onboarding

## Goal

Make Imajin installable, diagnosable, and usable by people beyond the original
developer.

The app depends on a heavy scientific/GUI stack:

- napari / Qt
- tifffile / bioio / CZI readers
- Cellpose / torch / CUDA
- btrack
- skimage / pandas / scipy
- optional LLM providers
- local display/GPU environments

Phase 8 should make setup failures understandable and give new users a clear
path from installation to a first successful analysis.

## Non-Goals

- Do not build a commercial installer unless packaging needs are clearer.
- Do not guarantee every OS/GPU combination.
- Do not bundle large proprietary sample data.
- Do not require LLM access for basic manual workflows.
- Do not make CUDA mandatory.

## Supported User Profiles

### Developer / Power User

- uses git and uv
- comfortable with terminal
- may run from source

### Lab User

- wants to open microscopy files and run analysis
- may not know Python packaging
- needs clear install and troubleshooting instructions

### Offline / No-LLM User

- uses manual dock only
- no API keys
- still needs file loading, segmentation, measurement, reports

### CPU-Only User

- no CUDA GPU
- Cellpose works but slower
- app should warn, not fail

## Platform Support Policy

Document support levels explicitly.

Recommended:

- Linux: primary supported development target
- Windows + WSLg: supported with caveats
- Windows native: possible but needs validation
- macOS: possible for CPU/manual workflows, CUDA unavailable

Do not imply full support until tested.

## Installation Paths

### From Source

Primary initial install path:

```bash
git clone <repo>
cd Imajin
uv sync
uv run imajin --doctor
uv run imajin
```

### CPU-Only Install

Consider a CPU-only dependency path if CUDA wheels cause issues.

Possible approaches:

- optional dependency group
- separate install instructions
- CPU torch index override

Do not block Phase 8 on perfect packaging, but document the current path.

### GPU/CUDA Install

Document:

- expected NVIDIA driver
- CUDA compatibility
- PyTorch wheel source
- how to check GPU availability

The app should degrade to CPU if CUDA is unavailable.

## Doctor Command

Expand `imajin --doctor`.

Current doctor checks imports, CUDA, display, providers. Improve it into a
structured environment diagnostic.

### Required Checks

- Python version
- package imports and versions
- napari import
- Qt backend
- OpenGL renderer
- WSL detection
- CUDA availability
- torch version
- Cellpose import
- Cellpose model availability/cache status
- tifffile LSM support
- bioio CZI support
- write access to config/data dirs
- API key/provider status
- local Ollama status if configured

### Output Levels

Use:

- OK
- WARN
- FAIL

Doctor should end with:

- overall status
- actionable next steps

Example:

```text
[WARN] CUDA not available
       Cellpose-SAM will run on CPU and may be slow.

[FAIL] Qt/OpenGL renderer unavailable
       napari may not display images. On WSLg, check ...
```

### Machine-Readable Mode

Add optional:

```bash
imajin --doctor --json
```

Useful for bug reports.

## First-Run Setup

Add a first-run checklist or setup dialog.

Minimum:

- verify writable config/data dirs
- show provider setup status
- allow manual/no-LLM mode
- show CUDA status
- show model/cache status
- link to sample/demo workflow

The app should be usable without entering API keys.

## Demo / Sample Workflow

Provide a small demo workflow.

Requirements:

- small file size
- legally distributable or generated synthetic data
- includes at least:
  - multichannel z-stack or synthetic z-stack
  - labels/segmentation path
  - measurement table
  - report generation

Possible approach:

- generate synthetic demo data programmatically
- avoid storing large binary files in repo

Command:

```bash
uv run imajin --demo
```

or UI button:

```text
Help -> Load Demo
```

Demo should not require LLM API keys.

## Model Download and Cache

Cellpose-SAM model download/cache should be explicit.

Requirements:

- show whether model is cached
- handle download failure gracefully
- do not download during import
- warn user before large download if possible
- document cache location

Doctor should check model state if practical.

## LLM Provider Onboarding

LLM should be optional.

Provider setup UI should explain:

- Anthropic key
- OpenAI-compatible endpoint
- Ollama/local endpoint
- no-LLM manual mode

Do not require API keys at launch.

Add provider diagnostics:

- key present
- endpoint reachable if local
- model selection valid if practical

Avoid logging API keys.

## Troubleshooting Guide

Create or expand docs for common problems.

Topics:

- napari window does not open
- black canvas / OpenGL error
- WSLg rendering slow
- CUDA unavailable
- Cellpose model download fails
- CZI reader missing
- LSM loads slowly
- memory fallback to memmap
- API key missing
- Ollama not running
- file path issues between Windows and WSL

Each should include:

- symptom
- likely cause
- command to diagnose
- fix

## Release Checklist

Before a release:

- `uv sync` from clean environment
- `uv run imajin --doctor`
- fast test suite
- import smoke test
- open GUI smoke test
- load LSM smoke test
- load OME-TIFF smoke test
- run segmentation on synthetic/small data
- run measurement
- generate report
- manual mode without API keys
- provider mode with configured test provider if available

## Version Compatibility Matrix

Document tested versions:

- Python
- napari
- Qt/PyQt
- torch
- CUDA
- cellpose
- tifffile
- bioio/bioio-czi
- OS/display environment

This can be a simple Markdown table at first.

## Bug Report Template

Provide a template asking for:

- OS
- install method
- `imajin --doctor --json`
- file type
- error message
- whether GUI opens
- whether CUDA is available
- whether API keys are involved

Do not ask users to share private microscopy data unless necessary.

## Documentation Structure

Recommended docs:

```text
docs/
  installation.md
  quickstart.md
  troubleshooting.md
  demo.md
  providers.md
  supported_formats.md
  release_checklist.md
```

Keep README concise and link to these.

## Tests

### Doctor

- doctor returns 0 when core imports available
- doctor reports missing optional provider key as WARN, not FAIL
- doctor JSON output is valid
- doctor detects CUDA unavailable as WARN
- doctor handles OpenGL probe failure gracefully

### Demo

- demo command creates/loads synthetic data
- demo analysis runs without API keys
- demo report generated

### Config

- first-run config dirs created
- settings save/load works
- secrets are not printed or written to project files

### CLI

- `imajin --doctor`
- `imajin --doctor --json`
- `imajin --demo`
- invalid args

## Acceptance Criteria

- A new developer can install from source and run doctor.
- App can launch without API keys.
- Manual demo workflow works without LLM.
- Doctor gives actionable warnings for CUDA/OpenGL/provider issues.
- Demo data/workflow validates file loading, segmentation/measurement, and
  report generation.
- Troubleshooting docs cover common environment failures.
- Release checklist exists.

## Suggested Implementation Order

1. Expand doctor checks and structured output.
2. Add `--doctor --json`.
3. Add synthetic demo dataset/workflow.
4. Implement `--demo` or Help -> Load Demo.
5. Add first-run/no-LLM onboarding messaging.
6. Improve provider diagnostics.
7. Write installation and troubleshooting docs.
8. Add release checklist and compatibility matrix.
9. Add tests.

## Open Questions

- Should the first distributed target be Linux/WSL only?
- Should CPU-only install have a separate dependency group?
- Should demo data be generated on the fly or stored as tiny files?
- Should model download be triggered by doctor or only checked?
- Should packaging eventually use PyInstaller, briefcase, conda, or stay uv-first?
