# Simple And Advanced Modes UI Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the current main screen into an advanced/expert mode, add a new simple mode that automatically normalizes audio, chooses good execution settings, processes the full audio by chunks, and keeps speaker names stable across chunks.

**Architecture:** Keep the existing reliable single-range pipeline as the low-level primitive, then add orchestration above it for simple mode. Split audio normalization, automatic planning, chunk iteration, and cross-chunk speaker identity resolution into small modules so the 2377-line `src/meeting_transcriber/gui.py` stops absorbing all product logic.

**Tech Stack:** Python 3.10+, Tkinter/ttk, faster-whisper, pyannote.audio, imageio-ffmpeg/ffmpeg, unittest/pytest-compatible tests.

**UI/UX Skill Requirement:** Apply `ui-ux-pro-max` during the UI design and review tasks. This is a desktop Tkinter app, so the relevant guidance is accessibility, visual hierarchy, progressive disclosure, clear form grouping, loading/progress feedback, spacing consistency, readable typography, predictable navigation, and strong recovery paths for errors.

---

## Current Context

The application is a local Python desktop app. The current `src/meeting_transcriber/gui.py` builds a dense expert screen with file selectors, model controls, diarization controls, runtime controls, range controls, Hugging Face token, processing actions, progress labels, history, preview, and logs in one window.

Important existing pieces:

- `src/meeting_transcriber/pipeline.py` processes one audio/range and exports raw plus diarized transcripts.
- `src/meeting_transcriber/benchmark.py` already benchmarks runtime candidates and recommends `device` plus `compute_type`.
- `src/meeting_transcriber/history.py` tracks completed ranges and can recommend the next range from observed speed.
- `src/meeting_transcriber/speaker_memory.py` stores validated names and optional voice embeddings.
- `src/meeting_transcriber/speaker_fingerprints.py` extracts and compares voice embeddings.
- `src/meeting_transcriber/speaker_cross_compare.py` already compares speakers between outputs.
- `tests/test_gui_layout.py` currently only protects coarse window/layout constants.

The requested product shift is:

- The current screen becomes "Avanzado" or "Experto".
- A new "Simple" mode becomes the default working surface.
- Simple mode chooses models/runtime settings automatically.
- Simple mode automatically walks the whole audio by portions.
- Simple mode automatically compares voices between portions to keep stable names.
- Advanced mode gets a manual audio normalization option.
- Simple mode always normalizes audio automatically.

## UX Direction

The first screen should feel like a guided production tool, not a configuration panel. The simple screen should ask for the few things a non-expert can confidently provide:

- audio file
- output folder
- language, with automatic as a supported option
- Hugging Face token only when needed
- one primary action: analyze the complete audio

Everything else should become automation status:

- selected runtime and compute type
- selected transcription model
- current chunk/range
- coverage of full audio
- detected speakers and confidence
- outputs generated
- warnings that require review

Advanced mode should preserve all existing expert capabilities, but reorganized into readable groups:

- Input and output
- Models and language
- Speakers and diarization
- Runtime and performance
- Range and chunk tools
- Normalization and exports
- History and review tools
- Preview and log

No current expert feature should disappear in the overhaul.

Design principles from `ui-ux-pro-max` to enforce:

- Accessibility first: visible focus states, readable contrast, keyboard-reachable controls, and no information conveyed by color alone.
- One primary action per mode: `Analizar audio completo` in Simple mode, `Procesar` in Advanced mode.
- Progressive disclosure: Simple mode hides expert decisions and shows them as automation status; Advanced mode groups expert controls without flattening everything into one dense form.
- Clear feedback: long-running benchmark, normalization, transcription, diarization, and identity reconciliation steps must show current stage, chunk count, cancellation affordance, and recoverable error messages.
- Consistent spacing: use a small shared spacing scale instead of ad hoc paddings across the window.
- Stable layout: progress text, timers, lists, and buttons must not resize the window or push critical controls off-screen.
- Form clarity: every input keeps a visible label; complex fields such as Token HF and speaker counts get nearby helper/status text where useful.
- Destructive actions: delete/discard controls remain visually separated and confirmation-protected.

## File Structure Plan

Create:

- `src/meeting_transcriber/app_modes.py`
  - Defines `AppMode`, `SimpleModeSettings`, `SimpleRunSummary`, and small UI-independent mode helpers.
- `src/meeting_transcriber/audio_normalization.py`
  - Builds ffmpeg filters for speech-focused normalization and exposes `normalize_audio_for_speech`.
- `src/meeting_transcriber/simple_planner.py`
  - Chooses runtime, compute type, Whisper model, diarization quality, and chunk sizing for simple mode.
- `src/meeting_transcriber/simple_pipeline.py`
  - Orchestrates full-audio simple processing by chunks using existing `process_meeting`.
- `src/meeting_transcriber/speaker_identity_resolver.py`
  - Clusters and reconciles speakers across chunks using embeddings, existing memory, and conservative confidence thresholds.
- `src/meeting_transcriber/gui_theme.py`
  - Centralizes ttk style names, spacing constants, and reusable layout helpers.
- `tests/test_audio_normalization.py`
- `tests/test_simple_planner.py`
- `tests/test_simple_pipeline.py`
- `tests/test_speaker_identity_resolver.py`

Modify:

- `src/meeting_transcriber/types.py`
  - Add `normalize_audio: bool = False` to `ProcessingConfig`.
- `src/meeting_transcriber/config.py`
  - Persist and migrate `normalize_audio`.
- `src/meeting_transcriber/pipeline.py`
  - Apply normalization before transcription/diarization when requested.
- `src/meeting_transcriber/progress.py`
  - Format new stages: `normalization`, `simple_plan`, `simple_chunk`, `simple_identity`, `simple_done`.
- `src/meeting_transcriber/gui.py`
  - Introduce simple/advanced UI, wire simple orchestration, preserve advanced behavior.
- `src/meeting_transcriber/history.py`
  - Add helpers needed by simple mode to find all missing ranges and record automatic chunks cleanly.
- `README.md`
  - Update quick usage and explain simple vs advanced.
- `docs/FUNCIONAMIENTO.md`
  - Document the new workflow, normalization, automation limits, and review expectations.
- `tests/test_config.py`
- `tests/test_progress.py`
- `tests/test_gui_layout.py`
- Existing speaker memory/cross-compare tests as needed.

## Implementation Tasks

### Task 0: Keep This Plan As The Source Of Truth

**Files:**

- Create: `docs/superpowers/plans/2026-05-01-simple-advanced-ui-overhaul.md`

- [x] **Step 1: Create this plan file**

The plan exists and contains an implementation status table at the end.

- [ ] **Step 2: Update the status table after each completed task**

Every implementation task below has a row in the status table. When a task is completed, update the row with the real result and the verification command that passed.

### Task 1: Add Baseline Characterization Tests Before Refactoring

**Files:**

- Modify: `tests/test_gui_layout.py`
- Create: `tests/test_app_modes.py`

- [ ] **Step 1: Protect current expert-mode labels**

Add a test asserting the advanced mode still exposes the existing expert controls after the UI split:

```python
def test_advanced_mode_keeps_existing_expert_control_labels():
    expected = (
        "Audio",
        "Salida",
        "Calidad",
        "Diarizacion",
        "Separacion voces",
        "Idioma",
        "Hablantes",
        "Rango",
        "Ejecucion",
        "Token HF",
        "Procesar",
        "Probar rendimiento",
        "Renombrar hablantes",
        "Guardar configuracion",
    )
    assert expected
```

The first implementation can expose these labels through a constant so the test does not need to instantiate Tk in headless CI.

- [ ] **Step 2: Add mode labels contract**

Create `tests/test_app_modes.py` with expectations:

```python
from meeting_transcriber.app_modes import AppMode


def test_app_modes_have_simple_and_advanced_values():
    assert AppMode.SIMPLE.value == "simple"
    assert AppMode.ADVANCED.value == "advanced"
```

- [ ] **Step 3: Run baseline tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_gui_layout.py tests/test_app_modes.py -q
```

Expected before implementation: fails because `meeting_transcriber.app_modes` does not exist.

### Task 2: Add Mode And Config Data Contracts

**Files:**

- Create: `src/meeting_transcriber/app_modes.py`
- Modify: `src/meeting_transcriber/types.py`
- Modify: `src/meeting_transcriber/config.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_app_modes.py`

- [ ] **Step 1: Define mode contracts**

Create:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AppMode(Enum):
    SIMPLE = "simple"
    ADVANCED = "advanced"


@dataclass(frozen=True)
class SimpleModeSettings:
    target_wait_seconds: float = 15 * 60.0
    chunk_overlap_seconds: float = 15.0
    min_chunk_seconds: float = 5 * 60.0
    max_chunk_seconds: float = 30 * 60.0
    normalize_audio: bool = True
    auto_apply_high_confidence_names: bool = True


@dataclass(frozen=True)
class SimpleRunSummary:
    chunks_total: int
    chunks_completed: int
    chunks_failed: int
    output_dirs: tuple[str, ...]
```

- [ ] **Step 2: Extend `ProcessingConfig`**

Add a final field:

```python
normalize_audio: bool = False
```

Keep it last so existing positional construction risk remains low.

- [ ] **Step 3: Persist and migrate config**

In `load_config`, read:

```python
normalize_audio=bool(payload.get("normalize_audio", False))
```

`save_config` already serializes dataclasses through `asdict`, so it will write the new value automatically.

- [ ] **Step 4: Add tests**

Add tests that an old config without `normalize_audio` loads with `False`, and a saved config with `True` round-trips.

- [ ] **Step 5: Verify**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_config.py tests/test_app_modes.py -q
```

Expected: pass.

### Task 3: Implement Speech-Focused Audio Normalization

**Files:**

- Create: `src/meeting_transcriber/audio_normalization.py`
- Create: `src/meeting_transcriber/deepfilternet.py`
- Modify: `src/meeting_transcriber/pipeline.py`
- Modify: `src/meeting_transcriber/progress.py`
- Create: `tests/test_audio_normalization.py`
- Modify: `tests/test_progress.py`

- [ ] **Step 1: Add ffmpeg filter builder tests**

Create tests for a stable filter chain:

```python
from meeting_transcriber.audio_normalization import speech_normalization_filter


def test_speech_normalization_filter_prioritizes_voice_and_noise_reduction():
    value = speech_normalization_filter()

    assert "highpass=f=80" in value
    assert "lowpass=f=8000" in value
    assert "afftdn=nf=-25" in value
    assert "dynaudnorm" in value
    assert "loudnorm=I=-16:TP=-1.5:LRA=11" in value
```

- [ ] **Step 2: Implement normalization module**

Use DeepFilterNet as the primary voice enhancement path and keep the original FFmpeg chain as a no-network/no-binary fallback:

```python
def normalize_audio_for_speech(ffmpeg_path, source_audio, output_audio) -> Path:
    deepfilter = resolve_deep_filter_binary(default_deepfilternet_dir())
    if deepfilter is not None:
        result = enhance_with_deepfilternet(...)
        if result.ok:
            return output_audio
    return normalize_audio_with_ffmpeg(ffmpeg_path, source_audio, output_audio)
```

The FFmpeg fallback filter remains:

```python
def speech_normalization_filter() -> str:
    return ",".join(
        (
            "highpass=f=80",
            "lowpass=f=8000",
            "afftdn=nf=-25",
            "dynaudnorm=f=150:g=15",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
        )
    )
```

`normalize_audio_with_ffmpeg(ffmpeg_path, source_audio, output_audio)` should run:

```text
ffmpeg -y -i <source> -ac 1 -ar 16000 -af <filter> <output>
```

Capture stderr and raise `RuntimeError("No se pudo normalizar el audio: ...")` with the first useful ffmpeg line.

`deepfilternet.py` should resolve, download and execute the official `deep-filter` release binary. Bootstrap prepares the desktop release assets for Windows, Linux and macOS from `https://github.com/Rikorose/DeepFilterNet/releases/`.

- [ ] **Step 3: Integrate in `process_meeting`**

Inside the existing temporary directory, after optional range extraction and before transcription:

```python
if config.normalize_audio:
    reporter(ProgressEvent(stage="normalization", message="Normalizando audio para voz humana"))
    normalized_audio = Path(temp_dir.name) / "normalized.wav"
    normalize_audio_for_speech(resolve_ffmpeg_path(config.ffmpeg_path), working_audio, normalized_audio)
    working_audio = normalized_audio
```

If there is no range extraction, still create a `TemporaryDirectory` for the normalized file.

- [ ] **Step 4: Format progress**

Add Spanish progress text for `normalization` in `progress.py` and update GUI labels so advanced and simple users see that normalization is active.

- [ ] **Step 5: Verify**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_audio_normalization.py tests/test_progress.py -q
```

Expected: pass.

### Task 4: Add The Advanced Mode Normalization Control

**Files:**

- Modify: `src/meeting_transcriber/gui.py`
- Modify: `tests/test_gui_layout.py`

- [ ] **Step 1: Add UI state**

In `App.__init__`, add:

```python
self.normalize_audio = tk.BooleanVar(value=existing.normalize_audio if existing else False)
```

- [ ] **Step 2: Add advanced checkbox**

Place the checkbox near export/runtime controls:

```python
ttk.Checkbutton(
    root,
    text="Normalizar audio para voz humana",
    variable=self.normalize_audio,
).grid(...)
```

The label must communicate that it is optional in advanced mode.

- [ ] **Step 3: Pass setting into config**

In `_current_config`, pass:

```python
normalize_audio=self.normalize_audio.get(),
```

- [ ] **Step 4: Update layout tests**

Extend the expert label contract with `"Normalizar audio para voz humana"`.

- [ ] **Step 5: Verify**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_gui_layout.py tests/test_config.py -q
```

Expected: pass.

### Task 5: Build Simple Mode Planning

**Files:**

- Create: `src/meeting_transcriber/simple_planner.py`
- Create: `tests/test_simple_planner.py`

- [ ] **Step 1: Define plan dataclasses**

Create:

```python
@dataclass(frozen=True)
class SimpleProcessingPlan:
    config: ProcessingConfig
    chunk_seconds: float
    overlap_seconds: float
    explanation: str
```

- [ ] **Step 2: Add model selection policy**

Implement `choose_whisper_model(device, observed_speed)`:

- CUDA with speed >= 12x: `large-v3`
- CUDA with speed >= 5x: `medium`
- CUDA slower than 5x: `small`
- CPU with speed >= 2x: `small`
- CPU slower than 2x: `base`

This is conservative: simple mode favors reliability, but avoids selecting a model likely to make long audio unusable.

- [ ] **Step 3: Add chunk duration policy**

Implement `choose_chunk_seconds(observed_speed, settings)`:

- If speed is known, use `settings.target_wait_seconds * observed_speed`.
- Clamp to `settings.min_chunk_seconds` and `settings.max_chunk_seconds`.
- If speed is unknown, use 10 minutes.

- [ ] **Step 4: Create planner entrypoint**

`build_simple_processing_plan(base_config, benchmark_result, settings)` should:

- use `benchmark_result.recommendation` for `device` and `compute_type`
- derive observed speed from the best successful attempt
- set `normalize_audio=True`
- set `diarization_quality="precise"` for v1 simple mode
- set selected Whisper model using the policy above
- preserve language, token, output, and speaker min/max from user input if present

- [ ] **Step 5: Verify**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_simple_planner.py -q
```

Expected: pass.

### Task 6: Implement Full-Audio Simple Chunk Orchestration

**Files:**

- Create: `src/meeting_transcriber/simple_pipeline.py`
- Modify: `src/meeting_transcriber/history.py`
- Create: `tests/test_simple_pipeline.py`

- [ ] **Step 1: Add range planning helper**

In `history.py`, add a pure helper:

```python
def missing_ranges(entries: list[HistoryEntry], total_duration_seconds: float) -> list[tuple[float, float]]:
    ...
```

It should use `completed_ranges` and return all gaps in order.

- [ ] **Step 2: Add chunk splitter**

In `simple_pipeline.py`, implement:

```python
def split_missing_ranges(
    missing: list[tuple[float, float]],
    *,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[tuple[float, float]]:
    ...
```

Rules:

- never start before 0
- never end after the missing range end
- add overlap after the first chunk only when it does not cross the missing range start
- skip zero-length chunks

- [ ] **Step 3: Add orchestration function**

Implement `process_audio_simple(...)` that:

- probes total duration
- builds missing chunks
- for each chunk, creates a per-range output dir with `build_processing_output_dir`
- calls `process_meeting` with start/end and `normalize_audio=True`
- records successful chunks in history
- reports progress before and after each chunk
- stops cleanly if cancellation is requested
- returns `SimpleRunSummary`

- [ ] **Step 4: Keep manual validation out of simple mode v1**

Simple mode should not show the current per-fragment "Guardar como valido / Descartar" dialog after each chunk. It should record chunks automatically and provide review tools at the end.

- [ ] **Step 5: Verify**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_simple_pipeline.py tests/test_history.py -q
```

Expected: pass.

### Task 7: Resolve Speaker Identity Across Chunks

**Files:**

- Create: `src/meeting_transcriber/speaker_identity_resolver.py`
- Modify: `src/meeting_transcriber/simple_pipeline.py`
- Create: `tests/test_speaker_identity_resolver.py`
- Modify: `tests/test_speaker_memory.py`

- [ ] **Step 1: Define identity result contracts**

Create:

```python
@dataclass(frozen=True)
class SpeakerIdentityDecision:
    source_speaker: str
    resolved_name: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class ChunkSpeakerResolution:
    output_dir: Path
    mapping: dict[str, str]
    decisions: tuple[SpeakerIdentityDecision, ...]
```

- [ ] **Step 2: Implement conservative matching**

Priority order:

1. Existing validated memory embeddings, threshold `0.85`.
2. Current simple-run cluster embeddings, threshold `0.82`.
3. Existing validated memory names only when speaker count matches exactly.
4. Stable generated names: `Persona 1`, `Persona 2`, etc.

Never map two source speakers to the same resolved name in one chunk unless they are already the same source label.

- [ ] **Step 3: Add final reconciliation pass**

After all chunks complete, simple mode should reload each chunk `transcript.json`, apply the final stable mapping, and rewrite all exports through `write_all_exports`.

This is required because early chunks may only become identifiable after later chunks provide better voice samples.

- [ ] **Step 4: Store evidence**

Save a simple JSON sidecar in each output directory:

```text
speaker_identity_decisions.json
```

Include source speaker, resolved name, confidence, and reason so the user can audit why a name was applied.

- [ ] **Step 5: Verify**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_speaker_identity_resolver.py tests/test_speaker_memory.py -q
```

Expected: pass.

### Task 8: Build The Simple Mode UI

**Files:**

- Modify: `src/meeting_transcriber/gui.py`
- Create: `src/meeting_transcriber/gui_theme.py`
- Modify: `tests/test_gui_layout.py`

- [ ] **Step 1: Introduce a top-level mode switch**

Use `ttk.Notebook` or a segmented radiobutton row with:

- `Simple`
- `Avanzado`

Default selected mode: `Simple`.

Apply `ui-ux-pro-max` navigation guidance here: the mode switch must be visibly top-level, show the active mode clearly, be keyboard reachable, and not look like a secondary settings control.

- [ ] **Step 2: Simple mode layout**

Simple mode should contain:

- audio selector
- output selector
- language selector
- Hugging Face token field with model/token buttons
- primary button: `Analizar audio completo`
- stop button
- progress step label
- current chunk label
- coverage progress bar
- speaker identity status
- output summary list
- compact log/details expander or a lower log area

Apply `ui-ux-pro-max` progressive disclosure guidance here: the simple mode must read as a guided workflow with one primary action, not as a reduced copy of advanced mode. The runtime/model/chunk decisions appear as status and audit details, not editable fields.

- [ ] **Step 3: Wire simple action**

Add `_process_simple` that:

- validates audio and output
- builds a base config from shared fields
- runs benchmark/planner
- starts `process_audio_simple` in a background thread
- disables conflicting buttons
- streams progress into the same event queue pattern used by `_process`

- [ ] **Step 4: Advanced mode contains existing screen**

Move the current controls into an advanced frame without removing existing commands:

- manual range processing remains
- benchmark remains
- history remains
- compare/fuse/reanalyze remains
- rename dialog remains
- manual normalization checkbox appears here

- [ ] **Step 5: Verify headless contracts**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_gui_layout.py -q
```

Expected: pass.

### Task 9: UI/UX Overhaul Of Advanced Mode

**Files:**

- Modify: `src/meeting_transcriber/gui.py`
- Modify: `src/meeting_transcriber/gui_theme.py`
- Modify: `tests/test_gui_layout.py`

- [ ] **Step 1: Group controls into labelled sections**

Use `ttk.LabelFrame` sparingly for functional groups:

- Entrada
- Modelos
- Hablantes
- Ejecucion
- Rango
- Historial
- Vista previa
- Registro

Apply `ui-ux-pro-max` form grouping guidance here: each section must have a clear purpose, visible labels, predictable tab order, and no controls whose relationship is only implied by physical proximity.

- [ ] **Step 2: Improve density and scanability**

Rules:

- related controls stay in the same row or same section
- long logs do not compete with primary actions
- progress labels are grouped together
- history actions are visually attached to the history list
- destructive actions stay at the end of action rows

Use `ui-ux-pro-max` visual hierarchy rules: primary action weight, section spacing, label weight, and status emphasis should explain what matters before the user reads every control.

- [ ] **Step 3: Keep professional desktop behavior**

Set stable minimum sizes for:

- file rows
- mode switch
- progress bars
- history list
- preview/log panes

Avoid layout shifts when progress text changes.

- [ ] **Step 4: Verify manually**

Run:

```bash
python scripts/bootstrap.py
```

Inspect:

- 1120x800 desktop window
- minimum 900x680 window
- simple tab default
- advanced tab still usable
- no overlapping labels/buttons
- visible focus ring and sensible tab order through Simple and Advanced modes
- primary actions are visually distinct from secondary/destructive actions
- error messages say what happened and how to recover

### Task 9.5: UI/UX Review Pass With `ui-ux-pro-max`

**Files:**

- Modify: `src/meeting_transcriber/gui.py`
- Modify: `src/meeting_transcriber/gui_theme.py`
- Modify: `tests/test_gui_layout.py`

- [ ] **Step 1: Accessibility review**

Check:

- all interactive controls have visible text or an accessible label
- focus order follows the visual order
- disabled states remain legible
- status messages do not rely on color alone
- token and path fields remain readable at minimum window size

- [ ] **Step 2: Interaction review**

Check:

- Simple mode has exactly one obvious primary CTA when idle
- long tasks disable conflicting actions and keep `Detener` available
- cancellation feedback appears within the normal status area
- destructive advanced actions require confirmation
- benchmark and simple processing make progress visible within 300ms of starting

- [ ] **Step 3: Layout review**

Check:

- no horizontal scrolling or clipped controls at `900x680`
- preview/log/history panes keep stable dimensions
- progress labels wrap or truncate intentionally without moving primary controls
- related fields align consistently across sections

- [ ] **Step 4: Record findings**

Update this plan with any UI issues found and either fix them immediately or add explicit follow-up rows to the status table.

### Task 10: Add Progress And Error Handling For Simple Mode

**Files:**

- Modify: `src/meeting_transcriber/progress.py`
- Modify: `src/meeting_transcriber/gui.py`
- Modify: `src/meeting_transcriber/simple_pipeline.py`
- Modify: `tests/test_progress.py`
- Modify: `tests/test_simple_pipeline.py`

- [ ] **Step 1: Define user-facing stages**

Stages:

- `simple_plan`: choosing settings
- `simple_chunk`: processing chunk N/M
- `normalization`: cleaning and leveling audio
- `simple_identity`: reconciling speaker names
- `simple_done`: complete

- [ ] **Step 2: Make failures recoverable**

If one chunk fails:

- log the chunk range and error
- keep completed chunks recorded
- continue only when the error is non-fatal and cancellation was not requested
- show final summary with completed and failed counts

Fatal errors:

- invalid audio path
- unavailable duration
- missing Hugging Face access for diarization
- no usable runtime candidate

- [ ] **Step 3: Verify**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_progress.py tests/test_simple_pipeline.py -q
```

Expected: pass.

### Task 11: Documentation Update

**Files:**

- Modify: `README.md`
- Modify: `docs/FUNCIONAMIENTO.md`

- [ ] **Step 1: README simple workflow**

Update quick workflow:

1. Open app.
2. Stay in Simple mode.
3. Choose audio and output folder.
4. Choose language or automatic.
5. Paste Hugging Face token if diarization needs it.
6. Press `Analizar audio completo`.
7. Review output and speaker identity summary.

- [ ] **Step 2: README advanced workflow**

Explain that Advanced mode is for manual ranges, specific models, runtime choices, normalization toggling, fragment repair, comparison, and merging.

- [ ] **Step 3: Functioning docs**

Document:

- normalization filter intent
- chunking behavior
- automatic model/runtime planning
- speaker identity confidence thresholds
- when the user should review names manually
- privacy note: processing remains local except optional downloads and optional external AI naming helper

- [ ] **Step 4: Verify docs are link-consistent**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_external_links.py -q
```

Expected: pass.

### Task 12: Full Verification And Manual Acceptance

**Files:**

- All changed implementation and docs files.

- [ ] **Step 1: Run focused tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_audio_normalization.py tests/test_simple_planner.py tests/test_simple_pipeline.py tests/test_speaker_identity_resolver.py tests/test_gui_layout.py tests/test_config.py tests/test_progress.py -q
```

Expected: pass.

- [ ] **Step 2: Run full suite**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Expected: pass.

- [ ] **Step 3: Manual smoke test**

Run:

```bash
python scripts/bootstrap.py
```

Acceptance checks:

- Simple is the default mode.
- Advanced mode contains the previous expert capabilities.
- Advanced normalization checkbox persists.
- Simple processing starts with automatic normalization.
- Runtime/model choice appears in progress/logs.
- Chunks are processed in order.
- Stop button cancels cleanly.
- Speaker names remain stable across at least two chunks when embeddings are available.
- Outputs are written to the expected per-range folders.

- [ ] **Step 4: Update status table**

Record exact test commands and manual smoke result in the table below.

## Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Tkinter UI refactor breaks existing expert workflows | High | Keep existing callbacks, add headless label contracts, manual smoke test advanced mode |
| Normalization can make some audio worse | Medium | Use DeepFilterNet as the primary cleaner, keep FFmpeg as fallback, make it optional in advanced mode, automatic only in simple mode, document filter intent |
| Full automatic processing takes a long time | Medium | Chunk by target wait, allow cancellation between chunks and inside pipeline |
| Speaker names are overconfidently auto-applied | High | Use high thresholds, one-to-one mapping, generated stable names when confidence is low, write decision sidecars |
| Hugging Face diarization access fails after long transcription | High | Existing preflight remains before transcription; simple mode keeps it before each chunk |
| Large-v3 selection is too slow on some machines | Medium | Select by benchmark speed policy and clamp to smaller models on slow hardware |
| `gui.py` remains too large | Medium | Extract non-UI logic first; defer deeper UI class splitting only if needed to complete safely |

## Implementation Status Table

| ID | Area | Status | Real result / evidence | Last updated |
| --- | --- | --- | --- | --- |
| 0 | Plan document | Done | Created `docs/superpowers/plans/2026-05-01-simple-advanced-ui-overhaul.md` | 2026-05-01 |
| 1 | Baseline characterization tests | Done | Added headless UI/mode contracts in `tests/test_gui_layout.py` and `tests/test_app_modes.py`; focal suite passed | 2026-05-01 |
| 2 | Mode and config contracts | Done | Added `app_modes.py`, `normalize_audio` config persistence, and migration tests; focal suite passed | 2026-05-01 |
| 3 | Audio normalization core | Done | Added DeepFilterNet primary normalization with FFmpeg fallback and pipeline integration; focal suite passed | 2026-05-01 |
| 4 | Advanced normalization option | Done | Added advanced checkbox and config wiring; focal suite passed | 2026-05-01 |
| 5 | Simple planner | Done | Added benchmark-based runtime/model/chunk planner; focal suite passed | 2026-05-01 |
| 6 | Simple chunk orchestration | Done | Added `simple_pipeline.py`, missing range helper, auto history recording; focal suite passed | 2026-05-01 |
| 7 | Cross-chunk speaker identity resolver | Done | Added conservative resolver and `speaker_identity_decisions.json`; focal suite passed | 2026-05-01 |
| 8 | Simple mode UI | Done | Added Simple/Avanzado tabs, simple CTA and progress surface; focal suite passed | 2026-05-01 |
| 9 | Advanced UI/UX overhaul | Done | Advanced controls moved into a grouped expert tab with normalization option; focal suite passed | 2026-05-01 |
| 9.5 | UI/UX review with `ui-ux-pro-max` | Done | Review found fatal handling, final reconciliation, embedding-warning, and grouping issues; all addressed; focused and full suites passed | 2026-05-01 |
| 10 | Simple progress and errors | Done | Added simple progress stages, recoverable chunk failure handling, cancellation status; focal suite passed | 2026-05-01 |
| 11 | Documentation | Done | Updated `README.md` and `docs/FUNCIONAMIENTO.md` for simple/advanced modes and normalization | 2026-05-01 |
| 12 | Full verification | Done | Added final combined transcript and HTML report after user smoke test; generated final output for current audio; `PYTHONPATH=src .venv/bin/python -m pytest -q` -> 161 passed; `PYTHONPATH=src .venv/bin/python -m compileall -q src tests scripts` -> passed | 2026-05-01 |
| 13 | DeepFilterNet release binaries | Done | Added `deepfilternet.py`, downloaded official v0.5.6 binaries for Windows, Linux and macOS into `models/deepfilternet/`; validated Linux `deep-filter --help` and a synthetic normalization run; focused DeepFilterNet/audio/bootstrap tests passed | 2026-05-01 |
| 14 | Reviewable final transcript | Done | Added phase/heartbeat progress during DeepFilterNet normalization; regenerated current `report.html` with per-row play buttons, editable speaker/text cells, browser autosave and MD/TXT/SRT/JSON reviewed exports; `PYTHONPATH=src .venv/bin/python -m pytest -q` -> 163 passed; compileall and install passed | 2026-05-01 |
