# Fragment Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a side-by-side merge review for two duplicate history entries and replace them with one visible merged result.

**Architecture:** Keep merge logic in pure modules and keep Tkinter as a thin orchestration layer. Extend history entries with backward-compatible metadata so accepted merges hide source entries without deleting output folders.

**Tech Stack:** Python dataclasses, unittest, Tkinter/ttk, existing transcript exporters and speaker memory.

---

### Task 1: Backward-Compatible History Metadata

**Files:**
- Modify: `src/meeting_transcriber/history.py`
- Modify: `tests/test_history.py`

- [x] Write failing tests for generated ids, visible filtering, and superseding two entries.
- [x] Implement `HistoryEntry.id`, `hidden`, `superseded_by`, `merge_source_ids`.
- [x] Implement `visible_entries_for` and `supersede_history_entries`.
- [x] Keep old JSON history files loadable.
- [x] Run `PYTHONPATH=src python -m unittest tests.test_history`.

### Task 2: Pure Transcript Merge Logic

**Files:**
- Create: `src/meeting_transcriber/transcript_merge.py`
- Create: `tests/test_transcript_merge.py`

- [x] Write failing tests for time-overlap alignment, unmatched rows, and default choice.
- [x] Implement `MergeRow`, `DraftMergeRow`, `align_turns_for_merge`, and `merged_turns_from_drafts`.
- [x] Run `PYTHONPATH=src python -m unittest tests.test_transcript_merge`.

### Task 3: Merge Output And History Acceptance

**Files:**
- Modify: `src/meeting_transcriber/exporters.py`
- Modify: `src/meeting_transcriber/history.py`
- Modify: `tests/test_exporters.py`
- Modify: `tests/test_history.py`

- [x] Write failing tests for `_merged` output folder suffixing and merged history creation.
- [x] Implement `build_merged_output_dir`.
- [x] Implement history write path that adds merged entry and hides sources.
- [x] Run `PYTHONPATH=src python -m unittest tests.test_exporters tests.test_history`.

### Task 4: Tkinter Merge Review UI

**Files:**
- Modify: `src/meeting_transcriber/gui.py`

- [x] Add multi-select support in the history listbox.
- [x] Add `Fusionar resultados` button enabled when exactly two visible entries are selected.
- [x] Load both transcripts and validate overlapping ranges.
- [x] Add `MergeReviewDialog` with left/right source columns and editable final speaker/text.
- [x] On save, write merged exports, update history, remember speaker memory, and refresh UI.

### Task 5: Documentation And Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/FUNCIONAMIENTO.md`

- [x] Document `Fusionar resultados`.
- [x] Run `PYTHONPATH=src python -m unittest discover -s tests`.
- [x] Run a syntax check with `PYTHONPATH=src python -m py_compile src/meeting_transcriber/*.py`.
