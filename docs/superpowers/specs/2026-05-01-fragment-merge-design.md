# Fragment Merge Review Design

## Goal

Add a history action for duplicate or overlapping fragment results that opens a side-by-side merge review. The user can choose the best transcription and speaker identification phrase by phrase, then accept a merged result that replaces the originals in the visible history.

## Chosen Approach

Use option A: an editable phrase-by-phrase reconciliation window.

The first version will focus on comparing two history entries at a time. This matches the current duplicate case created by reanalysis (`00-00-00_to_00-05-00` and `00-00-00_to_00-05-00_2`) and keeps the interface understandable. Multi-result merging can be added later by repeating pairwise merges.

## User Flow

1. The user selects two compatible history entries.
2. The user clicks `Fusionar resultados`.
3. The app loads both `transcript.json` files and aligns turns by time overlap.
4. A merge window shows each aligned row with:
   - left result text and speaker
   - right result text and speaker
   - chosen final speaker
   - chosen final text
   - controls to choose left, choose right, or edit manually
5. The user accepts the fusion.
6. The app writes a new merged output folder.
7. The visible history hides the two original entries and shows only the merged entry.

## History Semantics

Accepted fusions should not delete original output folders by default. Instead, originals become hidden/superseded entries. This keeps the UI clean while preserving rollback/debug evidence on disk.

History needs enough metadata to represent this:

- an entry id
- visible/hidden status
- optional superseded-by id
- optional merge source ids for the merged entry

Existing history files should continue to load. Entries without ids can receive stable ids when rewritten.

## Output Semantics

The merged result gets its own output directory under the same audio/range family, for example:

```text
output/Taula_Institucional_18_03_26/00-00-00_to_00-05-00_merged/
```

The app writes the usual exports:

- `transcript.md`
- `transcript.txt`
- `transcript.json`
- `transcript.srt`

The merged transcript should also be remembered as validated speaker memory once accepted, using the same memory flow as a normal validated result.

## Alignment Rules

The first implementation should use deterministic time-based alignment:

- pair turns when their time ranges overlap
- preserve unmatched left or right turns as rows with an empty opposite side
- sort rows by earliest start time
- default selection chooses the side with more text, unless one side has a non-generic remembered speaker and the other does not

This is predictable and testable. Text-similarity alignment can be added later if time alignment proves insufficient.

## Merge Window

The window should be usable in Tkinter without becoming a heavy editor:

- a scrollable table/list of aligned rows
- two read-only source columns
- one editable final speaker field
- one editable final text field
- small buttons per row: `Izquierda`, `Derecha`
- top summary: source folders, range, row count, speaker changes
- bottom actions: `Guardar fusion`, `Cancelar`

Speaker fields should use known speaker names as combobox suggestions when available.

## Validation And Errors

The app should prevent invalid merges:

- fewer than two selected history entries
- missing `transcript.json`
- different audio files
- ranges with no overlap

If a merge succeeds but history rewrite fails, the output folder should remain on disk and the user should see a clear error.

## Tests

Add focused tests for:

- loading old history entries without merge metadata
- hiding superseded entries from visible history
- aligning turns by time overlap
- choosing default row content
- writing a merged transcript and recording source ids

GUI behavior can stay thin and delegate most logic to pure modules.
