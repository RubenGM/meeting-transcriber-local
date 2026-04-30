# Speaker Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep speaker identities coherent across partial audio fragments and prepare the app for future voice-fingerprint matching.

**Architecture:** Add a small `speaker_memory.py` domain module that stores validated speaker identities per audio file and can apply those identities to later fragments. The first milestone uses validated transcript/name mappings and time-ranged speaker samples; the second milestone adds optional voice embedding fingerprints without changing the UI contract.

**Tech Stack:** Python dataclasses, JSON persistence in the existing config directory, Tkinter UI integration, existing `ConversationTurn` exports, optional future `pyannote.audio` embedding models.

---

### Task 1: Persist Speaker Memory

**Files:**
- Create: `src/meeting_transcriber/speaker_memory.py`
- Test: `tests/test_speaker_memory.py`

- [ ] **Step 1: Write the failing tests**

```python
import tempfile
import unittest
from pathlib import Path

from meeting_transcriber.speaker_memory import (
    SpeakerIdentity,
    SpeakerMemory,
    add_identity_samples,
    load_speaker_memory,
    save_speaker_memory,
)
from meeting_transcriber.types import ConversationTurn


class SpeakerMemoryTests(unittest.TestCase):
    def test_add_identity_samples_persists_validated_speaker_for_audio(self):
        turns = [
            ConversationTurn(start=0, end=5, speaker="Ruben", text="Hola."),
            ConversationTurn(start=10, end=20, speaker="Ruben", text="Seguimos."),
            ConversationTurn(start=20, end=22, speaker="Nuria", text="Bon dia."),
        ]

        memory = add_identity_samples(SpeakerMemory(audios={}), Path("/audio/a.m4a"), turns)

        identities = memory.identities_for(Path("/audio/a.m4a"))
        self.assertEqual([identity.name for identity in identities], ["Ruben", "Nuria"])
        self.assertEqual(identities[0].sample_ranges, ((0, 5), (10, 20)))

    def test_save_and_load_round_trips_memory(self):
        memory = SpeakerMemory(
            audios={
                "/audio/a.m4a": [
                    SpeakerIdentity(name="Ruben", sample_ranges=((0, 5),), embeddings=())
                ]
            }
        )

        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "speaker_memory.json"
            save_speaker_memory(path, memory)
            loaded = load_speaker_memory(path)

        self.assertEqual(loaded, memory)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests.test_speaker_memory`

Expected: FAIL with `ModuleNotFoundError: No module named 'meeting_transcriber.speaker_memory'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/meeting_transcriber/speaker_memory.py` with:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from meeting_transcriber.types import ConversationTurn


@dataclass(frozen=True)
class SpeakerIdentity:
    name: str
    sample_ranges: tuple[tuple[float, float], ...]
    embeddings: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class SpeakerMemory:
    audios: dict[str, list[SpeakerIdentity]]

    def identities_for(self, audio_path: Path) -> list[SpeakerIdentity]:
        return self.audios.get(str(audio_path), [])


def load_speaker_memory(path: Path) -> SpeakerMemory:
    if not path.exists():
        return SpeakerMemory(audios={})
    payload = json.loads(path.read_text(encoding="utf-8"))
    audios: dict[str, list[SpeakerIdentity]] = {}
    for audio_path, identities in payload.get("audios", {}).items():
        audios[audio_path] = [
            SpeakerIdentity(
                name=str(item["name"]),
                sample_ranges=tuple((float(start), float(end)) for start, end in item.get("sample_ranges", [])),
                embeddings=tuple(tuple(float(value) for value in embedding) for embedding in item.get("embeddings", [])),
            )
            for item in identities
        ]
    return SpeakerMemory(audios=audios)


def save_speaker_memory(path: Path, memory: SpeakerMemory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "audios": {
            audio_path: [
                {
                    "name": identity.name,
                    "sample_ranges": [[start, end] for start, end in identity.sample_ranges],
                    "embeddings": [list(embedding) for embedding in identity.embeddings],
                }
                for identity in identities
            ]
            for audio_path, identities in memory.audios.items()
        }
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def add_identity_samples(
    memory: SpeakerMemory,
    audio_path: Path,
    turns: list[ConversationTurn],
) -> SpeakerMemory:
    ranges_by_name: dict[str, list[tuple[float, float]]] = {}
    for turn in turns:
        name = turn.speaker.strip()
        if not name or name.startswith("SPEAKER_") or name.startswith("Persona "):
            continue
        if turn.end <= turn.start:
            continue
        ranges_by_name.setdefault(name, []).append((turn.start, turn.end))

    existing_by_name = {identity.name: identity for identity in memory.identities_for(audio_path)}
    updated: list[SpeakerIdentity] = []
    for name in sorted(set(existing_by_name) | set(ranges_by_name)):
        existing = existing_by_name.get(name)
        sample_ranges = tuple(ranges_by_name.get(name, []))
        if existing is not None:
            sample_ranges = tuple(dict.fromkeys((*existing.sample_ranges, *sample_ranges)))
            updated.append(SpeakerIdentity(name=name, sample_ranges=sample_ranges, embeddings=existing.embeddings))
        else:
            updated.append(SpeakerIdentity(name=name, sample_ranges=sample_ranges))

    audios = dict(memory.audios)
    audios[str(audio_path)] = updated
    return SpeakerMemory(audios=audios)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests.test_speaker_memory`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/meeting_transcriber/speaker_memory.py tests/test_speaker_memory.py docs/superpowers/plans/2026-05-01-speaker-memory.md
git commit -m "Add speaker memory persistence"
```

### Task 2: Save Validated Names Into Memory

**Files:**
- Modify: `src/meeting_transcriber/gui.py`
- Test: `tests/test_speaker_memory.py`

- [ ] **Step 1: Write the failing test**

Add:

```python
from meeting_transcriber.speaker_memory import remember_validated_turns


def test_remember_validated_turns_saves_memory_file(self):
    turns = [ConversationTurn(start=0, end=5, speaker="Ruben", text="Hola.")]
    with tempfile.TemporaryDirectory() as dirname:
        path = Path(dirname) / "memory.json"
        remember_validated_turns(path, Path("/audio/a.m4a"), turns)
        loaded = load_speaker_memory(path)

    self.assertEqual(loaded.identities_for(Path("/audio/a.m4a"))[0].name, "Ruben")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests.test_speaker_memory`

Expected: FAIL with `cannot import name 'remember_validated_turns'`.

- [ ] **Step 3: Implement the helper**

Add to `speaker_memory.py`:

```python
def remember_validated_turns(path: Path, audio_path: Path, turns: list[ConversationTurn]) -> SpeakerMemory:
    memory = load_speaker_memory(path)
    updated = add_identity_samples(memory, audio_path, turns)
    save_speaker_memory(path, updated)
    return updated
```

- [ ] **Step 4: Wire the GUI callback**

In `App.__init__`, add:

```python
self.speaker_memory_path = default_config_dir() / "speaker_memory.json"
```

In `_speaker_names_saved`, after updating `self.last_turns`, call:

```python
audio_text = self.audio_path.get().strip()
if audio_text:
    remember_validated_turns(self.speaker_memory_path, Path(audio_text), turns)
```

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=src python -m unittest tests.test_speaker_memory tests.test_gui_layout`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/meeting_transcriber/speaker_memory.py src/meeting_transcriber/gui.py tests/test_speaker_memory.py
git commit -m "Remember validated speaker names"
```

### Task 3: Suggest Existing Identities In The Rename Dialog

**Files:**
- Modify: `src/meeting_transcriber/speaker_memory.py`
- Modify: `src/meeting_transcriber/gui.py`
- Test: `tests/test_speaker_memory.py`

- [ ] **Step 1: Write failing test for suggestions**

Add:

```python
from meeting_transcriber.speaker_memory import identity_names


def test_identity_names_returns_known_names_for_audio(self):
    memory = SpeakerMemory(
        audios={
            "/audio/a.m4a": [
                SpeakerIdentity(name="Ruben", sample_ranges=((0, 5),)),
                SpeakerIdentity(name="Nuria", sample_ranges=((10, 15),)),
            ]
        }
    )

    self.assertEqual(identity_names(memory, Path("/audio/a.m4a")), ["Ruben", "Nuria"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests.test_speaker_memory`

Expected: FAIL with `cannot import name 'identity_names'`.

- [ ] **Step 3: Implement suggestions helper**

Add:

```python
def identity_names(memory: SpeakerMemory, audio_path: Path) -> list[str]:
    return [identity.name for identity in memory.identities_for(audio_path)]
```

- [ ] **Step 4: Update `SpeakerNameDialog`**

Pass known names from `App._open_speaker_editor`:

```python
memory = load_speaker_memory(self.speaker_memory_path)
known_names = identity_names(memory, Path(self.audio_path.get())) if self.audio_path.get().strip() else []
SpeakerNameDialog(self, turns, Path(self.output_dir.get()), self._speaker_names_saved, ai_response, known_names)
```

Change `SpeakerNameDialog.__init__` to accept `known_names: list[str] | None = None` and use `ttk.Combobox` instead of `ttk.Entry`:

```python
ttk.Combobox(root, textvariable=variable, values=self.known_names, width=24).grid(row=row, column=1, sticky="ew", padx=8, pady=4)
```

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=src python -m unittest tests.test_speaker_memory`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/meeting_transcriber/speaker_memory.py src/meeting_transcriber/gui.py tests/test_speaker_memory.py
git commit -m "Suggest remembered speaker names"
```

### Task 4: Apply Memory-Based Renaming Automatically When Safe

**Files:**
- Modify: `src/meeting_transcriber/speaker_memory.py`
- Modify: `src/meeting_transcriber/gui.py`
- Test: `tests/test_speaker_memory.py`

- [ ] **Step 1: Write failing test for conservative auto-mapping**

Add:

```python
from meeting_transcriber.speaker_memory import build_unique_name_mapping


def test_build_unique_name_mapping_only_maps_when_counts_match(self):
    memory = SpeakerMemory(
        audios={
            "/audio/a.m4a": [
                SpeakerIdentity(name="Ruben", sample_ranges=((0, 5),)),
                SpeakerIdentity(name="Nuria", sample_ranges=((10, 15),)),
            ]
        }
    )
    turns = [
        ConversationTurn(start=20, end=25, speaker="Persona 1", text="Hola."),
        ConversationTurn(start=25, end=30, speaker="Persona 2", text="Bon dia."),
    ]

    self.assertEqual(
        build_unique_name_mapping(memory, Path("/audio/a.m4a"), turns),
        {"Persona 1": "Ruben", "Persona 2": "Nuria"},
    )


def test_build_unique_name_mapping_refuses_ambiguous_count_mismatch(self):
    memory = SpeakerMemory(audios={"/audio/a.m4a": [SpeakerIdentity(name="Ruben", sample_ranges=((0, 5),))]})
    turns = [
        ConversationTurn(start=20, end=25, speaker="Persona 1", text="Hola."),
        ConversationTurn(start=25, end=30, speaker="Persona 2", text="Bon dia."),
    ]

    self.assertEqual(build_unique_name_mapping(memory, Path("/audio/a.m4a"), turns), {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests.test_speaker_memory`

Expected: FAIL with `cannot import name 'build_unique_name_mapping'`.

- [ ] **Step 3: Implement conservative mapping**

Add:

```python
from meeting_transcriber.speaker_names import speaker_labels


def build_unique_name_mapping(
    memory: SpeakerMemory,
    audio_path: Path,
    turns: list[ConversationTurn],
) -> dict[str, str]:
    labels = speaker_labels(turns)
    names = identity_names(memory, audio_path)
    if len(labels) != len(names):
        return {}
    if any(not label.startswith("Persona ") for label in labels):
        return {}
    return dict(zip(labels, names))
```

- [ ] **Step 4: Wire into completed processing**

In `_handle_completed_processing`, before asking to save/discard:

```python
memory = load_speaker_memory(self.speaker_memory_path)
mapping = build_unique_name_mapping(memory, audio_path, turns)
if mapping:
    turns = rename_speakers(turns, mapping)
```

Update status/log:

```python
if mapping:
    self.log.insert(tk.END, "Nombres recordados aplicados al fragmento\n")
```

- [ ] **Step 5: Run full tests**

Run: `PYTHONPATH=src python -m unittest discover -s tests`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/meeting_transcriber/speaker_memory.py src/meeting_transcriber/gui.py tests/test_speaker_memory.py
git commit -m "Apply remembered speaker names conservatively"
```

### Task 5: Prepare Voice Fingerprint Interface

**Files:**
- Modify: `src/meeting_transcriber/speaker_memory.py`
- Create: `src/meeting_transcriber/speaker_fingerprints.py`
- Test: `tests/test_speaker_fingerprints.py`

- [ ] **Step 1: Write failing tests for vector similarity**

```python
import unittest

from meeting_transcriber.speaker_fingerprints import cosine_similarity, best_embedding_match


class SpeakerFingerprintTests(unittest.TestCase):
    def test_cosine_similarity_scores_identical_vectors_as_one(self):
        self.assertAlmostEqual(cosine_similarity((1, 0, 0), (1, 0, 0)), 1.0)

    def test_best_embedding_match_returns_name_above_threshold(self):
        candidates = {"Ruben": ((1, 0),), "Nuria": ((0, 1),)}

        self.assertEqual(best_embedding_match((0.95, 0.05), candidates, threshold=0.8), "Ruben")

    def test_best_embedding_match_returns_none_below_threshold(self):
        candidates = {"Ruben": ((1, 0),)}

        self.assertIsNone(best_embedding_match((0.1, 0.2), candidates, threshold=0.8))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests.test_speaker_fingerprints`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement pure matching helpers**

Create `speaker_fingerprints.py`:

```python
from __future__ import annotations

import math


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (left_norm * right_norm)


def best_embedding_match(
    embedding: tuple[float, ...],
    candidates: dict[str, tuple[tuple[float, ...], ...]],
    *,
    threshold: float,
) -> str | None:
    best_name: str | None = None
    best_score = threshold
    for name, embeddings in candidates.items():
        for candidate in embeddings:
            score = cosine_similarity(embedding, candidate)
            if score > best_score:
                best_name = name
                best_score = score
    return best_name
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src python -m unittest tests.test_speaker_fingerprints`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/meeting_transcriber/speaker_fingerprints.py tests/test_speaker_fingerprints.py
git commit -m "Add speaker fingerprint matching helpers"
```

### Task 6: Research Pyannote Embedding Extraction

**Files:**
- Modify: `docs/FUNCIONAMIENTO.md`
- Modify later: `src/meeting_transcriber/speaker_fingerprints.py`

- [ ] **Step 1: Verify runtime API locally**

Run in the project venv:

```bash
.venv/bin/python - <<'PY'
from pyannote.audio import Model
print(Model)
PY
```

Expected: Prints the pyannote `Model` class if dependencies are installed.

- [ ] **Step 2: Decide extractor implementation**

If `pyannote.audio` exposes a stable embedding model in the installed version, add a function:

```python
def extract_speaker_embedding(audio_path: Path, start_seconds: float, end_seconds: float) -> tuple[float, ...] | None:
    ...
```

If it is not stable, keep embeddings as a future optional enhancement and rely on memory/name suggestions for now.

- [ ] **Step 3: Document current limitation**

Add to `docs/FUNCIONAMIENTO.md`:

```markdown
### Memoria de hablantes entre fragmentos

La aplicacion mantiene una memoria por archivo de audio con nombres validados y rangos de muestra.
La coherencia inicial reutiliza nombres confirmados. La comparacion por embeddings de voz queda preparada
como mejora opcional cuando la API de extraccion de huellas sea estable en el entorno instalado.
```

- [ ] **Step 4: Run full tests**

Run: `PYTHONPATH=src python -m unittest discover -s tests`

Expected: PASS.

---

## Self-Review

- Spec coverage: covers persistence, saving validated names, UI suggestions, conservative automatic reuse, and a clean interface for future voice fingerprints.
- Placeholder scan: no TBD/TODO language is used; optional embedding work is isolated behind an explicit research task.
- Type consistency: `SpeakerMemory`, `SpeakerIdentity`, `remember_validated_turns`, `identity_names`, and `build_unique_name_mapping` are introduced before use.
