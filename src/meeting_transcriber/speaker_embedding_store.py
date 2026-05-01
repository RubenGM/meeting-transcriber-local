from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpeakerEmbeddingStore:
    entries: dict[str, dict[str, dict[str, tuple[float, ...]]]]

    def embedding_for(self, audio_path: Path, source_id: str, speaker: str) -> tuple[float, ...] | None:
        return self.entries.get(str(audio_path), {}).get(source_id, {}).get(speaker)

    def embeddings_for_source(self, audio_path: Path, source_id: str) -> dict[str, tuple[float, ...]]:
        return dict(self.entries.get(str(audio_path), {}).get(source_id, {}))

    def with_embedding(
        self,
        *,
        audio_path: Path,
        source_id: str,
        speaker: str,
        embedding: tuple[float, ...],
    ) -> SpeakerEmbeddingStore:
        entries = {
            audio_key: {
                source_key: dict(speaker_embeddings)
                for source_key, speaker_embeddings in source_entries.items()
            }
            for audio_key, source_entries in self.entries.items()
        }
        entries.setdefault(str(audio_path), {}).setdefault(source_id, {})[speaker] = embedding
        return SpeakerEmbeddingStore(entries=entries)


def load_embedding_store(path: Path) -> SpeakerEmbeddingStore:
    if not path.exists():
        return SpeakerEmbeddingStore(entries={})
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries: dict[str, dict[str, dict[str, tuple[float, ...]]]] = {}
    for audio_path, sources in payload.get("entries", {}).items():
        entries[audio_path] = {}
        for source_id, speakers in sources.items():
            entries[audio_path][source_id] = {
                speaker: tuple(float(value) for value in embedding)
                for speaker, embedding in speakers.items()
            }
    return SpeakerEmbeddingStore(entries=entries)


def save_embedding_store(path: Path, store: SpeakerEmbeddingStore) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "entries": {
            audio_path: {
                source_id: {
                    speaker: list(embedding)
                    for speaker, embedding in speakers.items()
                }
                for source_id, speakers in sources.items()
            }
            for audio_path, sources in store.entries.items()
        }
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
