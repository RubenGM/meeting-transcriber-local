import tempfile
import unittest
from pathlib import Path

from meeting_transcriber.speaker_embedding_store import (
    SpeakerEmbeddingStore,
    load_embedding_store,
    save_embedding_store,
)


class SpeakerEmbeddingStoreTests(unittest.TestCase):
    def test_store_round_trips_embeddings_by_audio_source_and_speaker(self):
        store = SpeakerEmbeddingStore(entries={})
        updated = store.with_embedding(
            audio_path=Path("/audio/a.m4a"),
            source_id="entry-1",
            speaker="Persona 1",
            embedding=(0.1, 0.9),
        )

        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "embeddings.json"
            save_embedding_store(path, updated)
            loaded = load_embedding_store(path)

        self.assertEqual(
            loaded.embedding_for(Path("/audio/a.m4a"), "entry-1", "Persona 1"),
            (0.1, 0.9),
        )

    def test_missing_embedding_returns_none(self):
        store = SpeakerEmbeddingStore(entries={})

        self.assertIsNone(store.embedding_for(Path("/audio/a.m4a"), "entry-1", "Persona 1"))

    def test_count_embeddings_for_audio(self):
        store = (
            SpeakerEmbeddingStore(entries={})
            .with_embedding(audio_path=Path("/audio/a.m4a"), source_id="entry-1", speaker="Persona 1", embedding=(1.0,))
            .with_embedding(audio_path=Path("/audio/a.m4a"), source_id="entry-1", speaker="Persona 2", embedding=(0.5,))
            .with_embedding(audio_path=Path("/audio/b.m4a"), source_id="entry-2", speaker="Persona 1", embedding=(0.1,))
        )

        self.assertEqual(store.count_embeddings(Path("/audio/a.m4a")), 2)


if __name__ == "__main__":
    unittest.main()
