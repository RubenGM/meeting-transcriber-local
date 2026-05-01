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


if __name__ == "__main__":
    unittest.main()
