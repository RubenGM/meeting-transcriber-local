import tempfile
import unittest
from pathlib import Path

from meeting_transcriber.history import AnalysisHistory, HistoryEntry, add_history_entry, load_history


class HistoryTests(unittest.TestCase):
    def test_add_history_entry_persists_completed_range_by_audio_path(self):
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "history.json"
            audio_path = Path("/audio/meeting.m4a")

            add_history_entry(
                path,
                audio_path,
                HistoryEntry(start_seconds=60, end_seconds=120, output_dir=Path("/out")),
            )
            history = load_history(path)

        self.assertEqual(
            history.entries_for(audio_path),
            [HistoryEntry(start_seconds=60, end_seconds=120, output_dir=Path("/out"))],
        )

    def test_entries_for_unknown_audio_are_empty(self):
        history = AnalysisHistory(entries={})

        self.assertEqual(history.entries_for(Path("/missing.wav")), [])

