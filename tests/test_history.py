import tempfile
import unittest
from pathlib import Path

from meeting_transcriber.history import (
    AnalysisHistory,
    HistoryEntry,
    add_history_entry,
    completed_ranges,
    coverage_seconds,
    load_history,
    recommend_next_range,
    recent_processing_speed,
    remove_history_entry,
    output_dir_reference_count,
)


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

    def test_remove_history_entry_deletes_selected_entry_only(self):
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "history.json"
            audio_path = Path("/audio/meeting.m4a")
            add_history_entry(path, audio_path, HistoryEntry(0, 120, Path("/out/a")))
            add_history_entry(path, audio_path, HistoryEntry(120, 240, Path("/out/b")))

            removed = remove_history_entry(path, audio_path, 0)
            history = load_history(path)

        self.assertEqual(removed, HistoryEntry(0, 120, Path("/out/a")))
        self.assertEqual(history.entries_for(audio_path), [HistoryEntry(120, 240, Path("/out/b"))])

    def test_remove_history_entry_raises_index_error_for_missing_entry(self):
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "history.json"

            with self.assertRaises(IndexError):
                remove_history_entry(path, Path("/audio/meeting.m4a"), 0)

    def test_output_dir_reference_count_counts_reused_output_dirs(self):
        history = AnalysisHistory(
            entries={
                "/audio/a.m4a": [
                    HistoryEntry(0, 120, Path("/out/shared")),
                    HistoryEntry(120, 240, Path("/out/other")),
                ],
                "/audio/b.m4a": [HistoryEntry(0, 60, Path("/out/shared"))],
            }
        )

        self.assertEqual(output_dir_reference_count(history, Path("/out/shared")), 2)

    def test_completed_ranges_merges_adjacent_and_overlapping_entries(self):
        entries = [
            HistoryEntry(start_seconds=0, end_seconds=120, output_dir=Path("/out")),
            HistoryEntry(start_seconds=120, end_seconds=300, output_dir=Path("/out")),
            HistoryEntry(start_seconds=280, end_seconds=360, output_dir=Path("/out")),
            HistoryEntry(start_seconds=600, end_seconds=660, output_dir=Path("/out")),
        ]

        self.assertEqual(completed_ranges(entries, total_duration_seconds=900), [(0.0, 360.0), (600.0, 660.0)])
        self.assertEqual(coverage_seconds(entries, total_duration_seconds=900), 420.0)

    def test_recent_processing_speed_uses_weighted_recent_entries(self):
        entries = [
            HistoryEntry(start_seconds=0, end_seconds=120, output_dir=Path("/out"), elapsed_seconds=120),
            HistoryEntry(start_seconds=120, end_seconds=420, output_dir=Path("/out"), elapsed_seconds=150),
            HistoryEntry(start_seconds=420, end_seconds=720, output_dir=Path("/out"), elapsed_seconds=100),
        ]

        self.assertEqual(recent_processing_speed(entries, limit=2), 2.4)

    def test_recommend_next_range_uses_speed_and_first_gap(self):
        entries = [
            HistoryEntry(start_seconds=0, end_seconds=300, output_dir=Path("/out"), elapsed_seconds=600),
            HistoryEntry(start_seconds=900, end_seconds=1200, output_dir=Path("/out"), elapsed_seconds=600),
        ]

        recommendation = recommend_next_range(
            entries,
            total_duration_seconds=1800,
            target_wait_seconds=600,
            min_audio_seconds=120,
            max_audio_seconds=1800,
        )

        self.assertIsNotNone(recommendation)
        assert recommendation is not None
        self.assertEqual(recommendation.start_seconds, 300.0)
        self.assertEqual(recommendation.end_seconds, 600.0)
        self.assertEqual(recommendation.speed, 0.5)
        self.assertEqual(recommendation.estimated_elapsed_seconds, 600.0)

    def test_recommend_next_range_clamps_to_end_of_gap(self):
        entries = [
            HistoryEntry(start_seconds=0, end_seconds=300, output_dir=Path("/out"), elapsed_seconds=60),
            HistoryEntry(start_seconds=420, end_seconds=600, output_dir=Path("/out"), elapsed_seconds=60),
        ]

        recommendation = recommend_next_range(
            entries,
            total_duration_seconds=600,
            target_wait_seconds=900,
            min_audio_seconds=120,
            max_audio_seconds=1800,
        )

        self.assertIsNotNone(recommendation)
        assert recommendation is not None
        self.assertEqual(recommendation.start_seconds, 300.0)
        self.assertEqual(recommendation.end_seconds, 420.0)

    def test_recommend_next_range_returns_none_when_complete(self):
        entries = [
            HistoryEntry(start_seconds=None, end_seconds=None, output_dir=Path("/out"), elapsed_seconds=60),
        ]

        self.assertIsNone(recommend_next_range(entries, total_duration_seconds=600, target_wait_seconds=900))
