import unittest

from meeting_transcriber.progress import (
    ProgressEvent,
    format_diarization_progress,
    format_seconds,
    format_segment_preview,
    format_speaker_summary,
    format_transcription_progress,
)
from meeting_transcriber.types import ConversationTurn


class ProgressTests(unittest.TestCase):
    def test_format_seconds_uses_clock_format(self):
        self.assertEqual(format_seconds(65.4), "00:01:05")

    def test_format_transcription_progress_includes_time_and_text_count(self):
        event = ProgressEvent(
            stage="transcription",
            message="Transcribiendo",
            seconds=65.4,
            duration_seconds=130.0,
            elapsed_seconds=10.0,
            text_chars=120,
            segments=3,
        )

        self.assertEqual(
            format_transcription_progress(event),
            "Transcribiendo audio: 00:01:05 / 00:02:10, 6.5x, ETA 00:00:09, 3 segmentos, 120 caracteres",
        )

    def test_format_speaker_summary_counts_turns_and_talk_time(self):
        turns = [
            ConversationTurn(start=0, end=10, speaker="Persona 1", text="Hola"),
            ConversationTurn(start=10, end=25, speaker="Persona 2", text="Que tal"),
            ConversationTurn(start=25, end=30, speaker="Persona 1", text="Bien"),
        ]

        self.assertEqual(
            format_speaker_summary(turns),
            "Persona 1: 00:00:15 en 2 turnos | Persona 2: 00:00:15 en 1 turnos",
        )

    def test_format_segment_preview_includes_timestamp_and_text(self):
        event = ProgressEvent(
            stage="transcription_segment",
            message="Segmento transcrito",
            seconds=65.4,
            text="Bon dia a tothom.",
        )

        self.assertEqual(format_segment_preview(event), "[00:01:05] Bon dia a tothom.")

    def test_format_diarization_progress_includes_percentage(self):
        event = ProgressEvent(
            stage="diarization_progress",
            message="Separando voces: comparando huellas de voz",
            completed=3,
            total=6,
        )

        self.assertEqual(
            format_diarization_progress(event),
            "Separando voces: comparando huellas de voz: 3/6 (50%)",
        )


if __name__ == "__main__":
    unittest.main()
