import unittest

from meeting_transcriber.pipeline import offset_diarization, offset_transcript
from meeting_transcriber.types import DiarizationSegment, TranscriptSegment, TranscriptWord


class OffsetTests(unittest.TestCase):
    def test_offset_transcript_adds_seconds_to_segments(self):
        result = offset_transcript(
            [
                TranscriptSegment(
                    start=1.0,
                    end=2.5,
                    text="hola",
                    words=(TranscriptWord(start=1.2, end=1.5, text="hola"),),
                )
            ],
            60.0,
        )

        self.assertEqual(
            result,
            [
                TranscriptSegment(
                    start=61.0,
                    end=62.5,
                    text="hola",
                    words=(TranscriptWord(start=61.2, end=61.5, text="hola"),),
                )
            ],
        )

    def test_offset_diarization_adds_seconds_to_segments(self):
        result = offset_diarization(
            [DiarizationSegment(start=1.0, end=2.5, speaker="SPEAKER_00")],
            60.0,
        )

        self.assertEqual(result, [DiarizationSegment(start=61.0, end=62.5, speaker="SPEAKER_00")])


if __name__ == "__main__":
    unittest.main()
