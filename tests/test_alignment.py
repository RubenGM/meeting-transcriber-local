import unittest

from meeting_transcriber.pipeline import assign_speakers, turns_from_transcript
from meeting_transcriber.types import DiarizationSegment, TranscriptSegment, TranscriptWord


class AlignmentTests(unittest.TestCase):
    def test_assign_speakers_uses_largest_time_overlap(self):
        transcript = [
            TranscriptSegment(start=0.0, end=4.0, text="hello there"),
            TranscriptSegment(start=4.0, end=7.0, text="next point"),
        ]
        diarization = [
            DiarizationSegment(start=0.0, end=1.0, speaker="SPEAKER_02"),
            DiarizationSegment(start=1.0, end=4.0, speaker="SPEAKER_01"),
            DiarizationSegment(start=4.0, end=7.0, speaker="SPEAKER_02"),
        ]

        turns = assign_speakers(transcript, diarization)

        self.assertEqual([turn.speaker for turn in turns], ["Persona 1", "Persona 2"])
        self.assertEqual([turn.text for turn in turns], ["hello there", "next point"])

    def test_assign_speakers_merges_adjacent_segments_from_same_speaker(self):
        transcript = [
            TranscriptSegment(start=0.0, end=1.5, text="first"),
            TranscriptSegment(start=1.5, end=3.0, text="second"),
            TranscriptSegment(start=3.0, end=4.0, text="third"),
        ]
        diarization = [
            DiarizationSegment(start=0.0, end=3.0, speaker="SPEAKER_A"),
            DiarizationSegment(start=3.0, end=4.0, speaker="SPEAKER_B"),
        ]

        turns = assign_speakers(transcript, diarization)

        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0].start, 0.0)
        self.assertEqual(turns[0].end, 3.0)
        self.assertEqual(turns[0].speaker, "Persona 1")
        self.assertEqual(turns[0].text, "first second")
        self.assertEqual(turns[1].speaker, "Persona 2")
        self.assertEqual(turns[1].text, "third")

    def test_assign_speakers_uses_word_timestamps_when_available(self):
        transcript = [
            TranscriptSegment(
                start=0.0,
                end=4.0,
                text="one two three four",
                words=(
                    TranscriptWord(start=0.0, end=0.8, text="one"),
                    TranscriptWord(start=0.9, end=1.6, text="two"),
                    TranscriptWord(start=2.1, end=2.8, text="three"),
                    TranscriptWord(start=2.9, end=3.6, text="four"),
                ),
            )
        ]
        diarization = [
            DiarizationSegment(start=0.0, end=2.0, speaker="SPEAKER_A"),
            DiarizationSegment(start=2.0, end=4.0, speaker="SPEAKER_B"),
        ]

        turns = assign_speakers(transcript, diarization)

        self.assertEqual([turn.speaker for turn in turns], ["Persona 1", "Persona 2"])
        self.assertEqual([turn.text for turn in turns], ["one two", "three four"])

    def test_turns_from_transcript_creates_raw_fallback_turns(self):
        transcript = [
            TranscriptSegment(start=0.0, end=1.5, text=" first "),
            TranscriptSegment(start=1.5, end=3.0, text=""),
            TranscriptSegment(start=3.0, end=4.0, text="second"),
        ]

        turns = turns_from_transcript(transcript)

        self.assertEqual([turn.speaker for turn in turns], ["Sin diarizar", "Sin diarizar"])
        self.assertEqual([turn.text for turn in turns], ["first", "second"])


if __name__ == "__main__":
    unittest.main()
