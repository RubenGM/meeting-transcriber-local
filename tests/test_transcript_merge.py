import unittest

from meeting_transcriber.transcript_merge import (
    DraftMergeRow,
    align_turns_for_merge,
    merged_turns_from_drafts,
)
from meeting_transcriber.types import ConversationTurn


class TranscriptMergeTests(unittest.TestCase):
    def test_align_turns_pairs_overlapping_turns(self):
        left = [
            ConversationTurn(0, 5, "Persona 1", "Hola"),
            ConversationTurn(8, 12, "Persona 2", "Adeu"),
        ]
        right = [
            ConversationTurn(1, 6, "Nuria", "Hola a tots"),
            ConversationTurn(8.5, 12, "Jordi", "Adeu"),
        ]

        rows = align_turns_for_merge(left, right)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].left, left[0])
        self.assertEqual(rows[0].right, right[0])
        self.assertEqual(rows[0].chosen_speaker, "Nuria")
        self.assertEqual(rows[0].chosen_text, "Hola a tots")

    def test_align_turns_keeps_unmatched_turns(self):
        left = [ConversationTurn(0, 5, "Persona 1", "Hola")]
        right = [ConversationTurn(10, 15, "Persona 2", "Tard")]

        rows = align_turns_for_merge(left, right)

        self.assertEqual(len(rows), 2)
        self.assertIsNotNone(rows[0].left)
        self.assertIsNone(rows[0].right)
        self.assertIsNone(rows[1].left)
        self.assertIsNotNone(rows[1].right)

    def test_default_choice_prefers_non_generic_speaker(self):
        left = ConversationTurn(0, 5, "Persona 1", "Text llarg esquerra")
        right = ConversationTurn(0, 5, "Nuria", "Text")

        rows = align_turns_for_merge([left], [right])

        self.assertEqual(rows[0].chosen_speaker, "Nuria")
        self.assertEqual(rows[0].chosen_text, "Text")

    def test_row_marks_identical_and_difference_types(self):
        identical = align_turns_for_merge(
            [ConversationTurn(0, 5, "Nuria", "Hola")],
            [ConversationTurn(0, 5, "Nuria", "Hola")],
        )[0]
        speaker_diff = align_turns_for_merge(
            [ConversationTurn(0, 5, "Persona 1", "Hola")],
            [ConversationTurn(0, 5, "Nuria", "Hola")],
        )[0]
        text_diff = align_turns_for_merge(
            [ConversationTurn(0, 5, "Nuria", "Hola")],
            [ConversationTurn(0, 5, "Nuria", "Hola a tots")],
        )[0]

        self.assertTrue(identical.is_identical)
        self.assertFalse(identical.has_speaker_difference)
        self.assertFalse(identical.has_text_difference)
        self.assertFalse(speaker_diff.is_identical)
        self.assertTrue(speaker_diff.has_speaker_difference)
        self.assertFalse(speaker_diff.has_text_difference)
        self.assertTrue(text_diff.has_text_difference)

    def test_merged_turns_from_drafts_skips_empty_text(self):
        drafts = [
            DraftMergeRow(0, 5, "Nuria", "Hola"),
            DraftMergeRow(5, 7, "Jordi", " "),
        ]

        self.assertEqual(
            merged_turns_from_drafts(drafts),
            [ConversationTurn(0, 5, "Nuria", "Hola")],
        )


if __name__ == "__main__":
    unittest.main()
