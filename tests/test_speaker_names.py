import unittest

from meeting_transcriber.speaker_names import add_known_name


class SpeakerNamesTests(unittest.TestCase):
    def test_add_known_name_appends_clean_new_name(self):
        self.assertEqual(add_known_name(["Alan Bernan"], " Alícia "), ["Alan Bernan", "Alícia"])

    def test_add_known_name_ignores_existing_name(self):
        self.assertEqual(add_known_name(["Alícia"], "Alícia"), ["Alícia"])

    def test_add_known_name_ignores_blank_name(self):
        self.assertEqual(add_known_name(["Alícia"], " "), ["Alícia"])


if __name__ == "__main__":
    unittest.main()
