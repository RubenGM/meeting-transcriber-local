import unittest

from meeting_transcriber.languages import code_from_display_name, display_name_from_code


class LanguageTests(unittest.TestCase):
    def test_code_from_display_name_returns_whisper_language_code(self):
        self.assertEqual(code_from_display_name("Català"), "ca")
        self.assertEqual(code_from_display_name("Español"), "es")
        self.assertEqual(code_from_display_name("Detección automática"), None)

    def test_code_from_display_name_accepts_existing_codes_for_old_config_files(self):
        self.assertEqual(code_from_display_name("ca"), "ca")
        self.assertEqual(code_from_display_name("es"), "es")
        self.assertEqual(code_from_display_name("auto"), None)

    def test_display_name_from_code_returns_human_readable_name(self):
        self.assertEqual(display_name_from_code("ca"), "Català")
        self.assertEqual(display_name_from_code("es"), "Español")
        self.assertEqual(display_name_from_code(None), "Detección automática")


if __name__ == "__main__":
    unittest.main()
