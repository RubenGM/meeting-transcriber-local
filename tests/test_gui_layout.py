import unittest

from meeting_transcriber.gui import DEFAULT_WINDOW_GEOMETRY, MIN_WINDOW_SIZE, PROCESSING_OPTION_COLUMNS


class GuiLayoutTests(unittest.TestCase):
    def test_main_window_starts_larger_than_previous_default(self):
        width, height = (int(part) for part in DEFAULT_WINDOW_GEOMETRY.split("x"))

        self.assertGreaterEqual(width, 1100)
        self.assertGreaterEqual(height, 780)
        self.assertGreaterEqual(MIN_WINDOW_SIZE[0], 900)
        self.assertGreaterEqual(MIN_WINDOW_SIZE[1], 680)

    def test_processing_options_are_arranged_in_one_horizontal_row(self):
        self.assertEqual(
            PROCESSING_OPTION_COLUMNS,
            ("Calidad", "Diarización", "Separacion voces", "Idioma"),
        )


if __name__ == "__main__":
    unittest.main()
