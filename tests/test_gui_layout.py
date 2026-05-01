import unittest

from meeting_transcriber.gui import (
    ADVANCED_CONTROL_LABELS,
    DEFAULT_WINDOW_GEOMETRY,
    MIN_WINDOW_SIZE,
    MODE_LABELS,
    PROCESSING_OPTION_COLUMNS,
    SIMPLE_CONTROL_LABELS,
)


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

    def test_main_modes_are_simple_and_advanced(self):
        self.assertEqual(MODE_LABELS, ("Simple", "Avanzado"))

    def test_simple_mode_exposes_guided_workflow_labels(self):
        self.assertIn("Analizar audio completo", SIMPLE_CONTROL_LABELS)
        self.assertIn("Estado de hablantes", SIMPLE_CONTROL_LABELS)
        self.assertIn("Abrir informe HTML", SIMPLE_CONTROL_LABELS)
        self.assertIn("Abrir transcripcion final", SIMPLE_CONTROL_LABELS)
        self.assertIn("Abrir audio normalizado", SIMPLE_CONTROL_LABELS)

    def test_advanced_mode_keeps_existing_expert_control_labels(self):
        expected = (
            "Audio",
            "Salida",
            "Calidad",
            "Diarización",
            "Separacion voces",
            "Idioma",
            "Hablantes",
            "Rango",
            "Ejecucion",
            "Token HF",
            "Procesar",
            "Probar rendimiento",
            "Renombrar hablantes",
            "Guardar configuracion",
            "Normalizar audio para voz humana",
        )

        for label in expected:
            self.assertIn(label, ADVANCED_CONTROL_LABELS)


if __name__ == "__main__":
    unittest.main()
