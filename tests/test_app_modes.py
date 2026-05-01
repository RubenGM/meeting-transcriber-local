from meeting_transcriber.app_modes import AppMode, SimpleModeSettings


def test_app_modes_have_simple_and_advanced_values():
    assert AppMode.SIMPLE.value == "simple"
    assert AppMode.ADVANCED.value == "advanced"


def test_simple_mode_normalizes_audio_by_default():
    settings = SimpleModeSettings()

    assert settings.normalize_audio is True
    assert settings.chunk_overlap_seconds == 15.0

