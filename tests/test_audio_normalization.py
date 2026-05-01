from pathlib import Path
from unittest.mock import Mock, patch

from meeting_transcriber.audio_normalization import (
    normalize_audio_for_speech,
    normalize_audio_with_ffmpeg,
    speech_normalization_filter,
)


def test_speech_normalization_filter_prioritizes_voice_and_noise_reduction():
    value = speech_normalization_filter()

    assert "highpass=f=80" in value
    assert "lowpass=f=8000" in value
    assert "afftdn=nf=-25" in value
    assert "dynaudnorm" in value
    assert "loudnorm=I=-16:TP=-1.5:LRA=11" in value


def test_normalize_audio_with_ffmpeg_runs_ffmpeg_with_voice_filter(tmp_path):
    result = Mock(returncode=0, stderr="")

    with patch("meeting_transcriber.audio_normalization.subprocess.run", return_value=result) as run:
        output = normalize_audio_with_ffmpeg(Path("/ffmpeg"), Path("/audio/in.m4a"), tmp_path / "out.wav")

    assert output == tmp_path / "out.wav"
    command = run.call_args.args[0]
    assert command[:4] == ["/ffmpeg", "-y", "-i", "/audio/in.m4a"]
    assert "-af" in command
    assert speech_normalization_filter() in command


def test_normalize_audio_for_speech_uses_deepfilternet_when_available(tmp_path):
    with (
        patch("meeting_transcriber.audio_normalization.resolve_deep_filter_binary", return_value=Path("/deep-filter")),
        patch("meeting_transcriber.audio_normalization.enhance_with_deepfilternet") as enhance,
    ):
        enhance.return_value = Mock(ok=True)

        output = normalize_audio_for_speech(Path("/ffmpeg"), Path("/audio/in.m4a"), tmp_path / "out.wav")

    assert output == tmp_path / "out.wav"
    assert enhance.called


def test_normalize_audio_for_speech_falls_back_to_ffmpeg_when_deepfilternet_fails(tmp_path):
    result = Mock(returncode=0, stderr="")
    with (
        patch("meeting_transcriber.audio_normalization.resolve_deep_filter_binary", return_value=Path("/deep-filter")),
        patch("meeting_transcriber.audio_normalization.enhance_with_deepfilternet", return_value=Mock(ok=False)),
        patch("meeting_transcriber.audio_normalization.subprocess.run", return_value=result) as run,
    ):
        output = normalize_audio_for_speech(Path("/ffmpeg"), Path("/audio/in.m4a"), tmp_path / "out.wav")

    assert output == tmp_path / "out.wav"
    assert run.called


def test_normalize_audio_with_ffmpeg_reports_progress(tmp_path):
    events = []
    result = Mock(returncode=0, stderr="")

    with patch("meeting_transcriber.audio_normalization.subprocess.run", return_value=result):
        normalize_audio_with_ffmpeg(
            Path("/ffmpeg"),
            Path("/audio/in.m4a"),
            tmp_path / "out.wav",
            progress=events.append,
            duration_seconds=120,
        )

    assert [event.stage for event in events] == ["normalization_progress", "normalization_progress"]
    assert events[0].message == "FFmpeg: aplicando filtros de voz"
    assert events[-1].completed == 2
