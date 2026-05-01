from pathlib import Path

import pytest

from meeting_transcriber.cancellation import CancelledError
from meeting_transcriber.diarization import DiarizationPreflightError
from meeting_transcriber.exporters import write_all_exports
from meeting_transcriber.history import HistoryEntry, add_history_entry, load_history
from meeting_transcriber.simple_pipeline import process_audio_simple, split_missing_ranges
from meeting_transcriber.types import ConversationTurn, ProcessingConfig


def _config() -> ProcessingConfig:
    return ProcessingConfig(
        whisper_model="small",
        diarization_model="pyannote/speaker-diarization-community-1",
        huggingface_token=None,
        ffmpeg_path=None,
        language="ca",
        min_speakers=None,
        max_speakers=None,
        device="cpu",
        compute_type="int8",
        export_speaker_audio=False,
        normalize_audio=True,
    )


def test_split_missing_ranges_chunks_with_overlap():
    chunks = split_missing_ranges([(0, 1000)], chunk_seconds=400, overlap_seconds=20)

    assert chunks == [(0, 400), (380, 780), (760, 1000)]


def test_process_audio_simple_records_chunks_and_normalizes(tmp_path):
    calls = []

    def fake_process(audio_path, output_dir, config, progress, cancelled):
        calls.append(config)
        return [ConversationTurn(config.start_seconds or 0, config.end_seconds or 1, "Persona 1", "Hola")]

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "meeting_transcriber.simple_pipeline.normalize_audio_for_speech",
            lambda _ffmpeg, _source, output, **_kwargs: output.write_text("audio", encoding="utf-8") or output,
        )
        summary = process_audio_simple(
            Path("/audio/a.m4a"),
            tmp_path / "out",
            tmp_path / "history.json",
            tmp_path / "memory.json",
            _config(),
            chunk_seconds=300,
            overlap_seconds=0,
            process_func=fake_process,
            duration_probe=lambda _ffmpeg, _audio: 600,
            embedding_extractor_factory=lambda _ffmpeg, _config: (lambda _audio, start, end: (1.0, 0.0)),
        )

    assert summary.chunks_total == 2
    assert summary.chunks_completed == 2
    assert all(call.normalize_audio for call in calls)
    assert summary.final_output_dir is not None
    assert summary.final_transcript_path is not None
    assert summary.html_report_path is not None
    assert summary.normalized_audio_path is not None
    assert Path(summary.final_transcript_path).exists()
    assert Path(summary.html_report_path).exists()
    assert Path(summary.normalized_audio_path).exists()
    history = load_history(tmp_path / "history.json")
    assert len(history.entries_for(Path("/audio/a.m4a"))) == 2


def test_process_audio_simple_skips_completed_history_ranges(tmp_path):
    history_path = tmp_path / "history.json"
    add_history_entry(history_path, Path("/audio/a.m4a"), HistoryEntry(0, 300, tmp_path / "old"))
    calls = []

    def fake_process(audio_path, output_dir, config, progress, cancelled):
        calls.append((config.start_seconds, config.end_seconds))
        return [ConversationTurn(config.start_seconds or 0, config.end_seconds or 1, "Persona 1", "Hola")]

    process_audio_simple(
        Path("/audio/a.m4a"),
        tmp_path / "out",
        history_path,
        tmp_path / "memory.json",
        _config(),
        chunk_seconds=300,
        overlap_seconds=0,
        process_func=fake_process,
        duration_probe=lambda _ffmpeg, _audio: 600,
        embedding_extractor_factory=lambda _ffmpeg, _config: (lambda _audio, start, end: (1.0, 0.0)),
    )

    assert calls == [(300.0, 600)]


def test_process_audio_simple_generates_final_from_existing_history_when_complete(tmp_path):
    history_path = tmp_path / "history.json"
    output_dir = tmp_path / "existing"
    audio_path = Path("/audio/a.m4a")
    write_all_exports(output_dir, [ConversationTurn(0, 5, "Persona 1", "Hola")])
    add_history_entry(history_path, audio_path, HistoryEntry(0, 600, output_dir))

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "meeting_transcriber.simple_pipeline.normalize_audio_for_speech",
            lambda _ffmpeg, _source, output, **_kwargs: output.write_text("audio", encoding="utf-8") or output,
        )
        summary = process_audio_simple(
            audio_path,
            tmp_path / "out",
            history_path,
            tmp_path / "memory.json",
            _config(),
            chunk_seconds=300,
            overlap_seconds=0,
            process_func=lambda *_args: pytest.fail("should not process completed audio"),
            duration_probe=lambda _ffmpeg, _audio: 600,
        )

    assert summary.chunks_total == 0
    assert summary.final_transcript_path is not None
    assert Path(summary.final_transcript_path).exists()
    assert summary.html_report_path is not None
    assert Path(summary.html_report_path).exists()
    assert summary.normalized_audio_path is not None
    assert Path(summary.normalized_audio_path).exists()


def test_process_audio_simple_propagates_cancellation(tmp_path):
    def fake_process(audio_path, output_dir, config, progress, cancelled):
        raise CancelledError("stop")

    with pytest.raises(CancelledError):
        process_audio_simple(
            Path("/audio/a.m4a"),
            tmp_path / "out",
            tmp_path / "history.json",
            tmp_path / "memory.json",
            _config(),
            chunk_seconds=300,
            overlap_seconds=0,
            process_func=fake_process,
            duration_probe=lambda _ffmpeg, _audio: 600,
        )


def test_process_audio_simple_treats_diarization_preflight_as_fatal(tmp_path):
    def fake_process(audio_path, output_dir, config, progress, cancelled):
        raise DiarizationPreflightError("missing token")

    with pytest.raises(DiarizationPreflightError):
        process_audio_simple(
            Path("/audio/a.m4a"),
            tmp_path / "out",
            tmp_path / "history.json",
            tmp_path / "memory.json",
            _config(),
            chunk_seconds=300,
            overlap_seconds=0,
            process_func=fake_process,
            duration_probe=lambda _ffmpeg, _audio: 600,
        )
