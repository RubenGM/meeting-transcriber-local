from pathlib import Path

from meeting_transcriber.simple_report import (
    SimpleChunkReport,
    combine_chunk_turns,
    write_simple_final_artifacts,
)
from meeting_transcriber.types import ConversationTurn


def test_combine_chunk_turns_removes_overlap_using_midpoint_boundary():
    chunks = [
        SimpleChunkReport(
            Path("/out/1"),
            0,
            100,
            (
                ConversationTurn(10, 20, "Persona 1", "Primer tramo"),
                ConversationTurn(92, 98, "Persona 1", "Solape anterior"),
            ),
        ),
        SimpleChunkReport(
            Path("/out/2"),
            90,
            180,
            (
                ConversationTurn(92, 98, "Persona 1", "Solape anterior"),
                ConversationTurn(110, 120, "Persona 2", "Segundo tramo"),
            ),
        ),
    ]

    turns = combine_chunk_turns(chunks)

    assert [turn.text for turn in turns] == ["Primer tramo", "Segundo tramo"]


def test_write_simple_final_artifacts_creates_transcripts_and_html_report(tmp_path):
    chunks = [
        SimpleChunkReport(
            tmp_path / "chunk",
            0,
            60,
            (ConversationTurn(0, 5, "Ruben", "Hola."),),
        )
    ]

    artifacts = write_simple_final_artifacts(
        audio_path=Path("/audio/meeting.m4a"),
        base_output_dir=tmp_path / "out",
        chunks=chunks,
        chunks_failed=0,
        normalized_audio_path=tmp_path / "out" / "meeting" / "final" / "normalized_audio.wav",
    )

    assert artifacts.output_dir == tmp_path / "out" / "meeting" / "final"
    assert artifacts.transcript_path.exists()
    assert artifacts.report_path.exists()
    report = artifacts.report_path.read_text(encoding="utf-8")
    assert "Informe de transcripcion" in report
    assert "normalized_audio.wav" in report
    assert "class=\"play-button play-turn\"" in report
    assert "class=\"stop-button stop-turn\"" in report
    assert "class=\"segment-progress\"" in report
    assert "Posicion del tramo" in report
    assert "rowTimeFromProgress" in report
    assert "updateProgress" in report
    assert "data-state=\"stopped\"" in report
    assert "Pausar tramo" in report
    assert "Continuar tramo" in report
    assert "stopPlayback" in report
    assert "contenteditable=\"true\"" in report
    assert "id=\"save-review\"" in report
    assert "transcript_review.md" in report
    assert "localStorage" in report
