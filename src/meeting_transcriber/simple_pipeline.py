from __future__ import annotations

import contextlib
import io
import json
from dataclasses import replace
from pathlib import Path
from typing import Callable

from meeting_transcriber.app_modes import SimpleRunSummary
from meeting_transcriber.audio import probe_audio_duration
from meeting_transcriber.audio_normalization import normalize_audio_for_speech
from meeting_transcriber.cancellation import CancelCheck, CancelledError, raise_if_cancelled
from meeting_transcriber.diarization import DiarizationPreflightError
from meeting_transcriber.exporters import build_processing_output_dir, write_all_exports
from meeting_transcriber.ffmpeg import resolve_ffmpeg_path
from meeting_transcriber.history import HistoryEntry, add_history_entry, load_history, missing_ranges, visible_entries_for
from meeting_transcriber.pipeline import process_meeting
from meeting_transcriber.progress import ProgressEvent
from meeting_transcriber.simple_report import (
    SimpleChunkReport,
    build_simple_final_output_dir,
    write_simple_final_artifacts,
)
from meeting_transcriber.speaker_fingerprints import (
    EmbeddingExtractor,
    extract_speaker_embeddings,
    is_cuda_embedding_error,
    load_pyannote_embedding_extractor,
)
from meeting_transcriber.speaker_identity_resolver import (
    ChunkSpeakerEvidence,
    apply_chunk_speaker_resolution,
    resolve_chunk_speaker_identities,
    write_identity_decisions,
)
from meeting_transcriber.speaker_memory import load_speaker_memory, remember_validated_turns
from meeting_transcriber.types import ConversationTurn, ProcessingConfig


ProgressCallback = Callable[[ProgressEvent], None]
ProcessMeeting = Callable[[Path, Path, ProcessingConfig, ProgressCallback | None, CancelCheck | None], list[ConversationTurn]]
DurationProbe = Callable[[Path, Path], float | None]
EmbeddingExtractorFactory = Callable[[Path, ProcessingConfig], EmbeddingExtractor]


def split_missing_ranges(
    missing: list[tuple[float, float]],
    *,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[tuple[float, float]]:
    chunks: list[tuple[float, float]] = []
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    for range_start, range_end in missing:
        cursor = max(0.0, range_start)
        limit = max(cursor, range_end)
        while cursor < limit:
            end = min(limit, cursor + chunk_seconds)
            if end > cursor:
                chunks.append((cursor, end))
            if end >= limit:
                break
            cursor = max(range_start, end - max(0.0, overlap_seconds))
            if chunks and cursor <= chunks[-1][0]:
                cursor = chunks[-1][1]
    return chunks


def process_audio_simple(
    audio_path: Path,
    base_output_dir: Path,
    history_path: Path,
    speaker_memory_path: Path,
    config: ProcessingConfig,
    *,
    chunk_seconds: float,
    overlap_seconds: float,
    progress: ProgressCallback | None = None,
    cancelled: CancelCheck | None = None,
    process_func: ProcessMeeting = process_meeting,
    duration_probe: DurationProbe = probe_audio_duration,
    embedding_extractor_factory: EmbeddingExtractorFactory | None = None,
) -> SimpleRunSummary:
    reporter = progress if progress is not None else _ignore_progress
    raise_if_cancelled(cancelled)
    reporter(ProgressEvent(stage="simple_plan", message="Preparando analisis automatico"))
    duration = duration_probe(resolve_ffmpeg_path(config.ffmpeg_path), audio_path)
    if duration is None or duration <= 0:
        raise RuntimeError("No se pudo conocer la duracion del audio para el modo simple.")

    history = load_history(history_path)
    entries = visible_entries_for(history, audio_path)
    chunks = split_missing_ranges(
        missing_ranges(entries, duration),
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
    )
    completed: list[ChunkSpeakerEvidence] = []
    failed = 0
    output_dirs: list[str] = []

    for index, (start, end) in enumerate(chunks, start=1):
        raise_if_cancelled(cancelled)
        reporter(
            ProgressEvent(
                stage="simple_chunk",
                message=f"Procesando porcion {index}/{len(chunks)}",
                seconds=start,
                duration_seconds=duration,
                completed=index - 1,
                total=len(chunks),
            )
        )
        chunk_config = replace(config, start_seconds=start, end_seconds=end, normalize_audio=True)
        output_dir = build_processing_output_dir(
            base_output_dir,
            audio_path,
            start_seconds=start,
            end_seconds=end,
        )
        try:
            turns = process_func(audio_path, output_dir, chunk_config, reporter, cancelled)
        except CancelledError:
            raise
        except DiarizationPreflightError:
            raise
        except Exception as exc:
            failed += 1
            reporter(
                ProgressEvent(
                    stage="simple_chunk",
                    message=f"Fallo en porcion {index}/{len(chunks)}: {exc}",
                    completed=index,
                    total=len(chunks),
                )
            )
            continue
        add_history_entry(
            history_path,
            audio_path,
            HistoryEntry(start_seconds=start, end_seconds=end, output_dir=output_dir),
        )
        embeddings = _extract_embeddings_for_chunk(
            audio_path,
            turns,
            chunk_config,
            embedding_extractor_factory,
            reporter,
        )
        completed.append(
            ChunkSpeakerEvidence(
                output_dir=output_dir,
                turns=turns,
                embeddings=embeddings,
                start_seconds=start,
                end_seconds=end,
            )
        )
        output_dirs.append(str(output_dir))

    if completed:
        reporter(ProgressEvent(stage="simple_identity", message="Unificando nombres de hablantes entre porciones"))
        memory = load_speaker_memory(speaker_memory_path)
        resolutions = resolve_chunk_speaker_identities(audio_path, memory, completed)
        for evidence, resolution in zip(completed, resolutions, strict=False):
            updated_turns = apply_chunk_speaker_resolution(evidence.turns, resolution)
            write_all_exports(evidence.output_dir, updated_turns)
            write_identity_decisions(evidence.output_dir, resolution.decisions)
            remember_validated_turns(speaker_memory_path, audio_path, updated_turns)

    final_output_dir: str | None = None
    final_transcript_path: str | None = None
    html_report_path: str | None = None
    normalized_audio_path: str | None = None
    final_chunks = _final_chunks_from_history(history_path, audio_path, duration)
    if final_chunks:
        normalized_path = build_simple_final_output_dir(base_output_dir, audio_path) / "normalized_audio.wav"
        try:
            reporter(ProgressEvent(stage="simple_report", message="Generando audio normalizado revisable"))
            normalized_path.parent.mkdir(parents=True, exist_ok=True)
            normalize_audio_for_speech(
                resolve_ffmpeg_path(config.ffmpeg_path),
                audio_path,
                normalized_path,
                progress=reporter,
                duration_seconds=duration,
            )
            normalized_audio_path = str(normalized_path)
        except Exception as exc:
            reporter(
                ProgressEvent(
                    stage="simple_report",
                    message=f"No se pudo generar audio normalizado revisable: {exc}",
                )
            )
        reporter(ProgressEvent(stage="simple_report", message="Generando transcripcion final e informe HTML"))
        artifacts = write_simple_final_artifacts(
            audio_path=audio_path,
            base_output_dir=base_output_dir,
            chunks=final_chunks,
            chunks_failed=failed,
            normalized_audio_path=Path(normalized_audio_path) if normalized_audio_path else None,
        )
        final_output_dir = str(artifacts.output_dir)
        final_transcript_path = str(artifacts.transcript_path)
        html_report_path = str(artifacts.report_path)

    summary = SimpleRunSummary(
        chunks_total=len(chunks),
        chunks_completed=len(completed),
        chunks_failed=failed,
        output_dirs=tuple(output_dirs),
        final_output_dir=final_output_dir,
        final_transcript_path=final_transcript_path,
        html_report_path=html_report_path,
        normalized_audio_path=normalized_audio_path,
    )
    reporter(
        ProgressEvent(
            stage="simple_done",
            message=(
                "Analisis automatico completado: "
                f"{summary.chunks_completed}/{summary.chunks_total} porciones, "
                f"{summary.chunks_failed} fallos"
            ),
            completed=summary.chunks_completed,
            total=summary.chunks_total,
        )
    )
    return summary


def _final_chunks_from_history(
    history_path: Path,
    audio_path: Path,
    duration: float,
) -> list[SimpleChunkReport]:
    history = load_history(history_path)
    chunks: list[SimpleChunkReport] = []
    for entry in visible_entries_for(history, audio_path):
        turns = load_turns_from_transcript_json(entry.output_dir)
        if not turns:
            continue
        chunks.append(
            SimpleChunkReport(
                output_dir=entry.output_dir,
                start_seconds=float(entry.start_seconds or 0.0),
                end_seconds=float(duration if entry.end_seconds is None else entry.end_seconds),
                turns=tuple(turns),
            )
        )
    return chunks


def load_turns_from_transcript_json(output_dir: Path) -> list[ConversationTurn]:
    path = output_dir / "transcript.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        ConversationTurn(
            start=float(item["start"]),
            end=float(item["end"]),
            speaker=str(item["speaker"]),
            text=str(item["text"]),
        )
        for item in payload.get("turns", [])
    ]


def _extract_embeddings_for_chunk(
    audio_path: Path,
    turns: list[ConversationTurn],
    config: ProcessingConfig,
    factory: EmbeddingExtractorFactory | None,
    reporter: ProgressCallback,
) -> dict[str, tuple[float, ...]]:
    if not turns:
        return {}
    try:
        with _quiet_model_output():
            extractor = (
                factory(resolve_ffmpeg_path(config.ffmpeg_path), config)
                if factory is not None
                else load_pyannote_embedding_extractor(
                    resolve_ffmpeg_path(config.ffmpeg_path),
                    huggingface_token=config.huggingface_token,
                    device=config.device,
                )
            )
        try:
            with _quiet_model_output():
                return extract_speaker_embeddings(audio_path, turns, extractor)
        except Exception as exc:
            if factory is not None or config.device != "cuda" or not is_cuda_embedding_error(exc):
                raise
            with _quiet_model_output():
                cpu_extractor = load_pyannote_embedding_extractor(
                    resolve_ffmpeg_path(config.ffmpeg_path),
                    huggingface_token=config.huggingface_token,
                    device="cpu",
                )
                return extract_speaker_embeddings(audio_path, turns, cpu_extractor)
    except Exception as exc:
        reporter(
            ProgressEvent(
                stage="simple_identity",
                message=f"No se pudieron generar huellas de voz para una porcion: {exc}",
            )
        )
        return {}


def _ignore_progress(_event: ProgressEvent) -> None:
    return None


@contextlib.contextmanager
def _quiet_model_output() -> object:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield
