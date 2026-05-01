from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Iterable

from meeting_transcriber.audio import export_speaker_audio, extract_audio_range
from meeting_transcriber.audio_normalization import normalize_audio_for_speech
from meeting_transcriber.cancellation import CancelCheck, raise_if_cancelled
from meeting_transcriber.diarization import diarize_audio, load_diarization_pipeline
from meeting_transcriber.exporters import write_all_exports
from meeting_transcriber.ffmpeg import resolve_ffmpeg_path
from meeting_transcriber.progress import ProgressEvent, format_speaker_summary
from meeting_transcriber.transcription import transcribe_audio
from meeting_transcriber.types import (
    ConversationTurn,
    DiarizationSegment,
    ProcessingConfig,
    TranscriptSegment,
)

ProgressCallback = Callable[[ProgressEvent], None]


def assign_speakers(
    transcript: Iterable[TranscriptSegment],
    diarization: Iterable[DiarizationSegment],
) -> list[ConversationTurn]:
    transcript_segments = list(transcript)
    diarization_segments = list(diarization)
    if any(segment.words for segment in transcript_segments):
        return assign_speakers_by_words(transcript_segments, diarization_segments)

    speaker_names: dict[str, str] = {}
    turns: list[ConversationTurn] = []

    for segment in transcript_segments:
        speaker = _speaker_for_segment(segment, diarization_segments)
        speaker_names.setdefault(speaker, f"Persona {len(speaker_names) + 1}")
        display_speaker = speaker_names[speaker]
        text = segment.text.strip()
        if not text:
            continue
        if turns and turns[-1].speaker == display_speaker:
            previous = turns[-1]
            turns[-1] = ConversationTurn(
                start=previous.start,
                end=segment.end,
                speaker=previous.speaker,
                text=f"{previous.text} {text}",
            )
        else:
            turns.append(
                ConversationTurn(
                    start=segment.start,
                    end=segment.end,
                    speaker=display_speaker,
                    text=text,
                )
            )

    return turns


def assign_speakers_by_words(
    transcript: Iterable[TranscriptSegment],
    diarization: Iterable[DiarizationSegment],
) -> list[ConversationTurn]:
    diarization_segments = list(diarization)
    speaker_names: dict[str, str] = {}
    turns: list[ConversationTurn] = []

    for segment in transcript:
        fallback_speaker = _speaker_for_segment(segment, diarization_segments)
        for word in segment.words:
            text = word.text.strip()
            if not text:
                continue
            speaker = _speaker_for_time((word.start + word.end) / 2.0, diarization_segments)
            if speaker is None:
                speaker = fallback_speaker
            speaker_names.setdefault(speaker, f"Persona {len(speaker_names) + 1}")
            display_speaker = speaker_names[speaker]
            if turns and turns[-1].speaker == display_speaker and word.start - turns[-1].end <= 1.2:
                previous = turns[-1]
                turns[-1] = ConversationTurn(
                    start=previous.start,
                    end=word.end,
                    speaker=previous.speaker,
                    text=_append_word(previous.text, text),
                )
            else:
                turns.append(
                    ConversationTurn(
                        start=word.start,
                        end=word.end,
                        speaker=display_speaker,
                        text=text,
                    )
                )

    return turns


def process_meeting(
    audio_path: Path,
    output_dir: Path,
    config: ProcessingConfig,
    progress: ProgressCallback | None = None,
    cancelled: CancelCheck | None = None,
) -> list[ConversationTurn]:
    reporter = progress if progress is not None else _ignore_progress
    temp_dir: TemporaryDirectory[str] | None = None
    try:
        raise_if_cancelled(cancelled)
        reporter(ProgressEvent(stage="preflight", message="Comprobando modelos y permisos"))
        diarization_pipeline = load_diarization_pipeline(config)
        raise_if_cancelled(cancelled)
        reporter(ProgressEvent(stage="preflight_done", message="Modelos y permisos comprobados"))

        working_audio = audio_path
        offset_seconds = config.start_seconds or 0.0
        if config.start_seconds is not None or config.end_seconds is not None:
            raise_if_cancelled(cancelled)
            reporter(ProgressEvent(stage="prepare", message="Preparando recorte de audio"))
            temp_dir = TemporaryDirectory()
            working_audio = Path(temp_dir.name) / "selection.wav"
            extract_audio_range(
                resolve_ffmpeg_path(config.ffmpeg_path),
                audio_path,
                working_audio,
                config.start_seconds,
                config.end_seconds,
            )
        if config.normalize_audio:
            raise_if_cancelled(cancelled)
            reporter(ProgressEvent(stage="normalization", message="Normalizando audio para voz humana"))
            if temp_dir is None:
                temp_dir = TemporaryDirectory()
            normalized_audio = Path(temp_dir.name) / "normalized.wav"
            normalization_duration = None
            if config.start_seconds is not None or config.end_seconds is not None:
                end = config.end_seconds
                if end is not None:
                    normalization_duration = max(0.0, end - offset_seconds)
            normalize_audio_for_speech(
                resolve_ffmpeg_path(config.ffmpeg_path),
                working_audio,
                normalized_audio,
                progress=reporter,
                duration_seconds=normalization_duration,
            )
            working_audio = normalized_audio

        reporter(ProgressEvent(stage="transcription", message="Transcribiendo audio"))
        transcript = offset_transcript(
            transcribe_audio(
                working_audio,
                config,
                _offset_progress(reporter, offset_seconds),
                cancelled,
            ),
            offset_seconds,
        )
        raise_if_cancelled(cancelled)
        reporter(
            ProgressEvent(
                stage="transcription_done",
                message="Transcripcion completada",
                segments=len(transcript),
                text_chars=sum(len(segment.text.strip()) for segment in transcript),
            )
        )
        raw_turns = turns_from_transcript(transcript)
        reporter(ProgressEvent(stage="export", message="Guardando transcripcion bruta"))
        write_all_exports(output_dir, raw_turns, basename="transcript_raw")
        raise_if_cancelled(cancelled)
        reporter(ProgressEvent(stage="diarization", message="Separando hablantes"))
        try:
            diarization = offset_diarization(
                diarize_audio(working_audio, config, diarization_pipeline, reporter),
                offset_seconds,
            )
        except Exception:
            reporter(
                ProgressEvent(
                    stage="export",
                    message="Guardando transcripcion sin diarizar por fallo al separar hablantes",
                )
            )
            write_all_exports(output_dir, raw_turns)
            raise
        raise_if_cancelled(cancelled)
        reporter(
            ProgressEvent(
                stage="diarization_done",
                message="Diarizacion completada",
                speakers=len({segment.speaker for segment in diarization}),
                segments=len(diarization),
            )
        )
        reporter(ProgressEvent(stage="alignment", message="Alineando texto con hablantes"))
        turns = assign_speakers(transcript, diarization)
        raise_if_cancelled(cancelled)
        reporter(
            ProgressEvent(
                stage="alignment_done",
                message="Hablantes detectados",
                speakers=len({turn.speaker for turn in turns}),
                segments=len(turns),
                summary=format_speaker_summary(turns),
            )
        )
        reporter(ProgressEvent(stage="export", message="Guardando resultados"))
        write_all_exports(output_dir, turns)
        if config.export_speaker_audio:
            raise_if_cancelled(cancelled)
            reporter(ProgressEvent(stage="speaker_audio", message="Extrayendo audios por hablante"))
            ffmpeg_path = resolve_ffmpeg_path(config.ffmpeg_path)
            export_speaker_audio(ffmpeg_path, audio_path, turns, output_dir / "speaker_audio")
        reporter(ProgressEvent(stage="done", message="Terminado", summary=format_speaker_summary(turns)))
        return turns
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def offset_transcript(
    transcript: Iterable[TranscriptSegment],
    offset_seconds: float,
) -> list[TranscriptSegment]:
    if offset_seconds == 0:
        return list(transcript)
    return [
        TranscriptSegment(
            start=segment.start + offset_seconds,
            end=segment.end + offset_seconds,
            text=segment.text,
            words=tuple(
                type(word)(
                    start=word.start + offset_seconds,
                    end=word.end + offset_seconds,
                    text=word.text,
                )
                for word in segment.words
            ),
        )
        for segment in transcript
    ]


def turns_from_transcript(
    transcript: Iterable[TranscriptSegment],
    *,
    speaker: str = "Sin diarizar",
) -> list[ConversationTurn]:
    return [
        ConversationTurn(
            start=segment.start,
            end=segment.end,
            speaker=speaker,
            text=segment.text.strip(),
        )
        for segment in transcript
        if segment.text.strip()
    ]


def offset_diarization(
    diarization: Iterable[DiarizationSegment],
    offset_seconds: float,
) -> list[DiarizationSegment]:
    if offset_seconds == 0:
        return list(diarization)
    return [
        DiarizationSegment(
            start=segment.start + offset_seconds,
            end=segment.end + offset_seconds,
            speaker=segment.speaker,
        )
        for segment in diarization
    ]


def _offset_progress(reporter: ProgressCallback, offset_seconds: float) -> ProgressCallback:
    if offset_seconds == 0:
        return reporter

    def report(event: ProgressEvent) -> None:
        if event.stage in ("transcription", "transcription_segment") and event.seconds is not None:
            reporter(
                ProgressEvent(
                    stage=event.stage,
                    message=event.message,
                    seconds=event.seconds + offset_seconds,
                    duration_seconds=event.duration_seconds + offset_seconds
                    if event.duration_seconds is not None
                    else None,
                    elapsed_seconds=event.elapsed_seconds,
                    text_chars=event.text_chars,
                    segments=event.segments,
                    speakers=event.speakers,
                    summary=event.summary,
                    text=event.text,
                    completed=event.completed,
                    total=event.total,
                )
            )
            return
        reporter(event)

    return report


def _speaker_for_segment(
    transcript: TranscriptSegment,
    diarization: list[DiarizationSegment],
) -> str:
    overlaps: dict[str, float] = defaultdict(float)
    for diarized in diarization:
        overlap = min(transcript.end, diarized.end) - max(transcript.start, diarized.start)
        if overlap > 0:
            overlaps[diarized.speaker] += overlap

    if not overlaps:
        return "UNKNOWN"

    return max(overlaps.items(), key=lambda item: item[1])[0]


def _speaker_for_time(
    seconds: float,
    diarization: Iterable[DiarizationSegment],
) -> str | None:
    closest: DiarizationSegment | None = None
    closest_distance: float | None = None
    for segment in diarization:
        if segment.start <= seconds <= segment.end:
            return segment.speaker
        distance = min(abs(seconds - segment.start), abs(seconds - segment.end))
        if closest_distance is None or distance < closest_distance:
            closest = segment
            closest_distance = distance
    if closest is not None and closest_distance is not None and closest_distance <= 0.5:
        return closest.speaker
    return None


def _append_word(text: str, word: str) -> str:
    if not text:
        return word
    if word[:1] in ".,;:!?%)]}»”":
        return f"{text}{word}"
    if text[-1:] in "([{¿¡":
        return f"{text}{word}"
    return f"{text} {word}"


def _ignore_progress(_message: ProgressEvent) -> None:
    return None
