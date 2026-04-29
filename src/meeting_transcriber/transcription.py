from __future__ import annotations

from pathlib import Path
import time
from typing import Callable

from meeting_transcriber.cancellation import CancelCheck, raise_if_cancelled
from meeting_transcriber.cuda_runtime import configure_cuda_runtime
from meeting_transcriber.diarization_quality import uses_word_alignment
from meeting_transcriber.hf_auth import apply_huggingface_token
from meeting_transcriber.progress import ProgressEvent
from meeting_transcriber.types import ProcessingConfig, TranscriptSegment, TranscriptWord


ProgressCallback = Callable[[ProgressEvent], None]


def transcribe_audio(
    audio_path: Path,
    config: ProcessingConfig,
    progress: ProgressCallback | None = None,
    cancelled: CancelCheck | None = None,
) -> list[TranscriptSegment]:
    try:
        configure_cuda_runtime()
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Install project dependencies before transcribing."
        ) from exc

    try:
        raise_if_cancelled(cancelled)
        apply_huggingface_token(config.huggingface_token)
        model = WhisperModel(
            config.whisper_model,
            device=config.device,
            compute_type=config.compute_type,
            use_auth_token=config.huggingface_token,
        )
        started_at = time.monotonic()
        segments, info = model.transcribe(
            str(audio_path),
            language=config.language,
            vad_filter=True,
            word_timestamps=uses_word_alignment(config.diarization_quality),
        )
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        transcript_segments: list[TranscriptSegment] = []
        text_chars = 0
        for segment in segments:
            raise_if_cancelled(cancelled)
            transcript_segment = TranscriptSegment(
                start=float(segment.start),
                end=float(segment.end),
                text=segment.text,
                words=_segment_words(segment),
            )
            transcript_segments.append(transcript_segment)
            text_chars += len(segment.text.strip())
            if progress is not None:
                progress(
                    ProgressEvent(
                        stage="transcription_segment",
                        message="Segmento transcrito",
                        seconds=transcript_segment.start,
                        text=transcript_segment.text.strip(),
                    )
                )
                progress(
                    ProgressEvent(
                        stage="transcription",
                        message="Transcribiendo audio",
                        seconds=transcript_segment.end,
                        duration_seconds=duration,
                        elapsed_seconds=time.monotonic() - started_at,
                        text_chars=text_chars,
                        segments=len(transcript_segments),
                    )
                )
        return transcript_segments
    except Exception as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "CUDA se quedo sin memoria durante la transcripcion. "
                "Usa 'Probar rendimiento' para autoconfigurar o cambia a CPU / int8."
            ) from exc
        raise


def _segment_words(segment: object) -> tuple[TranscriptWord, ...]:
    raw_words = getattr(segment, "words", None)
    if raw_words is None:
        return ()
    words: list[TranscriptWord] = []
    for word in raw_words:
        text = str(getattr(word, "word", "") or "")
        if not text.strip():
            continue
        words.append(
            TranscriptWord(
                start=float(getattr(word, "start", getattr(segment, "start", 0.0)) or 0.0),
                end=float(getattr(word, "end", getattr(segment, "end", 0.0)) or 0.0),
                text=text,
            )
        )
    return tuple(words)
