from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TranscriptWord:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: tuple[TranscriptWord, ...] = ()


@dataclass(frozen=True)
class DiarizationSegment:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class ConversationTurn:
    start: float
    end: float
    speaker: str
    text: str


@dataclass(frozen=True)
class SpeakerAudioSegment:
    start: float
    end: float


@dataclass(frozen=True)
class SpeakerExtractPlan:
    source_audio: Path
    output_path: Path
    segments: tuple[SpeakerAudioSegment, ...]


@dataclass(frozen=True)
class ProcessingConfig:
    whisper_model: str
    diarization_model: str
    huggingface_token: str | None
    ffmpeg_path: Path | None
    language: str | None
    min_speakers: int | None
    max_speakers: int | None
    device: str
    compute_type: str
    export_speaker_audio: bool
    start_seconds: float | None = None
    end_seconds: float | None = None
    diarization_quality: str = "precise"
