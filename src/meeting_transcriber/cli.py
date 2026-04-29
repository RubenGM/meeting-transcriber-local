from __future__ import annotations

import argparse
from pathlib import Path

from meeting_transcriber.ffmpeg import resolve_ffmpeg_path
from meeting_transcriber.languages import code_from_display_name
from meeting_transcriber.pipeline import process_meeting
from meeting_transcriber.time_range import parse_optional_timestamp, validate_time_range
from meeting_transcriber.types import ProcessingConfig
from meeting_transcriber.diarization_models import DEFAULT_DIARIZATION_MODEL
from meeting_transcriber.diarization_quality import DEFAULT_DIARIZATION_QUALITY
from meeting_transcriber.whisper_models import DEFAULT_WHISPER_MODEL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--diarization-model", default=DEFAULT_DIARIZATION_MODEL)
    parser.add_argument("--diarization-quality", default=DEFAULT_DIARIZATION_QUALITY)
    parser.add_argument("--huggingface-token")
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--language", default="es")
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--speaker-audio", action="store_true")
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    args = parser.parse_args()
    start_seconds = parse_optional_timestamp(args.start)
    end_seconds = parse_optional_timestamp(args.end)
    validate_time_range(start_seconds, end_seconds)

    config = ProcessingConfig(
        whisper_model=args.whisper_model,
        diarization_model=args.diarization_model,
        huggingface_token=args.huggingface_token,
        ffmpeg_path=resolve_ffmpeg_path(args.ffmpeg),
        language=code_from_display_name(args.language),
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        device=args.device,
        compute_type=args.compute_type,
        export_speaker_audio=args.speaker_audio,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        diarization_quality=args.diarization_quality,
    )
    process_meeting(args.audio, args.output, config, print)


if __name__ == "__main__":
    main()
