from __future__ import annotations

import re
import subprocess
from pathlib import Path

from meeting_transcriber.types import ConversationTurn, SpeakerAudioSegment, SpeakerExtractPlan


def build_speaker_extract_plan(
    source_audio: Path,
    turns: list[ConversationTurn],
    output_dir: Path,
) -> dict[str, SpeakerExtractPlan]:
    segments_by_speaker: dict[str, list[SpeakerAudioSegment]] = {}
    for turn in turns:
        if turn.end <= turn.start:
            continue
        segments_by_speaker.setdefault(turn.speaker, []).append(
            SpeakerAudioSegment(start=turn.start, end=turn.end)
        )

    return {
        speaker: SpeakerExtractPlan(
            source_audio=source_audio,
            output_path=output_dir / f"{_safe_filename(speaker)}.wav",
            segments=tuple(segments),
        )
        for speaker, segments in segments_by_speaker.items()
    }


def export_speaker_audio(
    ffmpeg_path: Path,
    source_audio: Path,
    turns: list[ConversationTurn],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_speaker_extract_plan(source_audio, turns, output_dir)
    exported: dict[str, Path] = {}
    for speaker, speaker_plan in plan.items():
        if not speaker_plan.segments:
            continue
        _export_one_speaker(ffmpeg_path, speaker_plan, output_dir / f"{_safe_filename(speaker)}_parts")
        exported[speaker] = speaker_plan.output_path
    return exported


def extract_audio_range(
    ffmpeg_path: Path,
    source_audio: Path,
    output_audio: Path,
    start_seconds: float | None,
    end_seconds: float | None,
) -> Path:
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    command = [str(ffmpeg_path), "-y"]
    if start_seconds is not None:
        command.extend(["-ss", f"{start_seconds:.3f}"])
    if end_seconds is not None:
        if start_seconds is None:
            command.extend(["-to", f"{end_seconds:.3f}"])
        else:
            command.extend(["-t", f"{end_seconds - start_seconds:.3f}"])
    command.extend(
        [
            "-i",
            str(source_audio),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_audio),
        ]
    )
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_audio


def probe_audio_duration(ffmpeg_path: Path, source_audio: Path) -> float | None:
    command = [str(ffmpeg_path), "-i", str(source_audio)]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    return _parse_ffmpeg_duration(result.stderr)


def _parse_ffmpeg_duration(stderr: str) -> float | None:
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    if match is None:
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def _export_one_speaker(ffmpeg_path: Path, plan: SpeakerExtractPlan, work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    part_paths: list[Path] = []
    for index, segment in enumerate(plan.segments, start=1):
        part_path = work_dir / f"part_{index:04d}.wav"
        duration = segment.end - segment.start
        command = [
            str(ffmpeg_path),
            "-y",
            "-ss",
            f"{segment.start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(plan.source_audio),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(part_path),
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        part_paths.append(part_path)

    concat_file = work_dir / "concat.txt"
    concat_file.write_text(
        "".join(f"file '{path.as_posix()}'\n" for path in part_paths),
        encoding="utf-8",
    )
    concat_command = [
        str(ffmpeg_path),
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        str(plan.output_path),
    ]
    subprocess.run(concat_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    if cleaned == "":
        raise ValueError("Speaker name cannot produce an empty file name")
    return cleaned
