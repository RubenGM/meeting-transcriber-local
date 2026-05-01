from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

from meeting_transcriber.progress import ProgressEvent


DEEPFILTERNET_VERSION = "0.5.6"
DEEPFILTERNET_RELEASE_TAG = f"v{DEEPFILTERNET_VERSION}"
DEEPFILTERNET_RELEASE_BASE_URL = (
    "https://github.com/Rikorose/DeepFilterNet/releases/download"
    f"/{DEEPFILTERNET_RELEASE_TAG}"
)

DEEP_FILTER_ASSETS: dict[tuple[str, str], str] = {
    ("darwin", "arm64"): f"deep-filter-{DEEPFILTERNET_VERSION}-aarch64-apple-darwin",
    ("darwin", "aarch64"): f"deep-filter-{DEEPFILTERNET_VERSION}-aarch64-apple-darwin",
    ("darwin", "x86_64"): f"deep-filter-{DEEPFILTERNET_VERSION}-x86_64-apple-darwin",
    ("linux", "x86_64"): f"deep-filter-{DEEPFILTERNET_VERSION}-x86_64-unknown-linux-musl",
    ("linux", "amd64"): f"deep-filter-{DEEPFILTERNET_VERSION}-x86_64-unknown-linux-musl",
    ("linux", "aarch64"): f"deep-filter-{DEEPFILTERNET_VERSION}-aarch64-unknown-linux-gnu",
    ("linux", "arm64"): f"deep-filter-{DEEPFILTERNET_VERSION}-aarch64-unknown-linux-gnu",
    ("linux", "armv7l"): f"deep-filter-{DEEPFILTERNET_VERSION}-armv7-unknown-linux-gnueabihf",
    ("win32", "amd64"): f"deep-filter-{DEEPFILTERNET_VERSION}-x86_64-pc-windows-msvc.exe",
    ("win32", "x86_64"): f"deep-filter-{DEEPFILTERNET_VERSION}-x86_64-pc-windows-msvc.exe",
    ("cygwin", "amd64"): f"deep-filter-{DEEPFILTERNET_VERSION}-x86_64-pc-windows-msvc.exe",
    ("msys", "amd64"): f"deep-filter-{DEEPFILTERNET_VERSION}-x86_64-pc-windows-msvc.exe",
}

DESKTOP_DEEP_FILTER_ASSETS: tuple[str, ...] = tuple(dict.fromkeys(DEEP_FILTER_ASSETS.values()))


@dataclass(frozen=True)
class DeepFilterNetResult:
    ok: bool
    output_audio: Path | None = None
    detail: str | None = None


ProgressCallback = Callable[[ProgressEvent], None]


def default_deepfilternet_dir(project_dir: Path | None = None) -> Path:
    if project_dir is None:
        project_dir = Path(__file__).resolve().parents[2]
    return project_dir / "models" / "deepfilternet"


def deep_filter_asset_name(platform: str | None = None, machine: str | None = None) -> str:
    platform = _normalize_platform(platform or sys.platform)
    machine = _normalize_machine(machine or _machine())
    try:
        return DEEP_FILTER_ASSETS[(platform, machine)]
    except KeyError as exc:
        raise RuntimeError(f"DeepFilterNet no tiene binario preparado para {platform}/{machine}") from exc


def deep_filter_asset_url(asset_name: str) -> str:
    return f"{DEEPFILTERNET_RELEASE_BASE_URL}/{asset_name}"


def local_deep_filter_path(
    base_dir: Path | None = None,
    *,
    platform: str | None = None,
    machine: str | None = None,
) -> Path:
    asset_name = deep_filter_asset_name(platform, machine)
    return (base_dir or default_deepfilternet_dir()) / asset_name


def local_deep_filter_asset_path(asset_name: str, base_dir: Path | None = None) -> Path:
    return (base_dir or default_deepfilternet_dir()) / asset_name


def resolve_deep_filter_binary(base_dir: Path | None = None) -> Path | None:
    env_path = os.environ.get("DEEP_FILTER_BINARY")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
    path = local_deep_filter_path(base_dir)
    if path.exists():
        return path
    found = shutil.which("deep-filter")
    return Path(found) if found else None


def ensure_deep_filter_binary(base_dir: Path | None = None) -> Path:
    existing = resolve_deep_filter_binary(base_dir)
    if existing is not None:
        return existing
    base_dir = base_dir or default_deepfilternet_dir()
    return ensure_deep_filter_asset(deep_filter_asset_name(), base_dir)


def ensure_deep_filter_asset(asset_name: str, base_dir: Path | None = None) -> Path:
    target = local_deep_filter_asset_path(asset_name, base_dir)
    if target.exists():
        _make_executable(target)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = deep_filter_asset_url(asset_name)
    with urllib.request.urlopen(url, timeout=60) as response:
        target.write_bytes(response.read())
    _make_executable(target)
    return target


def ensure_deep_filter_desktop_binaries(base_dir: Path | None = None) -> tuple[Path, ...]:
    base_dir = base_dir or default_deepfilternet_dir()
    return tuple(ensure_deep_filter_asset(asset_name, base_dir) for asset_name in DESKTOP_DEEP_FILTER_ASSETS)


def enhance_with_deepfilternet(
    *,
    deep_filter_binary: Path,
    ffmpeg_path: Path,
    source_audio: Path,
    output_audio: Path,
    progress: ProgressCallback | None = None,
    duration_seconds: float | None = None,
) -> DeepFilterNetResult:
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as dirname:
        work_dir = Path(dirname)
        input_wav = work_dir / "deepfilter_input.wav"
        enhanced_dir = work_dir / "enhanced"
        enhanced_dir.mkdir()
        try:
            _report_normalization(
                progress,
                "DeepFilterNet: preparando WAV de trabajo",
                completed=1,
                total=4,
                duration_seconds=duration_seconds,
            )
            _convert_for_deepfilter(ffmpeg_path, source_audio, input_wav)
            command = [
                str(deep_filter_binary),
                "--pf",
                "-o",
                str(enhanced_dir),
                str(input_wav),
            ]
            _report_normalization(
                progress,
                "DeepFilterNet: reduciendo ruido y realzando voz",
                completed=2,
                total=4,
                duration_seconds=duration_seconds,
                elapsed_seconds=0.0,
            )
            result = _run_deepfilter_command(command, progress, duration_seconds)
            if result.returncode != 0:
                return DeepFilterNetResult(False, detail=_short_process_detail(result))
            enhanced = _find_enhanced_wav(enhanced_dir)
            if enhanced is None:
                return DeepFilterNetResult(False, detail="DeepFilterNet no genero ningun WAV")
            _report_normalization(
                progress,
                "DeepFilterNet: ajustando volumen final",
                completed=3,
                total=4,
                duration_seconds=duration_seconds,
            )
            _postprocess_enhanced_audio(ffmpeg_path, enhanced, output_audio)
        except Exception as exc:
            return DeepFilterNetResult(False, detail=str(exc))
    _report_normalization(
        progress,
        "Audio normalizado con DeepFilterNet",
        completed=4,
        total=4,
        duration_seconds=duration_seconds,
    )
    return DeepFilterNetResult(True, output_audio=output_audio)


def _run_deepfilter_command(
    command: list[str],
    progress: ProgressCallback | None,
    duration_seconds: float | None,
) -> subprocess.CompletedProcess[str]:
    started = time.monotonic()
    last_report = started
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while process.poll() is None:
        now = time.monotonic()
        if now - last_report >= 5.0:
            _report_normalization(
                progress,
                "DeepFilterNet: reduciendo ruido y realzando voz",
                completed=2,
                total=4,
                duration_seconds=duration_seconds,
                elapsed_seconds=now - started,
            )
            last_report = now
        time.sleep(0.25)
    stdout, stderr = process.communicate()
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _convert_for_deepfilter(ffmpeg_path: Path, source_audio: Path, output_wav: Path) -> None:
    command = [
        str(ffmpeg_path),
        "-y",
        "-i",
        str(source_audio),
        "-ac",
        "1",
        "-ar",
        "48000",
        str(output_wav),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _postprocess_enhanced_audio(ffmpeg_path: Path, source_audio: Path, output_audio: Path) -> None:
    command = [
        str(ffmpeg_path),
        "-y",
        "-i",
        str(source_audio),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        str(output_audio),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _find_enhanced_wav(directory: Path) -> Path | None:
    candidates = sorted(directory.glob("*.wav"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _short_process_detail(result: subprocess.CompletedProcess[str]) -> str:
    text = "\n".join(part for part in (result.stderr, result.stdout) if part)
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped[:500]
    return f"codigo de salida {result.returncode}"


def _report_normalization(
    progress: ProgressCallback | None,
    message: str,
    *,
    completed: int,
    total: int,
    duration_seconds: float | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    if progress is None:
        return
    progress(
        ProgressEvent(
            stage="normalization_progress",
            message=message,
            completed=completed,
            total=total,
            duration_seconds=duration_seconds,
            elapsed_seconds=elapsed_seconds,
        )
    )


def _make_executable(path: Path) -> None:
    if path.suffix.lower() == ".exe":
        return
    current = path.stat().st_mode
    path.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _normalize_platform(value: str) -> str:
    lowered = value.lower()
    if lowered.startswith("linux"):
        return "linux"
    if lowered.startswith("darwin"):
        return "darwin"
    if lowered.startswith("win"):
        return "win32"
    if lowered.startswith("cygwin"):
        return "cygwin"
    if lowered.startswith("msys"):
        return "msys"
    return lowered


def _normalize_machine(value: str) -> str:
    lowered = value.lower()
    aliases = {
        "x64": "x86_64",
        "x86-64": "x86_64",
        "amd64": "amd64",
        "arm64": "arm64",
    }
    return aliases.get(lowered, lowered)


def _machine() -> str:
    try:
        import platform

        return platform.machine()
    except Exception:
        return ""
