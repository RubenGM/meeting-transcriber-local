from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


_CONFIGURED = False


def configure_cuda_runtime() -> None:
    """Expose bundled NVIDIA runtime libraries to ctranslate2/faster-whisper."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    library_dirs = _nvidia_library_dirs()
    if not library_dirs:
        return

    _prepend_path_env("LD_LIBRARY_PATH", library_dirs)
    _prepend_path_env("PATH", library_dirs)

    if sys.platform.startswith("linux"):
        _preload_linux_libraries(library_dirs)


def _nvidia_library_dirs() -> list[Path]:
    dirs: list[Path] = []
    seen: set[Path] = set()
    for entry in map(Path, sys.path):
        nvidia_dir = entry / "nvidia"
        if not nvidia_dir.is_dir():
            continue
        for lib_dir in sorted(nvidia_dir.glob("*/lib")):
            resolved = lib_dir.resolve()
            if resolved not in seen and resolved.is_dir():
                dirs.append(resolved)
                seen.add(resolved)
    return dirs


def _prepend_path_env(name: str, dirs: list[Path]) -> None:
    existing = os.environ.get(name, "")
    parts = [str(path) for path in dirs]
    if existing:
        parts.append(existing)
    os.environ[name] = os.pathsep.join(parts)


def _preload_linux_libraries(library_dirs: list[Path]) -> None:
    for library_name in (
        "libnvrtc-builtins.so.13.0",
        "libnvrtc.so.13",
        "libcublas.so.13",
        "libcublasLt.so.13",
        "libnvrtc.so.12",
        "libnvrtc-builtins.so.12.9",
        "libcublas.so.12",
        "libcublasLt.so.12",
        "libcudnn.so.9",
    ):
        path = _find_library(library_dirs, library_name)
        if path is None:
            continue
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue


def _find_library(library_dirs: list[Path], library_name: str) -> Path | None:
    for lib_dir in library_dirs:
        path = lib_dir / library_name
        if path.exists():
            return path
    return None
