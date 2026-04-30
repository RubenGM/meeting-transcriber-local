from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import venv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BootstrapPaths:
    project_dir: Path
    venv_dir: Path


def main() -> int:
    paths = BootstrapPaths(
        project_dir=Path(__file__).resolve().parents[1],
        venv_dir=Path(__file__).resolve().parents[1] / ".venv",
    )
    python_path = venv_python_path(paths, sys.platform)
    runtime_env = _runtime_env(paths)
    if not should_run_setup(sys.argv, _environment_ready(python_path, runtime_env)):
        _run([str(python_path), "-m", "meeting_transcriber"], env=runtime_env)
        return 0

    _clear_screen()
    _title("Meeting Transcriber - instalador guiado")
    _say(
        """
        Este asistente prepara todo lo necesario en esta carpeta:

        - crea un entorno Python local en .venv
        - instala las dependencias de la aplicacion
        - instala ffmpeg embebido mediante imageio-ffmpeg
        - instala librerias CUDA locales para faster-whisper si estas en Linux
        - descarga/prepara el modelo Whisper equilibrado
        - arranca la aplicacion al terminar

        No modifica Python global ni instala paquetes fuera de esta carpeta.
        """
    )
    if not _confirm("Continuar con la instalacion/preparacion?"):
        _say("Instalacion cancelada.")
        return 1

    if not python_path.exists():
        _step("Creando entorno Python local")
        venv.EnvBuilder(with_pip=True, clear=False).create(paths.venv_dir)
    else:
        _step("Entorno Python local encontrado")

    _step("Instalando o actualizando dependencias")
    _run(build_pip_install_command(python_path, paths.project_dir))
    if sys.platform.startswith("linux"):
        _step("Instalando librerias CUDA locales para faster-whisper")
        _run(build_cuda_runtime_install_command(python_path))

    _step("Verificando ffmpeg embebido")
    _run(
        [str(python_path), "-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"],
        env=runtime_env,
    )

    _step("Preparando modelo Whisper equilibrado")
    _run(
        [
            str(python_path),
            "-c",
            (
                "from faster_whisper import WhisperModel; "
                "WhisperModel('small', device='cpu', compute_type='int8'); "
                "print('Modelo Whisper listo')"
            ),
        ],
        env=runtime_env,
    )

    _title("Preparacion completada")
    _say("Pulsa Enter para abrir Meeting Transcriber.")
    input()
    _run([str(python_path), "-m", "meeting_transcriber"], env=runtime_env)
    return 0


def venv_python_path(paths: BootstrapPaths, platform: str) -> Path:
    if platform.startswith("win"):
        return paths.venv_dir / "Scripts" / "python.exe"
    return paths.venv_dir / "bin" / "python"


def build_pip_install_command(python_path: Path, project_dir: Path) -> list[str]:
    return [
        str(python_path),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip",
        "wheel",
        "setuptools",
        str(project_dir),
    ]


def build_cuda_runtime_install_command(python_path: Path) -> list[str]:
    return [
        str(python_path),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12",
    ]


def should_run_setup(argv: list[str], environment_ready: bool) -> bool:
    if "--setup" in argv:
        return True
    return not environment_ready


def _environment_ready(python_path: Path, env: dict[str, str]) -> bool:
    if not python_path.exists():
        return False
    cuda_runtime_check = ""
    if sys.platform.startswith("linux"):
        cuda_runtime_check = (
            "import nvidia.cublas.lib; "
            "import nvidia.cudnn.lib; "
            "import nvidia.cuda_nvrtc.lib; "
        )
    command = [
        str(python_path),
        "-c",
        (
            "import meeting_transcriber; "
            "import faster_whisper; "
            "import pyannote.audio; "
            "import imageio_ffmpeg; "
            f"{cuda_runtime_check}"
            "from meeting_transcriber.cuda_runtime import configure_cuda_runtime; "
            "configure_cuda_runtime(); "
            "imageio_ffmpeg.get_ffmpeg_exe()"
        ),
    ]
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _runtime_env(paths: BootstrapPaths, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy() if base_env is None else base_env.copy()
    matplotlib_cache = paths.project_dir / ".cache" / "matplotlib"
    huggingface_cache = paths.project_dir / "models" / "huggingface"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    huggingface_cache.mkdir(parents=True, exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    env.setdefault("HF_HOME", str(huggingface_cache))
    source_dir = str(paths.project_dir / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{source_dir}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else source_dir
    )
    return env


def _run(command: list[str], env: dict[str, str] | None = None) -> None:
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _confirm(question: str) -> bool:
    while True:
        answer = input(f"{question} [S/n]: ").strip().lower()
        if answer in ("", "s", "si", "sí", "y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Responde S para continuar o N para cancelar.")


def _title(text: str) -> None:
    print("=" * len(text))
    print(text)
    print("=" * len(text))
    print()


def _step(text: str) -> None:
    print()
    print(f"> {text}")


def _say(text: str) -> None:
    print(textwrap.dedent(text).strip())
    print()


def _clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


if __name__ == "__main__":
    raise SystemExit(main())
