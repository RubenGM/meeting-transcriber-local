from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
from pathlib import Path

from meeting_transcriber.exporters import export_markdown_text
from meeting_transcriber.cancellation import CancelCheck, CancelledError, raise_if_cancelled
from meeting_transcriber.speaker_names import speaker_labels
from meeting_transcriber.types import ConversationTurn


class AiRunnerNotFound(RuntimeError):
    pass


class AiRunnerError(RuntimeError):
    pass


def build_speaker_identification_prompt(turns: list[ConversationTurn]) -> str:
    speakers = speaker_labels(turns)
    labels = "\n".join(f"- {speaker}" for speaker in speakers)
    transcript = export_markdown_text(turns)
    return (
        "Quiero identificar los nombres reales de los hablantes de una transcripcion.\n\n"
        "Tarea:\n"
        "1. Lee la transcripcion completa.\n"
        "2. Busca pistas explicitas de presentacion, por ejemplo 'me llamo', 'em dic', "
        "'jo soc', 'soc la/el', cargos, turnos de saludo o intervenciones posteriores.\n"
        "3. Devuelve SOLO un JSON valido, sin explicaciones fuera del JSON.\n"
        "4. Si no hay evidencia suficiente para un hablante, usa null.\n"
        "5. No inventes nombres. Si un mismo bloque contiene varias personas porque la "
        "diarizacion fallo, explicalo en evidence y deja el nombre en null o el mas probable.\n\n"
        "Formato exacto:\n"
        "{\n"
        '  "speakers": {\n'
        '    "Persona 1": {"name": "Nombre Apellidos", "confidence": "alta|media|baja", "evidence": "frase breve"},\n'
        '    "Persona 2": {"name": null, "confidence": "baja", "evidence": "motivo"}\n'
        "  }\n"
        "}\n\n"
        "Hablantes actuales:\n"
        f"{labels}\n\n"
        "Transcripcion:\n"
        f"{transcript}"
    )


def run_speaker_identification_ai(
    turns: list[ConversationTurn],
    output_dir: str,
    cancelled: CancelCheck | None = None,
) -> str:
    raise_if_cancelled(cancelled)
    prompt = build_speaker_identification_prompt(turns)
    prompt_path = _write_prompt_file(prompt, output_dir)
    opencode = shutil.which("opencode")
    if opencode:
        return _run_opencode(opencode, prompt_path, output_dir, cancelled)

    codex = shutil.which("codex")
    if codex:
        return _run_codex(codex, prompt, output_dir, cancelled)

    raise AiRunnerNotFound("No encuentro opencode ni codex en PATH.")


def has_ai_runner() -> bool:
    return shutil.which("opencode") is not None or shutil.which("codex") is not None


def parse_speaker_mapping_response(text: str) -> dict[str, str]:
    payload = json.loads(_extract_json_text(text))
    speakers = payload.get("speakers") if isinstance(payload, dict) else None
    if not isinstance(speakers, dict):
        raise ValueError("La respuesta debe contener un objeto JSON 'speakers'.")

    mapping: dict[str, str] = {}
    for speaker, value in speakers.items():
        name: object
        if isinstance(value, dict):
            name = value.get("name")
        else:
            name = value
        if isinstance(name, str) and name.strip():
            mapping[str(speaker)] = name.strip()
    return mapping


def _write_prompt_file(prompt: str, output_dir: str) -> Path:
    path = Path(output_dir) / "speaker_names_prompt.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt, encoding="utf-8")
    return path


def _run_opencode(
    executable: str,
    prompt_path: Path,
    output_dir: str,
    cancelled: CancelCheck | None,
) -> str:
    return _run_cancellable(
        [
            executable,
            "run",
            "Lee el archivo adjunto y responde SOLO con el JSON solicitado.",
            "--dir",
            output_dir,
            "--format",
            "default",
            "--file",
            str(prompt_path),
        ],
        None,
        cancelled,
        "opencode",
    )


def _run_codex(executable: str, prompt: str, output_dir: str, cancelled: CancelCheck | None) -> str:
    return _run_cancellable(
        [
            executable,
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "-C",
            output_dir,
            "-",
        ],
        prompt,
        cancelled,
        "codex",
    )


def _run_cancellable(
    command: list[str],
    stdin_text: str | None,
    cancelled: CancelCheck | None,
    label: str,
) -> str:
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if stdin_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    result: dict[str, object] = {}

    def communicate() -> None:
        try:
            stdout, stderr = process.communicate(input=stdin_text)
        except BaseException as exc:  # pragma: no cover - defensive subprocess bridge
            result["error"] = exc
            return
        result["stdout"] = stdout
        result["stderr"] = stderr

    worker = threading.Thread(target=communicate, daemon=True)
    worker.start()
    try:
        while worker.is_alive():
            if cancelled is not None and cancelled():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                worker.join(timeout=1)
                raise CancelledError("Deteccion IA cancelada por el usuario.")
            worker.join(timeout=0.5)
    except Exception:
        if process.poll() is None:
            process.kill()
        raise
    if "error" in result:
        raise result["error"]  # type: ignore[misc]

    stdout = str(result.get("stdout", ""))
    stderr = str(result.get("stderr", ""))

    if process.returncode != 0:
        detail = stderr.strip() or stdout.strip() or f"{label} fallo sin mensaje"
        raise AiRunnerError(detail)
    return stdout


def _extract_json_text(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        return fenced.group(1)
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        return stripped[first : last + 1]
    return stripped
