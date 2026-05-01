from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

from meeting_transcriber.exporters import write_all_exports
from meeting_transcriber.progress import format_seconds
from meeting_transcriber.speaker_identity_resolver import DECISIONS_FILENAME
from meeting_transcriber.types import ConversationTurn


@dataclass(frozen=True)
class SimpleChunkReport:
    output_dir: Path
    start_seconds: float
    end_seconds: float
    turns: tuple[ConversationTurn, ...]


@dataclass(frozen=True)
class SimpleFinalArtifacts:
    output_dir: Path
    transcript_path: Path
    report_path: Path
    turns: tuple[ConversationTurn, ...]
    normalized_audio_path: Path | None = None


def build_simple_final_output_dir(base_output_dir: Path, audio_path: Path) -> Path:
    return base_output_dir / _safe_path_part(audio_path.stem) / "final"


def write_simple_final_artifacts(
    *,
    audio_path: Path,
    base_output_dir: Path,
    chunks: list[SimpleChunkReport],
    chunks_failed: int,
    normalized_audio_path: Path | None = None,
) -> SimpleFinalArtifacts:
    output_dir = build_simple_final_output_dir(base_output_dir, audio_path)
    turns = tuple(combine_chunk_turns(chunks))
    write_all_exports(output_dir, list(turns))
    report_path = output_dir / "report.html"
    report_path.write_text(
        render_simple_html_report(
            audio_path=audio_path,
            chunks=chunks,
            turns=list(turns),
            chunks_failed=chunks_failed,
            output_dir=output_dir,
            normalized_audio_path=normalized_audio_path,
        ),
        encoding="utf-8",
    )
    return SimpleFinalArtifacts(
        output_dir=output_dir,
        transcript_path=output_dir / "transcript.md",
        report_path=report_path,
        turns=turns,
        normalized_audio_path=normalized_audio_path,
    )


def combine_chunk_turns(chunks: list[SimpleChunkReport]) -> list[ConversationTurn]:
    ordered = sorted(chunks, key=lambda chunk: (chunk.start_seconds, chunk.end_seconds))
    combined: list[ConversationTurn] = []
    for index, chunk in enumerate(ordered):
        valid_start = chunk.start_seconds
        valid_end = chunk.end_seconds
        if index > 0:
            previous = ordered[index - 1]
            valid_start = max(valid_start, (previous.end_seconds + chunk.start_seconds) / 2.0)
        if index + 1 < len(ordered):
            following = ordered[index + 1]
            valid_end = min(valid_end, (chunk.end_seconds + following.start_seconds) / 2.0)
        for turn in chunk.turns:
            midpoint = (turn.start + turn.end) / 2.0
            if midpoint < valid_start or midpoint >= valid_end:
                continue
            if index > 0 and abs(midpoint - valid_start) < 0.001:
                continue
            if combined and _same_turn(combined[-1], turn):
                continue
            combined.append(turn)
    return combined


def render_simple_html_report(
    *,
    audio_path: Path,
    chunks: list[SimpleChunkReport],
    turns: list[ConversationTurn],
    chunks_failed: int,
    output_dir: Path,
    normalized_audio_path: Path | None = None,
) -> str:
    speaker_seconds: dict[str, float] = {}
    speaker_turns: dict[str, int] = {}
    for turn in turns:
        speaker_seconds[turn.speaker] = speaker_seconds.get(turn.speaker, 0.0) + max(0.0, turn.end - turn.start)
        speaker_turns[turn.speaker] = speaker_turns.get(turn.speaker, 0) + 1
    decision_rows = _identity_decision_rows(chunks)
    transcript_rows = "\n".join(_transcript_row(index, turn) for index, turn in enumerate(turns))
    speaker_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(speaker)}</td>"
        f"<td>{html.escape(format_seconds(speaker_seconds[speaker]))}</td>"
        f"<td>{speaker_turns[speaker]}</td>"
        "</tr>"
        for speaker in sorted(speaker_seconds)
    )
    chunk_rows = "\n".join(
        "<tr>"
        f"<td>{index}</td>"
        f"<td>{html.escape(format_seconds(chunk.start_seconds))} - {html.escape(format_seconds(chunk.end_seconds))}</td>"
        f"<td>{len(chunk.turns)}</td>"
        f"<td>{html.escape(str(chunk.output_dir))}</td>"
        "</tr>"
        for index, chunk in enumerate(chunks, start=1)
    )
    normalized_section = _normalized_audio_section(output_dir, audio_path, normalized_audio_path)
    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Informe de transcripcion</title>
  <style>
    :root {{ color-scheme: light; --bg:#f5f7f8; --surface:#ffffff; --text:#172026; --muted:#5d6972; --line:#d9e0e4; --accent:#22577a; --accent-strong:#17415d; --focus:#d98b2b; }}
    body {{ margin:0; font-family: Arial, sans-serif; background:var(--bg); color:var(--text); line-height:1.45; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 28px; }}
    header {{ margin-bottom: 20px; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    .muted {{ color: var(--muted); }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:12px; margin: 18px 0; }}
    .metric, section {{ background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:16px; }}
    .metric strong {{ display:block; font-size:24px; margin-bottom:4px; }}
    section {{ margin: 14px 0; overflow:auto; }}
    table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    th, td {{ border-bottom:1px solid var(--line); padding:8px; text-align:left; vertical-align:top; }}
    th {{ color:var(--muted); font-weight:600; }}
    code {{ background:#edf2f5; padding:2px 4px; border-radius:4px; }}
    .transcript-toolbar {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; justify-content:space-between; margin-bottom:12px; }}
    .review-actions {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; }}
    button, .button {{ min-height:36px; border:1px solid var(--line); border-radius:6px; padding:7px 11px; background:#fff; color:var(--text); font:inherit; cursor:pointer; text-decoration:none; display:inline-flex; align-items:center; gap:6px; }}
    button:hover, .button:hover {{ border-color:var(--accent); color:var(--accent-strong); }}
    button:focus-visible, .button:focus-visible, [contenteditable="true"]:focus {{ outline:3px solid var(--focus); outline-offset:2px; }}
    .primary {{ background:var(--accent); border-color:var(--accent); color:#fff; }}
    .primary:hover {{ background:var(--accent-strong); color:#fff; }}
    .play-button {{ width:36px; min-width:36px; height:36px; justify-content:center; padding:0; font-size:13px; }}
    .stop-button {{ width:36px; min-width:36px; height:36px; justify-content:center; padding:0; font-size:13px; }}
    .stop-button[hidden] {{ display:none; }}
    .time-cell {{ white-space:nowrap; }}
    .time-wrap {{ display:grid; grid-template-columns:36px 36px auto; gap:6px 8px; align-items:center; min-width:210px; }}
    .segment-progress {{ grid-column:1 / -1; width:100%; accent-color:var(--accent); cursor:pointer; }}
    .segment-progress:focus-visible {{ outline:3px solid var(--focus); outline-offset:2px; border-radius:999px; }}
    .speaker-cell {{ min-width:120px; }}
    .text-cell {{ min-width:360px; }}
    [contenteditable="true"] {{ border-radius:6px; transition:background 150ms ease; }}
    [contenteditable="true"]:hover {{ background:#f7fafb; }}
    tr.is-playing {{ background:#edf6fb; }}
    #save-status {{ color:var(--muted); min-height:20px; }}
    @media (max-width: 720px) {{ main {{ padding:16px; }} .text-cell {{ min-width:220px; }} .transcript-toolbar {{ align-items:flex-start; }} .time-wrap {{ min-width:180px; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Informe de transcripcion</h1>
    <div class="muted">{html.escape(str(audio_path))}</div>
    <div class="muted">Salida final: <code>{html.escape(str(output_dir))}</code></div>
  </header>
  <div class="grid">
    <div class="metric"><strong>{len(turns)}</strong>turnos en transcripcion final</div>
    <div class="metric"><strong>{len(speaker_seconds)}</strong>hablantes detectados</div>
    <div class="metric"><strong>{len(chunks)}</strong>porciones completadas</div>
    <div class="metric"><strong>{chunks_failed}</strong>porciones con error</div>
  </div>
  {normalized_section}
  <section><h2>Hablantes</h2><table><thead><tr><th>Nombre</th><th>Tiempo hablado</th><th>Turnos</th></tr></thead><tbody>{speaker_rows}</tbody></table></section>
  <section><h2>Porciones procesadas</h2><table><thead><tr><th>#</th><th>Rango</th><th>Turnos</th><th>Carpeta</th></tr></thead><tbody>{chunk_rows}</tbody></table></section>
  <section><h2>Decisiones de identidad</h2><table><thead><tr><th>Porcion</th><th>Etiqueta</th><th>Nombre final</th><th>Confianza</th><th>Razon</th></tr></thead><tbody>{decision_rows}</tbody></table></section>
  <section>
    <div class="transcript-toolbar">
      <h2>Transcripcion final</h2>
      <div class="review-actions">
        <button type="button" class="primary" id="save-review">Guardar revision</button>
        <button type="button" id="reset-review">Restaurar original</button>
        <a class="button" id="download-md" download="transcript_review.md">MD</a>
        <a class="button" id="download-txt" download="transcript_review.txt">TXT</a>
        <a class="button" id="download-srt" download="transcript_review.srt">SRT</a>
        <a class="button" id="download-json" download="transcript_review.json">JSON</a>
      </div>
    </div>
    <div id="save-status" aria-live="polite"></div>
    <table id="review-table"><thead><tr><th>Tiempo</th><th>Hablante</th><th>Texto</th></tr></thead><tbody>{transcript_rows}</tbody></table>
  </section>
</main>
<script>
{_review_script()}
</script>
</body>
</html>
"""


def _transcript_row(index: int, turn: ConversationTurn) -> str:
    start = f"{turn.start:.3f}"
    end = f"{turn.end:.3f}"
    clock = html.escape(format_seconds(turn.start))
    return (
        f"<tr data-turn-index=\"{index}\" data-start=\"{start}\" data-end=\"{end}\">"
        "<td class=\"time-cell\">"
        "<div class=\"time-wrap\">"
        f"<button type=\"button\" class=\"play-button play-turn\" aria-label=\"Reproducir tramo {clock}\" "
        f"data-start=\"{start}\" data-end=\"{end}\" data-state=\"stopped\">&#9658;</button>"
        f"<button type=\"button\" class=\"stop-button stop-turn\" aria-label=\"Detener tramo {clock}\" hidden>&#9632;</button>"
        f"<span>{clock}</span>"
        f"<input type=\"range\" class=\"segment-progress\" aria-label=\"Posicion del tramo {clock}\" "
        f"min=\"0\" max=\"1000\" value=\"0\" data-start=\"{start}\" data-end=\"{end}\">"
        "</div>"
        "</td>"
        f"<td class=\"speaker-cell\" contenteditable=\"true\" data-field=\"speaker\">{html.escape(turn.speaker)}</td>"
        f"<td class=\"text-cell\" contenteditable=\"true\" data-field=\"text\">{html.escape(turn.text)}</td>"
        "</tr>"
    )


def _normalized_audio_section(output_dir: Path, audio_path: Path, normalized_audio_path: Path | None) -> str:
    audio_label = "Audio normalizado"
    if normalized_audio_path is None:
        href = audio_path.as_uri()
        audio_label = "Audio original"
        note = "No se genero un archivo normalizado; la revision usara el audio original como referencia."
    else:
        try:
            href = normalized_audio_path.relative_to(output_dir).as_posix()
        except ValueError:
            href = normalized_audio_path.as_uri()
        note = "Audio WAV mono 16 kHz con reduccion de ruido, filtrado de voz y normalizacion de volumen."
    try:
        link_label = Path(href).name if normalized_audio_path is not None else audio_path.name
    except ValueError:
        link_label = href
    return (
        f"<section><h2>{audio_label}</h2>"
        f"<p class=\"muted\">{html.escape(note)}</p>"
        f"<audio id=\"review-audio\" controls preload=\"metadata\" src=\"{html.escape(href)}\" style=\"width:100%\"></audio>"
        f"<p><a href=\"{html.escape(href)}\">Abrir {html.escape(link_label)}</a></p>"
        "</section>"
    )


def _review_script() -> str:
    return r"""
const storageKey = `meeting-transcriber-review:${location.pathname}`;
const rows = Array.from(document.querySelectorAll('#review-table tbody tr[data-turn-index]'));
const audio = document.getElementById('review-audio');
const statusNode = document.getElementById('save-status');
let stopAt = null;
let activeRow = null;
let activeButton = null;
let autoSaveTimer = null;
let downloadUrls = [];

function clock(seconds, millis = false) {
  const safe = Math.max(0, Number(seconds) || 0);
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const secs = Math.floor(safe % 60);
  const base = [hours, minutes, secs].map(value => String(value).padStart(2, '0')).join(':');
  if (!millis) return base;
  const ms = Math.floor((safe - Math.floor(safe)) * 1000);
  return `${base},${String(ms).padStart(3, '0')}`;
}

function collectTurns() {
  return rows.map(row => ({
    start: Number(row.dataset.start),
    end: Number(row.dataset.end),
    speaker: row.querySelector('[data-field="speaker"]').innerText.trim(),
    text: row.querySelector('[data-field="text"]').innerText.trim(),
  }));
}

function markdown(turns) {
  return turns.map(turn => `[${clock(turn.start)}] **${turn.speaker}:** ${turn.text}`).join('\n\n') + '\n';
}

function plainText(turns) {
  return turns.map(turn => `[${clock(turn.start)}] ${turn.speaker}: ${turn.text}`).join('\n') + '\n';
}

function srt(turns) {
  return turns.map((turn, index) => `${index + 1}\n${clock(turn.start, true)} --> ${clock(turn.end, true)}\n${turn.speaker}: ${turn.text}\n`).join('\n');
}

function jsonText(turns) {
  return JSON.stringify({ turns }, null, 2) + '\n';
}

function setDownload(id, filename, content, type) {
  const link = document.getElementById(id);
  if (!link) return;
  const url = URL.createObjectURL(new Blob([content], { type }));
  downloadUrls.push(url);
  link.href = url;
  link.download = filename;
}

function refreshDownloads() {
  downloadUrls.forEach(url => URL.revokeObjectURL(url));
  downloadUrls = [];
  const turns = collectTurns();
  setDownload('download-md', 'transcript_review.md', markdown(turns), 'text/markdown;charset=utf-8');
  setDownload('download-txt', 'transcript_review.txt', plainText(turns), 'text/plain;charset=utf-8');
  setDownload('download-srt', 'transcript_review.srt', srt(turns), 'text/plain;charset=utf-8');
  setDownload('download-json', 'transcript_review.json', jsonText(turns), 'application/json;charset=utf-8');
}

function saveReview(showStatus = true) {
  localStorage.setItem(storageKey, jsonText(collectTurns()));
  refreshDownloads();
  if (showStatus && statusNode) {
    statusNode.textContent = `Revision guardada en este navegador a las ${new Date().toLocaleTimeString()}.`;
  }
}

function loadReview() {
  const stored = localStorage.getItem(storageKey);
  if (!stored) {
    refreshDownloads();
    return;
  }
  try {
    const payload = JSON.parse(stored);
    (payload.turns || []).forEach((turn, index) => {
      const row = rows[index];
      if (!row) return;
      row.querySelector('[data-field="speaker"]').innerText = turn.speaker || '';
      row.querySelector('[data-field="text"]').innerText = turn.text || '';
    });
    refreshDownloads();
    if (statusNode) statusNode.textContent = 'Revision anterior cargada.';
  } catch {
    refreshDownloads();
  }
}

function scheduleAutoSave() {
  window.clearTimeout(autoSaveTimer);
  autoSaveTimer = window.setTimeout(() => saveReview(false), 500);
}

function buttonFor(row) {
  return row ? row.querySelector('.play-turn') : null;
}

function stopButtonFor(row) {
  return row ? row.querySelector('.stop-turn') : null;
}

function progressFor(row) {
  return row ? row.querySelector('.segment-progress') : null;
}

function rowTimeFromProgress(row) {
  const progress = progressFor(row);
  if (!progress) return Number(row.dataset.start);
  const start = Number(row.dataset.start);
  const end = Number(row.dataset.end);
  const ratio = Number(progress.value) / Number(progress.max || 1000);
  return start + Math.max(0, Math.min(1, ratio)) * Math.max(0, end - start);
}

function updateProgress(row, currentTime) {
  const progress = progressFor(row);
  if (!progress) return;
  const start = Number(row.dataset.start);
  const end = Number(row.dataset.end);
  const span = Math.max(0.001, end - start);
  const ratio = Math.max(0, Math.min(1, (currentTime - start) / span));
  progress.value = String(Math.round(ratio * Number(progress.max || 1000)));
}

function setPlaybackState(row, state) {
  const play = buttonFor(row);
  const stop = stopButtonFor(row);
  if (!play) return;
  play.dataset.state = state;
  if (state === 'playing') {
    play.innerHTML = '&#10073;&#10073;';
    play.setAttribute('aria-label', `Pausar tramo ${row.querySelector('.time-wrap span')?.textContent || ''}`.trim());
    if (stop) stop.hidden = false;
    row.classList.add('is-playing');
  } else if (state === 'paused') {
    play.innerHTML = '&#9658;';
    play.setAttribute('aria-label', `Continuar tramo ${row.querySelector('.time-wrap span')?.textContent || ''}`.trim());
    if (stop) stop.hidden = false;
    row.classList.add('is-playing');
  } else {
    play.innerHTML = '&#9658;';
    play.setAttribute('aria-label', `Reproducir tramo ${row.querySelector('.time-wrap span')?.textContent || ''}`.trim());
    if (stop) stop.hidden = true;
    updateProgress(row, Number(row.dataset.start));
    row.classList.remove('is-playing');
  }
}

function clearPlaybackState() {
  rows.forEach(row => setPlaybackState(row, 'stopped'));
  activeRow = null;
  activeButton = null;
  stopAt = null;
}

function stopPlayback() {
  if (audio) audio.pause();
  clearPlaybackState();
}

document.addEventListener('click', event => {
  const button = event.target.closest('.play-turn');
  if (!button || !audio) return;
  const row = button.closest('tr');
  if (!row) return;
  if (activeRow === row && button.dataset.state === 'playing') {
    audio.pause();
    setPlaybackState(row, 'paused');
    return;
  }
  if (activeRow === row && button.dataset.state === 'paused') {
    audio.play();
    setPlaybackState(row, 'playing');
    return;
  }
  const targetTime = rowTimeFromProgress(row);
  clearPlaybackState();
  activeRow = row;
  activeButton = button;
  stopAt = Number(button.dataset.end);
  audio.currentTime = targetTime;
  updateProgress(row, targetTime);
  audio.play();
  setPlaybackState(row, 'playing');
});

document.addEventListener('click', event => {
  const button = event.target.closest('.stop-turn');
  if (!button) return;
  stopPlayback();
});

document.addEventListener('input', event => {
  const progress = event.target.closest('.segment-progress');
  if (!progress || !audio) return;
  const row = progress.closest('tr');
  if (!row) return;
  const targetTime = rowTimeFromProgress(row);
  if (activeRow !== row) {
    if (!audio.paused) audio.pause();
    clearPlaybackState();
    activeRow = row;
    activeButton = buttonFor(row);
    stopAt = Number(row.dataset.end);
    setPlaybackState(row, 'paused');
  }
  audio.currentTime = targetTime;
  updateProgress(row, targetTime);
});

if (audio) {
  audio.addEventListener('timeupdate', () => {
    if (activeRow) updateProgress(activeRow, audio.currentTime);
    if (stopAt !== null && audio.currentTime >= stopAt) {
      audio.pause();
      clearPlaybackState();
    }
  });
  audio.addEventListener('pause', () => {
    if (activeRow && stopAt !== null && audio.currentTime < stopAt) {
      setPlaybackState(activeRow, 'paused');
    }
  });
  audio.addEventListener('play', () => {
    if (activeRow) setPlaybackState(activeRow, 'playing');
  });
  audio.addEventListener('ended', clearPlaybackState);
}

rows.forEach(row => {
  row.addEventListener('input', scheduleAutoSave);
});

document.getElementById('save-review')?.addEventListener('click', () => saveReview(true));
document.getElementById('reset-review')?.addEventListener('click', () => {
  localStorage.removeItem(storageKey);
  location.reload();
});

loadReview();
"""


def _identity_decision_rows(chunks: list[SimpleChunkReport]) -> str:
    rows: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        decisions_path = chunk.output_dir / DECISIONS_FILENAME
        if not decisions_path.exists():
            continue
        payload = json.loads(decisions_path.read_text(encoding="utf-8"))
        for decision in payload.get("decisions", []):
            rows.append(
                "<tr>"
                f"<td>{index}</td>"
                f"<td>{html.escape(str(decision.get('source_speaker', '')))}</td>"
                f"<td>{html.escape(str(decision.get('resolved_name', '')))}</td>"
                f"<td>{float(decision.get('confidence', 0.0)):.2f}</td>"
                f"<td>{html.escape(str(decision.get('reason', '')))}</td>"
                "</tr>"
            )
    if rows:
        return "\n".join(rows)
    return '<tr><td colspan="5">No hay decisiones de identidad disponibles.</td></tr>'


def _same_turn(left: ConversationTurn, right: ConversationTurn) -> bool:
    return (
        abs(left.start - right.start) < 0.25
        and abs(left.end - right.end) < 0.25
        and left.speaker == right.speaker
        and left.text == right.text
    )


def _safe_path_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_")
    return cleaned or "audio"
