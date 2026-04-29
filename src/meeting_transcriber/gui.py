from __future__ import annotations

import queue
import threading
import tkinter as tk
import webbrowser
import json
from pathlib import Path
from typing import Any
from tkinter import filedialog, messagebox, ttk

from meeting_transcriber.benchmark import BenchmarkResult, run_transcription_benchmark
from meeting_transcriber.cancellation import CancelledError
from meeting_transcriber.config import UiState, default_config_dir, load_config, load_ui_state, save_config
from meeting_transcriber.diarization_models import (
    DEFAULT_DIARIZATION_MODEL,
    diarization_model_id_from_display_name,
    diarization_model_label_from_id,
    diarization_model_labels,
)
from meeting_transcriber.diarization_quality import (
    DEFAULT_DIARIZATION_QUALITY,
    diarization_quality_id_from_display_name,
    diarization_quality_label_from_id,
    diarization_quality_labels,
)
from meeting_transcriber.ffmpeg import resolve_ffmpeg_path
from meeting_transcriber.external_links import HUGGINGFACE_TOKENS_URL, PYANNOTE_MODEL_URL
from meeting_transcriber.exporters import write_all_exports
from meeting_transcriber.history import HistoryEntry, add_history_entry, load_history
from meeting_transcriber.languages import (
    code_from_display_name,
    display_name_from_code,
    language_display_names,
)
from meeting_transcriber.pipeline import process_meeting
from meeting_transcriber.progress import ProgressEvent, format_progress_event
from meeting_transcriber.speaker_ai import (
    has_ai_runner,
    parse_speaker_mapping_response,
    run_speaker_identification_ai,
)
from meeting_transcriber.speaker_names import rename_speakers, speaker_labels
from meeting_transcriber.time_range import format_optional_range, hms_to_seconds, validate_time_range
from meeting_transcriber.types import ConversationTurn, ProcessingConfig
from meeting_transcriber.whisper_models import (
    DEFAULT_WHISPER_MODEL,
    model_id_from_display_name,
    model_label_from_id,
    whisper_model_labels,
)


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Meeting Transcriber")
        self.geometry("920x720")
        self.minsize(780, 620)
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.config_path = default_config_dir() / "config.json"
        self.history_path = default_config_dir() / "history.json"

        existing = load_config(self.config_path)
        self.ui_state = load_ui_state(self.config_path)
        self.audio_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path.cwd() / "output"))
        self.whisper_model = tk.StringVar(
            value=model_label_from_id(existing.whisper_model if existing else DEFAULT_WHISPER_MODEL)
        )
        self.diarization_model = tk.StringVar(
            value=diarization_model_label_from_id(
                existing.diarization_model if existing else DEFAULT_DIARIZATION_MODEL
            )
        )
        self.diarization_quality = tk.StringVar(
            value=diarization_quality_label_from_id(
                existing.diarization_quality if existing else DEFAULT_DIARIZATION_QUALITY
            )
        )
        self.huggingface_token = tk.StringVar(value=existing.huggingface_token if existing and existing.huggingface_token else "")
        self.language = tk.StringVar(
            value=display_name_from_code(existing.language if existing else "es")
        )
        self.min_speakers = tk.StringVar(
            value=str(existing.min_speakers) if existing and existing.min_speakers is not None else ""
        )
        self.max_speakers = tk.StringVar(
            value=str(existing.max_speakers) if existing and existing.max_speakers is not None else ""
        )
        self.start_hours = tk.StringVar(value="0")
        self.start_minutes = tk.StringVar(value="0")
        self.start_seconds = tk.StringVar(value="0")
        self.end_hours = tk.StringVar(value="0")
        self.end_minutes = tk.StringVar(value="0")
        self.end_seconds = tk.StringVar(value="0")
        self.device = tk.StringVar(value=existing.device if existing else "cpu")
        self.compute_type = tk.StringVar(value=existing.compute_type if existing else "int8")
        self.export_speaker_audio = tk.BooleanVar(
            value=existing.export_speaker_audio if existing else False
        )
        self.status = tk.StringVar(value="Listo")
        self.transcription_progress = tk.StringVar(value="Sin proceso activo")
        self.speaker_progress = tk.StringVar(value="Sin hablantes detectados")
        self.metrics_progress = tk.StringVar(value="Duracion y ETA disponibles al empezar")
        self.last_turns: list[ConversationTurn] = []
        self._auto_speaker_detection_running = False
        self._cancel_event = threading.Event()
        self._active_task: str | None = None
        self._build()
        self.after(200, self._poll_events)

    def _build(self) -> None:
        root = ttk.Frame(self, padding=16)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(1, weight=1)

        self._file_row(root, 0, "Audio", self.audio_path, self._choose_audio)
        self._file_row(root, 1, "Salida", self.output_dir, self._choose_output_dir)
        ttk.Label(root, text="Calidad").grid(row=2, column=0, sticky="w", pady=6)
        ttk.Combobox(
            root,
            textvariable=self.whisper_model,
            values=whisper_model_labels(),
            state="readonly",
            width=22,
        ).grid(row=2, column=1, sticky="w", pady=6)
        ttk.Label(root, text="Diarización").grid(row=3, column=0, sticky="w", pady=6)
        ttk.Combobox(
            root,
            textvariable=self.diarization_model,
            values=diarization_model_labels(),
            state="readonly",
            width=22,
        ).grid(row=3, column=1, sticky="w", pady=6)

        ttk.Label(root, text="Separacion voces").grid(row=4, column=0, sticky="w", pady=6)
        ttk.Combobox(
            root,
            textvariable=self.diarization_quality,
            values=diarization_quality_labels(),
            state="readonly",
            width=22,
        ).grid(row=4, column=1, sticky="w", pady=6)

        ttk.Label(root, text="Idioma").grid(row=5, column=0, sticky="w", pady=6)
        ttk.Combobox(
            root,
            textvariable=self.language,
            values=language_display_names(),
            state="readonly",
            width=22,
        ).grid(row=5, column=1, sticky="w", pady=6)

        speaker_frame = ttk.Frame(root)
        speaker_frame.grid(row=6, column=1, sticky="w", pady=6)
        ttk.Label(root, text="Hablantes").grid(row=6, column=0, sticky="w", pady=6)
        ttk.Label(speaker_frame, text="Min").pack(side=tk.LEFT)
        ttk.Entry(speaker_frame, textvariable=self.min_speakers, width=6).pack(side=tk.LEFT, padx=(6, 16))
        ttk.Label(speaker_frame, text="Max").pack(side=tk.LEFT)
        ttk.Entry(speaker_frame, textvariable=self.max_speakers, width=6).pack(side=tk.LEFT, padx=(6, 0))

        range_frame = ttk.Frame(root)
        range_frame.grid(row=7, column=1, sticky="w", pady=6)
        ttk.Label(root, text="Rango").grid(row=7, column=0, sticky="w", pady=6)
        self._time_selector(range_frame, "Inicio", self.start_hours, self.start_minutes, self.start_seconds)
        self._time_selector(range_frame, "Fin", self.end_hours, self.end_minutes, self.end_seconds)

        runtime = ttk.Frame(root)
        runtime.grid(row=8, column=1, sticky="w", pady=6)
        ttk.Label(root, text="Ejecucion").grid(row=8, column=0, sticky="w", pady=6)
        ttk.Combobox(runtime, textvariable=self.device, values=["cpu", "cuda"], width=10).pack(side=tk.LEFT)
        ttk.Combobox(
            runtime,
            textvariable=self.compute_type,
            values=["int8", "int8_float16", "float16", "float32"],
            width=10,
        ).pack(side=tk.LEFT, padx=8)

        ttk.Checkbutton(
            root,
            text="Exportar audio separado por hablante",
            variable=self.export_speaker_audio,
        ).grid(row=9, column=1, sticky="w", pady=10)

        token_frame = ttk.Frame(root)
        token_frame.grid(row=10, column=1, sticky="ew", pady=6)
        token_frame.columnconfigure(0, weight=1)
        ttk.Label(root, text="Token HF").grid(row=10, column=0, sticky="w", pady=6)
        ttk.Entry(token_frame, textvariable=self.huggingface_token, show="*").grid(
            row=0,
            column=0,
            sticky="ew",
        )
        ttk.Button(
            token_frame,
            text="Abrir modelo pyannote",
            command=lambda: webbrowser.open(PYANNOTE_MODEL_URL),
        ).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(
            token_frame,
            text="Crear token HF",
            command=lambda: webbrowser.open(HUGGINGFACE_TOKENS_URL),
        ).grid(row=0, column=2, padx=(8, 0))

        actions = ttk.Frame(root)
        actions.grid(row=11, column=0, columnspan=3, sticky="ew", pady=(18, 8))
        self.process_button = ttk.Button(actions, text="Procesar", command=self._process)
        self.process_button.pack(side=tk.LEFT)
        self.benchmark_button = ttk.Button(actions, text="Probar rendimiento", command=self._benchmark)
        self.benchmark_button.pack(side=tk.LEFT, padx=(8, 0))
        self.stop_button = ttk.Button(actions, text="Detener", command=self._stop_active_task, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))
        self.rename_button = ttk.Button(
            actions,
            text="Renombrar hablantes",
            command=self._open_speaker_editor,
            state=tk.DISABLED,
        )
        self.rename_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Guardar configuracion", command=self._save_current_config).pack(
            side=tk.LEFT, padx=8
        )

        ttk.Label(root, textvariable=self.status).grid(row=12, column=0, columnspan=3, sticky="w")
        ttk.Label(root, textvariable=self.transcription_progress).grid(
            row=13,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(4, 0),
        )
        ttk.Label(root, textvariable=self.speaker_progress).grid(
            row=14,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(4, 0),
        )
        ttk.Label(root, textvariable=self.metrics_progress).grid(
            row=15,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(4, 0),
        )
        self.busy_bar = ttk.Progressbar(root, mode="indeterminate")
        self.busy_bar.grid(row=16, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        history_frame = ttk.Frame(root)
        history_frame.grid(row=17, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        history_frame.columnconfigure(0, weight=1)
        ttk.Label(history_frame, text="Historial de este audio").grid(row=0, column=0, sticky="w")
        self.history_list = tk.Listbox(history_frame, height=4)
        self.history_list.grid(row=1, column=0, sticky="nsew")
        panes = ttk.PanedWindow(root, orient=tk.VERTICAL)
        panes.grid(row=18, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        root.rowconfigure(18, weight=1)

        preview_frame = ttk.Frame(panes)
        log_frame = ttk.Frame(panes)
        panes.add(preview_frame, weight=3)
        panes.add(log_frame, weight=2)

        ttk.Label(preview_frame, text="Vista previa").pack(anchor="w")
        self.preview = tk.Text(preview_frame, height=9, wrap="word")
        self.preview.pack(fill=tk.BOTH, expand=True)

        ttk.Label(log_frame, text="Registro").pack(anchor="w")
        self.log = tk.Text(log_frame, height=7, wrap="word")
        self.log.pack(fill=tk.BOTH, expand=True)

    def _time_selector(
        self,
        parent: ttk.Frame,
        label: str,
        hours: tk.StringVar,
        minutes: tk.StringVar,
        seconds: tk.StringVar,
    ) -> None:
        if label:
            ttk.Label(parent, text=label).pack(side=tk.LEFT)
        ttk.Spinbox(parent, from_=0, to=99, textvariable=hours, width=3).pack(side=tk.LEFT, padx=(6, 2))
        ttk.Label(parent, text="h").pack(side=tk.LEFT)
        ttk.Spinbox(parent, from_=0, to=59, textvariable=minutes, width=3).pack(side=tk.LEFT, padx=(6, 2))
        ttk.Label(parent, text="m").pack(side=tk.LEFT)
        ttk.Spinbox(parent, from_=0, to=59, textvariable=seconds, width=3).pack(side=tk.LEFT, padx=(6, 2))
        ttk.Label(parent, text="s").pack(side=tk.LEFT)

    def _file_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        command: object,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=6)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8, pady=6)
        ttk.Button(parent, text="Elegir", command=command).grid(row=row, column=2, sticky="e", pady=6)

    def _choose_audio(self) -> None:
        filename = filedialog.askopenfilename(
            title="Selecciona audio",
            initialdir=str(self.ui_state.last_audio_dir) if self.ui_state.last_audio_dir else None,
            filetypes=[("Audio/video", "*.wav *.mp3 *.m4a *.mp4 *.aac *.flac *.ogg"), ("Todos", "*.*")],
        )
        if filename:
            self.audio_path.set(filename)
            self.ui_state = UiState(last_audio_dir=Path(filename).parent)
            self._save_current_config()
            self._refresh_history()

    def _choose_output_dir(self) -> None:
        dirname = filedialog.askdirectory(title="Selecciona carpeta de salida")
        if dirname:
            self.output_dir.set(dirname)

    def _process(self) -> None:
        try:
            config = self._current_config()
        except ValueError as exc:
            messagebox.showerror("Configuracion invalida", str(exc))
            return

        audio_path = Path(self.audio_path.get())
        output_dir = Path(self.output_dir.get())
        if not audio_path.is_file():
            messagebox.showerror(
                "Audio no encontrado",
                "Selecciona primero un archivo de audio valido.",
            )
            return

        self._refresh_history()
        self._save_current_config()
        self._start_task("process")
        self.preview.delete("1.0", tk.END)
        self.log.delete("1.0", tk.END)
        self.speaker_progress.set("Hablantes: se detectaran al terminar la transcripcion")
        self.metrics_progress.set("Preparando analisis")
        self.busy_bar.start(12)
        thread = threading.Thread(
            target=self._run_processing,
            args=(audio_path, output_dir, config),
            daemon=True,
        )
        thread.start()

    def _benchmark(self) -> None:
        try:
            config = self._current_config()
        except ValueError as exc:
            messagebox.showerror("Configuracion invalida", str(exc))
            return

        audio_path = Path(self.audio_path.get())
        if not audio_path.is_file():
            messagebox.showerror(
                "Audio no encontrado",
                "Selecciona primero un archivo de audio valido.",
            )
            return

        self.log.delete("1.0", tk.END)
        self._start_task("benchmark")
        self.status.set("Probando configuraciones de rendimiento")
        self.transcription_progress.set("Benchmark activo")
        self.metrics_progress.set("Se probara un recorte corto con varias opciones")
        self.busy_bar.start(12)
        thread = threading.Thread(
            target=self._run_benchmark,
            args=(audio_path, config),
            daemon=True,
        )
        thread.start()

    def _run_processing(self, audio_path: Path, output_dir: Path, config: ProcessingConfig) -> None:
        try:
            turns = process_meeting(
                audio_path,
                output_dir,
                config,
                self._report_progress,
                self._cancel_event.is_set,
            )
            self.events.put(("done", (f"Resultados guardados en {output_dir}", turns)))
        except CancelledError as exc:
            self.events.put(("cancelled", str(exc)))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _run_benchmark(self, audio_path: Path, config: ProcessingConfig) -> None:
        try:
            result = run_transcription_benchmark(
                audio_path,
                config,
                progress=self._report_progress,
                cancelled=self._cancel_event.is_set,
            )
            self.events.put(("benchmark_done", result))
        except CancelledError as exc:
            self.events.put(("cancelled", str(exc)))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _report_progress(self, event: ProgressEvent) -> None:
        self.events.put(("progress", event))

    def _poll_events(self) -> None:
        while not self.events.empty():
            kind, payload = self.events.get()
            if kind == "speaker_ai_done" and isinstance(payload, str):
                self._auto_speaker_detection_running = False
                self._finish_task()
                self.speaker_progress.set("IA ha propuesto nombres de hablantes")
                self.status.set("Revisa los nombres propuestos por la IA")
                self.log.insert(tk.END, "IA ha propuesto nombres de hablantes\n")
                self.log.see(tk.END)
                self._open_speaker_editor(ai_response=payload)
                continue
            if kind == "speaker_ai_error":
                self._auto_speaker_detection_running = False
                self._finish_task()
                self.status.set("Deteccion IA de hablantes no disponible")
                self.log.insert(tk.END, f"Deteccion IA de hablantes no disponible: {payload}\n")
                self.log.see(tk.END)
                continue
            message = self._message_from_payload(payload)
            self.status.set(_short_status(message))
            self._update_progress_labels(payload)
            self._update_preview(payload)
            if not (isinstance(payload, ProgressEvent) and payload.stage == "transcription_segment"):
                self.log.insert(tk.END, message + "\n")
                self.log.see(tk.END)
            if kind in ("done", "error", "cancelled"):
                self._finish_task()
            if kind == "benchmark_done" and isinstance(payload, BenchmarkResult):
                self._finish_task()
                self._apply_benchmark_result(payload)
            if kind == "done":
                if isinstance(payload, tuple):
                    _message, turns = payload
                    if isinstance(turns, list):
                        self.last_turns = turns
                        self.rename_button.configure(state=tk.NORMAL)
                        self._start_auto_speaker_detection(turns)
                self._record_completed_range()
                self._refresh_history()
            if kind == "error":
                messagebox.showerror("Error", message)
            if kind == "cancelled":
                self._auto_speaker_detection_running = False
                self.status.set("Proceso detenido")
                self.metrics_progress.set("Proceso detenido por el usuario")
                self.log.insert(tk.END, "Proceso detenido por el usuario\n")
                self.log.see(tk.END)
        self.after(200, self._poll_events)

    def _start_task(self, task: str) -> None:
        self._cancel_event.clear()
        self._active_task = task
        self.busy_bar.configure(mode="indeterminate", maximum=100, value=0)
        self.stop_button.configure(state=tk.NORMAL)
        self.process_button.configure(state=tk.DISABLED)
        self.benchmark_button.configure(state=tk.DISABLED)

    def _finish_task(self) -> None:
        self.busy_bar.stop()
        self._active_task = None
        self._cancel_event.clear()
        self.stop_button.configure(state=tk.DISABLED)
        self.process_button.configure(state=tk.NORMAL)
        self.benchmark_button.configure(state=tk.NORMAL)

    def _stop_active_task(self) -> None:
        if self._active_task is None and not self._auto_speaker_detection_running:
            return
        self._cancel_event.set()
        self.stop_button.configure(state=tk.DISABLED)
        self.status.set("Deteniendo proceso...")
        self.metrics_progress.set("Deteniendo de forma segura")
        self.log.insert(tk.END, "Solicitud de detencion enviada\n")
        self.log.see(tk.END)

    def _message_from_payload(self, payload: Any) -> str:
        if isinstance(payload, tuple) and payload:
            return str(payload[0])
        if isinstance(payload, ProgressEvent):
            return format_progress_event(payload)
        if isinstance(payload, BenchmarkResult):
            return _format_benchmark_result(payload)
        return str(payload)

    def _update_progress_labels(self, payload: str | ProgressEvent) -> None:
        if not isinstance(payload, ProgressEvent):
            return
        if payload.stage == "transcription" and payload.seconds is not None:
            self.transcription_progress.set(format_progress_event(payload))
            self.metrics_progress.set(format_progress_event(payload))
        if payload.stage == "transcription_done":
            segments = payload.segments if payload.segments is not None else 0
            text_chars = payload.text_chars if payload.text_chars is not None else 0
            self.transcription_progress.set(
                f"Transcripcion completada: {segments} segmentos, {text_chars} caracteres"
            )
        if payload.stage == "diarization_done":
            speakers = payload.speakers if payload.speakers is not None else 0
            segments = payload.segments if payload.segments is not None else 0
            self.speaker_progress.set(
                f"Diarizacion completada: {speakers} hablantes crudos, {segments} segmentos"
            )
        if payload.stage in ("alignment_done", "done") and payload.summary:
            self.speaker_progress.set(payload.summary)
        if payload.stage == "diarization":
            self.speaker_progress.set("Hablantes: detectando y separando voces")
            self.metrics_progress.set("Separacion de voces iniciada")
            self.busy_bar.configure(mode="indeterminate", maximum=100)
            self.busy_bar.start(12)
        if payload.stage == "diarization_progress":
            formatted = format_progress_event(payload)
            self.speaker_progress.set(formatted)
            self.metrics_progress.set(formatted)
            if payload.completed is not None and payload.total is not None and payload.total > 0:
                self.busy_bar.stop()
                self.busy_bar.configure(mode="determinate", maximum=payload.total)
                self.busy_bar["value"] = min(payload.completed, payload.total)
        if payload.stage == "preflight":
            self.metrics_progress.set("Comprobando acceso a modelos antes de transcribir")
        if payload.stage == "preflight_done":
            self.metrics_progress.set("Modelos listos")
        if payload.stage == "benchmark":
            self.metrics_progress.set(payload.message)

    def _update_preview(self, payload: str | ProgressEvent) -> None:
        if not isinstance(payload, ProgressEvent):
            return
        if payload.stage != "transcription_segment" or not payload.text:
            return
        self.preview.insert(tk.END, format_progress_event(payload) + "\n")
        self.preview.see(tk.END)

    def _apply_benchmark_result(self, result: BenchmarkResult) -> None:
        if result.recommendation is None:
            self.transcription_progress.set("No se encontro una configuracion usable")
            messagebox.showwarning(
                "Benchmark sin resultado",
                "No se pudo completar ninguna configuracion. Prueba CPU / int8.",
            )
            return
        self.device.set(result.recommendation.device)
        self.compute_type.set(result.recommendation.compute_type)
        self.transcription_progress.set(
            f"Recomendado: {result.recommendation.device} / {result.recommendation.compute_type}"
        )
        self.metrics_progress.set("Configuracion recomendada aplicada")
        self._save_current_config()
        messagebox.showinfo(
            "Benchmark completado",
            _format_benchmark_result(result),
        )

    def _save_current_config(self) -> None:
        save_config(self.config_path, self._current_config(), self.ui_state)
        self.status.set(f"Configuracion guardada en {self.config_path}")

    def _current_config(self) -> ProcessingConfig:
        language = code_from_display_name(self.language.get())
        start_seconds = hms_to_seconds(
            self.start_hours.get(),
            self.start_minutes.get(),
            self.start_seconds.get(),
            blank_when_zero=True,
        )
        end_seconds = hms_to_seconds(
            self.end_hours.get(),
            self.end_minutes.get(),
            self.end_seconds.get(),
            blank_when_zero=True,
        )
        validate_time_range(start_seconds, end_seconds)
        return ProcessingConfig(
            whisper_model=model_id_from_display_name(self.whisper_model.get()),
            diarization_model=diarization_model_id_from_display_name(self.diarization_model.get()),
            huggingface_token=self.huggingface_token.get().strip() or None,
            ffmpeg_path=resolve_ffmpeg_path(None),
            language=language,
            min_speakers=_optional_int(self.min_speakers.get(), "Min hablantes"),
            max_speakers=_optional_int(self.max_speakers.get(), "Max hablantes"),
            device=self.device.get(),
            compute_type=self.compute_type.get(),
            export_speaker_audio=self.export_speaker_audio.get(),
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            diarization_quality=diarization_quality_id_from_display_name(self.diarization_quality.get()),
        )

    def _refresh_history(self) -> None:
        self.history_list.delete(0, tk.END)
        audio_text = self.audio_path.get().strip()
        if not audio_text:
            self.history_list.insert(tk.END, "Selecciona un audio para ver su historial")
            return
        entries = load_history(self.history_path).entries_for(Path(audio_text))
        if not entries:
            self.history_list.insert(tk.END, "Sin fragmentos completados para este audio")
            return
        for entry in entries:
            self.history_list.insert(
                tk.END,
                f"{format_optional_range(entry.start_seconds, entry.end_seconds)} -> {entry.output_dir}",
            )

    def _record_completed_range(self) -> None:
        audio_text = self.audio_path.get().strip()
        if not audio_text:
            return
        config = self._current_config()
        add_history_entry(
            self.history_path,
            Path(audio_text),
            HistoryEntry(
                start_seconds=config.start_seconds,
                end_seconds=config.end_seconds,
                output_dir=Path(self.output_dir.get()),
            ),
        )

    def _start_auto_speaker_detection(self, turns: list[ConversationTurn]) -> None:
        if self._auto_speaker_detection_running:
            return
        if not turns or not has_ai_runner():
            return
        self._auto_speaker_detection_running = True
        self._start_task("speaker_ai")
        self.speaker_progress.set("Detectando nombres de hablantes con IA")
        self.log.insert(tk.END, "Detectando nombres de hablantes con IA\n")
        self.log.see(tk.END)
        output_dir = Path(self.output_dir.get())
        thread = threading.Thread(
            target=self._run_auto_speaker_detection,
            args=(turns, output_dir),
            daemon=True,
        )
        thread.start()

    def _run_auto_speaker_detection(self, turns: list[ConversationTurn], output_dir: Path) -> None:
        try:
            response = run_speaker_identification_ai(
                turns,
                str(output_dir),
                self._cancel_event.is_set,
            )
        except CancelledError as exc:
            self.events.put(("cancelled", str(exc)))
            return
        except Exception as exc:
            self.events.put(("speaker_ai_error", str(exc)))
            return
        self.events.put(("speaker_ai_done", response))

    def _open_speaker_editor(self, ai_response: str | None = None) -> None:
        turns = self.last_turns or _load_turns_from_output(Path(self.output_dir.get()))
        if not turns:
            messagebox.showwarning(
                "Sin transcripcion",
                "Primero procesa un audio o selecciona una carpeta de salida con transcript.json.",
            )
            return
        self.last_turns = turns
        SpeakerNameDialog(self, turns, Path(self.output_dir.get()), self._speaker_names_saved, ai_response)

    def _speaker_names_saved(self, turns: list[ConversationTurn]) -> None:
        self.last_turns = turns
        self.speaker_progress.set("Nombres de hablantes guardados")
        self.status.set(f"Transcripcion actualizada en {self.output_dir.get()}")
        self.preview.delete("1.0", tk.END)
        for turn in turns[-20:]:
            self.preview.insert(tk.END, f"[{_clock_time(turn.start)}] {turn.speaker}: {turn.text}\n")
        self.preview.see(tk.END)


class SpeakerNameDialog(tk.Toplevel):
    def __init__(
        self,
        parent: App,
        turns: list[ConversationTurn],
        output_dir: Path,
        on_saved: object,
        ai_response: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.title("Renombrar hablantes")
        self.geometry("760x520")
        self.transient(parent)
        self.grab_set()
        self.turns = turns
        self.output_dir = output_dir
        self.on_saved = on_saved
        self.initial_ai_response = ai_response
        self.name_vars: dict[str, tk.StringVar] = {}
        self.ai_status = tk.StringVar(value="IA lista")
        self._build()
        if self.initial_ai_response:
            self._ai_finished(self.initial_ai_response)

    def _build(self) -> None:
        root = ttk.Frame(self, padding=16)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(1, weight=1)
        ttk.Label(
            root,
            text=(
                "Asigna nombres reales a las etiquetas detectadas. La app puede consultar "
                "opencode o codex y rellenar una propuesta para revisar."
            ),
            wraplength=700,
        ).grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 12))

        for row, speaker in enumerate(speaker_labels(self.turns), start=1):
            ttk.Label(root, text=speaker).grid(row=row, column=0, sticky="w", pady=4)
            variable = tk.StringVar(value=speaker)
            self.name_vars[speaker] = variable
            ttk.Entry(root, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8, pady=4)
            ttk.Label(root, text=_speaker_sample(self.turns, speaker), wraplength=260).grid(
                row=row,
                column=2,
                sticky="w",
                pady=4,
            )

        actions_row = len(self.name_vars) + 1
        actions = ttk.Frame(root)
        actions.grid(row=actions_row, column=0, columnspan=3, sticky="ew", pady=(14, 8))
        ttk.Button(actions, text="Detectar con IA", command=self._run_ai).pack(side=tk.LEFT)
        ttk.Button(actions, text="Pegar respuesta IA", command=self._paste_ai_response).pack(
            side=tk.LEFT,
            padx=(8, 0),
        )
        ttk.Button(actions, text="Guardar transcripcion", command=self._save).pack(
            side=tk.RIGHT,
            padx=(8, 0),
        )
        ttk.Button(actions, text="Cancelar", command=self.destroy).pack(side=tk.RIGHT)

        ttk.Label(root, text="Respuesta IA JSON").grid(
            row=actions_row + 1,
            column=0,
            sticky="w",
        )
        ttk.Label(root, textvariable=self.ai_status).grid(row=actions_row + 1, column=1, columnspan=2, sticky="e")
        self.response = tk.Text(root, height=9, wrap="word")
        self.response.grid(row=actions_row + 2, column=0, columnspan=3, sticky="nsew")
        root.rowconfigure(actions_row + 2, weight=1)

    def _run_ai(self) -> None:
        self.ai_status.set("Consultando IA...")
        thread = threading.Thread(target=self._run_ai_worker, daemon=True)
        thread.start()

    def _run_ai_worker(self) -> None:
        try:
            response = run_speaker_identification_ai(self.turns, str(self.output_dir))
        except Exception as exc:
            self.after(0, lambda: self._ai_failed(str(exc)))
            return
        self.after(0, lambda: self._ai_finished(response))

    def _ai_finished(self, response: str) -> None:
        self.response.delete("1.0", tk.END)
        self.response.insert(tk.END, response.strip())
        self.ai_status.set("Respuesta recibida")
        self._paste_ai_response()

    def _ai_failed(self, message: str) -> None:
        self.ai_status.set("IA fallida")
        messagebox.showerror("IA no disponible", message)

    def _paste_ai_response(self) -> None:
        text = self.response.get("1.0", tk.END).strip()
        if not text:
            try:
                text = self.clipboard_get().strip()
            except tk.TclError:
                text = ""
        if not text:
            messagebox.showwarning("Sin respuesta", "Pega primero la respuesta JSON de la IA.")
            return
        try:
            mapping = parse_speaker_mapping_response(text)
        except Exception as exc:
            messagebox.showerror("JSON no valido", str(exc))
            return
        for speaker, name in mapping.items():
            if speaker in self.name_vars:
                self.name_vars[speaker].set(name)

    def _save(self) -> None:
        mapping = {
            speaker: variable.get().strip()
            for speaker, variable in self.name_vars.items()
            if variable.get().strip()
        }
        turns = rename_speakers(self.turns, mapping)
        write_all_exports(self.output_dir, turns)
        callback = self.on_saved
        if callable(callback):
            callback(turns)
        self.destroy()


def _optional_int(value: str, label: str) -> int | None:
    stripped = value.strip()
    if stripped == "":
        return None
    try:
        parsed = int(stripped)
    except ValueError as exc:
        raise ValueError(f"{label} debe ser un numero entero") from exc
    if parsed <= 0:
        raise ValueError(f"{label} debe ser mayor que cero")
    return parsed


def _short_status(message: str) -> str:
    max_chars = 120
    if len(message) <= max_chars:
        return message
    return message[: max_chars - 3] + "..."


def _load_turns_from_output(output_dir: Path) -> list[ConversationTurn]:
    path = output_dir / "transcript.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    turns = payload.get("turns") if isinstance(payload, dict) else None
    if not isinstance(turns, list):
        return []
    loaded: list[ConversationTurn] = []
    for item in turns:
        if not isinstance(item, dict):
            continue
        try:
            loaded.append(
                ConversationTurn(
                    start=float(item["start"]),
                    end=float(item["end"]),
                    speaker=str(item["speaker"]),
                    text=str(item["text"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return loaded


def _speaker_sample(turns: list[ConversationTurn], speaker: str) -> str:
    for turn in turns:
        if turn.speaker == speaker and turn.text.strip():
            text = turn.text.strip()
            return text[:120] + ("..." if len(text) > 120 else "")
    return ""


def _clock_time(seconds: float) -> str:
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_benchmark_result(result: BenchmarkResult) -> str:
    lines = ["Benchmark completado"]
    if result.recommendation is not None:
        lines.append(
            f"Recomendacion: {result.recommendation.device} / {result.recommendation.compute_type}"
        )
    else:
        lines.append("Recomendacion: ninguna configuracion usable")
    for attempt in result.attempts:
        if attempt.ok and attempt.speed is not None:
            lines.append(f"- {attempt.candidate.label}: OK, {attempt.speed:.1f}x")
        else:
            lines.append(f"- {attempt.candidate.label}: fallo ({attempt.error})")
    return "\n".join(lines)


def main() -> None:
    app = App()
    app.mainloop()
