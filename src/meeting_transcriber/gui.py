from __future__ import annotations

import contextlib
import io
import queue
import shutil
import threading
import tkinter as tk
import webbrowser
import json
import time
from pathlib import Path
from typing import Any
from tkinter import filedialog, messagebox, simpledialog, ttk

from meeting_transcriber.audio import extract_audio_range, probe_audio_duration
from meeting_transcriber.audio_preview import preview_clip_path
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
from meeting_transcriber.exporters import build_merged_output_dir, build_processing_output_dir, write_all_exports
from meeting_transcriber.history import (
    HistoryEntry,
    add_history_entry,
    add_merged_history_entry,
    completed_ranges,
    coverage_seconds,
    load_history,
    output_dir_reference_count,
    recommend_next_range,
    reanalysis_range,
    remove_history_entry,
    visible_entries_for,
)
from meeting_transcriber.languages import (
    code_from_display_name,
    display_name_from_code,
    language_display_names,
)
from meeting_transcriber.pipeline import process_meeting
from meeting_transcriber.progress import ProgressEvent, format_progress_event, format_seconds
from meeting_transcriber.speaker_ai import (
    has_ai_runner,
    parse_speaker_mapping_response,
    run_speaker_identification_ai,
)
from meeting_transcriber.speaker_cross_compare import (
    SpeakerMatch,
    SpeakerProfile,
    SpeakerSource,
    build_speaker_profiles,
    compare_speaker_profiles,
    name_coherence_matrix,
)
from meeting_transcriber.speaker_embedding_store import (
    SpeakerEmbeddingStore,
    load_embedding_store,
    save_embedding_store,
)
from meeting_transcriber.speaker_fingerprints import (
    extract_speaker_embeddings,
    load_pyannote_embedding_extractor,
)
from meeting_transcriber.speaker_memory import (
    build_embedding_name_mapping,
    build_unique_name_mapping,
    format_speaker_memory_status,
    identity_names,
    load_speaker_memory,
    remember_validated_turns,
    speaker_memory_status,
)
from meeting_transcriber.speaker_names import add_known_name, rename_speakers, speaker_labels
from meeting_transcriber.transcript_merge import (
    DraftMergeRow,
    MergeRow,
    align_turns_for_merge,
    diff_text_segments,
    draft_from_source_turn,
    merged_turns_from_drafts,
)
from meeting_transcriber.time_range import format_optional_range, hms_to_seconds, validate_time_range
from meeting_transcriber.types import ConversationTurn, ProcessingConfig
from meeting_transcriber.whisper_models import (
    DEFAULT_WHISPER_MODEL,
    model_id_from_display_name,
    model_label_from_id,
    whisper_model_labels,
)


DEFAULT_WINDOW_GEOMETRY = "1120x800"
MIN_WINDOW_SIZE = (900, 680)
PROCESSING_OPTION_COLUMNS = ("Calidad", "Diarización", "Separacion voces", "Idioma")


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Meeting Transcriber")
        self.geometry(DEFAULT_WINDOW_GEOMETRY)
        self.minsize(*MIN_WINDOW_SIZE)
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.config_path = default_config_dir() / "config.json"
        self.history_path = default_config_dir() / "history.json"
        self.speaker_memory_path = default_config_dir() / "speaker_memory.json"
        self.embedding_store_path = default_config_dir() / "speaker_embeddings.json"
        self.preview_audio_dir = default_config_dir() / "preview_audio"
        self._reported_speaker_memory_errors: set[str] = set()

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
        self.history_summary = tk.StringVar(value="")
        self.history_recommendation = tk.StringVar(value="")
        self.target_wait_minutes = tk.StringVar(value="15 min")
        self.last_turns: list[ConversationTurn] = []
        self.last_output_dir: Path | None = None
        self._audio_duration_cache: dict[Path, float | None] = {}
        self._history_entries: list[HistoryEntry] = []
        self._coverage_ranges: list[tuple[float, float]] = []
        self._coverage_duration: float | None = None
        self._last_recommendation: tuple[float, float] | None = None
        self._pending_completion: tuple[Path, Path, ProcessingConfig] | None = None
        self._active_started_at: float | None = None
        self._last_process_elapsed_seconds: float | None = None
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
        self._processing_options_row(root, 2)

        speaker_frame = ttk.Frame(root)
        speaker_frame.grid(row=3, column=1, sticky="w", pady=6)
        ttk.Label(root, text="Hablantes").grid(row=3, column=0, sticky="w", pady=6)
        ttk.Label(speaker_frame, text="Min").pack(side=tk.LEFT)
        ttk.Entry(speaker_frame, textvariable=self.min_speakers, width=6).pack(side=tk.LEFT, padx=(6, 16))
        ttk.Label(speaker_frame, text="Max").pack(side=tk.LEFT)
        ttk.Entry(speaker_frame, textvariable=self.max_speakers, width=6).pack(side=tk.LEFT, padx=(6, 0))

        range_frame = ttk.Frame(root)
        range_frame.grid(row=4, column=1, sticky="w", pady=6)
        ttk.Label(root, text="Rango").grid(row=4, column=0, sticky="w", pady=6)
        self._time_selector(range_frame, "Inicio", self.start_hours, self.start_minutes, self.start_seconds)
        self._time_selector(range_frame, "Fin", self.end_hours, self.end_minutes, self.end_seconds)

        runtime = ttk.Frame(root)
        runtime.grid(row=5, column=1, sticky="w", pady=6)
        ttk.Label(root, text="Ejecucion").grid(row=5, column=0, sticky="w", pady=6)
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
        ).grid(row=6, column=1, sticky="w", pady=10)

        token_frame = ttk.Frame(root)
        token_frame.grid(row=7, column=1, sticky="ew", pady=6)
        token_frame.columnconfigure(0, weight=1)
        ttk.Label(root, text="Token HF").grid(row=7, column=0, sticky="w", pady=6)
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
        actions.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(18, 8))
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

        ttk.Label(root, textvariable=self.status).grid(row=9, column=0, columnspan=3, sticky="w")
        ttk.Label(root, textvariable=self.transcription_progress).grid(
            row=10,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(4, 0),
        )
        ttk.Label(root, textvariable=self.speaker_progress).grid(
            row=11,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(4, 0),
        )
        ttk.Label(root, textvariable=self.metrics_progress).grid(
            row=12,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(4, 0),
        )
        self.busy_bar = ttk.Progressbar(root, mode="indeterminate")
        self.busy_bar.grid(row=13, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        history_frame = ttk.Frame(root)
        history_frame.grid(row=14, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        history_frame.columnconfigure(0, weight=1)
        ttk.Label(history_frame, text="Historial de este audio").grid(row=0, column=0, sticky="w")
        self.coverage_canvas = tk.Canvas(history_frame, height=20, highlightthickness=1, highlightbackground="#9a9a9a")
        self.coverage_canvas.grid(row=1, column=0, sticky="ew", pady=(4, 2))
        self.coverage_canvas.bind("<Configure>", lambda _event: self._draw_history_coverage())
        ttk.Label(history_frame, textvariable=self.history_summary).grid(row=2, column=0, sticky="w")
        recommendation_frame = ttk.Frame(history_frame)
        recommendation_frame.grid(row=3, column=0, sticky="ew", pady=(4, 2))
        recommendation_frame.columnconfigure(1, weight=1)
        ttk.Label(recommendation_frame, text="Tanda aprox.").grid(row=0, column=0, sticky="w")
        self.target_wait_combo = ttk.Combobox(
            recommendation_frame,
            textvariable=self.target_wait_minutes,
            values=["10 min", "15 min", "30 min"],
            state="readonly",
            width=8,
        )
        self.target_wait_combo.grid(row=0, column=1, sticky="w", padx=(8, 12))
        self.target_wait_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_history())
        ttk.Label(recommendation_frame, textvariable=self.history_recommendation).grid(row=0, column=2, sticky="w")
        self.use_recommendation_button = ttk.Button(
            recommendation_frame,
            text="Usar recomendado",
            command=self._apply_recommended_range,
            state=tk.DISABLED,
        )
        self.use_recommendation_button.grid(row=0, column=3, sticky="e", padx=(12, 0))
        self.history_list = tk.Listbox(history_frame, height=4, selectmode=tk.EXTENDED)
        self.history_list.grid(row=4, column=0, sticky="nsew")
        self.history_list.bind("<<ListboxSelect>>", lambda _event: self._update_history_buttons())
        history_actions = ttk.Frame(history_frame)
        history_actions.grid(row=5, column=0, sticky="ew", pady=(4, 0))
        self.open_history_output_button = ttk.Button(
            history_actions,
            text="Abrir salida",
            command=self._open_selected_history_output,
            state=tk.DISABLED,
        )
        self.open_history_output_button.pack(side=tk.LEFT)
        self.reanalyze_history_button = ttk.Button(
            history_actions,
            text="Reanalizar",
            command=self._reanalyze_selected_history_entry,
            state=tk.DISABLED,
        )
        self.reanalyze_history_button.pack(side=tk.LEFT, padx=(8, 0))
        self.compare_history_speakers_button = ttk.Button(
            history_actions,
            text="Comparar personas",
            command=self._compare_selected_history_speakers,
            state=tk.DISABLED,
        )
        self.compare_history_speakers_button.pack(side=tk.LEFT, padx=(8, 0))
        self.merge_history_button = ttk.Button(
            history_actions,
            text="Fusionar resultados",
            command=self._merge_selected_history_entries,
            state=tk.DISABLED,
        )
        self.merge_history_button.pack(side=tk.LEFT, padx=(8, 0))
        self.delete_history_button = ttk.Button(
            history_actions,
            text="Eliminar seleccionado",
            command=self._delete_selected_history_entry,
            state=tk.DISABLED,
        )
        self.delete_history_button.pack(side=tk.LEFT, padx=(8, 0))
        panes = ttk.PanedWindow(root, orient=tk.VERTICAL)
        panes.grid(row=15, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        root.rowconfigure(15, weight=1)

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

    def _processing_options_row(self, parent: ttk.Frame, row: int) -> None:
        options = ttk.Frame(parent)
        options.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 6))
        for column in range(len(PROCESSING_OPTION_COLUMNS)):
            options.columnconfigure(column, weight=1, uniform="processing-options")

        specs = (
            ("Calidad", self.whisper_model, whisper_model_labels()),
            ("Diarización", self.diarization_model, diarization_model_labels()),
            ("Separacion voces", self.diarization_quality, diarization_quality_labels()),
            ("Idioma", self.language, language_display_names()),
        )
        for column, (label, variable, values) in enumerate(specs):
            group = ttk.Frame(options)
            group.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 10, 0))
            group.columnconfigure(0, weight=1)
            ttk.Label(group, text=label).grid(row=0, column=0, sticky="w")
            ttk.Combobox(
                group,
                textvariable=variable,
                values=values,
                state="readonly",
                width=18,
            ).grid(row=1, column=0, sticky="ew", pady=(3, 0))

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
        base_output_dir = Path(self.output_dir.get())
        if not audio_path.is_file():
            messagebox.showerror(
                "Audio no encontrado",
                "Selecciona primero un archivo de audio valido.",
            )
            return
        output_dir = build_processing_output_dir(
            base_output_dir,
            audio_path,
            start_seconds=config.start_seconds,
            end_seconds=config.end_seconds,
        )

        self._refresh_history()
        self._save_current_config()
        self._start_task("process")
        self.preview.delete("1.0", tk.END)
        self.log.delete("1.0", tk.END)
        self.speaker_progress.set("Hablantes: se detectaran al terminar la transcripcion")
        self.metrics_progress.set(f"Preparando analisis en {output_dir}")
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
            self.events.put(("done", (f"Resultados guardados en {output_dir}", audio_path, output_dir, config, turns)))
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
                self._open_speaker_editor(ai_response=payload, wait=True)
                self._finalize_pending_completion()
                continue
            if kind == "speaker_ai_error":
                self._auto_speaker_detection_running = False
                self._finish_task()
                self.status.set("Deteccion IA de hablantes no disponible")
                self.log.insert(tk.END, f"Deteccion IA de hablantes no disponible: {payload}\n")
                self.log.see(tk.END)
                if self._pending_completion is not None:
                    self._open_speaker_editor(wait=True)
                    self._finalize_pending_completion()
                continue
            if kind == "speaker_memory_error":
                message = str(payload)
                if message not in self._reported_speaker_memory_errors:
                    self._reported_speaker_memory_errors.add(message)
                    self.log.insert(tk.END, f"Memoria de voz no disponible: {message}\n")
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
                    self._handle_completed_processing(payload)
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
        self._active_started_at = time.monotonic()
        self.busy_bar.configure(mode="indeterminate", maximum=100, value=0)
        self.stop_button.configure(state=tk.NORMAL)
        self.process_button.configure(state=tk.DISABLED)
        self.benchmark_button.configure(state=tk.DISABLED)

    def _finish_task(self) -> None:
        if self._active_task == "process" and self._active_started_at is not None:
            self._last_process_elapsed_seconds = time.monotonic() - self._active_started_at
        self.busy_bar.stop()
        self._active_task = None
        self._active_started_at = None
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
        self._history_entries = []
        self._coverage_ranges = []
        self._coverage_duration = None
        self._last_recommendation = None
        self._update_history_buttons()
        self.use_recommendation_button.configure(state=tk.DISABLED)
        audio_text = self.audio_path.get().strip()
        if not audio_text:
            self.history_list.insert(tk.END, "Selecciona un audio para ver su historial")
            self.history_summary.set("")
            self.history_recommendation.set("")
            self._draw_history_coverage()
            return
        audio_path = Path(audio_text)
        duration = self._audio_duration(audio_path)
        history = load_history(self.history_path)
        entries = visible_entries_for(history, audio_path)
        self._history_entries = entries
        if duration is not None:
            self._coverage_duration = duration
            self._coverage_ranges = completed_ranges(entries, duration)
            covered = coverage_seconds(entries, duration)
            self.history_summary.set(
                f"Cobertura: {format_seconds(covered)} / {format_seconds(duration)}"
            )
        else:
            self.history_summary.set("Cobertura: duracion del audio no disponible")
        self._update_history_recommendation(entries, duration)
        self._draw_history_coverage()
        if not entries:
            self.history_list.insert(tk.END, "Sin fragmentos completados para este audio")
            return
        for entry in entries:
            self.history_list.insert(
                tk.END,
                f"{format_optional_range(entry.start_seconds, entry.end_seconds)} -> {entry.output_dir}",
            )
        self._update_history_buttons()

    def _record_completed_range(
        self,
        audio_path: Path,
        output_dir: Path,
        config: ProcessingConfig,
    ) -> None:
        add_history_entry(
            self.history_path,
            audio_path,
            HistoryEntry(
                start_seconds=config.start_seconds,
                end_seconds=config.end_seconds,
                output_dir=output_dir,
                elapsed_seconds=self._last_process_elapsed_seconds,
            ),
        )

    def _handle_completed_processing(self, payload: tuple[Any, ...]) -> None:
        if len(payload) != 5:
            return
        _message, audio_path, output_dir, config, turns = payload
        if not isinstance(audio_path, Path) or not isinstance(output_dir, Path):
            return
        if not isinstance(config, ProcessingConfig) or not isinstance(turns, list):
            return

        memory = load_speaker_memory(self.speaker_memory_path)
        speaker_embeddings = {}
        if _memory_has_embeddings(memory, audio_path):
            speaker_embeddings = self._try_extract_speaker_embeddings(audio_path, turns, config)
        mapping = build_embedding_name_mapping(
            memory,
            audio_path,
            speaker_embeddings,
            threshold=0.8,
        )
        if not mapping:
            mapping = build_unique_name_mapping(memory, audio_path, turns)
        if mapping:
            turns = rename_speakers(turns, mapping)
            write_all_exports(output_dir, turns)
            self.log.insert(tk.END, "Nombres recordados aplicados al fragmento\n")
            self.log.see(tk.END)

        self.last_turns = turns
        self.last_output_dir = output_dir
        self.rename_button.configure(state=tk.NORMAL)
        self._pending_completion = (audio_path, output_dir, config)
        if self._start_auto_speaker_detection(turns):
            return

        self._open_speaker_editor(wait=True)
        self._finalize_pending_completion()

    def _finalize_pending_completion(self) -> None:
        if self._pending_completion is None:
            return
        audio_path, output_dir, config = self._pending_completion
        self._pending_completion = None
        decision = self._ask_completed_result_decision(config, output_dir)
        if decision == "discard":
            self._discard_output_artifacts(output_dir)
            self.status.set("Resultado descartado")
            self._refresh_history()
            return

        self._record_completed_range(audio_path, output_dir, config)
        if self.last_turns:
            remember_validated_turns(self.speaker_memory_path, audio_path, self.last_turns)
            self._start_speaker_embedding_memory_update(audio_path, self.last_turns, config)
        self._refresh_history()

    def _ask_completed_result_decision(self, config: ProcessingConfig, output_dir: Path) -> str:
        result = tk.StringVar(value="")
        dialog = tk.Toplevel(self)
        dialog.title("Resultado completado")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)
        root = ttk.Frame(dialog, padding=16)
        root.pack(fill=tk.BOTH, expand=True)
        ttk.Label(root, text="Procesamiento completado").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(root, text=f"Rango: {format_optional_range(config.start_seconds, config.end_seconds)}").grid(
            row=1,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(8, 0),
        )
        ttk.Label(root, text=f"Salida: {output_dir}").grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 12))

        def choose(value: str) -> None:
            result.set(value)
            dialog.destroy()

        ttk.Button(root, text="Guardar como válido", command=lambda: choose("save")).grid(row=3, column=0, sticky="w")
        ttk.Button(root, text="Descartar y eliminar", command=lambda: choose("discard")).grid(
            row=3,
            column=1,
            padx=(8, 0),
        )
        ttk.Button(root, text="Revisar carpeta", command=lambda: self._open_output_dir(output_dir)).grid(
            row=3,
            column=2,
            padx=(8, 0),
        )
        dialog.protocol("WM_DELETE_WINDOW", lambda: choose("save"))
        dialog.wait_window()
        return result.get() or "save"

    def _audio_duration(self, audio_path: Path) -> float | None:
        if audio_path in self._audio_duration_cache:
            return self._audio_duration_cache[audio_path]
        try:
            duration = probe_audio_duration(resolve_ffmpeg_path(None), audio_path)
        except Exception:
            duration = None
        self._audio_duration_cache[audio_path] = duration
        return duration

    def _update_history_recommendation(
        self,
        entries: list[HistoryEntry],
        duration: float | None,
    ) -> None:
        if duration is None:
            self.history_recommendation.set("Recomendacion disponible al conocer la duracion")
            return
        target_wait_seconds = self._target_wait_seconds()
        recommendation = recommend_next_range(
            entries,
            total_duration_seconds=duration,
            target_wait_seconds=target_wait_seconds,
        )
        if recommendation is None:
            self.history_recommendation.set("Audio completo segun el historial")
            return
        self._last_recommendation = (recommendation.start_seconds, recommendation.end_seconds)
        self.use_recommendation_button.configure(state=tk.NORMAL)
        speed_text = "velocidad sin historial"
        if recommendation.speed is not None:
            speed_text = f"{recommendation.speed:.1f}x"
        eta_text = ""
        if recommendation.estimated_elapsed_seconds is not None:
            eta_text = f", aprox. {format_seconds(recommendation.estimated_elapsed_seconds)}"
        self.history_recommendation.set(
            "Siguiente: "
            f"{format_seconds(recommendation.start_seconds)} -> {format_seconds(recommendation.end_seconds)} "
            f"({speed_text}{eta_text})"
        )

    def _draw_history_coverage(self) -> None:
        self.coverage_canvas.delete("all")
        width = max(1, self.coverage_canvas.winfo_width())
        height = max(1, self.coverage_canvas.winfo_height())
        self.coverage_canvas.create_rectangle(0, 0, width, height, fill="#d9d9d9", outline="")
        if self._coverage_duration is None or self._coverage_duration <= 0:
            return
        for start, end in self._coverage_ranges:
            x1 = int(width * start / self._coverage_duration)
            x2 = int(width * end / self._coverage_duration)
            self.coverage_canvas.create_rectangle(x1, 0, max(x1 + 1, x2), height, fill="#4f7f9f", outline="")

    def _apply_recommended_range(self) -> None:
        if self._last_recommendation is None:
            return
        start_seconds, end_seconds = self._last_recommendation
        self._set_hms(self.start_hours, self.start_minutes, self.start_seconds, start_seconds)
        self._set_hms(self.end_hours, self.end_minutes, self.end_seconds, end_seconds)
        self.status.set("Rango recomendado aplicado")

    def _target_wait_seconds(self) -> float:
        value = self.target_wait_minutes.get().split()[0]
        try:
            return float(value) * 60
        except ValueError:
            return 15 * 60.0

    def _set_hms(
        self,
        hours_var: tk.StringVar,
        minutes_var: tk.StringVar,
        seconds_var: tk.StringVar,
        seconds: float,
    ) -> None:
        total = int(seconds)
        hours_var.set(str(total // 3600))
        minutes_var.set(str((total % 3600) // 60))
        seconds_var.set(str(total % 60))

    def _selected_history_index(self) -> int | None:
        indices = self._selected_history_indices()
        if len(indices) != 1:
            return None
        return indices[0]

    def _selected_history_indices(self) -> list[int]:
        indices = []
        for selection in self.history_list.curselection():
            index = int(selection)
            if 0 <= index < len(self._history_entries):
                indices.append(index)
        return indices

    def _update_history_buttons(self) -> None:
        selected_indices = self._selected_history_indices()
        single_state = tk.NORMAL if len(selected_indices) == 1 else tk.DISABLED
        merge_state = tk.NORMAL if len(selected_indices) == 2 else tk.DISABLED
        if hasattr(self, "open_history_output_button"):
            self.open_history_output_button.configure(state=single_state)
        if hasattr(self, "reanalyze_history_button"):
            self.reanalyze_history_button.configure(state=single_state)
        if hasattr(self, "compare_history_speakers_button"):
            self.compare_history_speakers_button.configure(state=single_state)
        if hasattr(self, "merge_history_button"):
            self.merge_history_button.configure(state=merge_state)
        if hasattr(self, "delete_history_button"):
            self.delete_history_button.configure(state=single_state)

    def _open_selected_history_output(self) -> None:
        index = self._selected_history_index()
        if index is None:
            return
        self._open_output_dir(self._history_entries[index].output_dir)

    def _reanalyze_selected_history_entry(self) -> None:
        index = self._selected_history_index()
        if index is None:
            return
        start_seconds, end_seconds = reanalysis_range(self._history_entries[index])
        self._set_hms(self.start_hours, self.start_minutes, self.start_seconds, start_seconds)
        if end_seconds is None:
            self._set_hms(self.end_hours, self.end_minutes, self.end_seconds, 0)
        else:
            self._set_hms(self.end_hours, self.end_minutes, self.end_seconds, end_seconds)
        self.status.set("Reanalizando fragmento seleccionado")
        self._process()

    def _compare_selected_history_speakers(self) -> None:
        index = self._selected_history_index()
        audio_text = self.audio_path.get().strip()
        if index is None or not audio_text:
            return
        audio_path = Path(audio_text)
        base_entry = self._history_entries[index]
        history_entries = load_history(self.history_path).entries_for(audio_path)
        memory = load_speaker_memory(self.speaker_memory_path)
        store = load_embedding_store(self.embedding_store_path)
        profiles_by_source: dict[str, list[SpeakerProfile]] = {}
        memory_source = SpeakerSource("memory", self.speaker_memory_path, "Memoria completa")
        sources: list[SpeakerSource] = [memory_source]
        profiles_by_source[memory_source.entry_id] = _memory_profiles(memory_source, memory, audio_path)
        for entry in history_entries:
            if entry.id is None:
                continue
            turns = _load_turns_from_output(entry.output_dir)
            if not turns:
                continue
            source = _speaker_source_from_entry(entry)
            sources.append(source)
            profiles_by_source[source.entry_id] = build_speaker_profiles(
                source,
                turns,
                embeddings=store.embeddings_for_source(audio_path, source.entry_id),
            )
        if base_entry.id is None or base_entry.id not in profiles_by_source:
            messagebox.showwarning(
                "Comparar personas",
                "No se encontro transcript.json en la salida seleccionada.",
            )
            return
        SpeakerComparisonDialog(
            self,
            audio_path,
            base_entry.id,
            sources,
            profiles_by_source,
            store,
            self._generate_speaker_embeddings_for_sources,
            self._save_comparison_speaker_corrections,
            self._play_comparison_profile,
        )

    def _merge_selected_history_entries(self) -> None:
        indices = self._selected_history_indices()
        audio_text = self.audio_path.get().strip()
        if len(indices) != 2 or not audio_text:
            return
        left_entry, right_entry = [self._history_entries[index] for index in indices]
        if not _history_entries_overlap(left_entry, right_entry):
            messagebox.showwarning(
                "Fusionar resultados",
                "Selecciona dos fragmentos del mismo rango o con solapamiento temporal.",
            )
            return
        left_turns = _load_turns_from_output(left_entry.output_dir)
        right_turns = _load_turns_from_output(right_entry.output_dir)
        if not left_turns or not right_turns:
            messagebox.showwarning(
                "Fusionar resultados",
                "No se encontro transcript.json en una de las salidas seleccionadas.",
            )
            return
        audio_path = Path(audio_text)
        memory = load_speaker_memory(self.speaker_memory_path)
        known_names = identity_names(memory, audio_path)
        rows = align_turns_for_merge(left_turns, right_turns)
        dialog = MergeReviewDialog(
            self,
            audio_path,
            self.preview_audio_dir,
            left_entry,
            right_entry,
            rows,
            known_names,
            lambda drafts: self._save_merged_history_result(audio_path, left_entry, right_entry, drafts),
        )
        dialog.wait_window()

    def _save_merged_history_result(
        self,
        audio_path: Path,
        left_entry: HistoryEntry,
        right_entry: HistoryEntry,
        drafts: list[DraftMergeRow],
    ) -> bool:
        turns = merged_turns_from_drafts(drafts)
        if not turns:
            messagebox.showwarning("Fusionar resultados", "La fusion no contiene texto.")
            return False
        output_dir = build_merged_output_dir(left_entry.output_dir)
        write_all_exports(output_dir, turns)
        merged_entry = add_merged_history_entry(
            self.history_path,
            audio_path,
            HistoryEntry(
                start_seconds=min(left_entry.start_seconds or 0.0, right_entry.start_seconds or 0.0),
                end_seconds=_merged_end_seconds(left_entry, right_entry),
                output_dir=output_dir,
            ),
            (left_entry.id, right_entry.id),
        )
        remember_validated_turns(self.speaker_memory_path, audio_path, turns)
        self.last_turns = turns
        self.last_output_dir = output_dir
        self.rename_button.configure(state=tk.NORMAL)
        self.status.set(f"Fusion guardada en {output_dir}")
        self.log.insert(tk.END, f"Fusion guardada: {merged_entry.output_dir}\n")
        self.log.see(tk.END)
        self._refresh_history()
        return True

    def _generate_speaker_embeddings_for_sources(
        self,
        audio_path: Path,
        source_ids: list[str],
        progress: object | None = None,
    ) -> SpeakerEmbeddingStore:
        config = self._current_config()
        store = load_embedding_store(self.embedding_store_path)
        history_entries = load_history(self.history_path).entries_for(audio_path)
        entries_by_id = {entry.id: entry for entry in history_entries}
        if callable(progress):
            progress("Cargando modelo de huellas de voz...")
        with _quiet_model_output():
            extractor = load_pyannote_embedding_extractor(
                resolve_ffmpeg_path(None),
                huggingface_token=config.huggingface_token,
                device=config.device,
            )
        unique_source_ids = list(dict.fromkeys(source_ids))
        for position, source_id in enumerate(unique_source_ids, start=1):
            entry = entries_by_id.get(source_id)
            if entry is None:
                continue
            turns = _load_turns_from_output(entry.output_dir)
            if callable(progress):
                progress(f"Generando huellas {position}/{len(unique_source_ids)}: {entry.output_dir.name}")
            with _quiet_model_output():
                embeddings = extract_speaker_embeddings(audio_path, turns, extractor)
            for speaker, embedding in embeddings.items():
                store = store.with_embedding(
                    audio_path=audio_path,
                    source_id=source_id,
                    speaker=speaker,
                    embedding=embedding,
                )
        save_embedding_store(self.embedding_store_path, store)
        if callable(progress):
            progress(f"Huellas guardadas: {store.count_embeddings(audio_path)}")
        return store

    def _save_comparison_speaker_corrections(
        self,
        audio_path: Path,
        source_id: str,
        mapping: dict[str, str],
    ) -> None:
        if not mapping:
            messagebox.showinfo("Comparar personas", "No hay correcciones para guardar.")
            return
        entry = next(
            (item for item in load_history(self.history_path).entries_for(audio_path) if item.id == source_id),
            None,
        )
        if entry is None:
            messagebox.showwarning("Comparar personas", "No se encontro la salida base.")
            return
        turns = _load_turns_from_output(entry.output_dir)
        updated_turns = rename_speakers(turns, mapping)
        write_all_exports(entry.output_dir, updated_turns)
        remember_validated_turns(self.speaker_memory_path, audio_path, updated_turns)
        self.last_turns = updated_turns
        self.last_output_dir = entry.output_dir
        self.rename_button.configure(state=tk.NORMAL)
        self.status.set("Correcciones de hablantes guardadas")

    def _play_comparison_profile(self, audio_path: Path, profile: SpeakerProfile | None) -> None:
        if profile is None or profile.sample_start is None or profile.sample_end is None:
            return
        clip_path = preview_clip_path(
            self.preview_audio_dir,
            audio_path,
            start_seconds=profile.sample_start,
            end_seconds=profile.sample_end,
        )
        try:
            if not clip_path.exists():
                extract_audio_range(
                    resolve_ffmpeg_path(None),
                    audio_path,
                    clip_path,
                    profile.sample_start,
                    profile.sample_end,
                )
            webbrowser.open(clip_path.resolve().as_uri())
        except Exception as exc:
            messagebox.showwarning("Reproducir audio", f"No se pudo reproducir esta muestra: {exc}")

    def _open_output_dir(self, output_dir: Path) -> None:
        try:
            webbrowser.open(output_dir.resolve().as_uri())
        except ValueError:
            webbrowser.open(str(output_dir))

    def _delete_selected_history_entry(self) -> None:
        index = self._selected_history_index()
        audio_text = self.audio_path.get().strip()
        if index is None or not audio_text:
            return
        answer = messagebox.askyesnocancel(
            "Eliminar fragmento",
            "Quieres quitar este fragmento del historial?\n\n"
            "Si: quitar del historial y eliminar archivos generados si no estan compartidos.\n"
            "No: quitar solo del historial.\n"
            "Cancelar: no hacer nada.",
        )
        if answer is None:
            return
        try:
            removed = remove_history_entry(self.history_path, Path(audio_text), self._full_history_index(Path(audio_text), index))
        except IndexError:
            messagebox.showerror("Historial", "La entrada seleccionada ya no existe.")
            self._refresh_history()
            return
        if answer:
            self._discard_output_artifacts(removed.output_dir)
        self.status.set("Fragmento eliminado del historial")
        self._refresh_history()

    def _full_history_index(self, audio_path: Path, visible_index: int) -> int:
        visible_entry = self._history_entries[visible_index]
        entries = load_history(self.history_path).entries_for(audio_path)
        for index, entry in enumerate(entries):
            if entry.id == visible_entry.id:
                return index
        raise IndexError("Entrada de historial no encontrada")

    def _discard_output_artifacts(self, output_dir: Path) -> None:
        history = load_history(self.history_path)
        if output_dir_reference_count(history, output_dir) > 0:
            messagebox.showwarning(
                "Salida compartida",
                "No se eliminaron archivos porque esta carpeta aparece en otros fragmentos del historial.",
            )
            return

        deleted_any = False
        for path in _generated_output_paths(output_dir):
            if not path.exists():
                continue
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                deleted_any = True
            except OSError as exc:
                messagebox.showwarning("No se pudo eliminar", f"No se pudo eliminar {path}: {exc}")
        if deleted_any:
            self.log.insert(tk.END, f"Archivos generados eliminados en {output_dir}\n")
            self.log.see(tk.END)

    def _start_auto_speaker_detection(self, turns: list[ConversationTurn]) -> bool:
        if self._auto_speaker_detection_running:
            return False
        if not turns or not has_ai_runner():
            return False
        self._auto_speaker_detection_running = True
        self._start_task("speaker_ai")
        self.speaker_progress.set("Detectando nombres de hablantes con IA")
        self.log.insert(tk.END, "Detectando nombres de hablantes con IA\n")
        self.log.see(tk.END)
        output_dir = self.last_output_dir or Path(self.output_dir.get())
        thread = threading.Thread(
            target=self._run_auto_speaker_detection,
            args=(turns, output_dir),
            daemon=True,
        )
        thread.start()
        return True

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

    def _open_speaker_editor(self, ai_response: str | None = None, wait: bool = False) -> None:
        output_dir = self.last_output_dir or Path(self.output_dir.get())
        turns = self.last_turns or _load_turns_from_output(output_dir)
        if not turns:
            messagebox.showwarning(
                "Sin transcripcion",
                "Primero procesa un audio o selecciona una carpeta de salida con transcript.json.",
            )
            return
        self.last_turns = turns
        audio_text = self.audio_path.get().strip()
        known_names: list[str] = []
        memory_status = "Memoria: sin audio seleccionado."
        if audio_text:
            memory = load_speaker_memory(self.speaker_memory_path)
            known_names = identity_names(memory, Path(audio_text))
            memory_status = format_speaker_memory_status(speaker_memory_status(memory, Path(audio_text)))
        dialog = SpeakerNameDialog(
            self,
            turns,
            output_dir,
            self._speaker_names_saved,
            ai_response,
            known_names,
            memory_status,
        )
        if wait:
            dialog.wait_window()

    def _speaker_names_saved(self, turns: list[ConversationTurn]) -> None:
        self.last_turns = turns
        audio_text = self.audio_path.get().strip()
        if audio_text and self._pending_completion is None:
            audio_path = Path(audio_text)
            remember_validated_turns(self.speaker_memory_path, audio_path, turns)
            try:
                config = self._current_config()
            except ValueError:
                config = None
            if config is not None:
                self._start_speaker_embedding_memory_update(audio_path, turns, config)
        self.speaker_progress.set("Nombres de hablantes guardados")
        output_dir = self.last_output_dir or Path(self.output_dir.get())
        self.status.set(f"Transcripcion actualizada en {output_dir}")
        self.preview.delete("1.0", tk.END)
        for turn in turns[-20:]:
            self.preview.insert(tk.END, f"[{_clock_time(turn.start)}] {turn.speaker}: {turn.text}\n")
        self.preview.see(tk.END)

    def _start_speaker_embedding_memory_update(
        self,
        audio_path: Path,
        turns: list[ConversationTurn],
        config: ProcessingConfig,
    ) -> None:
        thread = threading.Thread(
            target=self._update_speaker_embedding_memory,
            args=(audio_path, turns, config),
            daemon=True,
        )
        thread.start()

    def _update_speaker_embedding_memory(
        self,
        audio_path: Path,
        turns: list[ConversationTurn],
        config: ProcessingConfig,
    ) -> None:
        try:
            embeddings = self._try_extract_speaker_embeddings(audio_path, turns, config, report_errors=True)
            if not embeddings:
                self.events.put(
                    (
                        "speaker_memory_error",
                        "no se generaron huellas; se guardaran solo nombres y rangos validados",
                    )
                )
                return
            embeddings_by_name = {
                name: (embedding,)
                for name, embedding in embeddings.items()
            }
            remember_validated_turns(
                self.speaker_memory_path,
                audio_path,
                turns,
                embeddings_by_name=embeddings_by_name,
            )
        except Exception as exc:
            self.events.put(("speaker_memory_error", f"No se pudieron guardar huellas de voz: {exc}"))

    def _try_extract_speaker_embeddings(
        self,
        audio_path: Path,
        turns: list[ConversationTurn],
        config: ProcessingConfig,
        *,
        report_errors: bool = False,
    ) -> dict[str, tuple[float, ...]]:
        try:
            extractor = load_pyannote_embedding_extractor(
                resolve_ffmpeg_path(None),
                huggingface_token=config.huggingface_token,
                device=config.device,
            )
            return extract_speaker_embeddings(audio_path, turns, extractor)
        except Exception as exc:
            if report_errors:
                self.events.put(("speaker_memory_error", str(exc)))
            return {}


class SpeakerNameDialog(tk.Toplevel):
    def __init__(
        self,
        parent: App,
        turns: list[ConversationTurn],
        output_dir: Path,
        on_saved: object,
        ai_response: str | None = None,
        known_names: list[str] | None = None,
        memory_status: str = "",
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
        self.known_names = known_names or []
        self.memory_status = memory_status
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

        ttk.Label(
            root,
            text=self.memory_status,
            wraplength=700,
        ).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 12))

        for row, speaker in enumerate(speaker_labels(self.turns), start=2):
            ttk.Label(root, text=speaker).grid(row=row, column=0, sticky="w", pady=4)
            variable = tk.StringVar(value=speaker)
            self.name_vars[speaker] = variable
            ttk.Combobox(root, textvariable=variable, values=self.known_names).grid(
                row=row,
                column=1,
                sticky="ew",
                padx=8,
                pady=4,
            )
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


class SpeakerComparisonDialog(tk.Toplevel):
    def __init__(
        self,
        parent: App,
        audio_path: Path,
        base_source_id: str,
        sources: list[SpeakerSource],
        profiles_by_source: dict[str, list[SpeakerProfile]],
        store: SpeakerEmbeddingStore,
        on_generate_embeddings: object,
        on_save_corrections: object,
        on_play_profile: object,
    ) -> None:
        super().__init__(parent)
        self.title("Comparar hablantes entre salidas")
        self.geometry("1120x640")
        self.transient(parent)
        self.grab_set()
        self.audio_path = audio_path
        self.base_source_id = base_source_id
        self.sources = sources
        self.profiles_by_source = profiles_by_source
        self.store = store
        self.on_generate_embeddings = on_generate_embeddings
        self.on_save_corrections = on_save_corrections
        self.on_play_profile = on_play_profile
        self.pending_mapping: dict[str, str] = {}
        self.matches: list[SpeakerMatch] = []
        self.reference_var = tk.StringVar(value="Todas las salidas")
        self.filter_var = tk.StringVar(value="Todos")
        self.status_text = tk.StringVar(value="")
        self._build()
        self._refresh()

    def _build(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill=tk.BOTH, expand=True)
        controls = ttk.Frame(root)
        controls.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(controls, text="Referencia").pack(side=tk.LEFT)
        values = ["Todas las salidas", *[source.range_label for source in self.sources if source.entry_id != self.base_source_id]]
        self.reference_combo = ttk.Combobox(
            controls,
            textvariable=self.reference_var,
            values=values,
            state="readonly",
            width=42,
        )
        self.reference_combo.pack(side=tk.LEFT, padx=(8, 16))
        self.reference_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh())
        ttk.Label(controls, text="Filtro").pack(side=tk.LEFT)
        self.filter_combo = ttk.Combobox(
            controls,
            textvariable=self.filter_var,
            values=["Todos", "Sólo conflictos", "Sólo sin identificar", "Sólo alta confianza", "Sólo baja confianza"],
            state="readonly",
            width=22,
        )
        self.filter_combo.pack(side=tk.LEFT, padx=(8, 16))
        self.filter_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_table())
        self.generate_embeddings_button = ttk.Button(
            controls,
            text="Generar/actualizar huellas",
            command=self._generate_embeddings,
        )
        self.generate_embeddings_button.pack(side=tk.RIGHT)
        ttk.Label(root, textvariable=self.status_text).pack(anchor="w", pady=(0, 8))

        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True)
        matches_frame = ttk.Frame(notebook)
        matrix_frame = ttk.Frame(notebook)
        notebook.add(matches_frame, text="Coincidencias")
        notebook.add(matrix_frame, text="Matriz")

        columns = ("base", "candidate", "origin", "confidence", "name", "voice", "sample")
        self.table = ttk.Treeview(matches_frame, columns=columns, show="headings", height=12)
        headings = {
            "base": "En salida base",
            "candidate": "Mejor coincidencia",
            "origin": "Origen",
            "confidence": "Confianza",
            "name": "Nombre cuadra",
            "voice": "Voz",
            "sample": "Evidencia",
        }
        widths = {"base": 130, "candidate": 150, "origin": 170, "confidence": 130, "name": 160, "voice": 80, "sample": 320}
        for column in columns:
            self.table.heading(column, text=headings[column])
            self.table.column(column, width=widths[column], anchor="w")
        self.table.pack(fill=tk.BOTH, expand=True)

        matrix_columns = ("cluster", "names", "diagnosis")
        self.matrix = ttk.Treeview(matrix_frame, columns=matrix_columns, show="headings", height=12)
        self.matrix.heading("cluster", text="Voz")
        self.matrix.heading("names", text="Salidas/nombres")
        self.matrix.heading("diagnosis", text="Diagnóstico")
        self.matrix.column("cluster", width=120)
        self.matrix.column("names", width=650)
        self.matrix.column("diagnosis", width=220)
        self.matrix.pack(fill=tk.BOTH, expand=True)

        actions = ttk.Frame(root)
        actions.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(actions, text="Escuchar base", command=self._play_selected_base).pack(side=tk.LEFT)
        ttk.Button(actions, text="Escuchar referencia", command=self._play_selected_candidate).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Aplicar nombre", command=self._apply_selected_name).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Aplicar coincidencias seguras", command=self._apply_safe_matches).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Guardar correcciones", command=self._save_corrections).pack(side=tk.RIGHT)
        ttk.Button(actions, text="Cerrar", command=self.destroy).pack(side=tk.RIGHT, padx=(0, 8))

    def _refresh(self) -> None:
        base_profiles = self.profiles_by_source.get(self.base_source_id, [])
        candidates = self._candidate_profiles()
        self.matches = compare_speaker_profiles(base_profiles, candidates)
        with_embeddings = sum(1 for profiles in self.profiles_by_source.values() for profile in profiles if profile.embedding)
        total_profiles = sum(len(profiles) for profiles in self.profiles_by_source.values())
        self.status_text.set(f"Huellas disponibles: {with_embeddings} / {total_profiles}")
        self._refresh_table()
        self._refresh_matrix()

    def _candidate_profiles(self) -> list[SpeakerProfile]:
        selected_source = self._selected_reference_source()
        profiles = []
        for source_id, source_profiles in self.profiles_by_source.items():
            if source_id == self.base_source_id:
                continue
            if selected_source is not None and source_id != selected_source.entry_id:
                continue
            profiles.extend(source_profiles)
        return profiles

    def _refresh_table(self) -> None:
        self.table.delete(*self.table.get_children())
        for index, match in enumerate(self.matches):
            if not self._match_visible(match):
                continue
            candidate = match.candidate
            self.table.insert(
                "",
                tk.END,
                iid=str(index),
                values=(
                    match.base.display_name,
                    candidate.display_name if candidate else "-",
                    candidate.source.range_label if candidate else "-",
                    match.status,
                    match.name_status,
                    format_seconds(match.base.total_seconds),
                    match.base.sample,
                ),
            )

    def _refresh_matrix(self) -> None:
        self.matrix.delete(*self.matrix.get_children())
        all_profiles = [profile for profiles in self.profiles_by_source.values() for profile in profiles]
        for row in name_coherence_matrix(all_profiles):
            names = " | ".join(f"{source}: {name}" for source, name in row.names_by_source.items())
            self.matrix.insert("", tk.END, values=(row.cluster_id, names, row.diagnosis))

    def _match_visible(self, match: SpeakerMatch) -> bool:
        value = self.filter_var.get()
        if value == "Sólo conflictos":
            return "conflicto" in match.name_status.lower() or "distinta" in match.name_status.lower()
        if value == "Sólo sin identificar":
            return match.candidate is None or match.status == "Sin huellas disponibles"
        if value == "Sólo alta confianza":
            return match.status == "Coincidencia alta"
        if value == "Sólo baja confianza":
            return match.status in ("Coincidencia baja", "Sin huellas disponibles")
        return True

    def _selected_reference_source(self) -> SpeakerSource | None:
        value = self.reference_var.get()
        for source in self.sources:
            if source.range_label == value:
                return source
        return None

    def _selected_match(self) -> SpeakerMatch | None:
        selection = self.table.selection()
        if not selection:
            return None
        index = int(selection[0])
        if index < 0 or index >= len(self.matches):
            return None
        return self.matches[index]

    def _play_selected_base(self) -> None:
        match = self._selected_match()
        callback = self.on_play_profile
        if match is not None and callable(callback):
            callback(self.audio_path, match.base)

    def _play_selected_candidate(self) -> None:
        match = self._selected_match()
        callback = self.on_play_profile
        if match is not None and match.candidate is not None and callable(callback):
            callback(self.audio_path, match.candidate)

    def _apply_selected_name(self) -> None:
        match = self._selected_match()
        if match is None or match.candidate is None:
            return
        self.pending_mapping[match.base.label] = match.candidate.display_name
        self.status_text.set(f"Pendiente: {match.base.label} -> {match.candidate.display_name}")

    def _apply_safe_matches(self) -> None:
        for match in self.matches:
            if match.candidate is None:
                continue
            if match.status == "Coincidencia alta" and match.base.display_name != match.candidate.display_name:
                self.pending_mapping[match.base.label] = match.candidate.display_name
        self.status_text.set(f"Correcciones pendientes: {len(self.pending_mapping)}")

    def _save_corrections(self) -> None:
        callback = self.on_save_corrections
        if callable(callback):
            callback(self.audio_path, self.base_source_id, self.pending_mapping)
        self.destroy()

    def _generate_embeddings(self) -> None:
        callback = self.on_generate_embeddings
        if not callable(callback):
            return
        selected_source = self._selected_reference_source()
        source_ids = [self.base_source_id]
        if selected_source is None:
            source_ids.extend(source.entry_id for source in self.sources if source.entry_id != self.base_source_id)
        else:
            source_ids.append(selected_source.entry_id)
        self.generate_embeddings_button.configure(state=tk.DISABLED)
        self.status_text.set("Preparando generacion de huellas...")
        thread = threading.Thread(
            target=self._generate_embeddings_worker,
            args=(callback, source_ids),
            daemon=True,
        )
        thread.start()

    def _generate_embeddings_worker(self, callback: object, source_ids: list[str]) -> None:
        try:
            store = callback(
                self.audio_path,
                source_ids,
                lambda message: self.after(0, lambda: self.status_text.set(str(message))),
            )
        except Exception as exc:
            self.after(0, lambda: self._embedding_generation_failed(exc))
            return
        self.after(0, lambda: self._embedding_generation_finished(store))

    def _embedding_generation_failed(self, exc: Exception) -> None:
        self.generate_embeddings_button.configure(state=tk.NORMAL)
        self.status_text.set("No se pudieron generar huellas.")
        messagebox.showwarning("Comparar hablantes", f"No se pudieron generar huellas: {exc}")

    def _embedding_generation_finished(self, store: SpeakerEmbeddingStore) -> None:
        self.generate_embeddings_button.configure(state=tk.NORMAL)
        self.store = store
        for source in self.sources:
            source_embeddings = self.store.embeddings_for_source(self.audio_path, source.entry_id)
            if source.entry_id in self.profiles_by_source:
                turns = _load_turns_from_output(source.output_dir)
                self.profiles_by_source[source.entry_id] = build_speaker_profiles(
                    source,
                    turns,
                    embeddings=source_embeddings,
                )
        self._refresh()
        self.status_text.set(f"Huellas actualizadas: {self.store.count_embeddings(self.audio_path)} guardadas")


class MergeReviewDialog(tk.Toplevel):
    def __init__(
        self,
        parent: App,
        audio_path: Path,
        preview_audio_dir: Path,
        left_entry: HistoryEntry,
        right_entry: HistoryEntry,
        rows: list[MergeRow],
        known_names: list[str],
        on_saved: object,
    ) -> None:
        super().__init__(parent)
        self.title("Fusionar resultados")
        self.geometry("1180x720")
        self.transient(parent)
        self.grab_set()
        self.audio_path = audio_path
        self.preview_audio_dir = preview_audio_dir
        self.left_entry = left_entry
        self.right_entry = right_entry
        self.rows = rows
        self.known_names = list(known_names)
        self.on_saved = on_saved
        self.speaker_vars: list[tk.StringVar] = []
        self.speaker_combos: list[ttk.Combobox] = []
        self.text_widgets: list[tk.Text] = []
        self._build()

    def _build(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)
        identical_count = sum(1 for row in self.rows if row.is_identical)
        difference_count = len(self.rows) - identical_count
        ttk.Label(
            root,
            text=(
                f"Izquierda: {self.left_entry.output_dir}    "
                f"Derecha: {self.right_entry.output_dir}    "
                f"Filas: {len(self.rows)}    "
                f"Iguales: {identical_count}    "
                f"Con diferencias: {difference_count}"
            ),
            wraplength=1120,
        ).pack(anchor="w", pady=(0, 8))
        if difference_count == 0:
            tk.Label(
                root,
                text="Los dos resultados son identicos. Puedes guardar la fusion directamente.",
                bg="#d8f3dc",
                anchor="w",
                padx=8,
                pady=5,
            ).pack(fill=tk.X, pady=(0, 8))
        else:
            tk.Label(
                root,
                text=(
                    "Las palabras resaltadas son las diferencias. Haz clic en el texto de izquierda "
                    "o derecha para copiarlo a la version final; despues puedes corregirlo manualmente."
                ),
                bg="#ffe8a3",
                anchor="w",
                padx=8,
                pady=5,
            ).pack(fill=tk.X, pady=(0, 8))

        table_frame = ttk.Frame(root)
        table_frame.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(table_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=canvas.yview)
        body = ttk.Frame(canvas)
        body.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._bind_mousewheel_scroll(canvas)

        headers = ("Izquierda", "Derecha", "Hablante final", "Texto final", "Estado")
        widths = (28, 28, 18, 42, 12)
        for column, (header, width) in enumerate(zip(headers, widths)):
            ttk.Label(body, text=header, width=width).grid(row=0, column=column, sticky="w", padx=4, pady=(0, 6))

        for row_index, row in enumerate(self.rows, start=1):
            left_source = self._source_block(
                body,
                row,
                row_index - 1,
                row.left,
                "left",
                clickable=not row.is_identical,
            )
            left_source.grid(row=row_index, column=0, sticky="nsew", padx=4, pady=4)
            right_source = self._source_block(
                body,
                row,
                row_index - 1,
                row.right,
                "right",
                clickable=not row.is_identical,
            )
            right_source.grid(row=row_index, column=1, sticky="nsew", padx=4, pady=4)
            speaker_var = tk.StringVar(value=row.chosen_speaker)
            self.speaker_vars.append(speaker_var)
            speaker_frame = ttk.Frame(body)
            speaker_frame.grid(row=row_index, column=2, sticky="new", padx=4, pady=4)
            speaker_frame.columnconfigure(0, weight=1)
            combo = ttk.Combobox(speaker_frame, textvariable=speaker_var, values=self.known_names, width=18)
            combo.grid(row=0, column=0, sticky="ew")
            self.speaker_combos.append(combo)
            ttk.Button(
                speaker_frame,
                text="+",
                width=2,
                command=lambda i=row_index - 1: self._add_speaker_name(i),
            ).grid(row=0, column=1, sticky="e", padx=(4, 0))
            text = tk.Text(body, width=42, height=3, wrap="word")
            text.insert("1.0", row.chosen_text)
            text.grid(row=row_index, column=3, sticky="nsew", padx=4, pady=4)
            self.text_widgets.append(text)
            ttk.Label(body, text="Igual" if row.is_identical else "Clic texto").grid(
                row=row_index,
                column=4,
                sticky="nw",
                padx=4,
                pady=4,
            )

        actions = ttk.Frame(root)
        actions.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(actions, text="Guardar fusion", command=self._save).pack(side=tk.RIGHT)
        ttk.Button(actions, text="Cancelar", command=self.destroy).pack(side=tk.RIGHT, padx=(0, 8))

    def _source_block(
        self,
        parent: tk.Widget,
        row: MergeRow,
        row_index: int,
        turn: ConversationTurn | None,
        side: str,
        *,
        clickable: bool,
    ) -> tk.Frame:
        background = "#d8f3dc" if row.is_identical else "#ffffff"
        frame = tk.Frame(parent, bg=background)
        frame.columnconfigure(1, weight=1)
        play_button = ttk.Button(frame, text="▶", width=3, command=lambda: self._play_turn(turn))
        play_button.grid(row=0, column=0, sticky="nw", padx=(3, 4), pady=3)
        if turn is None:
            play_button.configure(state=tk.DISABLED)
        text = tk.Text(
            frame,
            width=28,
            height=_merge_source_line_count(turn),
            wrap="word",
            bg=background,
            relief=tk.FLAT,
            borderwidth=0,
            takefocus=False,
            cursor="hand2" if turn is not None and clickable else "",
        )
        text.tag_configure("changed", background="#ffb703")
        _insert_merge_source_text(text, row, turn, side)
        text.configure(state=tk.DISABLED)
        text.grid(row=0, column=1, sticky="nsew", padx=(0, 3), pady=3)
        if turn is not None and clickable:
            text.bind("<Button-1>", lambda _event, i=row_index, selected_side=side: self._choose_side(i, selected_side))
        return frame

    def _bind_mousewheel_scroll(self, canvas: tk.Canvas) -> None:
        def on_mousewheel(event: tk.Event) -> str:
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"

        canvas.bind_all("<MouseWheel>", on_mousewheel)
        canvas.bind_all("<Button-4>", on_mousewheel)
        canvas.bind_all("<Button-5>", on_mousewheel)

    def _play_turn(self, turn: ConversationTurn | None) -> None:
        if turn is None or turn.end <= turn.start:
            return
        clip_path = preview_clip_path(
            self.preview_audio_dir,
            self.audio_path,
            start_seconds=turn.start,
            end_seconds=turn.end,
        )
        try:
            if not clip_path.exists():
                extract_audio_range(resolve_ffmpeg_path(None), self.audio_path, clip_path, turn.start, turn.end)
            webbrowser.open(clip_path.resolve().as_uri())
        except Exception as exc:
            messagebox.showwarning("Reproducir audio", f"No se pudo reproducir este fragmento: {exc}")

    def _choose_side(self, index: int, side: str) -> None:
        draft = draft_from_source_turn(self.rows[index], side)
        if draft is None:
            return
        self.speaker_vars[index].set(draft.speaker)
        self.text_widgets[index].delete("1.0", tk.END)
        self.text_widgets[index].insert("1.0", draft.text)

    def _add_speaker_name(self, index: int) -> None:
        name = simpledialog.askstring("Nuevo hablante", "Nombre del hablante:", parent=self)
        if name is None:
            return
        updated = add_known_name(self.known_names, name)
        if updated == self.known_names:
            return
        self.known_names = updated
        for combo in self.speaker_combos:
            combo.configure(values=self.known_names)
        self.speaker_vars[index].set(self.known_names[-1])

    def _save(self) -> None:
        drafts = []
        for row, speaker_var, text_widget in zip(self.rows, self.speaker_vars, self.text_widgets):
            drafts.append(
                DraftMergeRow(
                    start=row.start,
                    end=row.end,
                    speaker=speaker_var.get(),
                    text=text_widget.get("1.0", tk.END),
                )
            )
        callback = self.on_saved
        if callable(callback):
            if callback(drafts) is False:
                return
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


def _history_entries_overlap(left: HistoryEntry, right: HistoryEntry) -> bool:
    left_start = left.start_seconds or 0.0
    right_start = right.start_seconds or 0.0
    left_end = left.end_seconds
    right_end = right.end_seconds
    if left_end is None or right_end is None:
        return True
    return min(left_end, right_end) > max(left_start, right_start)


def _speaker_source_from_entry(entry: HistoryEntry) -> SpeakerSource:
    return SpeakerSource(
        entry_id=entry.id or str(entry.output_dir),
        output_dir=entry.output_dir,
        range_label=f"{format_optional_range(entry.start_seconds, entry.end_seconds)} -> {entry.output_dir.name}",
    )


def _memory_profiles(source: SpeakerSource, memory: object, audio_path: Path) -> list[SpeakerProfile]:
    if not hasattr(memory, "identities_for"):
        return []
    profiles = []
    for identity in memory.identities_for(audio_path):  # type: ignore[attr-defined]
        sample_ranges = getattr(identity, "sample_ranges", ())
        total_seconds = sum(max(0.0, end - start) for start, end in sample_ranges)
        embeddings = tuple(getattr(identity, "embeddings", ()))
        profiles.append(
            SpeakerProfile(
                source=source,
                label=identity.name,
                display_name=identity.name,
                total_seconds=total_seconds,
                turn_count=len(sample_ranges),
                sample="Memoria validada",
                sample_start=sample_ranges[0][0] if sample_ranges else None,
                sample_end=sample_ranges[0][1] if sample_ranges else None,
                embedding=embeddings[0] if embeddings else None,
            )
        )
    return profiles


@contextlib.contextmanager
def _quiet_model_output() -> object:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield


def _merged_end_seconds(left: HistoryEntry, right: HistoryEntry) -> float | None:
    if left.end_seconds is None or right.end_seconds is None:
        return None
    return max(left.end_seconds, right.end_seconds)


def _format_merge_source(turn: ConversationTurn | None) -> str:
    if turn is None:
        return ""
    return f"[{format_seconds(turn.start)}] {turn.speaker}\n{turn.text}"


def _insert_merge_source_text(widget: tk.Text, row: MergeRow, turn: ConversationTurn | None, side: str) -> None:
    if turn is None:
        return
    widget.insert(tk.END, f"[{format_seconds(turn.start)}] {turn.speaker}\n")
    if row.left is None or row.right is None or not row.has_text_difference:
        widget.insert(tk.END, turn.text)
        return
    left_segments, right_segments = diff_text_segments(row.left.text, row.right.text)
    segments = left_segments if side == "left" else right_segments
    for text, changed in segments:
        widget.insert(tk.END, text, ("changed",) if changed else ())


def _merge_source_line_count(turn: ConversationTurn | None) -> int:
    if turn is None:
        return 3
    return 2 + (len(turn.text) // 42)


def _merge_row_background(row: MergeRow) -> str:
    if row.is_identical:
        return "#d8f3dc"
    return "#ffffff"


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


def _generated_output_paths(output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for basename in ("transcript", "transcript_raw"):
        for suffix in ("md", "txt", "json", "srt"):
            paths.append(output_dir / f"{basename}.{suffix}")
    paths.extend([output_dir / "speaker_audio", output_dir / "speaker_audio_parts"])
    return paths


def _memory_has_embeddings(memory: object, audio_path: Path) -> bool:
    if not hasattr(memory, "identities_for"):
        return False
    identities = memory.identities_for(audio_path)  # type: ignore[attr-defined]
    return any(getattr(identity, "embeddings", ()) for identity in identities)


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
