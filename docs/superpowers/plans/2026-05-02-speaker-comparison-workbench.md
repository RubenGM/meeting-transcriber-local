# Speaker Comparison Workbench Plan

> Plan para ejecutar mañana, 2026-05-02. Objetivo: convertir `Comparar personas` en una herramienta útil para contrastar voces, nombres y coherencia entre varias salidas ya analizadas.

## Problema Actual

La ventana actual de `Comparar personas` sólo compara una salida contra la memoria de nombres/huellas del audio. Eso deja varios huecos:

- No permite elegir otra salida como referencia.
- No muestra si `Persona 1` en una salida parece la misma voz que `Marta`, `Fina` o `Persona 3` en otra.
- No separa claramente tres conceptos distintos: nombre ya validado, coincidencia textual y coincidencia por voz.
- Si no hay embeddings guardados, la UI no ayuda a entender qué falta.
- No permite corregir nombres desde la propia comparación.

## Objetivo

Crear una sección nueva tipo “comparador de hablantes” donde podamos:

- Comparar las voces de una salida con cualquier otra salida analizada del mismo audio.
- Detectar si los nombres asignados cuadran entre fragmentos.
- Ver conflictos claros: misma voz con nombres distintos, mismo nombre con voces diferentes, voces sin identificar.
- Convertir una comparación buena en correcciones/memoria validada sin tener que salir de la pantalla.

## Propuesta De UX

Sustituir la ventana simple actual por una ventana más grande: `Comparar hablantes entre salidas`.

Arriba:

- salida base seleccionada
- selector de salida de referencia: `Memoria completa`, o cualquier entrada visible/oculta del historial
- estado de huellas: `X hablantes con huella`, `sin huellas`, `extraccion pendiente`, `error de pyannote`
- botón `Generar/actualizar huellas`

Tabla principal:

| En salida base | Mejor coincidencia | Origen | Confianza | Nombre cuadra | Voz | Evidencia |
|---|---|---|---|---|---|---|
| Persona 1 | Marta | 00:05:00 -> 00:27:47 | alta | conflicto | 00:02:15 | muestra + play |
| Marta | Marta | memoria | validado | ok | 00:04:37 | muestra + play |
| Persona 5 | sin match | - | baja | pendiente | 00:00:03 | muestra + play |

Acciones por fila:

- `Aplicar nombre`
- `Marcar como nueva persona`
- `Ignorar`
- `Escuchar base`
- `Escuchar referencia`

Acciones globales:

- `Aplicar coincidencias seguras`
- `Guardar correcciones`
- `Recalcular huellas`
- `Abrir salida`

## Mejoras Propuestas

### 1. Comparar Contra Cualquier Salida

Permitir seleccionar una salida de referencia desde el historial del mismo audio.

La comparación debe funcionar en tres modos:

- `Contra memoria`: compara con nombres/huellas ya validados.
- `Contra otra salida`: compara hablantes de salida A contra hablantes de salida B.
- `Contra todas`: busca la mejor coincidencia entre todas las salidas disponibles.

### 2. Matriz De Coherencia De Nombres

Añadir una vista secundaria `Matriz`:

| Voz detectada | Salida 1 | Salida 2 | Salida 3 | Diagnóstico |
|---|---|---|---|---|
| Cluster A | Núria | Núria | Persona 1 | falta aplicar nombre |
| Cluster B | Alan Bernan | Alícia | Alan Bernan | posible mezcla/conflicto |

Esto ayuda a responder rápido: “¿los nombres cuadran entre fragmentos?”.

### 3. Huellas Bajo Demanda

Si faltan embeddings, la app debe poder generarlos desde esta pantalla para:

- la salida seleccionada
- la salida de referencia
- todas las salidas validadas del audio

Debe mostrar errores accionables si pyannote/Token HF no permite extraer huellas.

### 4. Confianza Más Clara

Cambiar `Sin coincidencia` por estados más útiles:

- `Sin huellas disponibles`
- `No hay suficiente voz`
- `Coincidencia baja`
- `Coincidencia media`
- `Coincidencia alta`
- `Conflicto de nombre`
- `Nombre validado`

### 5. Revisión Con Audio

Cada fila debe tener play corto para:

- muestra de la voz base
- muestra de la voz candidata

Idealmente reutilizar el sistema de preescucha ya añadido a la fusión.

### 6. Correcciones Directas

Desde la comparación, permitir:

- renombrar un hablante de la salida base
- añadir nombre nuevo
- aplicar el nombre sugerido sólo a esa salida
- guardar cambios en `transcript.*`
- actualizar `speaker_memory.json`

### 7. Filtro De Problemas

Añadir filtros:

- `Todos`
- `Sólo conflictos`
- `Sólo sin identificar`
- `Sólo alta confianza`
- `Sólo baja confianza`

## Diseño Técnico

### Nuevos Módulos

`speaker_cross_compare.py`

Responsable de lógica pura:

- cargar hablantes agregados por salida
- construir perfiles por hablante
- comparar embeddings entre salidas
- detectar conflictos de nombre
- producir filas de diagnóstico

Tipos propuestos:

```python
@dataclass(frozen=True)
class SpeakerSource:
    entry_id: str
    output_dir: Path
    range_label: str

@dataclass(frozen=True)
class SpeakerProfile:
    source: SpeakerSource
    label: str
    display_name: str
    total_seconds: float
    turn_count: int
    sample: str
    embedding: tuple[float, ...] | None

@dataclass(frozen=True)
class SpeakerMatch:
    base: SpeakerProfile
    candidate: SpeakerProfile | None
    score: float | None
    status: str
    name_status: str
```

`speaker_embedding_store.py`

Responsable de cachear huellas por salida/hablante para no recalcular todo cada vez.

### Cambios En Módulos Existentes

`speaker_fingerprints.py`

- exponer extracción por salida/hablante con errores diferenciados
- devolver motivo cuando no hay muestras suficientes

`speaker_memory.py`

- mantener compatibilidad con memoria actual
- permitir asociar varias huellas por nombre y por rango/salida

`gui.py`

- reemplazar/expandir `SpeakerComparisonDialog`
- añadir selector de referencia
- añadir tabla de matches
- añadir acciones de corrección
- reutilizar preescucha de audio

`history.py`

- exponer historial visible y oculto para comparación
- permitir listar salidas por audio con id/rango/output_dir

## Plan De Ejecución

### Fase 1: Modelo Y Tests Puros

- Crear `tests/test_speaker_cross_compare.py`.
- Crear `speaker_cross_compare.py`.
- Testear:
  - agregación de turnos por hablante
  - cálculo de duración y muestras
  - comparación de embeddings
  - detección de `mismo nombre / voz distinta`
  - detección de `misma voz / nombre distinto`
  - estados cuando faltan huellas

Comando:

```bash
PYTHONPATH=src python -m unittest tests.test_speaker_cross_compare
```

### Fase 2: Cache/Huellas Bajo Demanda

- Crear o ampliar tests de `speaker_fingerprints`.
- Añadir cache por `audio_path + output_dir + speaker`.
- Guardar errores legibles:
  - modelo no accesible
  - token HF inválido
  - muestra demasiado corta
  - ffmpeg falló

Comando:

```bash
PYTHONPATH=src python -m unittest tests.test_speaker_fingerprints tests.test_speaker_memory
```

### Fase 3: Nueva Ventana De Comparación

- Cambiar `Comparar personas` para abrir el workbench.
- Añadir selector de referencia:
  - `Memoria completa`
  - cada salida del historial
  - `Todas las salidas`
- Mostrar filas con estado, confianza, nombre sugerido y acciones.
- Añadir filtros.
- Añadir play base/referencia.

Validación manual:

- seleccionar fragmento 00:00 -> 00:05
- comparar contra 00:05 -> 00:27
- comparar contra salida fusionada
- confirmar que conflictos aparecen claramente

### Fase 4: Correcciones Y Guardado

- Implementar `Aplicar nombre`.
- Implementar `Aplicar coincidencias seguras`.
- Reescribir `transcript.*` de la salida base.
- Actualizar memoria de hablantes.
- Mantener rollback: no borrar salidas ni tocar otras entradas.

Tests:

- aplicar nombre cambia todos los turnos de ese hablante
- guardar correcciones actualiza exports
- memoria recibe nombres validados

### Fase 5: Documentación Y Limpieza

- Actualizar `README.md`.
- Actualizar `docs/FUNCIONAMIENTO.md`.
- Añadir sección “Comparación de voces entre fragmentos”.
- Ejecutar suite completa.

Comandos finales:

```bash
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python -m py_compile src/meeting_transcriber/*.py
```

## Criterios De Aceptación

- Puedo comparar una salida con cualquier otra salida analizada del mismo audio.
- Puedo comparar una salida contra todas las salidas/memoria.
- La UI indica si faltan huellas y permite generarlas.
- La UI muestra conflictos de nombres entre salidas.
- Puedo escuchar muestras base y referencia.
- Puedo aplicar nombres sugeridos y guardar la transcripción corregida.
- Las coincidencias tienen estados comprensibles, no sólo `Sin coincidencia`.
- La suite de tests queda verde.

## Riesgos

- Pyannote embedding puede no estar accesible por permisos/token. La UI debe degradar bien y explicar el motivo.
- Comparar voces sin embeddings reales no debe inventar coincidencias.
- Si hay mezclas de voz dentro de una misma etiqueta, la confianza debe mostrarse como baja o conflicto.
- La ventana puede crecer demasiado; usar filtros y panel de detalle evitará saturarla.

## Orden Recomendado Para Mañana

1. Empezar por `speaker_cross_compare.py` con tests puros.
2. Integrar cache/estado de huellas.
3. Hacer una primera UI de sólo lectura.
4. Añadir acciones de corrección.
5. Probar con el audio real y ajustar estados/confianza.
