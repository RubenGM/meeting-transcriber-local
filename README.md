# Meeting Transcriber

Aplicacion de escritorio en Python para transcribir reuniones largas y separar la conversacion por hablante. Esta pensada para funcionar de forma local, sin enviar el audio a APIs externas de transcripcion.

## Que hace

- Transcribe audio o video con `faster-whisper`.
- Separa voces por hablante con `pyannote.audio`.
- Genera salidas en Markdown, texto, SRT y JSON.
- Permite procesar solo un rango del audio, util para reuniones de 2 horas o mas.
- Muestra progreso, vista previa de la conversacion e historial de fragmentos procesados.
- Puede exportar audio separado por hablante.
- Permite renombrar hablantes manualmente.
- Si detecta `opencode` o `codex`, puede pedir ayuda a una IA externa para proponer nombres de hablantes a partir de la transcripcion.

## Privacidad

La transcripcion y la diarizacion se ejecutan en la maquina local. No se usa ninguna API externa para procesar el audio.

Hay dos excepciones opcionales:

- Hugging Face puede usarse para descargar modelos la primera vez.
- `opencode` o `codex` pueden usarse si el usuario quiere proponer nombres reales de hablantes con IA.

## Requisitos

- Python 3.10 o superior.
- Conexion a internet durante la primera preparacion, para instalar dependencias y descargar modelos.
- Token de Hugging Face para usar el modelo de diarizacion de pyannote si no esta ya descargado.
- GPU NVIDIA opcional. La app tambien puede funcionar en CPU, aunque sera mas lenta.

## Uso Rapido

### Windows

Doble clic en:

```text
run_app.bat
```

o, si prefieres PowerShell:

```text
run_app.ps1
```

### Linux

```bash
./run_app.sh
```

El primer arranque abre un instalador guiado que:

- crea un entorno local `.venv`
- instala las dependencias
- prepara `ffmpeg` embebido mediante `imageio-ffmpeg`
- prepara librerias CUDA locales en Linux cuando procede
- descarga/prepara el modelo Whisper equilibrado
- abre la aplicacion al terminar

No modifica el Python global ni instala paquetes fuera de la carpeta del proyecto.

En ejecuciones posteriores, si el entorno ya esta listo, se abre directamente la ventana.

## Forzar Instalacion de Nuevo

```bash
python scripts/bootstrap.py --setup
```

## Flujo de Trabajo

1. Selecciona un archivo de audio o video.
2. Elige la carpeta de salida.
3. Selecciona idioma: automatico, catalan, espanol, ingles, frances, aleman, italiano o portugues.
4. Ajusta el rango de inicio/fin si solo quieres procesar una parte.
5. Indica minimo y maximo de hablantes si lo sabes.
6. Pulsa `Probar rendimiento` para que la app recomiende CPU/CUDA y tipo de computo.
7. Pulsa `Procesar`.

Durante el proceso veras:

- progreso de transcripcion
- velocidad y ETA aproximada
- vista previa del texto
- progreso de separacion de voces cuando pyannote lo permite
- resumen de hablantes detectados

## Calidad de Separacion de Voces

La opcion `Separacion voces` ofrece:

- `Rapida`: menos costosa, util para pruebas.
- `Precisa`: usa timestamps por palabra y alinea el texto con los hablantes con mas detalle.
- `Muy precisa`: intenta una configuracion mas estricta; puede tardar mas y en audios malos puede detectar hablantes de mas.

Para reuniones con muchas personas, ayuda mucho indicar un rango realista de hablantes. Por ejemplo, si esperas unas 12 personas, usar `Min 8` y `Max 15` suele ser mejor que dejarlo completamente libre.

## Token de Hugging Face

La diarizacion usa:

```text
pyannote/speaker-diarization-community-1
```

Si el modelo no esta descargado, puede ser necesario:

1. abrir la pagina del modelo desde la app
2. aceptar el acceso/licencia en Hugging Face
3. crear un token de Hugging Face
4. pegarlo en el campo `Token HF`

La app hace esta comprobacion al inicio del procesamiento para evitar esperar horas y fallar al final.

## Salidas Generadas

En la carpeta de salida se generan:

```text
transcript.md
transcript.txt
transcript.srt
transcript.json
transcript_raw.md
transcript_raw.txt
transcript_raw.srt
transcript_raw.json
```

`transcript_raw.*` contiene la transcripcion sin diarizar. Se guarda antes de separar hablantes para no perder el trabajo si pyannote falla.

Si activas `Exportar audio separado por hablante`, tambien se crea:

```text
speaker_audio/
```

con archivos como `Persona_1.wav`, `Persona_2.wav`, etc.

## Renombrar Hablantes

Despues de procesar, puedes usar `Renombrar hablantes` para sustituir `Persona 1`, `Persona 2`, etc. por nombres reales.

Si tienes `opencode` o `codex` instalados, la app puede lanzar automaticamente una deteccion de nombres al finalizar. Esta funcion no es necesaria para transcribir: solo ayuda a rellenar propuestas cuando la conversacion incluye presentaciones.

## Desarrollo

Instalar/preparar entorno:

```bash
python scripts/bootstrap.py --setup
```

Ejecutar app:

```bash
python scripts/bootstrap.py
```

Ejecutar tests:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

Reinstalar el paquete local despues de cambios:

```bash
.venv/bin/python -m pip install --upgrade --no-build-isolation .
```

## Notas Sobre CUDA

CUDA es opcional. En Linux, el instalador prepara librerias locales necesarias para `faster-whisper` y PyTorch/pyannote cuando es posible.

Si CUDA falla durante la diarizacion, la app intenta reintentar en CPU antes de rendirse. Si la GPU se queda sin memoria, prueba:

- reducir el rango de audio
- usar CPU para diarizacion
- usar una calidad de transcripcion menor
- cerrar otras aplicaciones que usen GPU

## Limitaciones

La separacion por hablante depende mucho de la calidad del audio. En reuniones con muchas personas, ruido, solapamientos o presentaciones muy rapidas, puede mezclar hablantes. El modo `Precisa` ayuda porque alinea palabra a palabra, pero no convierte un audio dificil en perfecto.

Para mejores resultados:

- usa audio lo mas limpio posible
- indica minimo y maximo de hablantes
- procesa fragmentos cortos si la reunion es muy larga
- revisa y renombra hablantes al final
