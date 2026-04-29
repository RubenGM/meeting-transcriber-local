@echo off
cd /d "%~dp0"
py scripts\bootstrap.py
if errorlevel 1 (
  echo.
  echo La preparacion no se completo. Pulsa una tecla para cerrar.
  pause > nul
)
