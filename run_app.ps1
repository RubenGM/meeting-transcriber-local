Set-Location -Path $PSScriptRoot
py scripts\bootstrap.py
if ($LASTEXITCODE -ne 0) {
  Write-Host ""
  Write-Host "La preparacion no se completo. Pulsa Enter para cerrar."
  Read-Host
}
