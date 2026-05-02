#Requires -Version 5.1
<#
  Собирает один файл dist\ScientificParser.exe (без установщика, без папки пакета).
  Запуск из каталога scientific_parser:
    powershell -ExecutionPolicy Bypass -File .\build_portable.ps1
#>
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
  Write-Host "Нет .venv. Создаю и ставлю зависимости..."
  py -3 -m venv .venv
  $Py = Join-Path $Root ".venv\Scripts\python.exe"
  & $Py -m pip install -U pip
  & $Py -m pip install -r requirements.txt
}

Write-Host "Зависимости portable (окно приложения pywebview)..."
& $Py -m pip install -r (Join-Path $Root "requirements-portable.txt")

Write-Host "Ставлю PyInstaller (если ещё нет)..."
& $Py -m pip install -r (Join-Path $Root "requirements-build.txt")

$Model = Join-Path $Root "models\block_classifier.joblib"
if (-not (Test-Path $Model)) {
  Write-Warning "Нет models\block_classifier.joblib — соберётся, но на целевой машине ML может не работать. Обучите: py train_evaluate.py"
}

Write-Host "Запуск PyInstaller (one-file .exe)..."
& $Py -m PyInstaller scientific_parser_portable.spec --noconfirm
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$DistExe = Join-Path $Root "dist\ScientificParser.exe"
if (-not (Test-Path $DistExe)) {
  Write-Error "Не найден $DistExe после сборки."
  exit 1
}

$Readme = Join-Path $Root "README_PORTABLE.txt"
if (Test-Path $Readme) {
  Copy-Item -Force $Readme (Join-Path $Root "dist\README_PORTABLE.txt")
}

$Zip = Join-Path $Root ("dist\ScientificParser_win64_" + (Get-Date -Format "yyyyMMdd") + ".zip")
if (Test-Path $Zip) { Remove-Item -Force $Zip }
Compress-Archive -Path $DistExe -DestinationPath $Zip -Force

Write-Host ""
Write-Host "Готово:"
Write-Host "  EXE:   $DistExe"
Write-Host "  (опционально) README: dist\README_PORTABLE.txt"
Write-Host "  ZIP:   $Zip  (только .exe внутри)"
Write-Host "Передайте человеку ScientificParser.exe или ZIP."
