# Сборка одного exe без тяжёлых библиотек (только tkinter + urllib).
# Запуск из каталога scientific_parser:
#   pip install pyinstaller
#   powershell -ExecutionPolicy Bypass -File desktop_client/build_windows.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    Write-Host "Установите: pip install pyinstaller"
    exit 1
}

pyinstaller --noconfirm --onefile --windowed --name "ScientificParserThin" "desktop_client/sp_desktop.py"

Write-Host "Готово: dist/ScientificParserThin.exe"
