# Sonelgaz - demarrage simple (Windows PowerShell)
# PowerShell dans ce dossier : Unblock-File -Path .\run_all.ps1  puis  .\run_all.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host "  Sonelgaz - demarrage du projet" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERREUR: Python n'est pas installe ou pas dans le PATH." -ForegroundColor Red
    Write-Host "Installez Python 3.10+ depuis https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host 'Cochez "Add python.exe to PATH" pendant l''installation.' -ForegroundColor Yellow
    Read-Host "Appuyez sur Entree pour fermer"
    exit 1
}

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "[1/4] Creation de l'environnement Python (.venv)..." -ForegroundColor Green
    python -m venv .venv
} else {
    Write-Host "[1/4] Environnement .venv deja present - OK" -ForegroundColor Green
}

Write-Host "[2/4] Installation des bibliotheques (premiere fois: plusieurs minutes)..." -ForegroundColor Green
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt

Write-Host "[3/4] Generation des donnees..." -ForegroundColor Green
& $py generate_data.py

Write-Host "[4/4] Entrainement des modeles (plusieurs minutes)..." -ForegroundColor Green
& $py train_models.py

Write-Host ""
Write-Host "Lancement de l'application..." -ForegroundColor Cyan
& $py app.py

Read-Host "Appuyez sur Entree pour fermer"
