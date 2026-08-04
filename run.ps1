# Start TherapyTrace locally on Windows.
# Backend on :8000, frontend dev server on :5173 with an /api proxy.
#
# Usage from Windows Terminal (PowerShell):
#     .\run.ps1
#
# If PowerShell refuses to run it, allow local scripts for this session only:
#     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Require-Command($name, $hint) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        Write-Host "$name was not found on your PATH." -ForegroundColor Red
        Write-Host "  $hint"
        exit 1
    }
}

Require-Command "python" "Install Python 3.10+ from python.org and tick 'Add python.exe to PATH'."
Require-Command "npm"    "Install Node 18+ from nodejs.org, then reopen Windows Terminal."

# --- backend virtualenv ----------------------------------------------------
if (-not (Test-Path "backend\.venv")) {
    Write-Host "Creating the backend virtualenv..." -ForegroundColor Cyan
    python -m venv backend\.venv
    & backend\.venv\Scripts\python.exe -m pip install --quiet --upgrade pip
    & backend\.venv\Scripts\python.exe -m pip install --quiet -r backend\requirements.txt
}

# --- frontend packages -----------------------------------------------------
if (-not (Test-Path "frontend\node_modules")) {
    Write-Host "Installing frontend packages..." -ForegroundColor Cyan
    Push-Location frontend
    npm install
    Pop-Location
}

# --- start both in their own windows --------------------------------------
Write-Host "Starting the API..." -ForegroundColor Cyan
Start-Process -FilePath "$PSScriptRoot\backend\.venv\Scripts\uvicorn.exe" `
    -ArgumentList "app.main:app", "--reload", "--port", "8000" `
    -WorkingDirectory "$PSScriptRoot\backend"

Start-Sleep -Seconds 3

Write-Host "Starting the app..." -ForegroundColor Cyan
Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/c", "npm run dev" `
    -WorkingDirectory "$PSScriptRoot\frontend"

Write-Host ""
Write-Host "  API   http://localhost:8000/docs" -ForegroundColor Green
Write-Host "  App   http://localhost:5173" -ForegroundColor Green
Write-Host ""
Write-Host "Both are running in their own windows. Close those windows to stop them."
