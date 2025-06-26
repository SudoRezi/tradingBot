# AI Crypto Trading Bot - Windows Installer Corretto
param(
    [string]$InstallPath = "$env:USERPROFILE\tradingBot",
    [switch]$Silent = $false
)

$ErrorActionPreference = "Stop"

Write-Host "=== AI Crypto Trading Bot - Installer Windows ===" -ForegroundColor Cyan
Write-Host "Installazione in: $InstallPath" -ForegroundColor Green

# Verifica privilegi amministratore
function Test-Admin {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Admin)) {
    Write-Host "ATTENZIONE: Privilegi amministratore raccomandati per installazione completa" -ForegroundColor Yellow
}

# Installazione Chocolatey
Write-Host "Controllo Chocolatey..." -ForegroundColor Yellow
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Installazione Chocolatey..." -ForegroundColor Yellow
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
}

# Installazione Python
Write-Host "Controllo Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -notmatch "Python 3\.[8-9]|Python 3\.1[0-9]") {
        Write-Host "Installazione Python 3.11..." -ForegroundColor Yellow
        choco install python --version=3.11.8 -y
    }
} catch {
    Write-Host "Installazione Python 3.11..." -ForegroundColor Yellow
    choco install python --version=3.11.8 -y
}

# Installazione Git
Write-Host "Controllo Git..." -ForegroundColor Yellow
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Installazione Git..." -ForegroundColor Yellow
    choco install git -y
}

# Aggiorna PATH
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Clonazione repository
Write-Host "Clonazione repository..." -ForegroundColor Yellow
if (Test-Path $InstallPath) {
    Remove-Item -Path $InstallPath -Recurse -Force
}

git clone https://github.com/SudoRezi/tradingBot.git $InstallPath
if (-not (Test-Path "$InstallPath\advanced_ai_system.py")) {
    throw "Errore clonazione repository"
}

# Setup Python environment
Write-Host "Setup ambiente Python..." -ForegroundColor Yellow
Set-Location -Path $InstallPath
python -m venv venv
& ".\venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip

# Installazione dipendenze
Write-Host "Installazione dipendenze..." -ForegroundColor Yellow
$packages = @(
    "streamlit==1.28.1",
    "pandas==2.0.3", 
    "numpy==1.24.3",
    "plotly==5.15.0",
    "requests==2.31.0",
    "python-binance==1.0.19",
    "yfinance==0.2.18",
    "scikit-learn==1.3.0",
    "apscheduler==3.10.4",
    "cryptography==41.0.3",
    "beautifulsoup4==4.12.2",
    "feedparser==6.0.10",
    "trafilatura==1.6.1",
    "sendgrid==6.10.0",
    "psutil==5.9.5",
    "joblib==1.3.1",
    "scipy==1.11.1"
)

foreach ($package in $packages) {
    pip install $package
}

# Creazione file .env
Write-Host "Creazione configurazione..." -ForegroundColor Yellow
$envContent = @'
# AI Crypto Trading Bot Configuration
TRADING_MODE=simulation
INITIAL_CAPITAL=10000
RISK_PERCENTAGE=2.0
MAX_POSITIONS=5

# Sistema
SYSTEM_OS=windows
LOG_LEVEL=INFO

# API Keys (configura per trading live)
# BINANCE_API_KEY=your_key_here
# BINANCE_SECRET_KEY=your_secret_here

# Performance
ENABLE_PERFORMANCE_MODE=true
CPU_OPTIMIZATION=true
MEMORY_OPTIMIZATION=true
'@

Set-Content -Path ".env" -Value $envContent -Encoding UTF8

# Creazione shortcut desktop
Write-Host "Creazione shortcut..." -ForegroundColor Yellow
$desktopPath = [System.Environment]::GetFolderPath('Desktop')
$shortcutPath = "$desktopPath\AI Trading Bot.lnk"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = "powershell.exe"
$shortcut.Arguments = "-Command `"cd '$InstallPath'; .\venv\Scripts\Activate.ps1; streamlit run advanced_ai_system.py --server.port 5000 --server.headless true --server.address 0.0.0.0`""
$shortcut.WorkingDirectory = $InstallPath
$shortcut.Description = "AI Crypto Trading Bot"
$shortcut.Save()

# Creazione batch file
$batchContent = @"
@echo off
cd /d "$InstallPath"
call venv\Scripts\activate.bat
streamlit run advanced_ai_system.py --server.port 5000 --server.headless true --server.address 0.0.0.0
pause
"@

Set-Content -Path "tradingbot.bat" -Value $batchContent -Encoding ASCII

# Test installazione
Write-Host "Test installazione..." -ForegroundColor Yellow
python -c "import streamlit, pandas, numpy; print('Dipendenze OK')"

Write-Host ""
Write-Host "INSTALLAZIONE COMPLETATA!" -ForegroundColor Green
Write-Host "=========================" -ForegroundColor Green
Write-Host "Directory: $InstallPath" -ForegroundColor Cyan
Write-Host "Avvio: Doppio click su 'AI Trading Bot' desktop" -ForegroundColor Cyan
Write-Host "URL: http://localhost:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "PROSSIMI PASSI:" -ForegroundColor Yellow
Write-Host "1. Configura API keys nel file .env" -ForegroundColor White
Write-Host "2. Avvia tramite shortcut desktop" -ForegroundColor White
Write-Host "3. Accedi a http://localhost:5000" -ForegroundColor White

if (-not $Silent) {
    Read-Host "Premi Invio per aprire la directory"
    Start-Process "explorer.exe" -ArgumentList $InstallPath
}