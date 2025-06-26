# Setup Completo Windows - AI Crypto Trading Bot
# Installa TUTTE le dipendenze necessarie per Windows

param(
    [string]$InstallPath = "$env:USERPROFILE\tradingBot",
    [switch]$Silent = $false
)

$ErrorActionPreference = "Continue"

Write-Host "=== Setup Completo Windows AI Trading Bot ===" -ForegroundColor Cyan
Write-Host "Installazione completa di tutte le dipendenze..." -ForegroundColor Green

# Funzione per verificare privilegi admin
function Test-Admin {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Admin)) {
    Write-Host "IMPORTANTE: Esegui come Amministratore per installazione completa!" -ForegroundColor Red
    Write-Host "Continuo con installazione limitata..." -ForegroundColor Yellow
}

# 1. CHOCOLATEY (Package Manager Windows)
Write-Host "`n[1/8] Installazione Chocolatey..." -ForegroundColor Blue
try {
    if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        Write-Host "✓ Chocolatey installato" -ForegroundColor Green
    } else {
        Write-Host "✓ Chocolatey già presente" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠ Errore Chocolatey: $($_.Exception.Message)" -ForegroundColor Yellow
}

# 2. PYTHON 3.11 + PIP
Write-Host "`n[2/8] Installazione Python 3.11..." -ForegroundColor Blue
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -notmatch "Python 3\.[8-9]|Python 3\.1[0-9]") {
        if (Get-Command choco -ErrorAction SilentlyContinue) {
            choco install python --version=3.11.8 -y
        } else {
            # Download diretto da Python.org
            $pythonUrl = "https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe"
            $tempPath = "$env:TEMP\python-installer.exe"
            Invoke-WebRequest -Uri $pythonUrl -OutFile $tempPath
            Start-Process -FilePath $tempPath -ArgumentList "/quiet", "InstallAllUsers=0", "PrependPath=1" -Wait
        }
        Write-Host "✓ Python 3.11 installato" -ForegroundColor Green
    } else {
        Write-Host "✓ Python già presente: $pythonVersion" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠ Errore Python: $($_.Exception.Message)" -ForegroundColor Yellow
}

# 3. GIT
Write-Host "`n[3/8] Installazione Git..." -ForegroundColor Blue
try {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        if (Get-Command choco -ErrorAction SilentlyContinue) {
            choco install git -y
        } else {
            # Download diretto Git
            $gitUrl = "https://github.com/git-scm/git/releases/download/v2.42.0.windows.2/Git-2.42.0.2-64-bit.exe"
            $tempPath = "$env:TEMP\git-installer.exe"
            Invoke-WebRequest -Uri $gitUrl -OutFile $tempPath
            Start-Process -FilePath $tempPath -ArgumentList "/SILENT" -Wait
        }
        Write-Host "✓ Git installato" -ForegroundColor Green
    } else {
        Write-Host "✓ Git già presente" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠ Errore Git: $($_.Exception.Message)" -ForegroundColor Yellow
}

# 4. VISUAL C++ BUILD TOOLS (necessari per alcune librerie Python)
Write-Host "`n[4/8] Installazione Visual C++ Build Tools..." -ForegroundColor Blue
try {
    if (Get-Command choco -ErrorAction SilentlyContinue) {
        choco install visualcpp-build-tools -y
        choco install vcredist140 -y
        Write-Host "✓ Visual C++ Build Tools installati" -ForegroundColor Green
    } else {
        Write-Host "⚠ Chocolatey non disponibile, installa manualmente Visual Studio Build Tools" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠ Errore Build Tools: continuando..." -ForegroundColor Yellow
}

# Aggiorna PATH environment
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# 5. CLONAZIONE REPOSITORY
Write-Host "`n[5/8] Clonazione repository..." -ForegroundColor Blue
try {
    if (Test-Path $InstallPath) {
        Remove-Item -Path $InstallPath -Recurse -Force
    }
    
    git clone https://github.com/SudoRezi/tradingBot.git $InstallPath
    
    if (-not (Test-Path "$InstallPath\advanced_ai_system.py")) {
        throw "File principale non trovato"
    }
    Write-Host "✓ Repository clonato" -ForegroundColor Green
} catch {
    Write-Host "✗ Errore clonazione: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Set-Location -Path $InstallPath

# 6. VIRTUAL ENVIRONMENT PYTHON
Write-Host "`n[6/8] Setup ambiente Python..." -ForegroundColor Blue
try {
    python -m venv venv
    & ".\venv\Scripts\Activate.ps1"
    python -m pip install --upgrade pip setuptools wheel
    Write-Host "✓ Virtual environment creato" -ForegroundColor Green
} catch {
    Write-Host "✗ Errore virtual environment: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# 7. INSTALLAZIONE LIBRERIE PYTHON SPECIFICHE
Write-Host "`n[7/8] Installazione librerie Python..." -ForegroundColor Blue

# Librerie core per il trading bot
$corePackages = @(
    "streamlit==1.28.1",
    "pandas==2.0.3",
    "numpy==1.24.3", 
    "plotly==5.15.0",
    "requests==2.31.0"
)

# Librerie trading e crypto
$tradingPackages = @(
    "python-binance==1.0.19",
    "yfinance==0.2.18",
    "ccxt==4.0.74"
)

# Librerie AI e Machine Learning
$aiPackages = @(
    "scikit-learn==1.3.0",
    "joblib==1.3.1",
    "scipy==1.11.1"
)

# Librerie sistema e utilità
$systemPackages = @(
    "apscheduler==3.10.4",
    "cryptography==41.0.3",
    "psutil==5.9.5"
)

# Librerie web scraping e data
$dataPackages = @(
    "beautifulsoup4==4.12.2",
    "feedparser==6.0.10",
    "trafilatura==1.6.1",
    "sendgrid==6.10.0"
)

# Librerie aggiuntive Windows-specific
$windowsPackages = @(
    "pywin32==306",
    "wmi==1.5.1"
)

$allPackages = $corePackages + $tradingPackages + $aiPackages + $systemPackages + $dataPackages + $windowsPackages

foreach ($package in $allPackages) {
    try {
        Write-Host "Installazione $package..." -ForegroundColor Gray
        pip install $package --no-cache-dir
    } catch {
        Write-Host "⚠ Errore $package, continuando..." -ForegroundColor Yellow
    }
}

# Installazione librerie opzionali per performance
Write-Host "Installazione librerie performance opzionali..." -ForegroundColor Gray
$optionalPackages = @(
    "ta-lib-binary",
    "numba",
    "bottleneck"
)

foreach ($package in $optionalPackages) {
    try {
        pip install $package --no-cache-dir
    } catch {
        Write-Host "⚠ $package non installato (opzionale)" -ForegroundColor Yellow
    }
}

Write-Host "✓ Librerie Python installate" -ForegroundColor Green

# 8. CONFIGURAZIONE SISTEMA
Write-Host "`n[8/8] Configurazione finale..." -ForegroundColor Blue

# File .env configurazione
$envContent = @'
# AI Crypto Trading Bot Configuration
# ==================================

# Modalità Trading
TRADING_MODE=simulation
INITIAL_CAPITAL=10000
RISK_PERCENTAGE=2.0
MAX_POSITIONS=5

# Configurazione Sistema
SYSTEM_OS=windows
SYSTEM_ARCH=x64
LOG_LEVEL=INFO

# Server Configuration
STREAMLIT_HOST=0.0.0.0
STREAMLIT_PORT=5000

# API Keys (configura per trading live)
# BINANCE_API_KEY=your_binance_api_key
# BINANCE_SECRET_KEY=your_binance_secret_key
# COINBASE_API_KEY=your_coinbase_api_key  
# COINBASE_SECRET_KEY=your_coinbase_secret_key
# ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
# NEWSAPI_KEY=your_news_api_key
# HUGGINGFACE_API_TOKEN=your_hf_token

# Email Notifications
# EMAIL_SMTP_SERVER=smtp.gmail.com
# EMAIL_SMTP_PORT=587
# EMAIL_USERNAME=your_email@gmail.com
# EMAIL_PASSWORD=your_app_password

# Performance Optimization
ENABLE_PERFORMANCE_MODE=true
CPU_OPTIMIZATION=true
MEMORY_OPTIMIZATION=true
MAX_WORKERS=4
CACHE_SIZE=1000

# Windows Specific
WINDOWS_PRIORITY=HIGH
ENABLE_WINDOWS_OPTIMIZATION=true
'@

Set-Content -Path ".env" -Value $envContent -Encoding UTF8

# Shortcut Desktop
try {
    $desktopPath = [System.Environment]::GetFolderPath('Desktop')
    $shortcutPath = "$desktopPath\AI Trading Bot.lnk"
    
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = "powershell.exe"
    $shortcut.Arguments = "-WindowStyle Normal -Command `"cd '$InstallPath'; .\venv\Scripts\Activate.ps1; streamlit run advanced_ai_system.py --server.port 5000 --server.headless true --server.address 0.0.0.0`""
    $shortcut.WorkingDirectory = $InstallPath
    $shortcut.Description = "AI Crypto Trading Bot"
    $shortcut.Save()
    Write-Host "✓ Shortcut desktop creato" -ForegroundColor Green
} catch {
    Write-Host "⚠ Errore shortcut desktop" -ForegroundColor Yellow
}

# File batch per avvio rapido
$batchContent = @"
@echo off
echo Avvio AI Crypto Trading Bot...
cd /d "$InstallPath"
call venv\Scripts\activate.bat
echo Sistema avviato su: http://localhost:5000
streamlit run advanced_ai_system.py --server.port 5000 --server.headless true --server.address 0.0.0.0
pause
"@

Set-Content -Path "avvia-trading-bot.bat" -Value $batchContent -Encoding ASCII

# Test finale installazione
Write-Host "`nTest installazione..." -ForegroundColor Yellow
try {
    python -c "
import streamlit, pandas, numpy, requests, plotly
import yfinance, sklearn, cryptography, psutil
print('✓ Tutte le librerie principali importate correttamente')
"
    Write-Host "✓ Test superato" -ForegroundColor Green
} catch {
    Write-Host "⚠ Alcuni moduli potrebbero non funzionare" -ForegroundColor Yellow
}

# Summary finale
Write-Host ""
Write-Host "🎉 INSTALLAZIONE WINDOWS COMPLETATA!" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host "Directory: $InstallPath" -ForegroundColor Cyan
Write-Host "Avvio rapido: Doppio click 'AI Trading Bot' sul desktop" -ForegroundColor Cyan
Write-Host "Avvio manuale: avvia-trading-bot.bat" -ForegroundColor Cyan
Write-Host "URL interfaccia: http://localhost:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 LIBRERIE INSTALLATE:" -ForegroundColor Yellow
Write-Host "• Python 3.11 + pip" -ForegroundColor White
Write-Host "• Streamlit (interfaccia web)" -ForegroundColor White
Write-Host "• Pandas + NumPy (data processing)" -ForegroundColor White  
Write-Host "• Plotly (grafici avanzati)" -ForegroundColor White
Write-Host "• Python-binance + yfinance (exchange)" -ForegroundColor White
Write-Host "• Scikit-learn (machine learning)" -ForegroundColor White
Write-Host "• Cryptography (sicurezza)" -ForegroundColor White
Write-Host "• APScheduler (automazione)" -ForegroundColor White
Write-Host "• BeautifulSoup + Requests (web data)" -ForegroundColor White
Write-Host "• PyWin32 (Windows integration)" -ForegroundColor White
Write-Host ""
Write-Host "📋 PROSSIMI PASSI:" -ForegroundColor Yellow
Write-Host "1. Configura API keys nel file .env" -ForegroundColor White
Write-Host "2. Avvia tramite shortcut desktop" -ForegroundColor White
Write-Host "3. Accedi a http://localhost:5000" -ForegroundColor White
Write-Host "4. Inizia con modalità simulation" -ForegroundColor White

if (-not $Silent) {
    Write-Host ""
    Read-Host "Premi Invio per aprire la directory di installazione"
    Start-Process "explorer.exe" -ArgumentList $InstallPath
}