# AI Crypto Trading Bot - GitHub Direct Installer (Windows)
# Installa direttamente da GitHub Repository

param(
    [string]$InstallPath = "$env:USERPROFILE\tradingBot",
    [string]$GitHubRepo = "https://github.com/SudoRezi/tradingBot.git",
    [switch]$Silent = $false
)

$ErrorActionPreference = "Stop"

Write-Host "=== AI Crypto Trading Bot - GitHub Installer ===" -ForegroundColor Cyan
Write-Host "Installing from GitHub to: $InstallPath" -ForegroundColor Green

# Funzione per verificare se l'utente é amministratore
function Test-Admin {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Verifica privilegi amministratore
if (-not (Test-Admin)) {
    Write-Host "ERRORE: Questo script richiede privilegi di amministratore!" -ForegroundColor Red
    Write-Host "Fai click destro su PowerShell e seleziona 'Esegui come amministratore'" -ForegroundColor Yellow
    Read-Host "Premi Invio per uscire"
    exit 1
}

# Funzione per installare Chocolatey se necessario
function Install-Chocolatey {
    Write-Host "Installazione Chocolatey..." -ForegroundColor Yellow
    try {
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        Write-Host "✓ Chocolatey installato" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "✗ Errore installazione Chocolatey: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Funzione per installare Python
function Install-Python {
    Write-Host "Verifica installazione Python..." -ForegroundColor Yellow
    
    try {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match "Python 3\.[8-9]|Python 3\.1[0-9]") {
            Write-Host "✓ Python trovato: $pythonVersion" -ForegroundColor Green
            return $true
        }
    }
    catch {
        Write-Host "Python non trovato o versione incompatibile" -ForegroundColor Red
    }
    
    Write-Host "Installazione Python 3.11 tramite Chocolatey..." -ForegroundColor Yellow
    try {
        choco install python --version=3.11.8 -y
        # Refresh environment
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        Write-Host "✓ Python 3.11 installato" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "✗ Errore installazione Python: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Funzione per installare Git
function Install-Git {
    Write-Host "Verifica installazione Git..." -ForegroundColor Yellow
    
    try {
        $gitVersion = git --version 2>&1
        if ($gitVersion -match "git version") {
            Write-Host "✓ Git trovato: $gitVersion" -ForegroundColor Green
            return $true
        }
    }
    catch {
        Write-Host "Git non trovato" -ForegroundColor Red
    }
    
    Write-Host "Installazione Git tramite Chocolatey..." -ForegroundColor Yellow
    try {
        choco install git -y
        # Refresh environment
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        Write-Host "✓ Git installato" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "✗ Errore installazione Git: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Funzione per clonare repository
function Clone-Repository {
    Write-Host "Clonazione repository da GitHub..." -ForegroundColor Yellow
    
    try {
        # Rimuovi directory esistente se presente
        if (Test-Path $InstallPath) {
            Write-Host "Rimozione installazione precedente..." -ForegroundColor Yellow
            Remove-Item -Path $InstallPath -Recurse -Force
        }
        
        # Clona repository
        git clone $GitHubRepo $InstallPath
        
        if (Test-Path "$InstallPath\advanced_ai_system.py") {
            Write-Host "✓ Repository clonato con successo" -ForegroundColor Green
            return $true
        } else {
            Write-Host "✗ File principali non trovati dopo clonazione" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "✗ Errore clonazione repository: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Funzione per installare dipendenze Python
function Install-PythonPackages {
    Write-Host "Installazione dipendenze Python..." -ForegroundColor Yellow
    
    try {
        Set-Location -Path $InstallPath
        
        # Crea virtual environment
        python -m venv venv
        
        # Attiva virtual environment
        & ".\venv\Scripts\Activate.ps1"
        
        # Upgrade pip
        python -m pip install --upgrade pip
        
        # Installa requirements se esiste
        if (Test-Path "requirements.txt") {
            pip install -r requirements.txt
        } else {
            # Installa pacchetti essenziali
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
                Write-Host "Installazione $package..." -ForegroundColor Gray
                pip install $package
            }
        }
        
        Write-Host "✓ Dipendenze Python installate" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "✗ Errore installazione dipendenze: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Funzione per creare file di configurazione
function Create-Configuration {
    Write-Host "Creazione file di configurazione..." -ForegroundColor Yellow
    
    try {
        # Crea file .env se non esiste
        $envPath = "$InstallPath\.env"
        if (-not (Test-Path $envPath)) {
            $envContent = @"
# AI Crypto Trading Bot Configuration
# =====================================

# Trading Configuration
TRADING_MODE=simulation
INITIAL_CAPITAL=10000
RISK_PERCENTAGE=2.0
MAX_POSITIONS=5

# System Configuration
SYSTEM_OS=windows
SYSTEM_ARCH=x64
LOG_LEVEL=INFO

# API Keys (da configurare per trading live)
# BINANCE_API_KEY=your_binance_api_key_here
# BINANCE_SECRET_KEY=your_binance_secret_key_here
# COINBASE_API_KEY=your_coinbase_api_key_here
# COINBASE_SECRET_KEY=your_coinbase_secret_key_here
# ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
# NEWSAPI_KEY=your_news_api_key_here

# Email Notifications (opzionale)
# EMAIL_SMTP_SERVER=smtp.gmail.com
# EMAIL_SMTP_PORT=587
# EMAIL_USERNAME=your_email@gmail.com
# EMAIL_PASSWORD=your_app_password_here

# Performance Settings
ENABLE_PERFORMANCE_MODE=true
CPU_OPTIMIZATION=true
MEMORY_OPTIMIZATION=true
"@
            Set-Content -Path $envPath -Value $envContent -Encoding UTF8
        }
        
        Write-Host "✓ File di configurazione creati" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "✗ Errore creazione configurazione: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Funzione per creare shortcut desktop
function Create-DesktopShortcut {
    Write-Host "Creazione shortcut desktop..." -ForegroundColor Yellow
    
    try {
        $desktopPath = [System.Environment]::GetFolderPath('Desktop')
        $shortcutPath = "$desktopPath\AI Trading Bot.lnk"
        
        $shell = New-Object -ComObject WScript.Shell
        $shortcut = $shell.CreateShortcut($shortcutPath)
        $shortcut.TargetPath = "powershell.exe"
        $shortcut.Arguments = "-WindowStyle Hidden -Command `"cd '$InstallPath'; .\venv\Scripts\Activate.ps1; python advanced_ai_system.py`""
        $shortcut.WorkingDirectory = $InstallPath
        $shortcut.Description = "AI Crypto Trading Bot"
        $shortcut.Save()
        
        Write-Host "✓ Shortcut desktop creato" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "✗ Errore creazione shortcut: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Funzione per creare comando CLI
function Create-CLICommand {
    Write-Host "Configurazione comando CLI..." -ForegroundColor Yellow
    
    try {
        $batchContent = @"
@echo off
cd /d "$InstallPath"
call venv\Scripts\activate.bat
python advanced_ai_system.py
pause
"@
        
        $batchPath = "$InstallPath\tradingbot.bat"
        Set-Content -Path $batchPath -Value $batchContent -Encoding ASCII
        
        # Aggiungi al PATH se non presente
        $currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
        if ($currentPath -notlike "*$InstallPath*") {
            $newPath = "$currentPath;$InstallPath"
            [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        }
        
        Write-Host "✓ Comando CLI 'tradingbot.bat' configurato" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "✗ Errore configurazione CLI: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Funzione per test post-installazione
function Test-Installation {
    Write-Host "Test post-installazione..." -ForegroundColor Yellow
    
    try {
        Set-Location -Path $InstallPath
        & ".\venv\Scripts\Activate.ps1"
        
        # Test import principales
        $testScript = @"
import sys
sys.path.insert(0, '.')

try:
    import streamlit
    print('✓ Streamlit OK')
except ImportError as e:
    print(f'✗ Streamlit Error: {e}')

try:
    import pandas
    print('✓ Pandas OK')
except ImportError as e:
    print(f'✗ Pandas Error: {e}')

try:
    import numpy
    print('✓ NumPy OK')
except ImportError as e:
    print(f'✗ NumPy Error: {e}')

try:
    import requests
    print('✓ Requests OK')
except ImportError as e:
    print(f'✗ Requests Error: {e}')

print('Installation test completed.')
"@
        
        $testPath = "$InstallPath\test_install.py"
        Set-Content -Path $testPath -Value $testScript -Encoding UTF8
        
        python test_install.py
        Remove-Item $testPath
        
        Write-Host "✓ Test installazione completato" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "✗ Errore test installazione: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# === MAIN INSTALLATION PROCESS ===

try {
    Write-Host "`n🚀 Avvio installazione AI Crypto Trading Bot..." -ForegroundColor Cyan
    
    # Step 1: Install Chocolatey
    Write-Host "`n[1/8] Installazione Chocolatey..." -ForegroundColor Blue
    if (-not (Install-Chocolatey)) {
        throw "Fallimento installazione Chocolatey"
    }
    
    # Step 2: Install Python
    Write-Host "`n[2/8] Installazione Python..." -ForegroundColor Blue
    if (-not (Install-Python)) {
        throw "Fallimento installazione Python"
    }
    
    # Step 3: Install Git
    Write-Host "`n[3/8] Installazione Git..." -ForegroundColor Blue
    if (-not (Install-Git)) {
        throw "Fallimento installazione Git"
    }
    
    # Step 4: Clone Repository
    Write-Host "`n[4/8] Clonazione repository..." -ForegroundColor Blue
    if (-not (Clone-Repository)) {
        throw "Fallimento clonazione repository"
    }
    
    # Step 5: Install Python packages
    Write-Host "`n[5/8] Installazione dipendenze Python..." -ForegroundColor Blue
    if (-not (Install-PythonPackages)) {
        throw "Fallimento installazione dipendenze"
    }
    
    # Step 6: Create configuration
    Write-Host "`n[6/8] Creazione configurazione..." -ForegroundColor Blue
    if (-not (Create-Configuration)) {
        throw "Fallimento creazione configurazione"
    }
    
    # Step 7: Create shortcuts
    Write-Host "`n[7/8] Creazione shortcuts..." -ForegroundColor Blue
    Create-DesktopShortcut | Out-Null
    Create-CLICommand | Out-Null
    
    # Step 8: Test installation
    Write-Host "`n[8/8] Test installazione..." -ForegroundColor Blue
    Test-Installation | Out-Null
    
    # Success message
    Write-Host "`n🎉 INSTALLAZIONE COMPLETATA CON SUCCESSO!" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Green
    Write-Host "Directory installazione: $InstallPath" -ForegroundColor Cyan
    Write-Host "Comando avvio: tradingbot.bat" -ForegroundColor Cyan
    Write-Host "Shortcut desktop: AI Trading Bot.lnk" -ForegroundColor Cyan
    Write-Host "Interfaccia web: http://localhost:5000" -ForegroundColor Cyan
    Write-Host "`n📋 PROSSIMI PASSI:" -ForegroundColor Yellow
    Write-Host "1. Configura le API keys nel file .env" -ForegroundColor White
    Write-Host "2. Avvia il bot tramite shortcut desktop o comando tradingbot.bat" -ForegroundColor White
    Write-Host "3. Accedi all'interfaccia web su http://localhost:5000" -ForegroundColor White
    Write-Host "4. Testa in modalità simulazione prima del trading live" -ForegroundColor White
    
    if (-not $Silent) {
        Write-Host "`nPremi Invio per aprire la directory di installazione..." -ForegroundColor Gray
        Read-Host
        Start-Process "explorer.exe" -ArgumentList $InstallPath
    }
    
}
catch {
    Write-Host "`n❌ ERRORE DURANTE L'INSTALLAZIONE!" -ForegroundColor Red
    Write-Host "Dettagli errore: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "`nPer supporto, controlla i log e contatta il supporto tecnico." -ForegroundColor Yellow
    
    if (-not $Silent) {
        Read-Host "Premi Invio per uscire"
    }
    exit 1
}