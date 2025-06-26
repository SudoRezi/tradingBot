# AI Crypto Trading Bot - Windows Installer (PowerShell)
# Questo script sarà convertito in .exe usando ps2exe o simili

param(
    [string]$InstallPath = "$env:USERPROFILE\ai-trading-bot",
    [switch]$Silent = $false
)

$ErrorActionPreference = "Stop"

Write-Host "=== AI Crypto Trading Bot - Windows Installer ===" -ForegroundColor Cyan
Write-Host "Installing to: $InstallPath" -ForegroundColor Green

# Funzione per verificare se l'utente è amministratore
function Test-Admin {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Funzione per installare Python se necessario
function Install-Python {
    Write-Host "Checking Python installation..." -ForegroundColor Yellow
    
    try {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match "Python 3\.[8-9]|Python 3\.1[0-9]") {
            Write-Host "Python found: $pythonVersion" -ForegroundColor Green
            return $true
        }
    }
    catch {
        Write-Host "Python not found or incompatible version" -ForegroundColor Red
    }
    
    Write-Host "Installing Python 3.11..." -ForegroundColor Yellow
    
    # Download Python installer
    $pythonUrl = "https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe"
    $tempPath = "$env:TEMP\python-installer.exe"
    
    try {
        Invoke-WebRequest -Uri $pythonUrl -OutFile $tempPath -UseBasicParsing
        
        # Install Python silently
        Start-Process -FilePath $tempPath -ArgumentList "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_test=0" -Wait
        
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        
        # Verify installation
        Start-Sleep -Seconds 5
        $pythonVersion = python --version 2>&1
        Write-Host "Python installed: $pythonVersion" -ForegroundColor Green
        
        Remove-Item $tempPath -Force
        return $true
    }
    catch {
        Write-Host "Failed to install Python: $_" -ForegroundColor Red
        return $false
    }
}

# Funzione per installare Git se necessario
function Install-Git {
    Write-Host "Checking Git installation..." -ForegroundColor Yellow
    
    try {
        $gitVersion = git --version 2>&1
        Write-Host "Git found: $gitVersion" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "Git not found, installing..." -ForegroundColor Yellow
        
        $gitUrl = "https://github.com/git-for-windows/git/releases/download/v2.43.0.windows.1/Git-2.43.0-64-bit.exe"
        $tempPath = "$env:TEMP\git-installer.exe"
        
        try {
            Invoke-WebRequest -Uri $gitUrl -OutFile $tempPath -UseBasicParsing
            Start-Process -FilePath $tempPath -ArgumentList "/SILENT", "/COMPONENTS=icons,ext\reg\shellhere,assoc,assoc_sh" -Wait
            
            # Refresh PATH
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
            
            Remove-Item $tempPath -Force
            Write-Host "Git installed successfully" -ForegroundColor Green
            return $true
        }
        catch {
            Write-Host "Failed to install Git: $_" -ForegroundColor Red
            return $false
        }
    }
}

# Funzione per creare la directory di installazione
function New-InstallDirectory {
    Write-Host "Creating installation directory..." -ForegroundColor Yellow
    
    if (Test-Path $InstallPath) {
        Write-Host "Directory already exists: $InstallPath" -ForegroundColor Yellow
    } else {
        New-Item -ItemType Directory -Path $InstallPath -Force | Out-Null
        Write-Host "Created directory: $InstallPath" -ForegroundColor Green
    }
}

# Funzione per installare le dipendenze Python
function Install-PythonDependencies {
    Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
    
    $packages = @(
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "plotly>=5.15.0",
        "scikit-learn>=1.3.0",
        "requests>=2.31.0",
        "cryptography>=41.0.0",
        "apscheduler>=3.10.0",
        "yfinance>=0.2.0",
        "beautifulsoup4>=4.12.0",
        "feedparser>=6.0.0",
        "psutil>=5.9.0",
        "joblib>=1.3.0",
        "scipy>=1.11.0",
        "trafilatura>=1.6.0",
        "sendgrid>=6.10.0"
    )
    
    foreach ($package in $packages) {
        try {
            Write-Host "Installing $package..." -NoNewline
            python -m pip install $package --quiet --no-warn-script-location
            Write-Host " ✓" -ForegroundColor Green
        }
        catch {
            Write-Host " ✗" -ForegroundColor Red
            Write-Host "Warning: Failed to install $package" -ForegroundColor Yellow
        }
    }
}

# Funzione per copiare i file dell'applicazione
function Copy-ApplicationFiles {
    Write-Host "Copying application files..." -ForegroundColor Yellow
    
    # Lista dei file core dell'applicazione
    $coreFiles = @(
        "advanced_ai_system.py",
        "advanced_quant_engine.py",
        "advanced_order_system.py",
        "arctic_data_manager.py",
        "smart_performance_optimizer.py",
        "autonomous_ai_trader.py",
        "real_ai_integration.py",
        "multilayer_api_protection.py",
        "system_health_check.py",
        "PERFORMANCE_CALCULATOR.py"
    )
    
    # Copia file core
    foreach ($file in $coreFiles) {
        if (Test-Path $file) {
            Copy-Item $file -Destination $InstallPath -Force
            Write-Host "Copied: $file" -ForegroundColor Green
        }
    }
    
    # Copia directory
    $directories = @("config", "data", "logs", "backups", "models", "strategies", "utils", "core")
    foreach ($dir in $directories) {
        if (Test-Path $dir) {
            Copy-Item $dir -Destination $InstallPath -Recurse -Force
            Write-Host "Copied directory: $dir" -ForegroundColor Green
        }
    }
}

# Funzione per gestire file di configurazione
function Setup-Configuration {
    Write-Host "Setting up configuration files..." -ForegroundColor Yellow
    
    $configDir = "$InstallPath\config"
    if (!(Test-Path $configDir)) {
        New-Item -ItemType Directory -Path $configDir -Force | Out-Null
    }
    
    # Setup .env file
    $envFile = "$InstallPath\.env"
    if (!(Test-Path $envFile)) {
        $envContent = @"
# AI Crypto Trading Bot - Environment Configuration
# Generated on $(Get-Date)

# Trading Configuration
TRADING_MODE=simulation
INITIAL_CAPITAL=10000
MAX_POSITIONS=5
RISK_PERCENTAGE=2.0

# API Keys (Configure for live trading)
# BINANCE_API_KEY=your_binance_api_key_here
# BINANCE_SECRET_KEY=your_binance_secret_key_here
# COINBASE_API_KEY=your_coinbase_api_key_here
# COINBASE_SECRET_KEY=your_coinbase_secret_key_here

# Data Sources
# ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
# NEWSAPI_KEY=your_news_api_key_here

# HuggingFace
# HUGGINGFACE_API_TOKEN=your_huggingface_token_here

# Notifications
# EMAIL_SMTP_SERVER=smtp.gmail.com
# EMAIL_SMTP_PORT=587
# EMAIL_USERNAME=your_email@gmail.com
# EMAIL_PASSWORD=your_app_password_here

# System Settings
SYSTEM_OS=windows
SYSTEM_ARCH=x64
LOG_LEVEL=INFO
"@
        
        Set-Content -Path $envFile -Value $envContent -Encoding UTF8
        Write-Host "Created .env file" -ForegroundColor Green
    } else {
        Write-Host ".env file already exists, preserving current configuration" -ForegroundColor Yellow
    }
    
    # Setup config.yaml
    $configFile = "$InstallPath\config\config.yaml"
    if (!(Test-Path $configFile)) {
        $configContent = @"
# AI Crypto Trading Bot Configuration
# Windows x64 Optimized Configuration

system:
  platform: windows
  architecture: x64
  install_path: "$InstallPath"
  log_directory: "$InstallPath\logs"
  data_directory: "$InstallPath\data"

trading:
  mode: simulation
  initial_capital: 10000
  max_positions: 5
  risk_percentage: 2.0
  
  exchanges:
    - name: binance
      enabled: false
      testnet: true
    - name: coinbase
      enabled: false
      sandbox: true

ai_models:
  optimization: cpu
  memory_limit: 2048
  cache_size: 512
  
  models:
    - name: trading_classifier
      type: sklearn
      enabled: true
    - name: sentiment_analyzer
      type: lightweight
      enabled: true

performance:
  cpu_optimization: true
  memory_optimization: true
  threading: true
  max_threads: 4

security:
  encryption: true
  key_rotation: true
  audit_logging: true

streamlit:
  host: 0.0.0.0
  port: 5000
  theme: dark
"@
        
        Set-Content -Path $configFile -Value $configContent -Encoding UTF8
        Write-Host "Created config.yaml file" -ForegroundColor Green
    }
}

# Funzione per creare shortcut desktop
function New-DesktopShortcut {
    Write-Host "Creating desktop shortcut..." -ForegroundColor Yellow
    
    $desktopPath = [Environment]::GetFolderPath("Desktop")
    $shortcutPath = "$desktopPath\AI Trading Bot.lnk"
    
    $WshShell = New-Object -comObject WScript.Shell
    $Shortcut = $WshShell.CreateShortcut($shortcutPath)
    $Shortcut.TargetPath = "python"
    $Shortcut.Arguments = "`"$InstallPath\advanced_ai_system.py`""
    $Shortcut.WorkingDirectory = $InstallPath
    $Shortcut.IconLocation = "python.exe,0"
    $Shortcut.Description = "AI Crypto Trading Bot"
    $Shortcut.Save()
    
    Write-Host "Desktop shortcut created" -ForegroundColor Green
}

# Funzione per creare comando CLI
function New-CLICommand {
    Write-Host "Setting up CLI command..." -ForegroundColor Yellow
    
    $batchContent = @"
@echo off
cd /d "$InstallPath"
python advanced_ai_system.py %*
"@
    
    $batchFile = "$InstallPath\tradingbot.bat"
    Set-Content -Path $batchFile -Value $batchContent
    
    # Add to PATH if not already there
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($userPath -notlike "*$InstallPath*") {
        [Environment]::SetEnvironmentVariable("Path", "$userPath;$InstallPath", "User")
        Write-Host "Added to PATH. Restart terminal to use 'tradingbot' command" -ForegroundColor Green
    }
}

# Funzione per test post-installazione
function Test-Installation {
    Write-Host "Running post-installation tests..." -ForegroundColor Yellow
    
    Push-Location $InstallPath
    
    try {
        # Test 1: Import core modules
        Write-Host "Testing core modules..." -NoNewline
        $testResult = python -c "
import sys
sys.path.insert(0, '.')
try:
    from advanced_ai_system import AdvancedAITradingSystem
    from advanced_quant_engine import get_quant_module_manager
    from advanced_order_system import get_order_system
    print('SUCCESS: Core modules loaded')
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
" 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host " ✓" -ForegroundColor Green
        } else {
            Write-Host " ✗" -ForegroundColor Red
            Write-Host $testResult -ForegroundColor Red
        }
        
        # Test 2: Configuration loading
        Write-Host "Testing configuration..." -NoNewline
        if (Test-Path ".env" -and (Test-Path "config\config.yaml")) {
            Write-Host " ✓" -ForegroundColor Green
        } else {
            Write-Host " ✗" -ForegroundColor Red
        }
        
        # Test 3: System health check
        if (Test-Path "system_health_check.py") {
            Write-Host "Running system health check..."
            python system_health_check.py
        }
        
        Write-Host "`nInstallation completed successfully!" -ForegroundColor Green -BackgroundColor Black
        Write-Host "Start the trading bot with: tradingbot" -ForegroundColor Cyan
        Write-Host "Or run: python `"$InstallPath\advanced_ai_system.py`"" -ForegroundColor Cyan
        
    }
    catch {
        Write-Host "Installation test failed: $_" -ForegroundColor Red
    }
    finally {
        Pop-Location
    }
}

# Funzione principale di installazione
function Start-Installation {
    Write-Host "Starting AI Crypto Trading Bot installation..." -ForegroundColor Cyan
    
    try {
        # Check admin rights for system-wide installations
        if (!(Test-Admin)) {
            Write-Host "Note: Running without administrator privileges. Some features may be limited." -ForegroundColor Yellow
        }
        
        # Step 1: Install Python
        if (!(Install-Python)) {
            throw "Python installation failed"
        }
        
        # Step 2: Install Git
        if (!(Install-Git)) {
            Write-Host "Warning: Git installation failed, continuing without Git support" -ForegroundColor Yellow
        }
        
        # Step 3: Create installation directory
        New-InstallDirectory
        
        # Step 4: Install Python dependencies
        Install-PythonDependencies
        
        # Step 5: Copy application files
        Copy-ApplicationFiles
        
        # Step 6: Setup configuration
        Setup-Configuration
        
        # Step 7: Create shortcuts
        New-DesktopShortcut
        New-CLICommand
        
        # Step 8: Run tests
        Test-Installation
        
        Write-Host "`n=== Installation Summary ===" -ForegroundColor Cyan
        Write-Host "Installation Path: $InstallPath" -ForegroundColor Green
        Write-Host "Python Version: $(python --version)" -ForegroundColor Green
        Write-Host "Desktop Shortcut: Created" -ForegroundColor Green
        Write-Host "CLI Command: tradingbot" -ForegroundColor Green
        Write-Host "`nTo start trading:" -ForegroundColor Yellow
        Write-Host "1. Configure API keys in .env file" -ForegroundColor White
        Write-Host "2. Run 'tradingbot' from command line" -ForegroundColor White
        Write-Host "3. Access web interface at http://localhost:5000" -ForegroundColor White
        
    }
    catch {
        Write-Host "Installation failed: $_" -ForegroundColor Red -BackgroundColor Black
        exit 1
    }
}

# Avvia l'installazione
Start-Installation