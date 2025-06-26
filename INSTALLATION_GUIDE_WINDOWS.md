# AI Trading Bot - Guida Installazione Windows

## Installazione Completa su Windows

### Prerequisiti Sistema

#### Installa Python
1. Vai su https://www.python.org/downloads/
2. Scarica Python 3.11 o superiore
3. **IMPORTANTE**: Durante installazione, spunta "Add Python to PATH"
4. Completa installazione

#### Verifica Installazione
Apri **PowerShell** o **Prompt dei Comandi** come Amministratore:
```cmd
python --version
pip --version
```

### Step 1: Download e Setup

#### Metodo 1: Download Diretto
1. Scarica `AI_Trading_Bot_Windows.zip`
2. Estrai in `C:\AI-Trading-Bot\`
3. Apri PowerShell nella cartella estratta

#### Metodo 2: PowerShell Download
```powershell
# Crea directory
New-Item -ItemType Directory -Force -Path "C:\AI-Trading-Bot"
Set-Location "C:\AI-Trading-Bot"

# Se hai il link diretto al ZIP
# Invoke-WebRequest -Uri "URL_DEL_ZIP" -OutFile "AI_Trading_Bot_Windows.zip"
# Expand-Archive -Path "AI_Trading_Bot_Windows.zip" -DestinationPath "."
```

### Step 2: Setup Environment Python

```powershell
# Naviga nella directory del bot
Set-Location "C:\AI-Trading-Bot"

# Crea ambiente virtuale
python -m venv trading_env

# Attiva ambiente virtuale
.\trading_env\Scripts\Activate.ps1

# Se ottieni errore di execution policy, esegui:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Aggiorna pip
python -m pip install --upgrade pip

# Installa dipendenze
pip install streamlit pandas numpy plotly scikit-learn asyncio
```

### Step 3: Configurazione Streamlit

Crea file di configurazione:
```powershell
# Crea directory config
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.streamlit"

# Crea file configurazione
@"
[server]
headless = true
address = "0.0.0.0"
port = 5000
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
"@ | Out-File -FilePath "$env:USERPROFILE\.streamlit\config.toml" -Encoding utf8
```

### Step 4: Avvio del Sistema

#### Avvio Manuale
```powershell
# Attiva ambiente virtuale (se non già attivo)
.\trading_env\Scripts\Activate.ps1

# Avvia l'applicazione
streamlit run advanced_ai_system.py --server.port 5000 --server.address 0.0.0.0
```

#### Avvio Automatico con Batch Script

Crea `start_trading_bot.bat`:
```batch
@echo off
cd /d "C:\AI-Trading-Bot"
call trading_env\Scripts\activate.bat
streamlit run advanced_ai_system.py --server.port 5000 --server.address 0.0.0.0
pause
```

#### Avvio come Servizio Windows (Avanzato)

Installa NSSM (Non-Sucking Service Manager):
```powershell
# Metodo 1: Con Chocolatey
# Installa Chocolatey prima se non ce l'hai
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

choco install nssm

# Metodo 2: Download manuale
# Vai su https://nssm.cc/download e scarica nssm
```

Configura servizio:
```cmd
# Apri cmd come Amministratore
nssm install "AI Trading Bot"

# Nella finestra di NSSM:
# Path: C:\AI-Trading-Bot\trading_env\Scripts\python.exe
# Startup directory: C:\AI-Trading-Bot
# Arguments: -m streamlit run advanced_ai_system.py --server.port 5000 --server.address 0.0.0.0

# Avvia servizio
nssm start "AI Trading Bot"
```

### Step 5: Configurazione Firewall Windows

```powershell
# Apri PowerShell come Amministratore
# Abilita regola firewall per porta 5000
New-NetFirewallRule -DisplayName "AI Trading Bot" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow

# Verifica regola creata
Get-NetFirewallRule -DisplayName "AI Trading Bot"
```

## Accesso alla Dashboard

### Accesso Locale
Apri browser e vai a:
- `http://localhost:5000`
- `http://127.0.0.1:5000`

### Accesso da Altri Dispositivi sulla Rete
1. Trova IP del PC Windows:
   ```cmd
   ipconfig
   ```
2. Da altri dispositivi: `http://IP_DEL_PC:5000`

### Configurazione Router (Accesso Esterno)
1. Accedi al router
2. Vai in **Port Forwarding** o **Virtual Server**
3. Aggiungi regola:
   - **Porta Esterna**: 5000
   - **Porta Interna**: 5000
   - **IP Interno**: IP del PC Windows
   - **Protocollo**: TCP
4. Accedi da esterno: `http://IP_PUBBLICO:5000`

## Gestione Sistema

### Comandi Utili PowerShell

```powershell
# Verifica processo Streamlit in esecuzione
Get-Process -Name "*streamlit*" -ErrorAction SilentlyContinue

# Termina processo se necessario
Stop-Process -Name "streamlit" -Force

# Verifica porta 5000 in uso
netstat -an | findstr "5000"

# Test connessione locale
Invoke-WebRequest -Uri "http://localhost:5000" -Method Head
```

### Script di Gestione Automatica

Crea `manage_bot.ps1`:
```powershell
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("start", "stop", "restart", "status")]
    [string]$Action
)

$BotPath = "C:\AI-Trading-Bot"
$ProcessName = "streamlit"

function Start-Bot {
    Set-Location $BotPath
    & .\trading_env\Scripts\Activate.ps1
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "& {Set-Location '$BotPath'; .\trading_env\Scripts\Activate.ps1; streamlit run advanced_ai_system.py --server.port 5000 --server.address 0.0.0.0}"
    Write-Host "Bot avviato!" -ForegroundColor Green
}

function Stop-Bot {
    Get-Process -Name $ProcessName -ErrorAction SilentlyContinue | Stop-Process -Force
    Write-Host "Bot fermato!" -ForegroundColor Red
}

function Get-BotStatus {
    $process = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue
    if ($process) {
        Write-Host "Bot in esecuzione (PID: $($process.Id))" -ForegroundColor Green
        return $true
    } else {
        Write-Host "Bot non in esecuzione" -ForegroundColor Red
        return $false
    }
}

switch ($Action) {
    "start" { Start-Bot }
    "stop" { Stop-Bot }
    "restart" { 
        Stop-Bot
        Start-Sleep -Seconds 3
        Start-Bot
    }
    "status" { Get-BotStatus }
}
```

Uso script:
```powershell
# Avvia bot
.\manage_bot.ps1 start

# Ferma bot
.\manage_bot.ps1 stop

# Riavvia bot
.\manage_bot.ps1 restart

# Verifica status
.\manage_bot.ps1 status
```

## Aggiornamenti

### Aggiornamento Manuale
```powershell
# Ferma bot se in esecuzione
.\manage_bot.ps1 stop

# Backup configurazione
Copy-Item -Path "config\" -Destination "config_backup\" -Recurse -Force

# Sostituisci file aggiornati
# Copia nuovi file sopra i vecchi

# Riattiva ambiente e riavvia
.\trading_env\Scripts\Activate.ps1
.\manage_bot.ps1 start
```

### Aggiornamento Dipendenze
```powershell
# Attiva ambiente
.\trading_env\Scripts\Activate.ps1

# Aggiorna tutte le dipendenze
pip install --upgrade streamlit pandas numpy plotly scikit-learn

# Verifica versioni
pip list
```

## Backup e Restore

### Script Backup Automatico

Crea `backup_bot.ps1`:
```powershell
$BackupPath = "C:\Backups\AI-Trading-Bot"
$SourcePath = "C:\AI-Trading-Bot"
$Date = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupFile = "$BackupPath\trading-bot-backup-$Date.zip"

# Crea directory backup se non esiste
New-Item -ItemType Directory -Force -Path $BackupPath

# Comprimi directory sorgente
Compress-Archive -Path "$SourcePath\*" -DestinationPath $BackupFile -Force

# Mantieni solo ultimi 7 backup
Get-ChildItem -Path $BackupPath -Filter "trading-bot-backup-*.zip" | 
    Sort-Object CreationTime -Descending | 
    Select-Object -Skip 7 | 
    Remove-Item -Force

Write-Host "Backup completato: $BackupFile" -ForegroundColor Green
```

### Backup Automatico con Task Scheduler
```powershell
# Crea task schedulato per backup giornaliero
$Action = New-ScheduledTaskAction -Execute "PowerShell.exe" -Argument "-File C:\AI-Trading-Bot\backup_bot.ps1"
$Trigger = New-ScheduledTaskTrigger -Daily -At "02:00"
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName "AI Trading Bot Backup" -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings
```

## Monitoraggio Performance

### Script Monitoraggio Sistema

Crea `monitor_system.ps1`:
```powershell
while ($true) {
    Clear-Host
    Write-Host "=== AI Trading Bot System Monitor ===" -ForegroundColor Cyan
    Write-Host "Timestamp: $(Get-Date)" -ForegroundColor Yellow
    
    # CPU Usage
    $cpu = Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 1
    Write-Host "CPU Usage: $([math]::Round($cpu.CounterSamples.CookedValue, 2))%" -ForegroundColor Green
    
    # Memory Usage
    $mem = Get-CimInstance Win32_OperatingSystem
    $memUsage = [math]::Round((($mem.TotalVisibleMemorySize - $mem.FreePhysicalMemory) / $mem.TotalVisibleMemorySize) * 100, 2)
    Write-Host "Memory Usage: $memUsage%" -ForegroundColor Green
    
    # Disk Usage
    $disk = Get-CimInstance -ClassName Win32_LogicalDisk -Filter "DeviceID='C:'"
    $diskUsage = [math]::Round((($disk.Size - $disk.FreeSpace) / $disk.Size) * 100, 2)
    Write-Host "Disk Usage: $diskUsage%" -ForegroundColor Green
    
    # Bot Status
    $botProcess = Get-Process -Name "streamlit" -ErrorAction SilentlyContinue
    if ($botProcess) {
        Write-Host "Bot Status: Running (PID: $($botProcess.Id))" -ForegroundColor Green
        Write-Host "Bot Memory: $([math]::Round($botProcess.WorkingSet / 1MB, 2)) MB" -ForegroundColor Green
    } else {
        Write-Host "Bot Status: Not Running" -ForegroundColor Red
    }
    
    # Network Connection
    $connection = Test-NetConnection -ComputerName "localhost" -Port 5000 -WarningAction SilentlyContinue
    if ($connection.TcpTestSucceeded) {
        Write-Host "Dashboard: Accessible on http://localhost:5000" -ForegroundColor Green
    } else {
        Write-Host "Dashboard: Not Accessible" -ForegroundColor Red
    }
    
    Write-Host "`nPress Ctrl+C to exit" -ForegroundColor Yellow
    Start-Sleep -Seconds 5
}
```

## Troubleshooting

### Problemi Comuni

```powershell
# Python non trovato
# Reinstalla Python e assicurati di aggiungere al PATH

# Pip non funziona
python -m ensurepip --upgrade

# Porta 5000 occupata
netstat -ano | findstr :5000
# Termina processo con PID mostrato:
taskkill /PID XXXX /F

# Errori permission PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Reset completo ambiente Python
Remove-Item -Recurse -Force trading_env
python -m venv trading_env
.\trading_env\Scripts\Activate.ps1
pip install streamlit pandas numpy plotly scikit-learn
```

### Logs e Debug

```powershell
# Avvia con logging dettagliato
streamlit run advanced_ai_system.py --logger.level debug --server.port 5000

# Verifica connessioni di rete
Get-NetTCPConnection -LocalPort 5000

# Test connessione HTTP
Invoke-WebRequest -Uri "http://localhost:5000" -UseBasicParsing
```

## Accesso Remoto Sicuro

### Configurazione VPN (Raccomandato)
1. Installa software VPN (OpenVPN, WireGuard)
2. Configura server VPN o usa servizio cloud
3. Connetti dispositivi alla VPN
4. Accedi al bot tramite IP privato VPN

### SSH Tunnel da Linux/macOS
```bash
# Da macOS/Linux verso Windows
ssh -L 5000:localhost:5000 user@IP_WINDOWS
# Poi accedi a http://localhost:5000
```

### TeamViewer/AnyDesk (Alternativa)
1. Installa TeamViewer o AnyDesk su Windows
2. Configura accesso non supervisionato
3. Connetti da remoto per gestire il bot

Il sistema è ora pronto per l'uso su Windows con accesso completo alla dashboard!