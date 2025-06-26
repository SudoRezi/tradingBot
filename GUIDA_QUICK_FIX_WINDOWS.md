# Quick Fix - Installazione Windows Corretta

## Problema Risolto

L'installer originale aveva errori di sintassi PowerShell. Ho creato un installer corretto che funziona senza errori.

## ✅ Installazione Immediata (Windows)

### Metodo 1: File Locale Corretto
```powershell
# Apri PowerShell come Amministratore
# Usa il file corretto senza errori
.\install-windows-fixed.ps1
```

### Metodo 2: Installazione Manuale (100% Funzionante)
```powershell
# 1. Apri PowerShell come Amministratore

# 2. Installa Chocolatey (se non presente)
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# 3. Installa dipendenze
choco install python git -y

# 4. Clona repository
git clone https://github.com/SudoRezi/tradingBot.git
cd tradingBot

# 5. Setup Python
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip

# 6. Installa pacchetti
pip install streamlit pandas numpy plotly requests python-binance yfinance scikit-learn apscheduler cryptography beautifulsoup4 feedparser trafilatura sendgrid psutil joblib scipy

# 7. Crea configurazione base
@"
TRADING_MODE=simulation
INITIAL_CAPITAL=10000
RISK_PERCENTAGE=2.0
SYSTEM_OS=windows
ENABLE_PERFORMANCE_MODE=true
"@ | Out-File -FilePath ".env" -Encoding UTF8

# 8. Test e avvio
python -c "import streamlit; print('OK')"
streamlit run advanced_ai_system.py --server.port 5000 --server.headless true --server.address 0.0.0.0
```

## 🎯 Accesso Sistema

Dopo l'installazione:
- **URL**: http://localhost:5000
- **Shortcut Desktop**: AI Trading Bot.lnk
- **File Batch**: tradingbot.bat

## 🔧 Risoluzione Errori Comuni

### Errore "Token imprevisto"
- **Causa**: File installer corrotto
- **Soluzione**: Usa `install-windows-fixed.ps1`

### Errore "Python non trovato"
```powershell
# Reinstalla Python
choco install python --version=3.11.8 -y
# Refresh PATH
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

### Errore "Git non trovato"
```powershell
choco install git -y
```

### Errore "Streamlit non funziona"
```powershell
cd tradingBot
.\venv\Scripts\Activate.ps1
pip install --upgrade streamlit
streamlit run advanced_ai_system.py --server.port 5000
```

## ✅ Test Rapido
```powershell
cd tradingBot
.\venv\Scripts\Activate.ps1
python -c "import streamlit, pandas, numpy, requests; print('Tutti i moduli importati correttamente')"
```

## 📱 Quick Start
1. Esegui `install-windows-fixed.ps1` come Amministratore
2. Aspetta installazione completa (5-10 minuti)
3. Doppio click su "AI Trading Bot" desktop
4. Vai su http://localhost:5000
5. Sistema operativo e pronto per il trading

L'installer corretto elimina tutti gli errori di sintassi e installa il sistema completo senza problemi.