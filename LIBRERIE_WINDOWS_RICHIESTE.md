# Librerie Windows per AI Trading Bot

## Librerie Essenziali Windows

### 1. Dipendenze Sistema
```powershell
# Visual C++ Redistributable (necessario per molte librerie Python)
choco install vcredist140 -y

# Visual Studio Build Tools (per compilare librerie native)
choco install visualstudio2022buildtools -y

# Windows SDK (opzionale ma raccomandato)
choco install windows-sdk-10-version-2004-all -y
```

### 2. Librerie Python Core
```bash
# Interfaccia web
streamlit==1.28.1

# Data processing
pandas==2.0.3
numpy==1.24.3

# Grafici e visualizzazione
plotly==5.15.0
matplotlib==3.7.1

# Network e web
requests==2.31.0
urllib3==2.0.4
```

### 3. Librerie Trading
```bash
# Exchange APIs
python-binance==1.0.19
yfinance==0.2.18
ccxt==4.0.74

# Analisi tecnica  
ta-lib-binary==0.4.25
pandas-ta==0.3.14b
```

### 4. Librerie AI/ML
```bash
# Machine Learning
scikit-learn==1.3.0
joblib==1.3.1
scipy==1.11.1

# Deep Learning (opzionale)
tensorflow==2.13.0
torch==2.0.1
```

### 5. Librerie Sistema Windows
```bash
# Windows integration
pywin32==306
wmi==1.5.1

# Processo e sistema
psutil==5.9.5
schedule==1.2.0
apscheduler==3.10.4
```

### 6. Librerie Sicurezza
```bash
# Crittografia
cryptography==41.0.3
pycryptodome==3.18.0

# Hashing
hashlib2==1.0.1
```

### 7. Librerie Web Scraping
```bash
# HTML parsing
beautifulsoup4==4.12.2
lxml==4.9.3

# Feed RSS
feedparser==6.0.10

# Content extraction
trafilatura==1.6.1
```

### 8. Librerie Comunicazione
```bash
# Email
sendgrid==6.10.0
smtplib2==0.2.1

# Notifiche
plyer==2.1.0
```

## Comando Installazione Completa

```powershell
# Apri PowerShell come Amministratore ed esegui:

# 1. Setup sistema
choco install python git visualcpp-build-tools vcredist140 -y

# 2. Setup ambiente Python
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel

# 3. Installa tutte le librerie
pip install streamlit==1.28.1 pandas==2.0.3 numpy==1.24.3 plotly==5.15.0 requests==2.31.0 python-binance==1.0.19 yfinance==0.2.18 scikit-learn==1.3.0 apscheduler==3.10.4 cryptography==41.0.3 beautifulsoup4==4.12.2 feedparser==6.0.10 trafilatura==1.6.1 sendgrid==6.10.0 psutil==5.9.5 joblib==1.3.1 scipy==1.11.1 pywin32==306 wmi==1.5.1 ccxt==4.0.74

# 4. Test installazione
python -c "import streamlit, pandas, numpy, requests, yfinance, sklearn; print('Setup completato!')"
```

## Librerie Opzionali Avanzate

```bash
# Performance boost
numba==0.57.1
bottleneck==1.3.7

# Database avanzato
arctic==1.82.0
sqlalchemy==2.0.19

# Analisi quantitativa
quantlib==1.31
zipline-reloaded==2.2.0
pyfolio==0.9.2
```

## Risoluzione Problemi Comuni

### Errore TA-Lib
```powershell
# Installa TA-Lib pre-compilato per Windows
pip install TA-Lib-0.4.25-cp311-cp311-win_amd64.whl
# O usa la versione binary
pip install ta-lib-binary
```

### Errore Visual C++
```powershell
# Installa Microsoft C++ Build Tools
choco install visualcpp-build-tools -y
# Riavvia PowerShell e riprova
```

### Errore PyWin32
```powershell
# Installazione specifica Windows
pip install pywin32==306
python venv/Scripts/pywin32_postinstall.py -install
```

### Errore Streamlit
```powershell
# Reinstalla Streamlit
pip uninstall streamlit -y
pip install streamlit==1.28.1
```

## Verifica Installazione

```python
# Test script - salva come test_windows.py
import sys
import importlib

required_modules = [
    'streamlit', 'pandas', 'numpy', 'plotly', 'requests',
    'binance', 'yfinance', 'sklearn', 'cryptography', 
    'psutil', 'win32api', 'bs4', 'feedparser'
]

print("Test librerie Windows:")
print("=" * 40)

for module in required_modules:
    try:
        importlib.import_module(module)
        print(f"✓ {module}")
    except ImportError as e:
        print(f"✗ {module} - {e}")

print("=" * 40)
print("Test completato!")
```

## Setup Automatico

Usa il file `SETUP_WINDOWS_COMPLETO.ps1` che installa automaticamente tutte le librerie necessarie per Windows con configurazione ottimale.