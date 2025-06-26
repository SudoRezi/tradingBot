# Installazione Diretta da GitHub - AI Trading Bot

## Problema Risolto

I file di installazione tradizionali possono avere problemi nel trovare i file necessari. Questa guida usa l'installazione diretta da GitHub che risolve tutti i problemi di file mancanti.

---

## 🚀 Installazione Ultra-Rapida da GitHub

### ✅ Opzione 1: Installazione Automatica (Raccomandato)

#### Windows
```powershell
# Apri PowerShell come Amministratore
# Usa l'installer corretto senza errori di sintassi
.\install-windows-fixed.ps1

# O scarica da GitHub:
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/SudoRezi/tradingBot/main/install-windows-fixed.ps1" -OutFile "install.ps1"
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\install.ps1
```

#### macOS
```bash
# Scarica e esegui l'installer
curl -O https://raw.githubusercontent.com/SudoRezi/tradingBot/main/github-installer-macos.sh
chmod +x github-installer-macos.sh
./github-installer-macos.sh

# O se hai il file localmente:
chmod +x github-installer-macos.sh
./github-installer-macos.sh
```

#### Linux
```bash
# Scarica e esegui l'installer
curl -O https://raw.githubusercontent.com/SudoRezi/tradingBot/main/github-installer-linux.sh
chmod +x github-installer-linux.sh
./github-installer-linux.sh

# O se hai il file localmente:
chmod +x github-installer-linux.sh
./github-installer-linux.sh
```

---

### ✅ Opzione 2: Installazione Manuale (Se preferisci il controllo completo)

#### Tutti i Sistemi - Installazione Git Standard

1. **Installa prerequisiti**
   ```bash
   # Windows (con Chocolatey)
   choco install git python3

   # macOS (con Homebrew)
   brew install git python@3.11

   # Ubuntu/Debian
   sudo apt update
   sudo apt install git python3 python3-pip python3-venv

   # CentOS/RHEL
   sudo yum install git python3 python3-pip
   ```

2. **Clona il repository**
   ```bash
   git clone https://github.com/SudoRezi/tradingBot.git
   cd tradingBot
   ```

3. **Crea virtual environment**
   ```bash
   # Tutti i sistemi
   python3 -m venv venv

   # Attiva virtual environment
   # Windows:
   venv\Scripts\activate

   # macOS/Linux:
   source venv/bin/activate
   ```

4. **Installa dipendenze**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt

   # Se requirements.txt non esiste, installa manualmente:
   pip install streamlit pandas numpy plotly requests python-binance yfinance scikit-learn apscheduler cryptography beautifulsoup4 feedparser trafilatura sendgrid psutil joblib scipy
   ```

5. **Configura file .env**
   ```bash
   # Copia template e modifica
   cp .env.template .env
   nano .env  # o notepad .env su Windows
   ```

6. **Test e avvio**
   ```bash
   # Test installazione
   python check_install.py

   # Avvia il bot
   python advanced_ai_system.py
   ```

---

## 🔧 Configurazione Rapida

### File .env Minimo
```env
# Modalità sicura per iniziare
TRADING_MODE=simulation
INITIAL_CAPITAL=10000
RISK_PERCENTAGE=2.0

# Sistema (auto-rilevato)
SYSTEM_OS=auto
SYSTEM_ARCH=auto
```

### File .env Completo (Trading Live)
```env
# Trading Configuration
TRADING_MODE=live
INITIAL_CAPITAL=10000
RISK_PERCENTAGE=1.0
MAX_POSITIONS=3

# Binance API
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET_KEY=your_binance_secret

# Coinbase API
COINBASE_API_KEY=your_coinbase_key
COINBASE_SECRET_KEY=your_coinbase_secret
COINBASE_PASSPHRASE=your_passphrase

# Data Sources
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
NEWSAPI_KEY=your_news_key
HUGGINGFACE_API_TOKEN=your_hf_token
```

---

## 🌐 Accesso Web Interface

Dopo l'installazione:

- **Windows/macOS**: http://localhost:5000
- **Linux Server**: http://server-ip:8501

### SSH Tunnel per Accesso Sicuro
```bash
# Dal tuo computer locale
ssh -L 5000:localhost:5000 user@server-ip
# Poi vai su: http://localhost:5000
```

---

## 🎯 Vantaggi dell'Installazione GitHub

### ✅ Risolve Tutti i Problemi
- ✅ File sempre aggiornati all'ultima versione
- ✅ Nessun problema di file mancanti
- ✅ Installazione completa garantita
- ✅ Dipendenze corrette automaticamente

### ✅ Caratteristiche
- **Auto-detection**: Rileva automaticamente OS e architettura
- **Dependency Management**: Installa tutte le dipendenze necessarie
- **Service Setup**: Configura servizio systemd su Linux
- **CLI Commands**: Comando `tradingbot` disponibile globalmente
- **Firewall**: Configura automaticamente porte necessarie
- **Health Checks**: Script di controllo sistema inclusi

---

## 🚨 Risoluzione Problemi

### Errore Git Clone
```bash
# Se git clone fallisce, prova:
git clone https://github.com/yourusername/ai-trading-bot.git --depth 1

# O scarica ZIP manualmente:
wget https://github.com/yourusername/ai-trading-bot/archive/main.zip
unzip main.zip
mv ai-trading-bot-main ai-trading-bot
```

### Errore Python/Pip
```bash
# Aggiorna Python e pip
python3 -m pip install --upgrade pip

# Se pip non funziona:
python3 -m ensurepip --upgrade

# Installa dipendenze una per una se batch fallisce:
pip install streamlit
pip install pandas
pip install numpy
# etc...
```

### Errore Permessi
```bash
# Linux/macOS - aggiungi sudo se necessario
sudo ./github-installer-linux.sh

# Windows - esegui PowerShell come Amministratore
# Click destro PowerShell → "Esegui come amministratore"
```

### Porta Occupata
```bash
# Trova processo che usa la porta
# Windows:
netstat -ano | findstr :5000
taskkill /PID [numero] /F

# Linux/macOS:
sudo lsof -i :5000
sudo kill -9 [PID]

# O cambia porta in .env:
STREAMLIT_PORT=5001
```

---

## 📊 Verificare Installazione

### Test Rapido
```bash
cd ai-trading-bot
python check_install.py

# Output atteso:
# ✅ Sistema principale: Operativo
# ✅ Moduli AI: Operativi
# ✅ Trading engine: Operativo
# ✅ Sistema ordini: Funzionale
# ✅ Storage dati: Operativo
```

### Health Check (Linux)
```bash
./healthcheck.sh

# Output atteso:
# ✓ Servizio: ATTIVO
# ✓ Porta web 8501: APERTA
# ✓ Processo Python: ATTIVO
# ✓ Interfaccia web: RAGGIUNGIBILE
```

---

## 🎯 Risultato Finale

Con l'installazione da GitHub avrai:

✅ **Sistema 100% operativo** senza file mancanti
✅ **Ultima versione** sempre aggiornata
✅ **Dipendenze corrette** installate automaticamente
✅ **Configurazione ottimale** per il tuo sistema
✅ **Servizi configurati** per operazione 24/7
✅ **Accesso remoto** configurato su Linux
✅ **Script di manutenzione** inclusi

---

## 📞 Link Utili per API Keys

- **Binance**: https://binance.com → Account → API Management
- **Coinbase**: https://pro.coinbase.com → Settings → API
- **Alpha Vantage**: https://www.alphavantage.co/support/#api-key
- **News API**: https://newsapi.org/register
- **HuggingFace**: https://huggingface.co/settings/tokens

---

**L'installazione da GitHub risolve completamente tutti i problemi di file mancanti e garantisce un sistema 100% funzionante.**