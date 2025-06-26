# Guida Installazione Completa - AI Crypto Trading Bot

## Panoramica

Questa guida ti accompagna passo dopo passo nell'installazione del sistema AI Crypto Trading Bot su Windows, macOS e Linux. Il bot include intelligence artificiale autonoma, analisi quantitative avanzate, supporto multi-exchange e funzionalità di trading automatizzato 24/7.

---

## 📦 Pacchetti di Installazione Disponibili

Prima di iniziare, scarica il pacchetto appropriato per il tuo sistema:

- **Windows 10/11 x64**: `AI_Trading_Bot_Windows_Installer.zip`
- **macOS Intel/Apple Silicon**: `AI_Trading_Bot_macOS_Installer.zip`
- **Linux Ubuntu/Debian/CentOS**: `AI_Trading_Bot_Linux_Installer.zip`
- **Pacchetto Universale**: `AI_Trading_Bot_Universal_Package.zip` (tutti i sistemi)

---

# 🪟 INSTALLAZIONE WINDOWS 10/11

## Requisiti di Sistema

- Windows 10 versione 1909 o superiore / Windows 11
- Architettura x64 (64-bit)
- 4GB RAM minimo (8GB raccomandato)
- 2GB spazio libero su disco
- Connessione internet attiva
- Privilegi amministratore (solo per installazione)

## Passo 1: Preparazione

1. **Scarica il pacchetto Windows**
   ```
   AI_Trading_Bot_Windows_Installer.zip
   ```

2. **Estrai il contenuto**
   - Fai clic destro sul file ZIP
   - Seleziona "Estrai tutto..."
   - Scegli una cartella temporanea (es. Desktop)

3. **Verifica i file estratti**
   - `install.bat` - Launcher di installazione
   - `install.ps1` - Script PowerShell principale
   - `README.txt` - Istruzioni specifiche Windows

## Passo 2: Installazione Automatica

1. **Esegui l'installer con privilegi amministratore**
   - Fai clic destro su `install.bat`
   - Seleziona **"Esegui come amministratore"**
   - Conferma il controllo UAC quando richiesto

2. **Segui il processo di installazione**
   ```
   AI Crypto Trading Bot - Windows Installer
   ==========================================
   
   Installazione in corso...
   ✓ Verifica sistema Windows
   ✓ Download e installazione Python 3.11
   ✓ Installazione Git for Windows
   ✓ Download dipendenze Python (16 pacchetti)
   ✓ Copia file applicazione
   ✓ Configurazione ambiente
   ✓ Creazione shortcut desktop
   ✓ Configurazione comando CLI
   ✓ Test post-installazione
   ```

3. **Installazione completata**
   - L'installer creerà la directory `C:\Users\[TuoNome]\ai-trading-bot`
   - Verrà creato un collegamento sul desktop "AI Trading Bot"
   - Il comando `tradingbot` sarà disponibile nel Prompt dei comandi

## Passo 3: Configurazione Iniziale

1. **Apri il file di configurazione**
   ```cmd
   notepad %USERPROFILE%\ai-trading-bot\.env
   ```

2. **Configura le impostazioni base**
   ```env
   # Modalità trading (lascia simulation per i test)
   TRADING_MODE=simulation
   INITIAL_CAPITAL=10000
   RISK_PERCENTAGE=2.0
   
   # Sistema
   SYSTEM_OS=windows
   SYSTEM_ARCH=x64
   ```

## Passo 4: Test dell'Installazione

1. **Apri il Prompt dei comandi**
   - Premi `Windows + R`
   - Digita `cmd` e premi Invio

2. **Esegui il test diagnostico**
   ```cmd
   cd %USERPROFILE%\ai-trading-bot
   python check_install.py
   ```

3. **Verifica il risultato**
   ```
   ✅ Sistema principale: Operativo
   ✅ Moduli AI: Operativi
   ✅ Trading engine: Operativo
   ✅ Sistema ordini: Funzionale
   ✅ Storage dati: Operativo
   ```

## Passo 5: Primo Avvio

1. **Avvia il trading bot**
   - **Metodo 1**: Fai doppio clic sul collegamento desktop "AI Trading Bot"
   - **Metodo 2**: Apri cmd e digita `tradingbot`
   - **Metodo 3**: 
     ```cmd
     cd %USERPROFILE%\ai-trading-bot
     python advanced_ai_system.py
     ```

2. **Accedi all'interfaccia web**
   - Apri il browser
   - Vai su `http://localhost:5000`
   - Vedrai la dashboard del trading bot

## Risoluzione Problemi Windows

### Errore "Python non trovato"
```cmd
# Installa Python manualmente
winget install Python.Python.3.11
# Riavvia il prompt e riprova
```

### Errore PowerShell "Execution Policy"
```powershell
# Apri PowerShell come amministratore
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Riprova l'installazione
```

### Porta 5000 occupata
```cmd
# Trova il processo che usa la porta
netstat -ano | findstr :5000
# Termina il processo (sostituisci PID)
taskkill /PID [numero_pid] /F
```

---

# 🍎 INSTALLAZIONE macOS

## Requisiti di Sistema

- macOS 10.15 Catalina o superiore
- Processore Intel x64 o Apple Silicon (M1/M2/M3)
- 4GB RAM minimo (8GB raccomandato per Apple Silicon)
- 3GB spazio libero su disco
- Xcode Command Line Tools
- Connessione internet attiva

## Passo 1: Preparazione

1. **Scarica il pacchetto macOS**
   ```
   AI_Trading_Bot_macOS_Installer.zip
   ```

2. **Estrai il contenuto**
   ```bash
   cd ~/Downloads
   unzip AI_Trading_Bot_macOS_Installer.zip
   cd macos/
   ```

3. **Verifica i file estratti**
   ```bash
   ls -la
   # Dovresti vedere:
   # install.sh - Script di installazione
   # README.txt - Istruzioni macOS
   ```

## Passo 2: Installazione delle Dipendenze di Sistema

1. **Installa Xcode Command Line Tools** (se non già installato)
   ```bash
   xcode-select --install
   ```
   - Segui le istruzioni a schermo
   - L'installazione richiede alcuni minuti

2. **Verifica l'installazione**
   ```bash
   xcode-select -p
   # Dovrebbe mostrare: /Applications/Xcode.app/Contents/Developer
   # o /Library/Developer/CommandLineTools
   ```

## Passo 3: Installazione Automatica

1. **Rendi eseguibile lo script**
   ```bash
   chmod +x install.sh
   ```

2. **Esegui l'installer**
   ```bash
   ./install.sh
   ```

3. **Segui il processo (circa 10-15 minuti)**
   ```
   === AI Crypto Trading Bot - macOS Installer ===
   Architettura rilevata: arm64 (Apple Silicon)
   
   ✅ Homebrew: Installato
   ✅ Python 3.11: Installato con ottimizzazioni ARM
   ✅ Dipendenze sistema: Installate
   ✅ Pacchetti Python: 16/16 installati
   ✅ File applicazione: Copiati
   ✅ Configurazione: Completata
   ✅ Comando CLI: Configurato
   ✅ App desktop: Creata
   ✅ Test funzionali: Superati
   ```

## Passo 4: Configurazione per Apple Silicon (M1/M2/M3)

Se hai un Mac con chip Apple Silicon, il sistema è automaticamente ottimizzato:

1. **Verifica ottimizzazioni ARM**
   ```bash
   cd ~/ai-trading-bot
   python -c "import platform; print(f'Architettura: {platform.machine()}')"
   ```

2. **Configurazione ottimizzata**
   ```bash
   cat config/config.yaml | grep -A 5 "ai_models:"
   # Dovrebbe mostrare:
   # optimization: arm
   # memory_limit: 4096
   # metal_acceleration: true
   ```

## Passo 5: Test dell'Installazione

1. **Esegui il test diagnostico**
   ```bash
   cd ~/ai-trading-bot
   python check_install.py
   ```

2. **Test rapido**
   ```bash
   ./healthcheck.sh
   ```

## Passo 6: Primo Avvio

1. **Avvia il trading bot**
   ```bash
   # Metodo 1: Comando CLI
   tradingbot
   
   # Metodo 2: Script diretto
   cd ~/ai-trading-bot
   python advanced_ai_system.py
   
   # Metodo 3: App desktop
   open "/Applications/AI Trading Bot.app"
   ```

2. **Accedi all'interfaccia**
   - Browser: `http://localhost:5000`
   - L'interfaccia si aprirà automaticamente

## Risoluzione Problemi macOS

### Homebrew non si installa
```bash
# Installazione manuale Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Aggiungi al PATH (Apple Silicon)
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
source ~/.zprofile
```

### Errore permessi Python
```bash
# Fix permessi
sudo chown -R $(whoami) ~/ai-trading-bot
chmod +x ~/ai-trading-bot/tradingbot
```

### App bloccata da Gatekeeper
```bash
# Sblocca l'app (solo se necessario)
sudo xattr -rd com.apple.quarantine "/Applications/AI Trading Bot.app"
```

---

# 🐧 INSTALLAZIONE LINUX

## Sistemi Supportati

- **Ubuntu**: 18.04 LTS, 20.04 LTS, 22.04 LTS, 24.04 LTS
- **Debian**: 10 (Buster), 11 (Bullseye), 12 (Bookworm)
- **CentOS**: 7, 8, Stream
- **RHEL**: 7, 8, 9
- **Fedora**: 35, 36, 37, 38, 39

## Requisiti di Sistema

- Kernel Linux 4.15 o superiore
- Architettura x86_64 (64-bit)
- 2GB RAM minimo (4GB raccomandato per server)
- 3GB spazio libero su disco
- Accesso sudo per installazione pacchetti di sistema
- Connessione internet attiva

## Passo 1: Preparazione

1. **Scarica il pacchetto Linux**
   ```bash
   wget https://[tuo-server]/AI_Trading_Bot_Linux_Installer.zip
   # oppure trasferisci il file via SCP/SFTP
   ```

2. **Estrai il contenuto**
   ```bash
   unzip AI_Trading_Bot_Linux_Installer.zip
   cd linux/
   ```

3. **Verifica i file**
   ```bash
   ls -la
   # install.sh - Script principale
   # healthcheck.sh - Check rapido sistema
   # README.txt - Istruzioni Linux
   ```

## Passo 2: Installazione Automatica

1. **Rendi eseguibile lo script**
   ```bash
   chmod +x install.sh
   chmod +x healthcheck.sh
   ```

2. **Esegui l'installer**
   ```bash
   ./install.sh
   ```

3. **Segui il processo automatico**
   ```
   === AI Crypto Trading Bot - Linux Installer ===
   Distribuzione rilevata: Ubuntu 22.04
   
   ✅ Aggiornamento pacchetti sistema
   ✅ Installazione dipendenze (build-essential, git, python3, etc.)
   ✅ Configurazione Python e virtual environment
   ✅ Installazione 16 pacchetti Python
   ✅ Copia file applicazione
   ✅ Configurazione .env e config.yaml
   ✅ Creazione servizio systemd
   ✅ Configurazione firewall (porta 8501)
   ✅ Setup comando CLI globale
   ✅ Test funzionali completati
   ```

## Passo 3: Configurazione Servizio (24/7)

1. **Il servizio systemd viene creato automaticamente**
   ```bash
   # Verifica lo status
   sudo systemctl status ai-trading-bot
   
   # Il servizio è già abilitato per l'avvio automatico
   sudo systemctl is-enabled ai-trading-bot
   ```

2. **Comandi di gestione servizio**
   ```bash
   # Avvia il servizio
   sudo systemctl start ai-trading-bot
   
   # Ferma il servizio
   sudo systemctl stop ai-trading-bot
   
   # Riavvia il servizio
   sudo systemctl restart ai-trading-bot
   
   # Visualizza log in tempo reale
   sudo journalctl -u ai-trading-bot -f
   ```

## Passo 4: Configurazione Accesso Remoto

### Accesso Locale
```bash
# Il bot è configurato per accesso remoto sulla porta 8501
# Verifica che il servizio sia in ascolto
sudo netstat -tlnp | grep 8501
```

### Accesso da Internet (Metodo Diretto)
```bash
# Verifica firewall
sudo ufw status
# Dovrebbe mostrare: 8501 ALLOW Anywhere

# Trova l'IP del server
hostname -I
# Accedi da browser: http://[IP-SERVER]:8501
```

### Accesso Sicuro (SSH Tunnel)
```bash
# Dal tuo computer locale, crea tunnel SSH
ssh -L 8501:localhost:8501 nomeutente@ip-server

# Poi accedi a: http://localhost:8501
```

## Passo 5: Test dell'Installazione

1. **Test completo**
   ```bash
   cd ~/ai-trading-bot
   source venv/bin/activate
   python check_install.py
   ```

2. **Test rapido**
   ```bash
   cd ~/ai-trading-bot
   ./healthcheck.sh
   ```

3. **Test connettività web**
   ```bash
   curl -I http://localhost:8501
   # Dovrebbe restituire: HTTP/1.1 200 OK
   ```

## Passo 6: Configurazione Avanzata Server

### Per Server Remoto/VPS

1. **Configura accesso sicuro**
   ```bash
   # Modifica configurazione SSH
   sudo nano /etc/ssh/sshd_config
   
   # Aggiungi/modifica:
   PermitRootLogin no
   PasswordAuthentication no
   PubkeyAuthentication yes
   
   # Riavvia SSH
   sudo systemctl restart sshd
   ```

2. **Setup certificato SSL (opzionale)**
   ```bash
   # Installa nginx
   sudo apt install nginx certbot python3-certbot-nginx
   
   # Configura reverse proxy
   sudo nano /etc/nginx/sites-available/trading-bot
   ```

3. **Monitoraggio automatico**
   ```bash
   # Aggiungi al crontab per check ogni 5 minuti
   crontab -e
   
   # Aggiungi:
   */5 * * * * ~/ai-trading-bot/healthcheck.sh >> ~/ai-trading-bot/logs/monitor.log 2>&1
   ```

## Risoluzione Problemi Linux

### Dipendenze mancanti
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3-dev build-essential libssl-dev libffi-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel openssl-devel libffi-devel
```

### Porta 8501 occupata
```bash
# Trova processo che usa la porta
sudo lsof -i :8501
sudo netstat -tlnp | grep 8501

# Termina processo se necessario
sudo kill -9 [PID]
```

### Virtual environment corrotto
```bash
# Ricrea virtual environment
cd ~/ai-trading-bot
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

# ⚙️ CONFIGURAZIONE POST-INSTALLAZIONE (TUTTI I SISTEMI)

## Passo 1: Configurazione File .env

Dopo l'installazione, devi configurare le API keys per il trading live:

```bash
# Windows
notepad %USERPROFILE%\ai-trading-bot\.env

# macOS/Linux
nano ~/ai-trading-bot/.env
```

### Configurazione Base (Modalità Simulazione)
```env
# Modalità trading - inizia sempre con simulation
TRADING_MODE=simulation
INITIAL_CAPITAL=10000
MAX_POSITIONS=5
RISK_PERCENTAGE=2.0

# Sistema (compilato automaticamente)
SYSTEM_OS=auto
SYSTEM_ARCH=auto
LOG_LEVEL=INFO
```

### Configurazione Avanzata (Trading Live)
```env
# ⚠️ SOLO dopo aver testato in modalità simulation

# Binance (Exchange principale)
BINANCE_API_KEY=tua_chiave_binance_qui
BINANCE_SECRET_KEY=tua_chiave_segreta_binance_qui
BINANCE_TESTNET=false

# Coinbase (Exchange secondario)
COINBASE_API_KEY=tua_chiave_coinbase_qui
COINBASE_SECRET_KEY=tua_chiave_segreta_coinbase_qui
COINBASE_SANDBOX=false

# Feed dati di mercato
ALPHA_VANTAGE_API_KEY=tua_chiave_alpha_vantage_qui
NEWSAPI_KEY=tua_chiave_news_api_qui

# AI Models (opzionale)
HUGGINGFACE_API_TOKEN=tuo_token_huggingface_qui

# Notifiche (opzionale)
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=tua_email@gmail.com
EMAIL_PASSWORD=tua_password_app_gmail

# ⚠️ Cambia solo quando sei pronto
TRADING_MODE=live
```

## Passo 2: Ottenere le API Keys

### Binance (Raccomandato)
1. Vai su [binance.com](https://binance.com)
2. Registrati e completa la verifica KYC
3. Vai su Account → API Management
4. Crea nuova API key
   - Abilita "Enable Reading"
   - Abilita "Enable Spot & Margin Trading"
   - **NON** abilitare "Enable Withdrawals"
5. Configura restrizioni IP per sicurezza
6. Copia API Key e Secret Key nel file .env

### Coinbase Pro/Advanced Trade
1. Vai su [pro.coinbase.com](https://pro.coinbase.com)
2. Vai su Settings → API
3. Crea nuova API key
   - Abilita "View" e "Trade"
   - **NON** abilitare "Transfer"
4. Salva API Key, Secret, e Passphrase

### Alpha Vantage (Dati di mercato)
1. Vai su [alphavantage.co](https://www.alphavantage.co/support/#api-key)
2. Richiedi chiave gratuita (500 chiamate/giorno)
3. Copia la chiave nel file .env

### News API (Sentiment analysis)
1. Vai su [newsapi.org](https://newsapi.org/register)
2. Registrazione gratuita (1000 richieste/giorno)
3. Copia la chiave nel file .env

## Passo 3: Test della Configurazione

1. **Test in modalità simulazione**
   ```bash
   # Avvia il bot
   tradingbot
   
   # Vai su http://localhost:5000
   # Verifica che tutti i moduli siano "Operativi"
   ```

2. **Test delle API keys**
   ```bash
   # Nella dashboard, vai su "Setup & Control"
   # Clicca "Test API Connections"
   # Verifica che le connessioni siano verdi
   ```

3. **Test di trading simulato**
   ```bash
   # Nella dashboard:
   # 1. Vai su "Live Trading"
   # 2. Seleziona "Paper Trading Mode"
   # 3. Avvia un trade di test con capitale simulato
   # 4. Verifica che gli ordini vengano eseguiti
   ```

## Passo 4: Passaggio al Trading Live

⚠️ **ATTENZIONE**: Passa al trading live solo dopo aver:
- Testato completamente in modalità simulazione
- Compreso il funzionamento del bot
- Configurato limiti di rischio appropriati
- Testato con importi piccoli

1. **Modifica .env per trading live**
   ```env
   TRADING_MODE=live
   BINANCE_TESTNET=false
   COINBASE_SANDBOX=false
   ```

2. **Configura limiti di sicurezza**
   ```env
   INITIAL_CAPITAL=1000        # Inizia con importi piccoli
   RISK_PERCENTAGE=1.0         # Riduci rischio all'1%
   MAX_POSITIONS=2             # Limita posizioni simultanee
   MAX_DAILY_LOSS=100          # Stop loss giornaliero
   ```

3. **Riavvia il bot**
   ```bash
   # Windows: Chiudi e riapri il bot
   # macOS: Ctrl+C e poi tradingbot
   # Linux: sudo systemctl restart ai-trading-bot
   ```

---

# 🔧 GESTIONE E MANUTENZIONE

## Comandi Essenziali

### Tutti i sistemi
```bash
# Avvio manuale
tradingbot

# Test diagnostico completo
python check_install.py

# Accesso interfaccia web
# http://localhost:5000 (Windows/macOS)
# http://server-ip:8501 (Linux)
```

### Linux (Server)
```bash
# Gestione servizio
sudo systemctl start|stop|restart ai-trading-bot
sudo systemctl status ai-trading-bot

# Log in tempo reale
sudo journalctl -u ai-trading-bot -f

# Check rapido sistema
./healthcheck.sh
```

## Monitoraggio

### Dashboard Web
- **Portfolio**: Valore totale, P&L, allocazioni
- **Trading Live**: Posizioni aperte, ordini, storia trades
- **AI Intelligence**: Decisioni AI, sentiment, segnali
- **Performance**: Metriche Sharpe, drawdown, win rate
- **System Monitor**: CPU, RAM, connessioni API

### Log Files
- **Windows**: `%USERPROFILE%\ai-trading-bot\logs\`
- **macOS**: `~/ai-trading-bot/logs/`
- **Linux**: `~/ai-trading-bot/logs/` + journalctl

## Backup e Sicurezza

### Backup Automatico
Il sistema crea backup automatici di:
- Database AI models
- Configurazioni
- Log delle transazioni
- Performance storiche

### Backup Manuale
```bash
# Crea backup completo
cd ~/ai-trading-bot
tar -czf backup-$(date +%Y%m%d).tar.gz \
  --exclude=venv \
  --exclude=logs/*.log \
  .
```

## Aggiornamenti

### Aggiornamento Dipendenze
```bash
cd ~/ai-trading-bot
source venv/bin/activate  # Linux/macOS only
pip install --upgrade -r requirements.txt
```

### Aggiornamento Sistema
- Windows: Riesegui installer
- macOS: Riesegui install.sh
- Linux: Riesegui install.sh

---

# 📞 SUPPORTO E RISOLUZIONE PROBLEMI

## Diagnosi Automatica

### Check Completo
```bash
cd ~/ai-trading-bot
python check_install.py
```

### Check Rapido (Linux/macOS)
```bash
./healthcheck.sh
```

### Check Performance
```bash
python PERFORMANCE_CALCULATOR.py
```

## Problemi Comuni

### Bot non si avvia
1. Verifica Python: `python --version`
2. Verifica dipendenze: `pip list`
3. Controlla logs per errori
4. Riesegui test: `python check_install.py`

### Interfaccia web non accessibile
1. Verifica porta: `netstat -an | grep 5000`
2. Controlla firewall
3. Prova browser diverso
4. Verifica che il bot sia in esecuzione

### API connections falliscono
1. Verifica chiavi API nel file .env
2. Controlla scadenza chiavi
3. Verifica restrizioni IP
4. Test connettività: ping api.binance.com

### Performance degradate
1. Controlla uso CPU/RAM
2. Pulisci cache: riavvia bot
3. Verifica spazio disco
4. Ottimizza configurazione

## Contatto Supporto

### Informazioni da Includere
1. Sistema operativo e versione
2. Output di `python check_install.py`
3. Log files recenti
4. Descrizione del problema
5. Passi per riprodurre il problema

### File di Log Utili
- `logs/trading.log` - Log principale
- `logs/error.log` - Errori sistema
- `logs/ai_decisions.log` - Decisioni AI
- `installation_report_*.json` - Report installazione

---

## 🎯 Risultato Finale

Seguendo questa guida avrai:

✅ **Sistema completamente installato** e configurato per il tuo OS
✅ **Trading bot AI operativo** con tutte le funzionalità attive
✅ **Accesso web sicuro** locale o remoto secondo le tue esigenze
✅ **Configurazione API** per trading simulato o live
✅ **Monitoraggio** e strumenti diagnostici funzionanti
✅ **Backup automatici** e procedure di manutenzione attive

Il tuo AI Crypto Trading Bot è ora pronto per operazioni di trading automatizzato 24/7!