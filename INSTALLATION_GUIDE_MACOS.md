# AI Trading Bot - Guida Installazione macOS

## Installazione Completa su macOS

### Prerequisiti Sistema

#### Installa Homebrew (se non già installato)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Installa Python e Dipendenze
```bash
# Installa Python 3.11
brew install python@3.11

# Verifica installazione
python3 --version
pip3 --version

# Installa git se necessario
brew install git
```

### Step 1: Download e Setup

#### Metodo 1: Download Diretto
```bash
# Crea directory progetto
mkdir ~/ai-trading-bot
cd ~/ai-trading-bot

# Download e estrazione (sostituisci URL con il tuo)
curl -L "URL_DEL_ZIP" -o AI_Trading_Bot_macOS.zip
unzip AI_Trading_Bot_macOS.zip
cd AI_Trading_Bot_macOS
```

#### Metodo 2: Setup Manuale
```bash
# Se hai i file in locale
mkdir ~/ai-trading-bot
cd ~/ai-trading-bot
# Copia i file del bot nella directory
```

### Step 2: Setup Environment Python

```bash
# Crea ambiente virtuale
python3 -m venv trading_env

# Attiva ambiente virtuale
source trading_env/bin/activate

# Aggiorna pip
pip install --upgrade pip

# Installa dipendenze
pip install streamlit pandas numpy plotly scikit-learn asyncio

# Per Apple Silicon (M1/M2/M3) potrebbero servire comandi specifici:
# export ARCHFLAGS="-arch arm64"
# pip install --no-cache-dir numpy pandas
```

### Step 3: Configurazione Streamlit

```bash
# Crea directory configurazione
mkdir -p ~/.streamlit

# Crea file configurazione
cat > ~/.streamlit/config.toml << EOF
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
EOF
```

### Step 4: Avvio del Sistema

#### Avvio Manuale
```bash
# Attiva ambiente virtuale
source trading_env/bin/activate

# Avvia l'applicazione
streamlit run advanced_ai_system.py --server.port 5000 --server.address 0.0.0.0
```

#### Script di Avvio Automatico

Crea `start_trading_bot.sh`:
```bash
#!/bin/bash
cd ~/ai-trading-bot/AI_Trading_Bot_macOS
source trading_env/bin/activate
streamlit run advanced_ai_system.py --server.port 5000 --server.address 0.0.0.0
```

Rendi eseguibile:
```bash
chmod +x start_trading_bot.sh
```

#### Avvio Automatico con LaunchDaemon

Crea file plist per avvio automatico:
```bash
# Crea directory se non esiste
mkdir -p ~/Library/LaunchAgents

# Crea file LaunchAgent
cat > ~/Library/LaunchAgents/com.aitradingbot.app.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.aitradingbot.app</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-c</string>
        <string>cd $HOME/ai-trading-bot/AI_Trading_Bot_macOS && source trading_env/bin/activate && streamlit run advanced_ai_system.py --server.port 5000 --server.address 0.0.0.0</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/aitradingbot.out</string>
    <key>StandardErrorPath</key>
    <string>/tmp/aitradingbot.err</string>
</dict>
</plist>
EOF

# Carica LaunchAgent
launchctl load ~/Library/LaunchAgents/com.aitradingbot.app.plist

# Avvia servizio
launchctl start com.aitradingbot.app
```

### Step 5: Configurazione Firewall macOS

```bash
# Abilita Application Firewall se necessario
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on

# Aggiungi Python alle app consentite (se richiesto)
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/bin/python3
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp /usr/bin/python3
```

## Accesso alla Dashboard

### Accesso Locale
Apri browser e vai a:
- `http://localhost:5000`
- `http://127.0.0.1:5000`

### Accesso da Altri Dispositivi sulla Rete
```bash
# Trova IP del Mac
ifconfig | grep "inet " | grep -v 127.0.0.1

# Da altri dispositivi: http://IP_DEL_MAC:5000
```

### Configurazione Router (Accesso Esterno)
1. Accedi al router
2. Configura Port Forwarding:
   - **Porta Esterna**: 5000
   - **Porta Interna**: 5000
   - **IP Interno**: IP del Mac
   - **Protocollo**: TCP

## Gestione Sistema

### Comandi Utili

```bash
# Verifica processo Streamlit
ps aux | grep streamlit

# Termina processo se necessario
pkill -f streamlit

# Verifica porta 5000 in uso
lsof -i :5000
netstat -an | grep 5000

# Test connessione locale
curl -I http://localhost:5000
```

### Script di Gestione

Crea `manage_bot.sh`:
```bash
#!/bin/bash

BOT_DIR="$HOME/ai-trading-bot/AI_Trading_Bot_macOS"
VENV_PATH="$BOT_DIR/trading_env/bin/activate"

start_bot() {
    echo "Avvio AI Trading Bot..."
    cd "$BOT_DIR"
    source "$VENV_PATH"
    nohup streamlit run advanced_ai_system.py --server.port 5000 --server.address 0.0.0.0 > /tmp/tradingbot.log 2>&1 &
    echo "Bot avviato! PID: $!"
}

stop_bot() {
    echo "Fermando AI Trading Bot..."
    pkill -f "streamlit.*advanced_ai_system"
    echo "Bot fermato!"
}

status_bot() {
    if pgrep -f "streamlit.*advanced_ai_system" > /dev/null; then
        echo "Bot in esecuzione"
        ps aux | grep "streamlit.*advanced_ai_system" | grep -v grep
    else
        echo "Bot non in esecuzione"
    fi
}

restart_bot() {
    stop_bot
    sleep 3
    start_bot
}

case "$1" in
    start)
        start_bot
        ;;
    stop)
        stop_bot
        ;;
    restart)
        restart_bot
        ;;
    status)
        status_bot
        ;;
    *)
        echo "Uso: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
```

Rendi eseguibile e usa:
```bash
chmod +x manage_bot.sh

# Avvia bot
./manage_bot.sh start

# Ferma bot
./manage_bot.sh stop

# Riavvia bot
./manage_bot.sh restart

# Verifica status
./manage_bot.sh status
```

### Gestione LaunchAgent

```bash
# Verifica status LaunchAgent
launchctl list | grep aitradingbot

# Stop LaunchAgent
launchctl stop com.aitradingbot.app

# Start LaunchAgent
launchctl start com.aitradingbot.app

# Rimuovi LaunchAgent
launchctl unload ~/Library/LaunchAgents/com.aitradingbot.app.plist
rm ~/Library/LaunchAgents/com.aitradingbot.app.plist

# Visualizza logs
tail -f /tmp/aitradingbot.out
tail -f /tmp/aitradingbot.err
```

## Aggiornamenti

### Aggiornamento Manuale
```bash
# Ferma bot
./manage_bot.sh stop

# Backup configurazione
cp -r config config_backup

# Sostituisci file aggiornati
# Copia nuovi file sopra i vecchi

# Aggiorna dipendenze se necessario
source trading_env/bin/activate
pip install --upgrade streamlit pandas numpy plotly scikit-learn

# Riavvia bot
./manage_bot.sh start
```

### Aggiornamento Homebrew e Python
```bash
# Aggiorna Homebrew
brew update && brew upgrade

# Aggiorna Python se disponibile
brew upgrade python@3.11
```

## Backup e Restore

### Script Backup Automatico

Crea `backup_bot.sh`:
```bash
#!/bin/bash

BACKUP_DIR="$HOME/Backups/AI-Trading-Bot"
SOURCE_DIR="$HOME/ai-trading-bot"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/trading-bot-backup-$DATE.tar.gz"

# Crea directory backup se non esiste
mkdir -p "$BACKUP_DIR"

# Comprimi directory sorgente
tar -czf "$BACKUP_FILE" -C "$SOURCE_DIR" .

# Mantieni solo ultimi 7 backup
find "$BACKUP_DIR" -name "trading-bot-backup-*.tar.gz" -mtime +7 -delete

echo "Backup completato: $BACKUP_FILE"
```

### Backup Automatico con Cron
```bash
# Rendi eseguibile script backup
chmod +x backup_bot.sh

# Aggiungi a crontab per backup giornaliero alle 2:00
(crontab -l 2>/dev/null; echo "0 2 * * * $HOME/ai-trading-bot/backup_bot.sh") | crontab -

# Verifica crontab
crontab -l
```

### Restore da Backup
```bash
# Ferma bot
./manage_bot.sh stop

# Restore
cd ~/ai-trading-bot
tar -xzf ~/Backups/AI-Trading-Bot/trading-bot-backup-YYYYMMDD_HHMMSS.tar.gz

# Riavvia bot
./manage_bot.sh start
```

## Monitoraggio Performance

### Script Monitoraggio Sistema

Crea `monitor_system.sh`:
```bash
#!/bin/bash

while true; do
    clear
    echo "=== AI Trading Bot System Monitor ==="
    echo "Timestamp: $(date)"
    echo
    
    # CPU Usage
    CPU_USAGE=$(top -l 1 | grep "CPU usage" | awk '{print $3}' | sed 's/%//')
    echo "CPU Usage: ${CPU_USAGE}%"
    
    # Memory Usage
    MEMORY_PRESSURE=$(memory_pressure | grep "System-wide memory free percentage" | awk '{print $5}' | sed 's/%//')
    MEMORY_USED=$((100 - MEMORY_PRESSURE))
    echo "Memory Usage: ${MEMORY_USED}%"
    
    # Disk Usage
    DISK_USAGE=$(df -h / | tail -1 | awk '{print $5}' | sed 's/%//')
    echo "Disk Usage: ${DISK_USAGE}%"
    
    # Bot Status
    if pgrep -f "streamlit.*advanced_ai_system" > /dev/null; then
        BOT_PID=$(pgrep -f "streamlit.*advanced_ai_system")
        BOT_MEMORY=$(ps -o rss= -p $BOT_PID | awk '{print $1/1024}')
        echo "Bot Status: Running (PID: $BOT_PID)"
        printf "Bot Memory: %.2f MB\n" $BOT_MEMORY
    else
        echo "Bot Status: Not Running"
    fi
    
    # Network Connection
    if nc -z localhost 5000 2>/dev/null; then
        echo "Dashboard: Accessible on http://localhost:5000"
    else
        echo "Dashboard: Not Accessible"
    fi
    
    echo
    echo "Press Ctrl+C to exit"
    sleep 5
done
```

Uso:
```bash
chmod +x monitor_system.sh
./monitor_system.sh
```

## Configurazioni Specifiche per Apple Silicon

### Per Mac M1/M2/M3
```bash
# Verifica architettura
uname -m

# Se hai M1/M2/M3 (arm64), potrebbero servire configurazioni specifiche
# Installa Rosetta 2 se necessario per compatibilità
sudo softwareupdate --install-rosetta

# Variabili ambiente per compilazione ottimizzata
export ARCHFLAGS="-arch arm64"
export _PYTHON_HOST_PLATFORM="macosx-11.0-arm64"
export MACOSX_DEPLOYMENT_TARGET="11.0"

# Reinstalla dipendenze con ottimizzazioni ARM
pip uninstall numpy pandas -y
pip install --no-cache-dir numpy pandas
```

## Accesso Remoto

### SSH Tunnel
```bash
# Da altro Mac/Linux verso questo Mac
ssh -L 5000:localhost:5000 username@IP_DEL_MAC
# Poi accedi a http://localhost:5000

# Configurazione SSH per accesso più facile
echo "Host trading-bot
    HostName IP_DEL_MAC
    User username
    LocalForward 5000 localhost:5000" >> ~/.ssh/config
    
# Connessione semplificata
ssh trading-bot
```

### VNC per Accesso Desktop Completo
```bash
# Abilita Screen Sharing nelle Preferenze di Sistema
# Vai in Sistema → Condivisione → Condivisione Schermo

# Accesso da altri Mac
open vnc://IP_DEL_MAC

# Da Windows/Linux usa VNC Viewer
```

### TeamViewer/AnyDesk (Alternativa)
```bash
# Installa con Homebrew
brew install --cask teamviewer
# oppure
brew install --cask anydesk

# Configura accesso non supervisionato
```

## Troubleshooting

### Problemi Comuni

```bash
# Python non trovato
# Reinstalla con Homebrew
brew install python@3.11

# Problemi con pip
python3 -m ensurepip --upgrade

# Porta 5000 occupata
lsof -ti:5000 | xargs kill -9

# Problemi permessi
sudo chown -R $(whoami) ~/ai-trading-bot

# Reset ambiente Python
rm -rf trading_env
python3 -m venv trading_env
source trading_env/bin/activate
pip install streamlit pandas numpy plotly scikit-learn
```

### Specifici per Apple Silicon
```bash
# Errori di compilazione
export ARCHFLAGS="-arch arm64"
pip install --no-cache-dir --upgrade pip setuptools wheel

# Problemi con librerie native
brew install openblas lapack
export OPENBLAS=$(brew --prefix openblas)
```

### Logs e Debug
```bash
# Logs LaunchAgent
tail -f /tmp/aitradingbot.out
tail -f /tmp/aitradingbot.err

# Debug manuale
source trading_env/bin/activate
streamlit run advanced_ai_system.py --logger.level debug --server.port 5000

# Verifica connessioni
lsof -i :5000
netstat -an | grep 5000
```

## Configurazione Avanzata

### Ottimizzazione Performance
```bash
# Aumenta limiti sistema
echo "kern.maxfiles=65536" | sudo tee -a /etc/sysctl.conf
echo "kern.maxfilesperproc=65536" | sudo tee -a /etc/sysctl.conf

# Riavvia per applicare
sudo reboot
```

### Configurazione Firewall Avanzata
```bash
# Configura pfctl (firewall avanzato macOS)
# Crea file regole
sudo tee /etc/pf.anchors/trading-bot << EOF
# Allow inbound connections on port 5000
pass in on en0 proto tcp from any to any port 5000
EOF

# Carica regole
sudo pfctl -f /etc/pf.conf
sudo pfctl -e
```

Il sistema è ora completamente configurato per macOS con accesso locale e remoto alla dashboard!