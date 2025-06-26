# AI Trading Bot - Quick Start Guide

## Avvio Rapido per Tutti i Sistemi

### Linux Server (Raccomandato per 24/7)
```bash
# Setup rapido
mkdir ~/ai-trading-bot && cd ~/ai-trading-bot
python3 -m venv trading_env
source trading_env/bin/activate
pip install streamlit pandas numpy plotly scikit-learn

# Avvio
streamlit run advanced_ai_system.py --server.port 5000 --server.address 0.0.0.0

# Accesso dashboard: http://IP_SERVER:5000
```

### Windows (Desktop)
```powershell
# Setup rapido
python -m venv trading_env
.\trading_env\Scripts\Activate.ps1
pip install streamlit pandas numpy plotly scikit-learn

# Avvio
streamlit run advanced_ai_system.py --server.port 5000

# Accesso: http://localhost:5000
```

### macOS
```bash
# Setup rapido
python3 -m venv trading_env
source trading_env/bin/activate
pip install streamlit pandas numpy plotly scikit-learn

# Avvio
streamlit run advanced_ai_system.py --server.port 5000

# Accesso: http://localhost:5000
```

## Configurazione Iniziale

### 1. Connetti Exchange
- Vai nel tab "Setup & Control"
- Configura almeno un exchange con API Key + Secret
- **NON servono passphrase** - solo Key + Secret

### 2. Modalità Trading
- **Paper Trading**: Modalità sicura per test (raccomandato per iniziare)
- **Live Trading**: Modalità reale con soldi veri
- **AI Full Control**: L'AI gestisce tutto automaticamente

### 3. Configurazione Alert (Opzionale)
- **Telegram**: Bot Token + Chat ID
- **Email**: Configurazione SMTP

### 4. Avvia Sistema
- Clicca "START SYSTEM"
- Monitora dashboard in tempo reale

## Accesso Remoto (Server Linux)

### SSH Tunnel (Sicuro)
```bash
# Da Windows/macOS/Linux
ssh -L 5000:localhost:5000 user@IP_SERVER

# Poi accedi a: http://localhost:5000
```

### Accesso Diretto
```bash
# Configura firewall server
sudo ufw allow 5000/tcp

# Accedi da qualsiasi dispositivo: http://IP_SERVER:5000
```

## API Exchange - Cosa Serve

### Binance
- ✅ API Key
- ✅ Secret Key  
- ✅ Permessi: Spot Trading + Read Info
- ❌ Passphrase NON necessaria

### Bybit
- ✅ API Key
- ✅ Secret Key
- ✅ Permessi: Trade + Read
- ❌ Passphrase NON necessaria

### Altri Exchange
- Stessi requisiti: solo API Key + Secret
- Abilita permessi di trading (NO withdraw per sicurezza)

## Comandi Utili

### Linux
```bash
# Verifica bot attivo
ps aux | grep streamlit

# Stop bot
pkill -f streamlit

# Restart bot
./manage_bot.sh restart

# Logs
journalctl -u ai-trading-bot -f
```

### Windows
```powershell
# Verifica bot attivo
Get-Process -Name "*streamlit*"

# Stop bot
Stop-Process -Name "streamlit" -Force

# Restart bot
.\manage_bot.ps1 restart
```

### macOS
```bash
# Verifica bot attivo
ps aux | grep streamlit

# Stop bot
pkill -f streamlit

# Restart bot
./manage_bot.sh restart
```

## Sicurezza

### Setup Sicuro
1. **Mai abilitare withdraw** nelle API
2. **Usa IP whitelist** quando possibile
3. **Testa sempre in paper trading** prima
4. **Backup configurazioni** regolarmente
5. **Monitora logs** per attività sospette

### Accesso Remoto Sicuro
1. **SSH Tunnel** sempre quando possibile
2. **VPN** per accesso esterno
3. **Cambia porta SSH** default (22)
4. **Disable root login** SSH
5. **Usa fail2ban** contro brute force

## Troubleshooting Rapido

### Bot Non Si Avvia
```bash
# Verifica Python
python3 --version

# Reinstalla dipendenze
pip install --upgrade streamlit

# Verifica porta libera
netstat -tulpn | grep :5000
```

### Dashboard Non Accessibile
```bash
# Verifica processo
ps aux | grep streamlit

# Verifica firewall
sudo ufw status  # Linux
netsh advfirewall show allprofiles  # Windows

# Test locale
curl http://localhost:5000
```

### Errori API Exchange
1. Verifica API Key corrette
2. Controlla permessi abilitati
3. Verifica IP whitelist
4. Testa con Testnet prima

## Supporto

### Log Locations
- **Linux**: `/var/log/` o `journalctl`
- **Windows**: Event Viewer o file locali
- **macOS**: `/tmp/` o Console.app

### File Importanti
- `config/system_config.json` - Configurazione principale
- `config/advanced_config.json` - Configurazioni avanzate
- `logs/` - Log applicazione
- `.streamlit/config.toml` - Configurazione Streamlit

### Reset Sistema
```bash
# Reset completo (ATTENZIONE: perde configurazioni)
rm -rf config/
rm -rf logs/
# Riavvia bot per rigenerare configurazioni default
```

Il sistema è pronto per l'uso! Inizia sempre con **Paper Trading** per familiarizzare con l'interfaccia.