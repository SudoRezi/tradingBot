# AI Trading Bot - Guida Rapida di Riferimento

## 🚀 Installazione Ultra-Rapida

### Windows
```cmd
1. Scarica AI_Trading_Bot_Windows_Installer.zip
2. Estrai → Click destro install.bat → "Esegui come amministratore"
3. Attendi installazione automatica (5-10 minuti)
4. Avvia: tradingbot
```

### macOS
```bash
1. Scarica AI_Trading_Bot_macOS_Installer.zip
2. unzip AI_Trading_Bot_macOS_Installer.zip && cd macos/
3. ./install.sh
4. tradingbot
```

### Linux
```bash
1. Scarica AI_Trading_Bot_Linux_Installer.zip
2. unzip AI_Trading_Bot_Linux_Installer.zip && cd linux/
3. ./install.sh
4. sudo systemctl start ai-trading-bot
```

## ⚡ Primo Avvio (3 minuti)

### 1. Accesso Web Interface
- **Windows/macOS**: http://localhost:5000
- **Linux**: http://server-ip:8501

### 2. Configurazione Base
```bash
# Apri file configurazione
nano ~/ai-trading-bot/.env    # macOS/Linux
notepad %USERPROFILE%\ai-trading-bot\.env    # Windows

# Impostazioni minime
TRADING_MODE=simulation
INITIAL_CAPITAL=10000
RISK_PERCENTAGE=2.0
```

### 3. Test Sistema
```bash
cd ~/ai-trading-bot
python check_install.py
# Verifica: ✅ 6+ componenti operativi
```

## 🔑 API Keys (Trading Live)

### Exchange Principali
```env
# Binance (raccomandato)
BINANCE_API_KEY=your_key_here
BINANCE_SECRET_KEY=your_secret_here

# Coinbase
COINBASE_API_KEY=your_key_here
COINBASE_SECRET_KEY=your_secret_here
COINBASE_PASSPHRASE=your_passphrase_here
```

### Dati di Mercato
```env
# Alpha Vantage (gratuito)
ALPHA_VANTAGE_API_KEY=your_key_here

# News API (gratuito)
NEWSAPI_KEY=your_key_here
```

### Dove Ottenere le Chiavi
- **Binance**: binance.com → Account → API Management
- **Coinbase**: pro.coinbase.com → Settings → API
- **Alpha Vantage**: alphavantage.co/support/#api-key
- **News API**: newsapi.org/register

## 🎛️ Comandi Essenziali

### Gestione Bot
```bash
# Avvio
tradingbot                           # Tutti i sistemi

# Windows
python %USERPROFILE%\ai-trading-bot\advanced_ai_system.py

# macOS/Linux manuale
cd ~/ai-trading-bot && python advanced_ai_system.py

# Linux servizio
sudo systemctl start ai-trading-bot     # Avvia
sudo systemctl stop ai-trading-bot      # Ferma
sudo systemctl restart ai-trading-bot   # Riavvia
sudo systemctl status ai-trading-bot    # Status
```

### Diagnostica
```bash
python check_install.py      # Test completo
./healthcheck.sh             # Test rapido (Linux/macOS)
python system_health_check.py # Test operativo
```

### Log e Monitoraggio
```bash
# Log files
tail -f ~/ai-trading-bot/logs/trading.log

# Linux servizio log
sudo journalctl -u ai-trading-bot -f

# Performance
python PERFORMANCE_CALCULATOR.py
```

## 🌐 Accesso Remoto

### SSH Tunnel (Sicuro)
```bash
# Dal tuo computer
ssh -L 5000:localhost:5000 user@server-ip
# Poi vai su: http://localhost:5000
```

### Diretto (Server pubblico)
```bash
# Verifica firewall aperto
sudo ufw allow 8501
# Accesso: http://server-ip:8501
```

### Ngrok (Tunnel rapido)
```bash
ngrok http 5000
# Usa URL HTTPS fornito
```

## ⚙️ Configurazioni Chiave

### Sicurezza Trading
```env
TRADING_MODE=simulation    # Inizia sempre così
INITIAL_CAPITAL=10000      # Capitale iniziale
RISK_PERCENTAGE=1.0        # 1% rischio per trade
MAX_POSITIONS=3            # Max 3 posizioni aperte
MAX_DAILY_LOSS=500         # Stop loss giornaliero
```

### Performance
```env
ENABLE_PERFORMANCE_MODE=true
CPU_OPTIMIZATION=true
MEMORY_OPTIMIZATION=true
MAX_THREADS=4
```

### Interfaccia
```env
STREAMLIT_HOST=0.0.0.0     # Accesso remoto
STREAMLIT_PORT=5000        # Porta web
STREAMLIT_THEME=dark       # Tema scuro
```

## 🔧 Risoluzione Problemi Rapida

### Bot non si avvia
```bash
# 1. Verifica Python
python --version           # Deve essere 3.8+

# 2. Test installazione
python check_install.py

# 3. Reinstalla dipendenze
pip install -r requirements.txt
```

### Porta occupata
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID [numero] /F

# Linux/macOS
sudo lsof -i :5000
sudo kill -9 [PID]
```

### API non funzionano
```bash
# 1. Verifica chiavi in .env
cat ~/ai-trading-bot/.env | grep API

# 2. Test connessione
curl -I https://api.binance.com/api/v3/ping

# 3. Verifica restrizioni IP su exchange
```

### Performance lente
```bash
# 1. Riavvia bot
sudo systemctl restart ai-trading-bot

# 2. Pulisci cache
rm -rf ~/ai-trading-bot/data/cache/*

# 3. Verifica risorse
htop
df -h
```

## 📊 Dashboard - Funzioni Principali

### Tab Principali
1. **Setup & Control** - Configurazione iniziale
2. **Live Trading** - Trading in tempo reale
3. **AI Intelligence** - Decisioni AI
4. **Data Feeds** - Feed dati mercato
5. **Advanced Config** - Configurazioni avanzate
6. **HuggingFace Models** - Modelli AI
7. **QuantConnect** - Backtesting
8. **Security & Orders** - Sicurezza e ordini
9. **Microcap Gems** - Analisi altcoin
10. **Smart Performance** - Ottimizzazioni
11. **System Monitor** - Monitoraggio sistema

### Controlli Chiave
- **Emergency Stop** - Ferma tutto immediatamente
- **Test API Connections** - Verifica connessioni
- **Backup AI Models** - Salva configurazioni
- **Export Performance** - Esporta risultati

## 🎯 Checklist Pre-Trading Live

### Prima di passare da simulation a live:

- [ ] Sistema testato in modalità simulation per almeno 1 settimana
- [ ] Tutte le API connections verdi nella dashboard
- [ ] Backup completo della configurazione effettuato
- [ ] Limiti di rischio configurati (max 2% per trade)
- [ ] Capitale di test limitato (max €1000 iniziali)
- [ ] Monitoraggio attivo configurato
- [ ] Stop loss giornaliero impostato
- [ ] Compreso il funzionamento di tutti i controlli

### Cambia in .env:
```env
TRADING_MODE=live
BINANCE_TESTNET=false
COINBASE_SANDBOX=false
```

## 📞 Supporto Rapido

### File da Controllare
1. `~/ai-trading-bot/logs/trading.log` - Log principale
2. `~/ai-trading-bot/.env` - Configurazione
3. `~/ai-trading-bot/config/config.yaml` - Impostazioni avanzate

### Comandi Diagnostici
```bash
# Sistema generale
python check_install.py > diagnostics.txt

# Performance
python PERFORMANCE_CALCULATOR.py > performance.txt

# Health check
./healthcheck.sh > health.txt
```

### Reset Completo (Emergency)
```bash
# Ferma bot
sudo systemctl stop ai-trading-bot

# Backup dati importanti
cp ~/ai-trading-bot/.env ~/ai-trading-bot/.env.backup

# Reset configurazione
cd ~/ai-trading-bot
git stash  # Se installato via git
# oppure
rm -rf data/cache/* logs/*.log

# Riavvia
sudo systemctl start ai-trading-bot
```

## 🚀 Stato Sistema Attuale

✅ **Core operativo al 100%**
✅ **8/8 componenti verificati**
✅ **Advanced Order System testato**
✅ **AI Models Hub completo**
✅ **Smart Performance attivo**
✅ **Installer multi-piattaforma pronti**

**Solo API keys richieste per trading live**

---

*Guida aggiornata: Giugno 2025 - Sistema completamente operativo e pronto per distribuzione*