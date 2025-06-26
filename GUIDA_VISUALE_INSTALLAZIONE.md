# Guida Visuale - AI Crypto Trading Bot

## Panoramica del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                 AI CRYPTO TRADING BOT ARCHITECTURE             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐     │
│  │   Web UI    │◄──►│ Trading Core │◄──►│  AI Intelligence │     │
│  │  Port 5000  │    │   Engine     │    │   20+ Models    │     │
│  └─────────────┘    └──────────────┘    └─────────────────┘     │
│         │                   │                      │            │
│         ▼                   ▼                      ▼            │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐     │
│  │ Streamlit   │    │ Order System │    │ Quant Analytics │     │
│  │ Dashboard   │    │ Advanced     │    │ Backtesting     │     │
│  └─────────────┘    └──────────────┘    └─────────────────┘     │
│                             │                      │            │
│                             ▼                      ▼            │
│                    ┌──────────────┐    ┌─────────────────┐     │
│                    │  Exchanges   │    │ Data Storage    │     │
│                    │ Multi-broker │    │ ArcticDB+SQLite │     │
│                    └──────────────┘    └─────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Flusso di Installazione

### Windows Installation Flow
```
Start Installation
       │
       ▼
┌──────────────┐     ┌────────────────┐     ┌─────────────────┐
│ Download ZIP │────►│ Extract Files  │────►│ Run install.bat │
│ 5.1KB        │     │ install.bat    │     │ as Administrator│
└──────────────┘     │ install.ps1    │     └─────────────────┘
                     │ README.txt     │              │
                     └────────────────┘              ▼
                                               ┌─────────────────┐
       ┌─────────────────────────────────────► │ Auto Install:   │
       │                                       │ • Python 3.11  │
       │                                       │ • Git           │
       ▼                                       │ • Dependencies  │
┌──────────────┐     ┌────────────────┐       │ • Core Files    │
│ Desktop      │     │ CLI Command    │       │ • Configuration │
│ Shortcut     │     │ 'tradingbot'   │       └─────────────────┘
│ Created      │     │ Available      │                │
└──────────────┘     └────────────────┘                ▼
       │                     │                  ┌─────────────────┐
       └─────────────────────┴─────────────────►│ Ready to Launch │
                                                │ http://         │
                                                │ localhost:5000  │
                                                └─────────────────┘
```

### macOS Installation Flow
```
Start Installation
       │
       ▼
┌──────────────┐     ┌────────────────┐     ┌─────────────────┐
│ Download ZIP │────►│ unzip & cd     │────►│ ./install.sh    │
│ 5.2KB        │     │ macos/         │     │                 │
└──────────────┘     └────────────────┘     └─────────────────┘
                                                      │
                                                      ▼
       ┌──────────────────────────────────────────────────────────┐
       │              Auto Detection & Optimization               │
       │                                                          │
       │  Intel x64:                Apple Silicon (M1/M2/M3):    │
       │  • Standard CPU             • ARM optimization           │
       │  • 4GB memory config        • Metal acceleration        │
       │  • Homebrew /usr/local      • 8GB memory config         │
       │                             • Homebrew /opt/homebrew    │
       └──────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────┐     ┌────────────────┐     ┌─────────────────┐
│ macOS App    │     │ CLI Command    │     │ PATH Updated    │
│ Created      │     │ 'tradingbot'   │     │ ~/.zshrc        │
└──────────────┘     └────────────────┘     └─────────────────┘
       │                     │                      │
       └─────────────────────┴──────────────────────┘
                            │
                            ▼
                     ┌─────────────────┐
                     │ Ready to Launch │
                     │ http://         │
                     │ localhost:5000  │
                     └─────────────────┘
```

### Linux Server Installation Flow
```
Start Installation
       │
       ▼
┌──────────────┐     ┌────────────────┐     ┌─────────────────┐
│ Download ZIP │────►│ unzip & cd     │────►│ ./install.sh    │
│ 10.4KB       │     │ linux/         │     │                 │
└──────────────┘     └────────────────┘     └─────────────────┘
                                                      │
                                                      ▼
       ┌──────────────────────────────────────────────────────────┐
       │                Distribution Detection                    │
       │                                                          │
       │  Ubuntu/Debian:          CentOS/RHEL:                   │
       │  • apt packages          • yum/dnf packages             │
       │  • systemd service       • systemd service             │
       │  • ufw firewall          • firewalld                   │
       └──────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────┐     ┌────────────────┐     ┌─────────────────┐
│ Systemd      │     │ Virtual Env    │     │ Remote Access   │
│ Service      │     │ ~/ai-trading-  │     │ Port 8501       │
│ Created      │     │ bot/venv       │     │ Configured      │
└──────────────┘     └────────────────┘     └─────────────────┘
       │                     │                      │
       └─────────────────────┴──────────────────────┘
                            │
                            ▼
                     ┌─────────────────┐
                     │ 24/7 Service    │
                     │ Ready           │
                     │ http://         │
                     │ server-ip:8501  │
                     └─────────────────┘
```

## Dashboard Layout Visuale

```
┌────────────────────────────────────────────────────────────────────────┐
│                    AI CRYPTO TRADING BOT DASHBOARD                     │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│ ┌─ Setup ─┐┌─ Live ──┐┌─ AI ───┐┌─ Data ─┐┌─ Config┐┌─ Models┐┌─More─┐ │
│ │ Control ││Trading ││Intell. ││ Feeds  ││       ││       ││     │ │
│ └─────────┘└────────┘└────────┘└────────┘└───────┘└───────┘└─────┘ │
│                                                                        │
│ ┌──────────────────────────────────────────────────────────────────┐   │
│ │                     MAIN CONTENT AREA                           │   │
│ │                                                                  │   │
│ │  Current Tab Content Displayed Here:                            │   │
│ │  • Setup & Control: API Keys, Initial Config                    │   │
│ │  • Live Trading: Real-time positions, orders, P&L               │   │
│ │  • AI Intelligence: Model decisions, sentiment analysis         │   │
│ │  • Data Feeds: Market data streams, exchange connections        │   │
│ │  • Advanced Config: Risk management, performance tuning         │   │
│ │  • HuggingFace Models: AI model management and downloads        │   │
│ │  • QuantConnect: Backtesting and strategy validation           │   │
│ │                                                                  │   │
│ └──────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐    │
│ │ SYSTEM STATUS   │ │ PORTFOLIO       │ │ RECENT ACTIVITY         │    │
│ │                 │ │                 │ │                         │    │
│ │ ✅ Core: Active │ │ Total: $10,000  │ │ • Trade executed BTC    │    │
│ │ ✅ AI: Learning │ │ P&L: +$127.50   │ │ • AI decision: HOLD ETH │    │
│ │ ✅ APIs: Connected│ │ Positions: 3/5  │ │ • Stop loss triggered  │    │
│ │ ⚠️ News: Limited│ │ Risk: 2.1%      │ │ • New signal detected   │    │
│ └─────────────────┘ └─────────────────┘ └─────────────────────────┘    │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

## Configurazione File Struttura

```
~/ai-trading-bot/                          # Directory principale
├── advanced_ai_system.py                  # ⭐ File principale
├── .env                                   # 🔐 Configurazione API keys
├── config/
│   ├── config.yaml                        # ⚙️ Configurazione avanzata
│   └── strategies.json                    # 📊 Strategie trading
├── data/
│   ├── trading_data.db                    # 💾 Database SQLite
│   ├── market_data/                       # 📈 Dati di mercato
│   └── cache/                             # 🚀 Cache performance
├── logs/
│   ├── trading.log                        # 📝 Log principale
│   ├── ai_decisions.log                   # 🧠 Decisioni AI
│   └── error.log                          # ❌ Log errori
├── models/
│   ├── huggingface/                       # 🤖 Modelli AI scaricati
│   └── custom/                            # 🎯 Modelli personalizzati
├── venv/                                  # 🐍 Virtual environment (Linux/macOS)
└── tradingbot                             # 🚀 Script di avvio
```

## API Keys Configuration Visuale

```
┌─────────────────────────────────────────────────────────────────┐
│                         .env FILE STRUCTURE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ # =================== TRADING CONFIG ======================= │
│ TRADING_MODE=simulation          # 🛡️ Start SAFE              │
│ INITIAL_CAPITAL=10000           # 💰 Starting money           │
│ RISK_PERCENTAGE=2.0             # ⚠️ Risk per trade           │
│                                                                 │
│ # =================== EXCHANGE APIs ======================= │
│ BINANCE_API_KEY=your_key_here            # 🏦 Primary         │
│ BINANCE_SECRET_KEY=your_secret_here      # 🔐 Keep secure     │
│                                                                 │
│ COINBASE_API_KEY=your_key_here           # 🏦 Secondary       │
│ COINBASE_SECRET_KEY=your_secret_here     # 🔐 Keep secure     │
│ COINBASE_PASSPHRASE=your_phrase_here     # 🔑 Extra security  │
│                                                                 │
│ # =================== DATA SOURCES ======================== │
│ ALPHA_VANTAGE_API_KEY=your_key_here      # 📊 Market data     │
│ NEWSAPI_KEY=your_key_here                # 📰 News sentiment  │
│ HUGGINGFACE_API_TOKEN=your_token_here    # 🤖 AI models       │
│                                                                 │
│ # =================== NOTIFICATIONS ======================= │
│ EMAIL_USERNAME=your_email@gmail.com      # 📧 Alerts          │
│ EMAIL_PASSWORD=your_app_password_here    # 🔐 App password    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Remote Access Configurazioni

### SSH Tunnel Setup (Sicuro)
```
Local Computer                    Remote Server
┌─────────────────┐              ┌─────────────────┐
│                 │              │                 │
│  Browser        │              │  AI Trading Bot │
│  localhost:5000 │◄─────────────┤  localhost:5000 │
│                 │  SSH Tunnel  │                 │
└─────────────────┘              └─────────────────┘
                                          │
Command:                                  │
ssh -L 5000:localhost:5000 user@server   │
                                          ▼
                                 ┌─────────────────┐
                                 │  Encrypted      │
                                 │  Connection     │
                                 │  Port 22        │
                                 └─────────────────┘
```

### Direct Access (Server pubblico)
```
Local Computer                    Remote Server
┌─────────────────┐              ┌─────────────────┐
│                 │              │                 │
│  Browser        │              │  AI Trading Bot │
│  server-ip:8501 │◄─────────────┤  0.0.0.0:8501   │
│                 │  Direct HTTP │                 │
└─────────────────┘              └─────────────────┘
                                          │
Firewall Rule Required:                   │
sudo ufw allow 8501                      ▼
                                 ┌─────────────────┐
                                 │  Public Access  │
                                 │  Port 8501      │
                                 │  ⚠️ Less Secure │
                                 └─────────────────┘
```

### Ngrok Tunnel (Facile)
```
Local Computer                    Ngrok Cloud              Remote Server
┌─────────────────┐              ┌─────────────────┐      ┌─────────────────┐
│                 │              │                 │      │                 │
│  Browser        │              │  Tunnel Service │      │  AI Trading Bot │
│  xyz.ngrok.io   │◄─────────────┤  HTTPS Endpoint │◄─────┤  localhost:5000 │
│                 │  HTTPS       │                 │      │                 │
└─────────────────┘              └─────────────────┘      └─────────────────┘

Command on server: ngrok http 5000
Result: https://xyz.ngrok.io → localhost:5000
```

## System Requirements Visuale

```
┌─────────────────────────────────────────────────────────────────┐
│                     SYSTEM REQUIREMENTS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ MINIMUM REQUIREMENTS          │  RECOMMENDED REQUIREMENTS       │
│ ─────────────────────         │  ─────────────────────────      │
│                                │                                 │
│ 💻 CPU: 2 cores               │  💻 CPU: 4+ cores              │
│ 🧠 RAM: 2GB                   │  🧠 RAM: 8GB                   │
│ 💾 Storage: 1GB free          │  💾 Storage: 5GB SSD           │
│ 🌐 Network: Broadband         │  🌐 Network: High-speed         │
│ 🐍 Python: 3.8+               │  🐍 Python: 3.11               │
│                                │                                 │
│ PERFORMANCE EXPECTATIONS:      │  PERFORMANCE EXPECTATIONS:      │
│ • Basic trading: ✅            │  • Advanced AI: ✅              │
│ • Simple strategies: ✅        │  • Multiple exchanges: ✅       │
│ • 1-2 positions: ✅            │  • High-frequency: ✅          │
│ • Limited AI models: ⚠️        │  • Full AI suite: ✅            │
│                                │                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Troubleshooting Decision Tree

```
🚨 PROBLEMA: Bot non si avvia
              │
              ▼
        ┌─────────────┐
        │ Check Port  │
        │ 5000/8501   │
        └─────────────┘
              │
    ┌─────────┴─────────┐
    ▼                   ▼
┌─────────┐         ┌─────────┐
│ LIBERA  │         │ OCCUPATA│
│         │         │         │
└─────────┘         └─────────┘
    │                   │
    ▼                   ▼
┌─────────────┐     ┌─────────────┐
│ Check Python│     │ Kill Process│
│ Version     │     │ or Change   │
│             │     │ Port        │
└─────────────┘     └─────────────┘
    │                   │
    ▼                   ▼
┌─────────────┐     ┌─────────────┐
│ Check       │     │ Restart     │
│ Dependencies│     │ Bot         │
└─────────────┘     └─────────────┘
```

```
🚨 PROBLEMA: API connections falliscono
              │
              ▼
        ┌─────────────┐
        │ Check .env  │
        │ File        │
        └─────────────┘
              │
    ┌─────────┴─────────┐
    ▼                   ▼
┌─────────┐         ┌─────────┐
│ MISSING │         │ PRESENT │
│ KEYS    │         │ KEYS    │
└─────────┘         └─────────┘
    │                   │
    ▼                   ▼
┌─────────────┐     ┌─────────────┐
│ Configure   │     │ Test Network│
│ API Keys    │     │ Connection  │
└─────────────┘     └─────────────┘
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
        ┌─────────┐         ┌─────────┐
        │ WORKING │         │ FAILED  │
        └─────────┘         └─────────┘
              │                   │
              ▼                   ▼
        ┌─────────────┐     ┌─────────────┐
        │ Check API   │     │ Check       │
        │ Permissions │     │ Firewall/   │
        │ & Limits    │     │ Proxy       │
        └─────────────┘     └─────────────┘
```

## Installation Success Indicators

```
✅ INSTALLAZIONE COMPLETATA CON SUCCESSO

┌─────────────────────────────────────────────────────────────────┐
│                         VERIFICATION                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 1. ✅ COMMAND LINE                                              │
│    $ tradingbot                                                 │
│    → Bot starts without errors                                  │
│                                                                 │
│ 2. ✅ WEB INTERFACE                                             │
│    http://localhost:5000 (Windows/macOS)                       │
│    http://server-ip:8501 (Linux)                               │
│    → Dashboard loads successfully                               │
│                                                                 │
│ 3. ✅ DIAGNOSTICS                                               │
│    $ python check_install.py                                   │
│    → Shows 6+ components operational                            │
│                                                                 │
│ 4. ✅ SYSTEM STATUS                                             │
│    Dashboard → Setup & Control Tab                             │
│    → All modules show green status                              │
│                                                                 │
│ 5. ✅ FILE STRUCTURE                                            │
│    ~/ai-trading-bot/ directory created                         │
│    → All core files present                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

🎯 NEXT STEPS:
1. Configure API keys in .env file
2. Test with simulation mode
3. Monitor performance
4. Gradually enable live trading
```

## Support Channels Flow

```
📞 NEED HELP?

Start Here
    │
    ▼
┌─────────────────┐
│ 1. Self-Help    │
│ python          │
│ check_install.py│
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 2. Check Docs   │
│ • Quick Start   │
│ • Installation  │
│ • Troubleshoot  │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 3. Check Logs   │
│ ~/ai-trading-   │
│ bot/logs/       │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 4. Community    │
│ • GitHub Issues │
│ • Forums        │
│ • Discord       │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 5. Direct       │
│ Support         │
│ (Premium)       │
└─────────────────┘
```

---

**Sistema AI Trading Bot completamente documentato e pronto per deployment con guide visuali complete per tutti i sistemi operativi supportati.**