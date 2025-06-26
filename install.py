#!/usr/bin/env python3
"""
AI Trading Bot - Cross-Platform Installer
Supports Windows, macOS, and Linux
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

# Required packages for the AI Trading Bot
REQUIRED_PACKAGES = [
    "streamlit>=1.28.0",
    "pandas>=2.0.0", 
    "numpy>=1.24.0",
    "plotly>=5.15.0",
    "scikit-learn>=1.3.0",
    "requests>=2.31.0",
    "cryptography>=41.0.0",
    "apscheduler>=3.10.0",
    "psutil>=5.9.0",
    "yfinance>=0.2.18",
    "beautifulsoup4>=4.12.0",
    "trafilatura>=1.6.0",
    "feedparser>=6.0.0",
    "joblib>=1.3.0",
    "scipy>=1.11.0"
]

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_packages():
    """Install required packages"""
    print("\n🔧 Installing required packages...")
    
    for package in REQUIRED_PACKAGES:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    print("✅ All packages installed successfully!")
    return True

def create_launcher_scripts():
    """Create platform-specific launcher scripts"""
    system = platform.system().lower()
    
    if system == "windows":
        # Windows batch file
        batch_content = '''@echo off
echo Starting AI Trading Bot...
python simple_app.py
if %errorlevel% neq 0 (
    echo.
    echo Error starting the application. Make sure Python is installed.
    pause
)
'''
        with open("start_trading_bot.bat", "w") as f:
            f.write(batch_content)
        print("✅ Created start_trading_bot.bat for Windows")
        
    elif system == "darwin":
        # macOS shell script
        script_content = '''#!/bin/bash
echo "Starting AI Trading Bot..."
cd "$(dirname "$0")"
python3 simple_app.py
if [ $? -ne 0 ]; then
    echo ""
    echo "Error starting the application. Make sure Python 3 is installed."
    read -p "Press Enter to continue..."
fi
'''
        with open("start_trading_bot.sh", "w") as f:
            f.write(script_content)
        os.chmod("start_trading_bot.sh", 0o755)
        print("✅ Created start_trading_bot.sh for macOS")
        
    else:
        # Linux shell script
        script_content = '''#!/bin/bash
echo "Starting AI Trading Bot..."
cd "$(dirname "$0")"
python3 simple_app.py
if [ $? -ne 0 ]; then
    echo ""
    echo "Error starting the application. Make sure Python 3 is installed."
    read -p "Press Enter to continue..."
fi
'''
        with open("start_trading_bot.sh", "w") as f:
            f.write(script_content)
        os.chmod("start_trading_bot.sh", 0o755)
        print("✅ Created start_trading_bot.sh for Linux")

def create_streamlit_runner():
    """Create a simple runner that launches with Streamlit"""
    runner_content = '''#!/usr/bin/env python3
"""
AI Trading Bot - Streamlit Runner
Launches the trading bot with proper Streamlit configuration
"""

import subprocess
import sys
import os

def main():
    """Main entry point"""
    try:
        # Change to script directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Launch Streamlit app
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "simple_app.py", 
            "--server.port", "5000",
            "--server.address", "0.0.0.0",
            "--server.headless", "true"
        ]
        
        print("🚀 Starting AI Trading Bot...")
        print("📱 Open your browser to: http://localhost:5000")
        print("⏹️  Press Ctrl+C to stop")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\\n🛑 Trading Bot stopped by user")
    except Exception as e:
        print(f"❌ Error starting Trading Bot: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
'''
    
    with open("run_trading_bot.py", "w") as f:
        f.write(runner_content)
    print("✅ Created run_trading_bot.py")

def create_readme():
    """Create installation and usage README"""
    readme_content = '''# AI Trading Bot - Installation & Usage

## 🚀 Quick Start

### Windows
1. Run `install.py` to install dependencies
2. Double-click `start_trading_bot.bat` to launch
3. Open browser to http://localhost:5000

### macOS / Linux  
1. Run `python3 install.py` to install dependencies
2. Run `./start_trading_bot.sh` to launch
3. Open browser to http://localhost:5000

## 📋 System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended  
- **Disk Space**: 2GB free space
- **Internet**: Stable connection for real-time trading

## 🏦 Supported Exchanges

- **Binance** - Global cryptocurrency exchange
- **Bybit** - Derivatives and spot trading
- **KuCoin** - Wide selection of altcoins
- **Coinbase** - US-based regulated exchange  
- **Kraken** - European regulated exchange
- **OKX** - Advanced trading features

## 💰 Trading Pairs Supported

### Major Pairs
- BTC/USDT, BTC/USDC, BTC/ETH
- ETH/USDT, ETH/USDC  
- KAS/USDT (Kaspa)

### Altcoins
- SOL/USDT, DOGE/USDT, AVAX/USDT
- MATIC/USDT, LINK/USDT, UNI/USDT
- AAVE/USDT

## ⚙️ Configuration

1. **API Keys**: Enter your exchange API keys in the setup page
2. **AI Mode**: Select "AI Gestisce Tutto Automaticamente" for full autonomy
3. **Trading Pairs**: Choose which pairs to trade
4. **Risk Level**: Set conservative/moderate/aggressive
5. **Capital**: System auto-detects your available funds

## 🤖 AI Features

- **Machine Learning**: LSTM, Transformer, DQN ensemble models
- **Social Intelligence**: Real-time sentiment analysis
- **Multi-Exchange Arbitrage**: Cross-exchange profit opportunities  
- **Dynamic Leverage**: ATR-based position sizing (1-10x)
- **Risk Management**: Automatic stop losses and portfolio balancing

## 📊 Dashboard Tabs

1. **Portfolio Overview** - Total value, P&L, active positions
2. **Social Intelligence** - News and social sentiment analysis
3. **Multi-Exchange Arbitrage** - Cross-exchange opportunities
4. **Dynamic Leverage** - Leverage trading dashboard
5. **AI/ML Performance** - Model accuracy and predictions
6. **System Monitor** - CPU, RAM, disk usage, alerts
7. **Trading Strategy** - Configure trading modes and allocations
8. **Performance Charts** - Real-time portfolio charts
9. **Settings** - System configuration and controls

## 🔒 Security Features

- **AES-256 Encryption** - All API keys encrypted at rest
- **No Cloud Dependencies** - Runs entirely on your machine
- **Audit Trail** - Complete logging of all activities
- **Emergency Stops** - Automatic system shutdown on anomalies

## 📞 Troubleshooting

### Common Issues

**"Module not found" errors**
- Run the installer again: `python install.py`

**"Port 5000 already in use"**
- Close other applications using port 5000
- Or edit `run_trading_bot.py` to use a different port

**Trading not starting**
- Check API key permissions (trading enabled)
- Verify exchange connectivity
- Check minimum balance requirements

### Support Files

- **Logs**: Check `logs/` directory for detailed activity logs
- **Config**: `institutional_config.json` contains your settings
- **Monitoring**: Real-time system stats in System Monitor tab

## ⚖️ Legal Disclaimer

This software is for educational and research purposes. Trading cryptocurrencies involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk and never invest more than you can afford to lose.

## 🔄 Updates

The system includes automatic model retraining and strategy optimization. Manual updates will be released periodically with new features and improvements.
'''
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    print("✅ Created README.md with installation instructions")

def main():
    """Main installation process"""
    print("🤖 AI Trading Bot - Cross-Platform Installer")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install packages
    if not install_packages():
        print("❌ Installation failed. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Create platform-specific launchers
    create_launcher_scripts()
    
    # Create Streamlit runner
    create_streamlit_runner()
    
    # Create documentation
    create_readme()
    
    print("\n" + "=" * 50)
    print("✅ Installation completed successfully!")
    print("\n📋 Next Steps:")
    
    system = platform.system().lower()
    if system == "windows":
        print("1. Double-click 'start_trading_bot.bat' to launch")
    else:
        print("1. Run './start_trading_bot.sh' to launch")
    
    print("2. Open browser to http://localhost:5000")
    print("3. Configure your exchange API keys")
    print("4. Select 'AI Gestisce Tutto Automaticamente'")
    print("5. Click 'START SYSTEM' to begin trading")
    print("\n🔒 Your API keys will be encrypted and stored locally")
    print("🤖 The AI will manage your portfolio 24/7")

if __name__ == "__main__":
    main()