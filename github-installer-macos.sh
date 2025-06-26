#!/bin/bash
# AI Crypto Trading Bot - GitHub Direct Installer (macOS)
# Installa direttamente da GitHub Repository

set -e

# Configurazione
INSTALL_DIR="$HOME/tradingBot"
GITHUB_REPO="https://github.com/SudoRezi/tradingBot.git"
PYTHON_VERSION="3.11"

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Funzioni di utilità
print_header() {
    echo -e "${CYAN}=== AI Crypto Trading Bot - GitHub Installer ===${NC}"
    echo -e "${GREEN}Installing from GitHub to: $INSTALL_DIR${NC}"
    echo ""
}

print_step() {
    echo -e "${BLUE}[Step $1] $2${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Rilevamento architettura
detect_architecture() {
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
        echo "Apple Silicon (M1/M2/M3)"
        IS_ARM=true
        HOMEBREW_PREFIX="/opt/homebrew"
    else
        echo "Intel x64"
        IS_ARM=false
        HOMEBREW_PREFIX="/usr/local"
    fi
}

# Installazione Homebrew
install_homebrew() {
    if command -v brew >/dev/null 2>&1; then
        print_success "Homebrew già installato"
        return 0
    fi
    
    print_warning "Installazione Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Configura PATH per Homebrew
    if [[ "$IS_ARM" == true ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    else
        echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/usr/local/bin/brew shellenv)"
    fi
    
    print_success "Homebrew installato"
}

# Installazione Python
install_python() {
    if command -v python3 >/dev/null 2>&1; then
        CURRENT_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
        if [[ "$CURRENT_VERSION" == "3.11" ]] || [[ "$CURRENT_VERSION" == "3.10" ]] || [[ "$CURRENT_VERSION" == "3.9" ]]; then
            print_success "Python $CURRENT_VERSION già installato"
            return 0
        fi
    fi
    
    print_warning "Installazione Python $PYTHON_VERSION..."
    brew install python@$PYTHON_VERSION
    
    # Crea symlink se necessario
    if [[ ! -L "$HOMEBREW_PREFIX/bin/python3" ]]; then
        ln -sf "$HOMEBREW_PREFIX/bin/python$PYTHON_VERSION" "$HOMEBREW_PREFIX/bin/python3"
    fi
    
    print_success "Python $PYTHON_VERSION installato"
}

# Installazione Git
install_git() {
    if command -v git >/dev/null 2>&1; then
        print_success "Git già installato"
        return 0
    fi
    
    print_warning "Installazione Git..."
    brew install git
    print_success "Git installato"
}

# Clonazione repository
clone_repository() {
    print_warning "Clonazione repository da GitHub..."
    
    # Rimuovi directory esistente
    if [[ -d "$INSTALL_DIR" ]]; then
        print_warning "Rimozione installazione precedente..."
        rm -rf "$INSTALL_DIR"
    fi
    
    # Clona repository
    git clone "$GITHUB_REPO" "$INSTALL_DIR"
    
    if [[ ! -f "$INSTALL_DIR/advanced_ai_system.py" ]]; then
        print_error "File principali non trovati dopo clonazione"
        return 1
    fi
    
    print_success "Repository clonato con successo"
}

# Installazione dipendenze Python
install_python_packages() {
    print_warning "Installazione dipendenze Python..."
    
    cd "$INSTALL_DIR"
    
    # Crea virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Installa requirements se esiste
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
    else
        # Installa pacchetti essenziali
        pip install streamlit==1.28.1
        pip install pandas==2.0.3
        pip install numpy==1.24.3
        pip install plotly==5.15.0
        pip install requests==2.31.0
        pip install python-binance==1.0.19
        pip install yfinance==0.2.18
        pip install scikit-learn==1.3.0
        pip install apscheduler==3.10.4
        pip install cryptography==41.0.3
        pip install beautifulsoup4==4.12.2
        pip install feedparser==6.0.10
        pip install trafilatura==1.6.1
        pip install sendgrid==6.10.0
        pip install psutil==5.9.5
        pip install joblib==1.3.1
        pip install scipy==1.11.1
    fi
    
    print_success "Dipendenze Python installate"
}

# Creazione file di configurazione
create_configuration() {
    print_warning "Creazione file di configurazione..."
    
    cd "$INSTALL_DIR"
    
    # Crea file .env se non esiste
    if [[ ! -f ".env" ]]; then
        cat > .env << 'EOF'
# AI Crypto Trading Bot Configuration
# =====================================

# Trading Configuration
TRADING_MODE=simulation
INITIAL_CAPITAL=10000
RISK_PERCENTAGE=2.0
MAX_POSITIONS=5

# System Configuration
SYSTEM_OS=macos
SYSTEM_ARCH=arm64
LOG_LEVEL=INFO

# API Keys (da configurare per trading live)
# BINANCE_API_KEY=your_binance_api_key_here
# BINANCE_SECRET_KEY=your_binance_secret_key_here
# COINBASE_API_KEY=your_coinbase_api_key_here
# COINBASE_SECRET_KEY=your_coinbase_secret_key_here
# ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
# NEWSAPI_KEY=your_news_api_key_here

# Email Notifications (opzionale)
# EMAIL_SMTP_SERVER=smtp.gmail.com
# EMAIL_SMTP_PORT=587
# EMAIL_USERNAME=your_email@gmail.com
# EMAIL_PASSWORD=your_app_password_here

# Performance Settings
ENABLE_PERFORMANCE_MODE=true
CPU_OPTIMIZATION=true
MEMORY_OPTIMIZATION=true
EOF
    fi
    
    # Ottimizzazioni specifiche per architettura
    if [[ "$IS_ARM" == true ]]; then
        cat >> .env << 'EOF'

# Apple Silicon Optimizations
METAL_ACCELERATION=true
ARM_OPTIMIZATION=true
MEMORY_LIMIT=8192
EOF
    else
        cat >> .env << 'EOF'

# Intel x64 Optimizations
INTEL_OPTIMIZATION=true
MEMORY_LIMIT=4096
EOF
    fi
    
    print_success "File di configurazione creati"
}

# Creazione comando CLI
create_cli_command() {
    print_warning "Configurazione comando CLI..."
    
    # Crea script di avvio
    cat > "$INSTALL_DIR/tradingbot" << 'EOF'
#!/bin/bash
cd "$HOME/tradingBot"
source venv/bin/activate
python advanced_ai_system.py
EOF
    
    chmod +x "$INSTALL_DIR/tradingbot"
    
    # Aggiungi al PATH
    SHELL_RC=""
    if [[ "$SHELL" == *"zsh"* ]]; then
        SHELL_RC="$HOME/.zshrc"
    elif [[ "$SHELL" == *"bash"* ]]; then
        SHELL_RC="$HOME/.bashrc"
    fi
    
    if [[ -n "$SHELL_RC" ]]; then
        if ! grep -q "$INSTALL_DIR" "$SHELL_RC" 2>/dev/null; then
            echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> "$SHELL_RC"
        fi
    fi
    
    print_success "Comando CLI 'tradingbot' configurato"
}

# Creazione app macOS
create_macos_app() {
    print_warning "Creazione app macOS..."
    
    APP_DIR="/Applications/AI Trading Bot.app"
    
    # Crea struttura app
    sudo mkdir -p "$APP_DIR/Contents/MacOS"
    sudo mkdir -p "$APP_DIR/Contents/Resources"
    
    # Crea Info.plist
    sudo cat > "$APP_DIR/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>ai-trading-bot</string>
    <key>CFBundleIdentifier</key>
    <string>com.ai-trading-bot.app</string>
    <key>CFBundleName</key>
    <string>AI Trading Bot</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
</dict>
</plist>
EOF
    
    # Crea script eseguibile
    sudo cat > "$APP_DIR/Contents/MacOS/ai-trading-bot" << EOF
#!/bin/bash
cd "$INSTALL_DIR"
source venv/bin/activate
python advanced_ai_system.py
EOF
    
    sudo chmod +x "$APP_DIR/Contents/MacOS/ai-trading-bot"
    
    print_success "App macOS creata"
}

# Test installazione
test_installation() {
    print_warning "Test installazione..."
    
    cd "$INSTALL_DIR"
    source venv/bin/activate
    
    # Test import principali
    python3 << 'EOF'
import sys
sys.path.insert(0, '.')

try:
    import streamlit
    print('✓ Streamlit OK')
except ImportError as e:
    print(f'✗ Streamlit Error: {e}')

try:
    import pandas
    print('✓ Pandas OK')
except ImportError as e:
    print(f'✗ Pandas Error: {e}')

try:
    import numpy
    print('✓ NumPy OK')
except ImportError as e:
    print(f'✗ NumPy Error: {e}')

try:
    import requests
    print('✓ Requests OK')
except ImportError as e:
    print(f'✗ Requests Error: {e}')

print('Installation test completed.')
EOF
    
    print_success "Test installazione completato"
}

# === MAIN INSTALLATION PROCESS ===

main() {
    print_header
    
    echo "Rilevamento architettura..."
    ARCH_INFO=$(detect_architecture)
    echo "Architettura rilevata: $ARCH_INFO"
    echo ""
    
    # Step 1: Install Homebrew
    print_step "1/8" "Installazione Homebrew..."
    install_homebrew
    echo ""
    
    # Step 2: Install Python
    print_step "2/8" "Installazione Python..."
    install_python
    echo ""
    
    # Step 3: Install Git
    print_step "3/8" "Installazione Git..."
    install_git
    echo ""
    
    # Step 4: Clone Repository
    print_step "4/8" "Clonazione repository..."
    clone_repository
    echo ""
    
    # Step 5: Install Python packages
    print_step "5/8" "Installazione dipendenze Python..."
    install_python_packages
    echo ""
    
    # Step 6: Create configuration
    print_step "6/8" "Creazione configurazione..."
    create_configuration
    echo ""
    
    # Step 7: Create shortcuts
    print_step "7/8" "Creazione shortcuts..."
    create_cli_command
    create_macos_app
    echo ""
    
    # Step 8: Test installation
    print_step "8/8" "Test installazione..."
    test_installation
    echo ""
    
    # Success message
    echo -e "${GREEN}🎉 INSTALLAZIONE COMPLETATA CON SUCCESSO!${NC}"
    echo "================================================"
    echo -e "${CYAN}Directory installazione: $INSTALL_DIR${NC}"
    echo -e "${CYAN}Comando avvio: tradingbot${NC}"
    echo -e "${CYAN}App macOS: /Applications/AI Trading Bot.app${NC}"
    echo -e "${CYAN}Interfaccia web: http://localhost:5000${NC}"
    echo ""
    echo -e "${YELLOW}📋 PROSSIMI PASSI:${NC}"
    echo "1. Configura le API keys nel file .env"
    echo "2. Avvia: tradingbot o apri l'app AI Trading Bot"
    echo "3. Accedi all'interfaccia web su http://localhost:5000"
    echo "4. Testa in modalità simulazione prima del trading live"
    echo ""
    echo -e "${YELLOW}Per aprire la directory di installazione:${NC}"
    echo "open $INSTALL_DIR"
}

# Esegui installazione
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi