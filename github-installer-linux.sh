#!/bin/bash
# AI Crypto Trading Bot - GitHub Direct Installer (Linux)
# Installa direttamente da GitHub Repository

set -e

# Configurazione
INSTALL_DIR="$HOME/tradingBot"
GITHUB_REPO="https://github.com/SudoRezi/tradingBot.git"
SERVICE_NAME="ai-trading-bot"
WEB_PORT="8501"

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

# Rilevamento distribuzione
detect_distro() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        DISTRO=$ID
        VERSION=$VERSION_ID
    elif [[ -f /etc/redhat-release ]]; then
        DISTRO="centos"
    elif [[ -f /etc/debian_version ]]; then
        DISTRO="debian"
    else
        DISTRO="unknown"
    fi
    
    echo "Distribuzione rilevata: $DISTRO $VERSION"
}

# Aggiornamento pacchetti sistema
update_system() {
    print_warning "Aggiornamento pacchetti sistema..."
    
    case $DISTRO in
        "ubuntu"|"debian")
            sudo apt update
            sudo apt upgrade -y
            ;;
        "centos"|"rhel"|"fedora")
            if command -v dnf >/dev/null 2>&1; then
                sudo dnf update -y
            else
                sudo yum update -y
            fi
            ;;
        *)
            print_warning "Distribuzione non completamente supportata, continuando..."
            ;;
    esac
    
    print_success "Sistema aggiornato"
}

# Installazione dipendenze sistema
install_system_dependencies() {
    print_warning "Installazione dipendenze sistema..."
    
    case $DISTRO in
        "ubuntu"|"debian")
            sudo apt install -y \
                python3 \
                python3-pip \
                python3-venv \
                python3-dev \
                git \
                build-essential \
                libssl-dev \
                libffi-dev \
                libbz2-dev \
                libreadline-dev \
                libsqlite3-dev \
                wget \
                curl \
                llvm \
                libncurses5-dev \
                libncursesw5-dev \
                xz-utils \
                tk-dev \
                libxml2-dev \
                libxmlsec1-dev \
                libffi-dev \
                liblzma-dev
            ;;
        "centos"|"rhel")
            if command -v dnf >/dev/null 2>&1; then
                sudo dnf groupinstall -y "Development Tools"
                sudo dnf install -y \
                    python3 \
                    python3-pip \
                    python3-devel \
                    git \
                    openssl-devel \
                    libffi-devel \
                    bzip2-devel \
                    readline-devel \
                    sqlite-devel \
                    wget \
                    curl \
                    llvm \
                    ncurses-devel \
                    xz \
                    tk-devel \
                    libxml2-devel \
                    xmlsec1-devel \
                    xz-devel
            else
                sudo yum groupinstall -y "Development Tools"
                sudo yum install -y \
                    python3 \
                    python3-pip \
                    python3-devel \
                    git \
                    openssl-devel \
                    libffi-devel \
                    bzip2-devel \
                    readline-devel \
                    sqlite-devel \
                    wget \
                    curl \
                    llvm \
                    ncurses-devel \
                    xz \
                    tk-devel \
                    libxml2-devel \
                    xmlsec1-devel \
                    xz-devel
            fi
            ;;
        "fedora")
            sudo dnf groupinstall -y "Development Tools"
            sudo dnf install -y \
                python3 \
                python3-pip \
                python3-devel \
                git \
                openssl-devel \
                libffi-devel \
                bzip2-devel \
                readline-devel \
                sqlite-devel \
                wget \
                curl \
                llvm \
                ncurses-devel \
                xz \
                tk-devel \
                libxml2-devel \
                xmlsec1-devel \
                xz-devel
            ;;
    esac
    
    print_success "Dipendenze sistema installate"
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
SYSTEM_OS=linux
SYSTEM_ARCH=x64
LOG_LEVEL=INFO

# Server Configuration (for remote access)
STREAMLIT_HOST=0.0.0.0
STREAMLIT_PORT=8501

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
    
    print_success "File di configurazione creati"
}

# Configurazione firewall
configure_firewall() {
    print_warning "Configurazione firewall..."
    
    # UFW (Ubuntu/Debian)
    if command -v ufw >/dev/null 2>&1; then
        sudo ufw allow $WEB_PORT/tcp
        print_success "UFW configurato per porta $WEB_PORT"
    # Firewalld (CentOS/RHEL/Fedora)
    elif command -v firewall-cmd >/dev/null 2>&1; then
        sudo firewall-cmd --permanent --add-port=$WEB_PORT/tcp
        sudo firewall-cmd --reload
        print_success "Firewalld configurato per porta $WEB_PORT"
    # iptables fallback
    elif command -v iptables >/dev/null 2>&1; then
        sudo iptables -A INPUT -p tcp --dport $WEB_PORT -j ACCEPT
        # Salva regole iptables se possibile
        if command -v iptables-save >/dev/null 2>&1; then
            sudo iptables-save > /etc/iptables/rules.v4 2>/dev/null || true
        fi
        print_success "iptables configurato per porta $WEB_PORT"
    else
        print_warning "Firewall non rilevato, configurazione manuale potrebbe essere necessaria"
    fi
}

# Creazione servizio systemd
create_systemd_service() {
    print_warning "Creazione servizio systemd..."
    
    sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null << EOF
[Unit]
Description=AI Crypto Trading Bot
After=network.target
StartLimitBurst=5
StartLimitIntervalSec=10

[Service]
Type=simple
Restart=always
RestartSec=1
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment=PATH=$INSTALL_DIR/venv/bin
ExecStart=$INSTALL_DIR/venv/bin/python advanced_ai_system.py
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    # Ricarica systemd e abilita servizio
    sudo systemctl daemon-reload
    sudo systemctl enable $SERVICE_NAME
    
    print_success "Servizio systemd creato e abilitato"
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
    
    # Crea symlink globale se possibile
    if [[ -w "/usr/local/bin" ]]; then
        sudo ln -sf "$INSTALL_DIR/tradingbot" /usr/local/bin/tradingbot
    fi
    
    # Aggiungi al PATH dell'utente
    SHELL_RC=""
    if [[ "$SHELL" == *"bash"* ]]; then
        SHELL_RC="$HOME/.bashrc"
    elif [[ "$SHELL" == *"zsh"* ]]; then
        SHELL_RC="$HOME/.zshrc"
    fi
    
    if [[ -n "$SHELL_RC" ]] && [[ -f "$SHELL_RC" ]]; then
        if ! grep -q "$INSTALL_DIR" "$SHELL_RC" 2>/dev/null; then
            echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> "$SHELL_RC"
        fi
    fi
    
    print_success "Comando CLI 'tradingbot' configurato"
}

# Creazione script di check salute
create_health_check() {
    print_warning "Creazione script di controllo..."
    
    cat > "$INSTALL_DIR/healthcheck.sh" << 'EOF'
#!/bin/bash
# Health check script per AI Trading Bot

echo "=== AI Trading Bot Health Check ==="
echo "Timestamp: $(date)"
echo ""

# Check servizio systemd
if systemctl is-active ai-trading-bot >/dev/null 2>&1; then
    echo "✓ Servizio: ATTIVO"
else
    echo "✗ Servizio: INATTIVO"
fi

# Check porta web
if netstat -tlnp 2>/dev/null | grep -q ":8501 "; then
    echo "✓ Porta web 8501: APERTA"
else
    echo "✗ Porta web 8501: CHIUSA"
fi

# Check processo Python
if pgrep -f "advanced_ai_system.py" >/dev/null; then
    echo "✓ Processo Python: ATTIVO"
else
    echo "✗ Processo Python: INATTIVO"
fi

# Check connettività web
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8501 | grep -q "200"; then
    echo "✓ Interfaccia web: RAGGIUNGIBILE"
else
    echo "✗ Interfaccia web: NON RAGGIUNGIBILE"
fi

# Check spazio disco
DISK_USAGE=$(df "$HOME" | awk 'NR==2 {print $5}' | sed 's/%//')
if [[ $DISK_USAGE -lt 90 ]]; then
    echo "✓ Spazio disco: OK ($DISK_USAGE% utilizzato)"
else
    echo "⚠ Spazio disco: ATTENZIONE ($DISK_USAGE% utilizzato)"
fi

# Check memoria
MEM_USAGE=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
if [[ $MEM_USAGE -lt 90 ]]; then
    echo "✓ Memoria: OK ($MEM_USAGE% utilizzata)"
else
    echo "⚠ Memoria: ATTENZIONE ($MEM_USAGE% utilizzata)"
fi

echo ""
echo "Health check completato."
EOF
    
    chmod +x "$INSTALL_DIR/healthcheck.sh"
    
    print_success "Script di controllo creato"
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

# Avvio servizio
start_service() {
    print_warning "Avvio servizio AI Trading Bot..."
    
    sudo systemctl start $SERVICE_NAME
    sleep 3
    
    if systemctl is-active $SERVICE_NAME >/dev/null 2>&1; then
        print_success "Servizio avviato con successo"
    else
        print_error "Errore avvio servizio"
        sudo systemctl status $SERVICE_NAME
        return 1
    fi
}

# === MAIN INSTALLATION PROCESS ===

main() {
    print_header
    
    # Verifica privilegi sudo
    if ! sudo -n true 2>/dev/null; then
        echo "Questo script richiede privilegi sudo per installare dipendenze sistema."
        echo "Inserisci la password quando richiesto."
        sudo true
    fi
    
    echo "Rilevamento distribuzione..."
    detect_distro
    echo ""
    
    # Step 1: Update system
    print_step "1/10" "Aggiornamento sistema..."
    update_system
    echo ""
    
    # Step 2: Install system dependencies
    print_step "2/10" "Installazione dipendenze sistema..."
    install_system_dependencies
    echo ""
    
    # Step 3: Clone Repository
    print_step "3/10" "Clonazione repository..."
    clone_repository
    echo ""
    
    # Step 4: Install Python packages
    print_step "4/10" "Installazione dipendenze Python..."
    install_python_packages
    echo ""
    
    # Step 5: Create configuration
    print_step "5/10" "Creazione configurazione..."
    create_configuration
    echo ""
    
    # Step 6: Configure firewall
    print_step "6/10" "Configurazione firewall..."
    configure_firewall
    echo ""
    
    # Step 7: Create systemd service
    print_step "7/10" "Creazione servizio systemd..."
    create_systemd_service
    echo ""
    
    # Step 8: Create CLI command
    print_step "8/10" "Configurazione comando CLI..."
    create_cli_command
    echo ""
    
    # Step 9: Create health check
    print_step "9/10" "Creazione script di controllo..."
    create_health_check
    echo ""
    
    # Step 10: Test and start
    print_step "10/10" "Test e avvio servizio..."
    test_installation
    start_service
    echo ""
    
    # Success message
    echo -e "${GREEN}🎉 INSTALLAZIONE COMPLETATA CON SUCCESSO!${NC}"
    echo "================================================"
    echo -e "${CYAN}Directory installazione: $INSTALL_DIR${NC}"
    echo -e "${CYAN}Servizio systemd: $SERVICE_NAME${NC}"
    echo -e "${CYAN}Comando CLI: tradingbot${NC}"
    echo -e "${CYAN}Interfaccia web: http://$(hostname -I | awk '{print $1}'):$WEB_PORT${NC}"
    echo -e "${CYAN}Accesso locale: http://localhost:$WEB_PORT${NC}"
    echo ""
    echo -e "${YELLOW}📋 GESTIONE SERVIZIO:${NC}"
    echo "sudo systemctl start|stop|restart $SERVICE_NAME"
    echo "sudo systemctl status $SERVICE_NAME"
    echo "sudo journalctl -u $SERVICE_NAME -f"
    echo "./healthcheck.sh"
    echo ""
    echo -e "${YELLOW}📋 PROSSIMI PASSI:${NC}"
    echo "1. Configura le API keys nel file $INSTALL_DIR/.env"
    echo "2. Il servizio è già attivo e disponibile 24/7"
    echo "3. Accedi all'interfaccia web per configurare il trading"
    echo "4. Testa in modalità simulazione prima del trading live"
    echo ""
    echo -e "${YELLOW}📋 ACCESSO REMOTO:${NC}"
    echo "SSH Tunnel: ssh -L 8501:localhost:8501 $(whoami)@$(hostname -I | awk '{print $1}')"
    echo "Diretto: http://$(hostname -I | awk '{print $1}'):$WEB_PORT"
}

# Esegui installazione
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi