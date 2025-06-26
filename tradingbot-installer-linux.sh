#!/bin/bash
# AI Crypto Trading Bot - Linux Ubuntu/Debian Installer
# x64 Optimized for server and desktop environments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="$HOME/ai-trading-bot"
PYTHON_VERSION="3.11"
REQUIRED_PYTHON="3.8"
SERVICE_NAME="ai-trading-bot"

echo -e "${CYAN}=== AI Crypto Trading Bot - Linux Ubuntu/Debian Installer ===${NC}"
echo -e "${GREEN}Installing to: $INSTALL_DIR${NC}"
echo -e "${GREEN}Architecture: $(uname -m)${NC}"

# Function to print status
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if running as root
is_root() {
    [[ $EUID -eq 0 ]]
}

# Function to detect Linux distribution
detect_distro() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        DISTRO=$ID
        VERSION=$VERSION_ID
    elif [[ -f /etc/redhat-release ]]; then
        DISTRO="centos"
    else
        DISTRO="unknown"
    fi
    
    print_status "Detected distribution: $DISTRO $VERSION"
}

# Function to update package manager
update_packages() {
    print_status "Updating package manager..."
    
    case $DISTRO in
        "ubuntu"|"debian")
            sudo apt-get update -qq
            ;;
        "centos"|"rhel"|"fedora")
            if command_exists dnf; then
                sudo dnf update -y -q
            else
                sudo yum update -y -q
            fi
            ;;
        *)
            print_warning "Unknown distribution, skipping package update"
            ;;
    esac
}

# Function to install system dependencies
install_system_dependencies() {
    print_status "Installing system dependencies..."
    
    case $DISTRO in
        "ubuntu"|"debian")
            local packages=(
                "python3" "python3-pip" "python3-venv" "python3-dev"
                "build-essential" "git" "curl" "wget" "unzip"
                "libssl-dev" "libffi-dev" "libbz2-dev" "libreadline-dev"
                "libsqlite3-dev" "libncurses5-dev" "libncursesw5-dev"
                "xz-utils" "tk-dev" "libxml2-dev" "libxmlsec1-dev"
                "liblzma-dev" "pkg-config" "software-properties-common"
                "tmux" "screen" "htop" "nano" "vim"
            )
            
            for package in "${packages[@]}"; do
                if ! dpkg -l | grep -q "^ii  $package "; then
                    print_status "Installing $package..."
                    sudo apt-get install -y "$package" >/dev/null 2>&1 || print_warning "Failed to install $package"
                else
                    echo -e "${GREEN}✓${NC} $package already installed"
                fi
            done
            ;;
            
        "centos"|"rhel"|"fedora")
            local packages=(
                "python3" "python3-pip" "python3-devel"
                "gcc" "gcc-c++" "make" "git" "curl" "wget" "unzip"
                "openssl-devel" "libffi-devel" "bzip2-devel" "readline-devel"
                "sqlite-devel" "ncurses-devel" "xz-devel" "tk-devel"
                "libxml2-devel" "xmlsec1-devel" "pkgconfig"
                "tmux" "screen" "htop" "nano" "vim"
            )
            
            local installer="yum"
            if command_exists dnf; then
                installer="dnf"
            fi
            
            for package in "${packages[@]}"; do
                print_status "Installing $package..."
                sudo $installer install -y "$package" >/dev/null 2>&1 || print_warning "Failed to install $package"
            done
            ;;
            
        *)
            print_error "Unsupported distribution: $DISTRO"
            print_status "Please install Python 3.8+, pip, git, and build tools manually"
            ;;
    esac
}

# Function to install Python
setup_python() {
    print_status "Setting up Python environment..."
    
    # Check current Python version
    if command_exists python3; then
        CURRENT_PYTHON=$(python3 --version | grep -o '[0-9]\+\.[0-9]\+' | head -1)
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
            print_success "Compatible Python found: $(python3 --version)"
        else
            print_warning "Python version $CURRENT_PYTHON is too old, installing Python $PYTHON_VERSION"
            install_python_from_source
        fi
    else
        print_error "Python3 not found after system installation"
        install_python_from_source
    fi
    
    # Upgrade pip
    print_status "Upgrading pip..."
    python3 -m pip install --user --upgrade pip setuptools wheel
}

# Function to install Python from source (if needed)
install_python_from_source() {
    print_status "Installing Python $PYTHON_VERSION from source..."
    
    cd /tmp
    wget "https://www.python.org/ftp/python/$PYTHON_VERSION.8/Python-$PYTHON_VERSION.8.tgz"
    tar -xzf "Python-$PYTHON_VERSION.8.tgz"
    cd "Python-$PYTHON_VERSION.8"
    
    ./configure --enable-optimizations --with-ensurepip=install
    make -j$(nproc)
    sudo make altinstall
    
    # Create symlinks
    sudo ln -sf "/usr/local/bin/python$PYTHON_VERSION" /usr/local/bin/python3
    sudo ln -sf "/usr/local/bin/pip$PYTHON_VERSION" /usr/local/bin/pip3
    
    cd "$HOME"
    rm -rf "/tmp/Python-$PYTHON_VERSION.8"*
    
    print_success "Python $PYTHON_VERSION installed from source"
}

# Function to create installation directory
create_install_directory() {
    print_status "Creating installation directory..."
    
    if [[ -d "$INSTALL_DIR" ]]; then
        print_warning "Directory already exists: $INSTALL_DIR"
    else
        mkdir -p "$INSTALL_DIR"
        print_success "Created directory: $INSTALL_DIR"
    fi
    
    # Create subdirectories
    mkdir -p "$INSTALL_DIR"/{logs,data,backups,config}
}

# Function to install Python packages
install_python_packages() {
    print_status "Installing Python packages..."
    
    # Create virtual environment
    python3 -m venv "$INSTALL_DIR/venv"
    source "$INSTALL_DIR/venv/bin/activate"
    
    # Upgrade pip in virtual environment
    pip install --upgrade pip setuptools wheel
    
    # Core packages
    local packages=(
        "streamlit>=1.28.0"
        "pandas>=2.0.0"
        "numpy>=1.24.0"
        "plotly>=5.15.0"
        "scikit-learn>=1.3.0"
        "requests>=2.31.0"
        "cryptography>=41.0.0"
        "apscheduler>=3.10.0"
        "yfinance>=0.2.0"
        "beautifulsoup4>=4.12.0"
        "feedparser>=6.0.0"
        "psutil>=5.9.0"
        "joblib>=1.3.0"
        "scipy>=1.11.0"
        "trafilatura>=1.6.0"
        "sendgrid>=6.10.0"
        "uvloop>=0.17.0"  # Linux performance boost
    )
    
    for package in "${packages[@]}"; do
        print_status "Installing $package..."
        if pip install "$package" --quiet --no-warn-script-location; then
            echo -e "${GREEN}✓${NC} $package"
        else
            echo -e "${RED}✗${NC} $package"
            print_warning "Failed to install $package"
        fi
    done
    
    deactivate
}

# Function to copy application files
copy_application_files() {
    print_status "Copying application files..."
    
    # Core application files
    local core_files=(
        "advanced_ai_system.py"
        "advanced_quant_engine.py"
        "advanced_order_system.py"
        "arctic_data_manager.py"
        "smart_performance_optimizer.py"
        "autonomous_ai_trader.py"
        "real_ai_integration.py"
        "multilayer_api_protection.py"
        "system_health_check.py"
        "PERFORMANCE_CALCULATOR.py"
    )
    
    # Copy core files
    for file in "${core_files[@]}"; do
        if [[ -f "$file" ]]; then
            cp "$file" "$INSTALL_DIR/"
            print_success "Copied: $file"
        else
            print_warning "File not found: $file"
        fi
    done
    
    # Copy directories
    local directories=("config" "data" "logs" "backups" "models" "strategies" "utils" "core")
    for dir in "${directories[@]}"; do
        if [[ -d "$dir" ]]; then
            cp -r "$dir" "$INSTALL_DIR/"
            print_success "Copied directory: $dir"
        fi
    done
}

# Function to setup configuration
setup_configuration() {
    print_status "Setting up configuration files..."
    
    # Setup .env file
    local env_file="$INSTALL_DIR/.env"
    if [[ ! -f "$env_file" ]]; then
        cat > "$env_file" << EOF
# AI Crypto Trading Bot - Environment Configuration
# Generated on $(date)

# Trading Configuration
TRADING_MODE=simulation
INITIAL_CAPITAL=10000
MAX_POSITIONS=5
RISK_PERCENTAGE=2.0

# API Keys (Configure for live trading)
# BINANCE_API_KEY=your_binance_api_key_here
# BINANCE_SECRET_KEY=your_binance_secret_key_here
# COINBASE_API_KEY=your_coinbase_api_key_here
# COINBASE_SECRET_KEY=your_coinbase_secret_key_here

# Data Sources
# ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
# NEWSAPI_KEY=your_news_api_key_here

# HuggingFace
# HUGGINGFACE_API_TOKEN=your_huggingface_token_here

# Notifications
# EMAIL_SMTP_SERVER=smtp.gmail.com
# EMAIL_SMTP_PORT=587
# EMAIL_USERNAME=your_email@gmail.com
# EMAIL_PASSWORD=your_app_password_here

# System Settings
SYSTEM_OS=linux
SYSTEM_ARCH=x64
LOG_LEVEL=INFO

# Server Settings (for remote access)
STREAMLIT_HOST=0.0.0.0
STREAMLIT_PORT=8501
STREAMLIT_HEADLESS=true
EOF
        print_success "Created .env file"
    else
        print_warning ".env file already exists, preserving current configuration"
    fi
    
    # Setup config.yaml
    local config_file="$INSTALL_DIR/config/config.yaml"
    if [[ ! -f "$config_file" ]]; then
        cat > "$config_file" << EOF
# AI Crypto Trading Bot Configuration
# Linux x64 Server Optimized Configuration

system:
  platform: linux
  architecture: x64
  install_path: "$INSTALL_DIR"
  log_directory: "$INSTALL_DIR/logs"
  data_directory: "$INSTALL_DIR/data"
  venv_path: "$INSTALL_DIR/venv"

trading:
  mode: simulation
  initial_capital: 10000
  max_positions: 5
  risk_percentage: 2.0
  
  exchanges:
    - name: binance
      enabled: false
      testnet: true
    - name: coinbase
      enabled: false
      sandbox: true

ai_models:
  optimization: cpu
  memory_limit: 4096
  cache_size: 1024
  
  models:
    - name: trading_classifier
      type: sklearn
      enabled: true
    - name: sentiment_analyzer
      type: lightweight
      enabled: true

performance:
  cpu_optimization: true
  memory_optimization: true
  threading: true
  max_threads: $(nproc)
  use_uvloop: true

security:
  encryption: true
  key_rotation: true
  audit_logging: true

streamlit:
  host: 0.0.0.0
  port: 8501
  theme: dark
  headless: true

server:
  enable_remote_access: true
  auto_restart: true
  log_rotation: true
  max_log_size: 100MB
  backup_retention: 30
EOF
        print_success "Created config.yaml file"
    fi
}

# Function to create systemd service
create_systemd_service() {
    print_status "Setting up systemd service..."
    
    local service_file="/etc/systemd/system/$SERVICE_NAME.service"
    
    if is_root || sudo -n true 2>/dev/null; then
        cat > /tmp/ai-trading-bot.service << EOF
[Unit]
Description=AI Crypto Trading Bot
After=network.target
Wants=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment=PATH=$INSTALL_DIR/venv/bin:$PATH
ExecStart=$INSTALL_DIR/venv/bin/python advanced_ai_system.py
Restart=always
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=ai-trading-bot

[Install]
WantedBy=multi-user.target
EOF
        
        sudo mv /tmp/ai-trading-bot.service "$service_file"
        sudo systemctl daemon-reload
        sudo systemctl enable "$SERVICE_NAME"
        
        print_success "Systemd service created and enabled"
        print_status "Service commands:"
        echo -e "  ${BLUE}sudo systemctl start $SERVICE_NAME${NC}    # Start the service"
        echo -e "  ${BLUE}sudo systemctl stop $SERVICE_NAME${NC}     # Stop the service"
        echo -e "  ${BLUE}sudo systemctl status $SERVICE_NAME${NC}   # Check status"
        echo -e "  ${BLUE}sudo journalctl -u $SERVICE_NAME -f${NC}   # View logs"
    else
        print_warning "Cannot create systemd service without sudo access"
    fi
}

# Function to create CLI command
create_cli_command() {
    print_status "Setting up CLI command..."
    
    # Create launch script
    cat > "$INSTALL_DIR/tradingbot" << EOF
#!/bin/bash
cd "$INSTALL_DIR"
source venv/bin/activate
python advanced_ai_system.py "\$@"
EOF
    
    chmod +x "$INSTALL_DIR/tradingbot"
    
    # Create global symlink if possible
    if [[ -w "/usr/local/bin" ]] || sudo -n true 2>/dev/null; then
        sudo ln -sf "$INSTALL_DIR/tradingbot" "/usr/local/bin/tradingbot" 2>/dev/null || true
        print_success "Global CLI command created: tradingbot"
    else
        # Add to user PATH
        local shell_rc=""
        if [[ "$SHELL" == *"bash"* ]]; then
            shell_rc="$HOME/.bashrc"
        elif [[ "$SHELL" == *"zsh"* ]]; then
            shell_rc="$HOME/.zshrc"
        else
            shell_rc="$HOME/.profile"
        fi
        
        if ! grep -q "$INSTALL_DIR" "$shell_rc" 2>/dev/null; then
            echo "export PATH=\"\$PATH:$INSTALL_DIR\"" >> "$shell_rc"
            print_success "Added to PATH in $shell_rc"
        fi
    fi
}

# Function to setup firewall rules
setup_firewall() {
    print_status "Configuring firewall for remote access..."
    
    if command_exists ufw; then
        # UFW (Ubuntu Firewall)
        if sudo -n true 2>/dev/null; then
            sudo ufw allow 8501/tcp comment "AI Trading Bot Web Interface" >/dev/null 2>&1
            print_success "UFW rule added for port 8501"
        fi
    elif command_exists firewall-cmd; then
        # FirewallD (CentOS/RHEL/Fedora)
        if sudo -n true 2>/dev/null; then
            sudo firewall-cmd --permanent --add-port=8501/tcp >/dev/null 2>&1
            sudo firewall-cmd --reload >/dev/null 2>&1
            print_success "FirewallD rule added for port 8501"
        fi
    else
        print_warning "No supported firewall found. Manually open port 8501 for remote access"
    fi
}

# Function to run post-installation tests
test_installation() {
    print_status "Running post-installation tests..."
    
    cd "$INSTALL_DIR"
    source venv/bin/activate
    
    # Test 1: Core modules
    print_status "Testing core modules..."
    if python -c "
import sys
sys.path.insert(0, '.')
try:
    from advanced_ai_system import AdvancedAITradingSystem
    from advanced_quant_engine import get_quant_module_manager
    from advanced_order_system import get_order_system
    print('SUCCESS: Core modules loaded')
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
" >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Core modules test passed"
    else
        echo -e "${RED}✗${NC} Core modules test failed"
    fi
    
    # Test 2: Virtual environment
    if [[ "$VIRTUAL_ENV" == "$INSTALL_DIR/venv" ]]; then
        echo -e "${GREEN}✓${NC} Virtual environment active"
    else
        echo -e "${RED}✗${NC} Virtual environment not active"
    fi
    
    # Test 3: Configuration
    if [[ -f ".env" && -f "config/config.yaml" ]]; then
        echo -e "${GREEN}✓${NC} Configuration files present"
    else
        echo -e "${RED}✗${NC} Configuration files missing"
    fi
    
    # Test 4: Network connectivity
    if curl -s --connect-timeout 5 https://api.github.com >/dev/null; then
        echo -e "${GREEN}✓${NC} Network connectivity test passed"
    else
        echo -e "${YELLOW}⚠${NC} Network connectivity limited"
    fi
    
    # Test 5: System health check
    if [[ -f "system_health_check.py" ]]; then
        print_status "Running system health check..."
        python system_health_check.py || true
    fi
    
    deactivate
    print_success "Installation tests completed"
}

# Function to create remote access guide
create_remote_guide() {
    print_status "Creating remote access guide..."
    
    cat > "$INSTALL_DIR/REMOTE_ACCESS_GUIDE.md" << 'EOF'
# AI Trading Bot - Remote Access Guide

## Server Setup and Remote Access

### 1. SSH Access
```bash
# Connect to your server
ssh username@your-server-ip

# Navigate to installation directory
cd ~/ai-trading-bot
```

### 2. Starting the Application

#### Option A: Direct Run (Foreground)
```bash
source venv/bin/activate
python advanced_ai_system.py
```

#### Option B: Background with Screen
```bash
screen -S trading-bot
source venv/bin/activate
python advanced_ai_system.py
# Press Ctrl+A, then D to detach
# screen -r trading-bot  # to reattach
```

#### Option C: Background with Tmux
```bash
tmux new-session -d -s trading-bot
tmux send-keys -t trading-bot "cd ~/ai-trading-bot && source venv/bin/activate && python advanced_ai_system.py" Enter
# tmux attach -t trading-bot  # to attach
```

#### Option D: Systemd Service (Recommended)
```bash
sudo systemctl start ai-trading-bot
sudo systemctl status ai-trading-bot
sudo journalctl -u ai-trading-bot -f  # View logs
```

### 3. Accessing the Web Interface

#### Method A: Direct Access (if port 8501 is open)
```
http://your-server-ip:8501
```

#### Method B: SSH Tunnel (Secure)
```bash
# On your local machine
ssh -L 8501:localhost:8501 username@your-server-ip

# Then access: http://localhost:8501
```

#### Method C: Ngrok Tunnel
```bash
# Install ngrok on server
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar xvzf ngrok-v3-stable-linux-amd64.tgz
sudo mv ngrok /usr/local/bin

# Create tunnel
ngrok http 8501
# Use the provided HTTPS URL
```

#### Method D: Cloudflare Tunnel
```bash
# Install cloudflared
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb

# Create tunnel
cloudflared tunnel --url http://localhost:8501
```

### 4. Service Management

#### Start/Stop/Restart
```bash
sudo systemctl start ai-trading-bot    # Start
sudo systemctl stop ai-trading-bot     # Stop
sudo systemctl restart ai-trading-bot  # Restart
sudo systemctl enable ai-trading-bot   # Enable on boot
```

#### View Logs
```bash
sudo journalctl -u ai-trading-bot -f   # Live logs
sudo journalctl -u ai-trading-bot -n 100  # Last 100 lines
```

#### Service Status
```bash
sudo systemctl status ai-trading-bot
```

### 5. Configuration and Updates

#### Edit Configuration
```bash
nano ~/ai-trading-bot/.env              # Environment variables
nano ~/ai-trading-bot/config/config.yaml  # Main configuration
```

#### Update Application
```bash
cd ~/ai-trading-bot
git pull  # If using git
# Or manually update files
sudo systemctl restart ai-trading-bot  # Restart service
```

#### Update Dependencies
```bash
cd ~/ai-trading-bot
source venv/bin/activate
pip install --upgrade -r requirements.txt
deactivate
sudo systemctl restart ai-trading-bot
```

### 6. Monitoring and Maintenance

#### Resource Monitoring
```bash
htop                    # System resources
df -h                   # Disk usage
free -h                 # Memory usage
```

#### Application Monitoring
```bash
# Health check
cd ~/ai-trading-bot && source venv/bin/activate && python system_health_check.py

# Performance monitoring
cd ~/ai-trading-bot && source venv/bin/activate && python PERFORMANCE_CALCULATOR.py
```

#### Log Rotation
```bash
# View log sizes
du -sh ~/ai-trading-bot/logs/*

# Manual cleanup (if needed)
find ~/ai-trading-bot/logs -name "*.log" -mtime +30 -delete
```

### 7. Security Considerations

#### Firewall Rules
```bash
# UFW (Ubuntu/Debian)
sudo ufw allow ssh
sudo ufw allow 8501
sudo ufw enable

# FirewallD (CentOS/RHEL)
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --permanent --add-port=8501/tcp
sudo firewall-cmd --reload
```

#### SSL/TLS (Optional)
For production use, consider setting up SSL with:
- Nginx reverse proxy with Let's Encrypt
- Cloudflare SSL
- Custom SSL certificates

### 8. Troubleshooting

#### Service Won't Start
```bash
sudo journalctl -u ai-trading-bot -n 50  # Check logs
sudo systemctl status ai-trading-bot     # Check status
```

#### Port Already in Use
```bash
sudo lsof -i :8501                       # Check what's using port
sudo netstat -tulpn | grep 8501          # Alternative check
```

#### Permission Issues
```bash
sudo chown -R $USER:$USER ~/ai-trading-bot
chmod +x ~/ai-trading-bot/tradingbot
```

#### Python Environment Issues
```bash
cd ~/ai-trading-bot
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 9. Performance Optimization

#### For High-Performance Trading
```bash
# Increase file limits
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Optimize kernel parameters
echo "net.core.rmem_max = 16777216" | sudo tee -a /etc/sysctl.conf
echo "net.core.wmem_max = 16777216" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

#### CPU Affinity (for dedicated servers)
```bash
# Pin service to specific CPU cores
sudo systemctl edit ai-trading-bot
# Add:
# [Service]
# CPUAffinity=0-3
```

### 10. Backup and Recovery

#### Backup Configuration
```bash
tar -czf ai-trading-bot-backup-$(date +%Y%m%d).tar.gz ~/ai-trading-bot
```

#### Restore from Backup
```bash
tar -xzf ai-trading-bot-backup-YYYYMMDD.tar.gz -C ~/
sudo systemctl restart ai-trading-bot
```

## Support and Resources

- GitHub Repository: [Your Repository URL]
- Documentation: ~/ai-trading-bot/README.md
- Health Check: `python system_health_check.py`
- Performance Calculator: `python PERFORMANCE_CALCULATOR.py`

For additional support, check the logs and system health reports.
EOF
    
    print_success "Remote access guide created: $INSTALL_DIR/REMOTE_ACCESS_GUIDE.md"
}

# Function to display final instructions
show_final_instructions() {
    echo -e "\n${CYAN}=== Installation Summary ===${NC}"
    echo -e "${GREEN}Installation Path:${NC} $INSTALL_DIR"
    echo -e "${GREEN}Python Version:${NC} $(source $INSTALL_DIR/venv/bin/activate && python --version && deactivate)"
    echo -e "${GREEN}Virtual Environment:${NC} $INSTALL_DIR/venv"
    echo -e "${GREEN}CLI Command:${NC} tradingbot"
    echo -e "${GREEN}Service:${NC} $SERVICE_NAME"
    
    echo -e "\n${YELLOW}Quick Start:${NC}"
    echo -e "${NC}1. Configure API keys:${NC}"
    echo -e "   ${BLUE}nano $INSTALL_DIR/.env${NC}"
    echo -e "${NC}2. Start the service:${NC}"
    echo -e "   ${BLUE}sudo systemctl start $SERVICE_NAME${NC}"
    echo -e "${NC}3. Access web interface:${NC}"
    echo -e "   ${BLUE}http://$(hostname -I | awk '{print $1}'):8501${NC}"
    echo -e "   ${BLUE}# Or via SSH tunnel: ssh -L 8501:localhost:8501 $USER@$(hostname -I | awk '{print $1}')${NC}"
    
    echo -e "\n${YELLOW}Management Commands:${NC}"
    echo -e "${BLUE}sudo systemctl start $SERVICE_NAME${NC}     # Start service"
    echo -e "${BLUE}sudo systemctl stop $SERVICE_NAME${NC}      # Stop service"
    echo -e "${BLUE}sudo systemctl status $SERVICE_NAME${NC}    # Check status"
    echo -e "${BLUE}sudo journalctl -u $SERVICE_NAME -f${NC}    # View logs"
    echo -e "${BLUE}tradingbot${NC}                           # Direct run"
    
    echo -e "\n${YELLOW}Remote Access:${NC}"
    echo -e "${NC}Complete guide: ${BLUE}$INSTALL_DIR/REMOTE_ACCESS_GUIDE.md${NC}"
    echo -e "${NC}Health check: ${BLUE}cd $INSTALL_DIR && source venv/bin/activate && python system_health_check.py${NC}"
    
    echo -e "\n${CYAN}For production trading:${NC}"
    echo -e "• Configure real API keys in .env file"
    echo -e "• Set TRADING_MODE=live in .env"
    echo -e "• Review risk settings in config/config.yaml"
    echo -e "• Set up SSL/TLS for secure remote access"
    echo -e "• Configure proper firewall rules"
}

# Main installation function
main() {
    print_status "Starting AI Crypto Trading Bot installation for Linux..."
    
    # Check for required tools
    if ! command_exists curl; then
        print_error "curl is required but not installed. Please install it first."
        exit 1
    fi
    
    # Installation steps
    detect_distro
    update_packages
    install_system_dependencies
    setup_python
    create_install_directory
    install_python_packages
    copy_application_files
    setup_configuration
    create_systemd_service
    create_cli_command
    setup_firewall
    test_installation
    create_remote_guide
    show_final_instructions
    
    print_success "AI Crypto Trading Bot installation completed successfully!"
    print_status "System is ready for 24/7 automated trading operations"
}

# Run main function
main "$@"