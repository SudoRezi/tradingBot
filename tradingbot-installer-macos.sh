#!/bin/bash
# AI Crypto Trading Bot - macOS Installer
# Compatible with Intel (x64) and Apple Silicon (ARM/M1/M2/M3)

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

# Detect system architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    echo -e "${CYAN}=== Detected Apple Silicon (ARM/M1/M2/M3) ===${NC}"
    IS_ARM=true
    HOMEBREW_PREFIX="/opt/homebrew"
else
    echo -e "${CYAN}=== Detected Intel x64 ===${NC}"
    IS_ARM=false
    HOMEBREW_PREFIX="/usr/local"
fi

echo -e "${CYAN}=== AI Crypto Trading Bot - macOS Installer ===${NC}"
echo -e "${GREEN}Installing to: $INSTALL_DIR${NC}"
echo -e "${GREEN}Architecture: $ARCH${NC}"

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

# Function to install Homebrew
install_homebrew() {
    print_status "Checking Homebrew installation..."
    
    if command_exists brew; then
        print_success "Homebrew already installed"
        return 0
    fi
    
    print_status "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH
    if [[ "$IS_ARM" == true ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    else
        echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/usr/local/bin/brew shellenv)"
    fi
    
    print_success "Homebrew installed successfully"
}

# Function to install Python
install_python() {
    print_status "Checking Python installation..."
    
    # Check if Python is already installed and compatible
    if command_exists python3; then
        CURRENT_PYTHON=$(python3 --version | grep -o '[0-9]\+\.[0-9]\+' | head -1)
        if [[ $(echo "$CURRENT_PYTHON >= $REQUIRED_PYTHON" | bc -l) -eq 1 ]]; then
            print_success "Compatible Python found: $(python3 --version)"
            return 0
        fi
    fi
    
    print_status "Installing Python $PYTHON_VERSION via Homebrew..."
    brew install python@$PYTHON_VERSION
    
    # Create symlinks
    if [[ "$IS_ARM" == true ]]; then
        ln -sf /opt/homebrew/bin/python3 /opt/homebrew/bin/python 2>/dev/null || true
        ln -sf /opt/homebrew/bin/pip3 /opt/homebrew/bin/pip 2>/dev/null || true
    else
        ln -sf /usr/local/bin/python3 /usr/local/bin/python 2>/dev/null || true
        ln -sf /usr/local/bin/pip3 /usr/local/bin/pip 2>/dev/null || true
    fi
    
    print_success "Python installed: $(python3 --version)"
}

# Function to install system dependencies
install_system_dependencies() {
    print_status "Installing system dependencies..."
    
    # Core dependencies
    local deps=("git" "curl" "wget" "openssl" "readline" "sqlite3" "xz" "zlib")
    
    # Architecture-specific optimizations
    if [[ "$IS_ARM" == true ]]; then
        deps+=("llvm" "libomp")  # For ARM optimization
    fi
    
    for dep in "${deps[@]}"; do
        if ! brew list "$dep" >/dev/null 2>&1; then
            print_status "Installing $dep..."
            brew install "$dep" || print_warning "Failed to install $dep"
        else
            print_success "$dep already installed"
        fi
    done
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
}

# Function to install Python packages
install_python_packages() {
    print_status "Installing Python packages..."
    
    # Upgrade pip first
    python3 -m pip install --upgrade pip
    
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
    )
    
    # ARM-specific optimizations
    if [[ "$IS_ARM" == true ]]; then
        export OPENBLAS_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export VECLIB_MAXIMUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        
        # Install ARM-optimized packages
        packages+=("tensorflow-macos" "tensorflow-metal")
    fi
    
    for package in "${packages[@]}"; do
        print_status "Installing $package..."
        if python3 -m pip install "$package" --quiet --no-warn-script-location; then
            echo -e "${GREEN}✓${NC} $package"
        else
            echo -e "${RED}✗${NC} $package"
            print_warning "Failed to install $package"
        fi
    done
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
    
    # Create config directory
    mkdir -p "$INSTALL_DIR/config"
    
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
SYSTEM_OS=macos
SYSTEM_ARCH=$ARCH
LOG_LEVEL=INFO
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
# macOS $ARCH Optimized Configuration

system:
  platform: macos
  architecture: $ARCH
  install_path: "$INSTALL_DIR"
  log_directory: "$INSTALL_DIR/logs"
  data_directory: "$INSTALL_DIR/data"
  is_arm: $IS_ARM

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
  optimization: $(if [[ "$IS_ARM" == true ]]; then echo "arm"; else echo "cpu"; fi)
  memory_limit: $(if [[ "$IS_ARM" == true ]]; then echo "4096"; else echo "2048"; fi)
  cache_size: $(if [[ "$IS_ARM" == true ]]; then echo "1024"; else echo "512"; fi)
  
  models:
    - name: trading_classifier
      type: sklearn
      enabled: true
    - name: sentiment_analyzer
      type: lightweight
      enabled: true
$(if [[ "$IS_ARM" == true ]]; then cat << EOL
    - name: neural_network
      type: tensorflow
      enabled: true
      optimization: metal
EOL
fi)

performance:
  cpu_optimization: true
  memory_optimization: true
  threading: true
  max_threads: $(if [[ "$IS_ARM" == true ]]; then echo "8"; else echo "4"; fi)
$(if [[ "$IS_ARM" == true ]]; then cat << EOL
  arm_optimization: true
  metal_acceleration: true
EOL
fi)

security:
  encryption: true
  key_rotation: true
  audit_logging: true

streamlit:
  host: 0.0.0.0
  port: 5000
  theme: dark
EOF
        print_success "Created config.yaml file"
    fi
}

# Function to create CLI command
create_cli_command() {
    print_status "Setting up CLI command..."
    
    # Create launch script
    cat > "$INSTALL_DIR/tradingbot" << EOF
#!/bin/bash
cd "$INSTALL_DIR"
python3 advanced_ai_system.py "\$@"
EOF
    
    chmod +x "$INSTALL_DIR/tradingbot"
    
    # Add to PATH
    local shell_rc=""
    if [[ "$SHELL" == *"zsh"* ]]; then
        shell_rc="$HOME/.zshrc"
    elif [[ "$SHELL" == *"bash"* ]]; then
        shell_rc="$HOME/.bashrc"
    else
        shell_rc="$HOME/.profile"
    fi
    
    if ! grep -q "$INSTALL_DIR" "$shell_rc" 2>/dev/null; then
        echo "export PATH=\"\$PATH:$INSTALL_DIR\"" >> "$shell_rc"
        print_success "Added to PATH in $shell_rc"
    fi
    
    # Create symlink for immediate use
    if [[ -w "/usr/local/bin" ]]; then
        ln -sf "$INSTALL_DIR/tradingbot" "/usr/local/bin/tradingbot" 2>/dev/null || true
    fi
}

# Function to create desktop alias
create_desktop_alias() {
    print_status "Creating desktop alias..."
    
    local app_path="/Applications/AI Trading Bot.app"
    mkdir -p "$app_path/Contents/Resources"
    
    # Create Info.plist
    cat > "$app_path/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>tradingbot</string>
    <key>CFBundleIdentifier</key>
    <string>com.aitradingbot.app</string>
    <key>CFBundleName</key>
    <string>AI Trading Bot</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
</dict>
</plist>
EOF
    
    # Create launcher script
    mkdir -p "$app_path/Contents/MacOS"
    cat > "$app_path/Contents/MacOS/tradingbot" << EOF
#!/bin/bash
cd "$INSTALL_DIR"
open -a Terminal "$INSTALL_DIR/tradingbot"
EOF
    
    chmod +x "$app_path/Contents/MacOS/tradingbot"
    print_success "Desktop application created"
}

# Function to run post-installation tests
test_installation() {
    print_status "Running post-installation tests..."
    
    cd "$INSTALL_DIR"
    
    # Test 1: Core modules
    print_status "Testing core modules..."
    if python3 -c "
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
    
    # Test 2: Configuration
    print_status "Testing configuration..."
    if [[ -f ".env" && -f "config/config.yaml" ]]; then
        echo -e "${GREEN}✓${NC} Configuration files present"
    else
        echo -e "${RED}✗${NC} Configuration files missing"
    fi
    
    # Test 3: System health check
    if [[ -f "system_health_check.py" ]]; then
        print_status "Running system health check..."
        python3 system_health_check.py || true
    fi
    
    print_success "Installation tests completed"
}

# Function to display final instructions
show_final_instructions() {
    echo -e "\n${CYAN}=== Installation Summary ===${NC}"
    echo -e "${GREEN}Installation Path:${NC} $INSTALL_DIR"
    echo -e "${GREEN}Python Version:${NC} $(python3 --version)"
    echo -e "${GREEN}Architecture:${NC} $ARCH $(if [[ "$IS_ARM" == true ]]; then echo "(ARM optimized)"; fi)"
    echo -e "${GREEN}CLI Command:${NC} tradingbot"
    echo -e "${GREEN}Desktop App:${NC} AI Trading Bot.app"
    
    echo -e "\n${YELLOW}To start trading:${NC}"
    echo -e "${NC}1. Configure API keys in .env file:${NC}"
    echo -e "   ${BLUE}nano $INSTALL_DIR/.env${NC}"
    echo -e "${NC}2. Run the trading bot:${NC}"
    echo -e "   ${BLUE}tradingbot${NC}"
    echo -e "${NC}3. Access web interface:${NC}"
    echo -e "   ${BLUE}http://localhost:5000${NC}"
    
    echo -e "\n${YELLOW}Additional commands:${NC}"
    echo -e "${BLUE}tradingbot --help${NC}          Show help"
    echo -e "${BLUE}cd $INSTALL_DIR${NC}    Navigate to installation"
    echo -e "${BLUE}python3 system_health_check.py${NC}  Run health check"
    
    if [[ "$IS_ARM" == true ]]; then
        echo -e "\n${CYAN}ARM Optimization Active:${NC}"
        echo -e "• Metal acceleration enabled for neural networks"
        echo -e "• ARM-optimized mathematical libraries"
        echo -e "• Enhanced performance for Apple Silicon"
    fi
}

# Main installation function
main() {
    print_status "Starting AI Crypto Trading Bot installation for macOS..."
    
    # Check for required tools
    if ! command_exists curl; then
        print_error "curl is required but not installed. Please install Xcode Command Line Tools."
        exit 1
    fi
    
    # Installation steps
    install_homebrew
    install_python
    install_system_dependencies
    create_install_directory
    install_python_packages
    copy_application_files
    setup_configuration
    create_cli_command
    create_desktop_alias
    test_installation
    show_final_instructions
    
    print_success "AI Crypto Trading Bot installation completed successfully!"
}

# Run main function
main "$@"