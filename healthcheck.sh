#!/bin/bash
# AI Crypto Trading Bot - Quick Health Check Script
# Fast system verification for Linux/macOS environments

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
INSTALL_DIR="$HOME/ai-trading-bot"
SERVICE_NAME="ai-trading-bot"

print_status() {
    case $2 in
        "pass") echo -e "${GREEN}✅ $1${NC}" ;;
        "fail") echo -e "${RED}❌ $1${NC}" ;;
        "warn") echo -e "${YELLOW}⚠️  $1${NC}" ;;
        "info") echo -e "${BLUE}ℹ️  $1${NC}" ;;
    esac
    [[ -n "$3" ]] && echo -e "   ${NC}$3"
}

echo -e "${BLUE}🔍 AI Trading Bot - Quick Health Check${NC}"
echo "=============================================="

# 1. Check installation directory
if [[ -d "$INSTALL_DIR" ]]; then
    print_status "Installation Directory" "pass" "$INSTALL_DIR"
else
    print_status "Installation Directory" "fail" "Not found: $INSTALL_DIR"
    exit 1
fi

cd "$INSTALL_DIR" || exit 1

# 2. Check Python and virtual environment
if [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
    print_status "Virtual Environment" "pass" "Activated"
    
    python_version=$(python --version 2>&1)
    if [[ $python_version == *"3."* ]]; then
        print_status "Python Version" "pass" "$python_version"
    else
        print_status "Python Version" "fail" "$python_version"
    fi
else
    print_status "Virtual Environment" "warn" "Using system Python"
    python_version=$(python3 --version 2>&1)
    print_status "Python Version" "info" "$python_version"
fi

# 3. Check core files
core_files=("advanced_ai_system.py" "advanced_quant_engine.py" "advanced_order_system.py")
missing_files=0

for file in "${core_files[@]}"; do
    if [[ -f "$file" ]]; then
        print_status "Core File: $file" "pass"
    else
        print_status "Core File: $file" "fail" "Missing"
        ((missing_files++))
    fi
done

# 4. Check configuration
if [[ -f ".env" ]]; then
    env_lines=$(grep -v '^#' .env | grep -v '^$' | wc -l)
    print_status "Environment Configuration" "pass" "$env_lines active settings"
else
    print_status "Environment Configuration" "warn" "Using defaults"
fi

if [[ -f "config/config.yaml" ]]; then
    print_status "YAML Configuration" "pass"
else
    print_status "YAML Configuration" "warn" "Using defaults"
fi

# 5. Check database
if [[ -f "ai_models.db" ]]; then
    db_size=$(du -h ai_models.db | cut -f1)
    print_status "AI Database" "pass" "Size: $db_size"
else
    print_status "AI Database" "info" "Will be created on first run"
fi

# 6. Check service (Linux only)
if command -v systemctl >/dev/null 2>&1; then
    if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        uptime=$(systemctl show "$SERVICE_NAME" --property=ActiveEnterTimestamp --value)
        print_status "System Service" "pass" "Running since $uptime"
    elif systemctl list-unit-files | grep -q "$SERVICE_NAME"; then
        print_status "System Service" "warn" "Installed but not running"
    else
        print_status "System Service" "info" "Not installed (manual start only)"
    fi
fi

# 7. Check network connectivity
if curl -s --connect-timeout 5 https://api.github.com >/dev/null; then
    print_status "Network Connectivity" "pass" "Internet accessible"
else
    print_status "Network Connectivity" "warn" "Limited internet access"
fi

# 8. Check web interface
if curl -f http://localhost:5000 >/dev/null 2>&1; then
    print_status "Web Interface" "pass" "Responding on port 5000"
elif pgrep -f "python.*advanced_ai_system" >/dev/null; then
    print_status "Web Interface" "warn" "Process running but port not responding"
else
    print_status "Web Interface" "info" "Not currently running"
fi

# 9. Check system resources
memory_usage=$(free | awk 'NR==2{printf "%.1f%%", $3*100/$2 }')
disk_usage=$(df . | awk 'NR==2{print $5}' | sed 's/%//')

if (( $(echo "$memory_usage" | cut -d'%' -f1 | cut -d'.' -f1) < 80 )); then
    print_status "Memory Usage" "pass" "$memory_usage"
else
    print_status "Memory Usage" "warn" "$memory_usage (high)"
fi

if (( disk_usage < 90 )); then
    print_status "Disk Usage" "pass" "${disk_usage}%"
else
    print_status "Disk Usage" "warn" "${disk_usage}% (high)"
fi

# 10. Quick functional test
if [[ $missing_files -eq 0 ]]; then
    echo -e "\n${BLUE}🧪 Quick Functional Test${NC}"
    
    if python -c "
import sys
sys.path.insert(0, '.')
try:
    from advanced_ai_system import AdvancedAITradingSystem
    print('✅ Core modules import successfully')
except Exception as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
" 2>/dev/null; then
        print_status "Module Import Test" "pass"
    else
        print_status "Module Import Test" "fail" "Core modules cannot be imported"
    fi
fi

# Summary
echo -e "\n${BLUE}📊 Health Check Summary${NC}"
echo "=============================================="

if [[ $missing_files -eq 0 ]]; then
    echo -e "${GREEN}🎉 System Status: HEALTHY${NC}"
    echo "   Ready for trading operations"
    echo ""
    echo -e "${BLUE}Quick Start Commands:${NC}"
    echo "   Start manually: cd $INSTALL_DIR && python advanced_ai_system.py"
    [[ -f "/usr/local/bin/tradingbot" ]] && echo "   Start via CLI: tradingbot"
    if systemctl list-unit-files | grep -q "$SERVICE_NAME"; then
        echo "   Start service: sudo systemctl start $SERVICE_NAME"
    fi
    echo "   Web interface: http://localhost:5000"
else
    echo -e "${RED}⚠️  System Status: NEEDS ATTENTION${NC}"
    echo "   Missing $missing_files core files"
    echo "   Run installation script to fix issues"
fi

echo ""
echo -e "${BLUE}📋 Next Steps:${NC}"
echo "   1. Configure API keys in .env file (for live trading)"
echo "   2. Review settings in config/config.yaml"
echo "   3. Run full diagnostic: python check_install.py"

# Return appropriate exit code
[[ $missing_files -eq 0 ]] && exit 0 || exit 1