# Production Deployment Guide - Advanced AI Trading System

## Overview

This guide covers deploying the Advanced AI Trading System in production environments across different operating systems with enterprise-grade reliability and security.

## System Requirements

### Minimum Requirements
- **CPU**: 4+ cores (Intel i5/AMD Ryzen 5 or better)
- **RAM**: 8GB+ (16GB recommended for AI models)
- **Storage**: 50GB+ SSD space
- **Network**: Stable internet (10+ Mbps)
- **OS**: Ubuntu 20.04+, Windows 10+, macOS 10.15+

### Recommended Production Setup
- **CPU**: 8+ cores (Intel i7/AMD Ryzen 7 or better)
- **RAM**: 32GB+ for AI model caching
- **Storage**: 200GB+ NVMe SSD
- **Network**: Redundant internet connections
- **GPU**: Optional CUDA-compatible for AI acceleration

## Pre-Installation Setup

### 1. Security Hardening

#### Linux (Ubuntu/Debian)
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install security tools
sudo apt install -y fail2ban ufw

# Configure firewall
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp  # SSH
sudo ufw allow 5000/tcp  # Streamlit

# Configure fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

#### Windows
```powershell
# Run as Administrator
# Enable Windows Defender
Set-MpPreference -DisableRealtimeMonitoring $false

# Configure Windows Firewall
New-NetFirewallRule -DisplayName "AI Trading System" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow

# Install Windows Subsystem for Linux (recommended)
wsl --install -d Ubuntu
```

#### macOS
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install security tools
brew install --cask lulu  # Network monitor
```

### 2. Environment Setup

#### Install Python 3.11+
```bash
# Linux/macOS
sudo apt install python3.11 python3.11-pip python3.11-venv  # Ubuntu
brew install python@3.11  # macOS

# Windows (using chocolatey)
choco install python311
```

#### Install Node.js (for additional tools)
```bash
# Linux/macOS
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs  # Ubuntu

brew install node  # macOS

# Windows
choco install nodejs
```

## Installation Methods

### Method 1: Automated Installer (Recommended)

#### Linux Production Setup
```bash
#!/bin/bash
# production_install_linux.sh

set -e

echo "🚀 Installing Advanced AI Trading System - Production"

# Create system user
sudo useradd -m -s /bin/bash aitrader
sudo usermod -aG sudo aitrader

# Create application directory
sudo mkdir -p /opt/ai-trading-system
sudo chown aitrader:aitrader /opt/ai-trading-system

# Switch to aitrader user
sudo -u aitrader bash << 'EOF'
cd /opt/ai-trading-system

# Clone repository
git clone https://github.com/your-repo/ai-trading-system.git .

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install additional production dependencies
pip install gunicorn supervisor redis celery

# Create configuration
mkdir -p config logs data

# Set permissions
chmod 750 config
chmod 755 logs
chmod 750 data

EOF

# Create systemd service
sudo tee /etc/systemd/system/ai-trading.service > /dev/null << 'EOF'
[Unit]
Description=Advanced AI Trading System
After=network.target

[Service]
Type=simple
User=aitrader
Group=aitrader
WorkingDirectory=/opt/ai-trading-system
Environment=PATH=/opt/ai-trading-system/venv/bin
ExecStart=/opt/ai-trading-system/venv/bin/streamlit run advanced_ai_system.py --server.port 5000 --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable ai-trading.service
sudo systemctl start ai-trading.service

echo "✅ Installation complete! Service running at http://localhost:5000"
```

#### Windows Production Setup
```powershell
# production_install_windows.ps1
# Run as Administrator

Write-Host "🚀 Installing Advanced AI Trading System - Production" -ForegroundColor Green

# Create application directory
New-Item -ItemType Directory -Force -Path "C:\ai-trading-system"
Set-Location "C:\ai-trading-system"

# Download and extract system
# (Assuming you have a Windows package available)
Invoke-WebRequest -Uri "https://github.com/your-repo/ai-trading-system/archive/main.zip" -OutFile "system.zip"
Expand-Archive -Path "system.zip" -DestinationPath "." -Force

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install Windows Service Manager
pip install pywin32

# Create Windows service script
@"
import win32serviceutil
import win32service
import win32event
import subprocess
import sys

class AITradingService(win32serviceutil.ServiceFramework):
    _svc_name_ = "AITradingSystem"
    _svc_display_name_ = "Advanced AI Trading System"
    _svc_description_ = "AI-powered cryptocurrency trading system"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)

    def SvcDoRun(self):
        subprocess.run([
            r"C:\ai-trading-system\venv\Scripts\streamlit.exe",
            "run", "advanced_ai_system.py",
            "--server.port", "5000",
            "--server.address", "0.0.0.0", 
            "--server.headless", "true"
        ], cwd=r"C:\ai-trading-system")

if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(AITradingService)
"@ | Out-File -FilePath "ai_trading_service.py" -Encoding UTF8

# Install and start service
python ai_trading_service.py install
python ai_trading_service.py start

Write-Host "✅ Installation complete! Service running at http://localhost:5000" -ForegroundColor Green
```

#### macOS Production Setup
```bash
#!/bin/bash
# production_install_macos.sh

echo "🚀 Installing Advanced AI Trading System - Production"

# Create application directory
sudo mkdir -p /usr/local/ai-trading-system
sudo chown $(whoami):staff /usr/local/ai-trading-system
cd /usr/local/ai-trading-system

# Clone repository
git clone https://github.com/your-repo/ai-trading-system.git .

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create LaunchDaemon for system-wide service
sudo tee /Library/LaunchDaemons/com.aitrading.system.plist > /dev/null << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.aitrading.system</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/ai-trading-system/venv/bin/streamlit</string>
        <string>run</string>
        <string>advanced_ai_system.py</string>
        <string>--server.port</string>
        <string>5000</string>
        <string>--server.address</string>
        <string>0.0.0.0</string>
        <string>--server.headless</string>
        <string>true</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/usr/local/ai-trading-system</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/usr/local/ai-trading-system/logs/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/usr/local/ai-trading-system/logs/stderr.log</string>
</dict>
</plist>
EOF

# Load and start service
sudo launchctl load /Library/LaunchDaemons/com.aitrading.system.plist

echo "✅ Installation complete! Service running at http://localhost:5000"
```

### Method 2: Docker Deployment

#### Production Dockerfile
```dockerfile
# Dockerfile.production
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 aitrader
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
RUN chown -R aitrader:aitrader /app

# Switch to non-root user
USER aitrader

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/_stcore/health || exit 1

# Start application
CMD ["streamlit", "run", "advanced_ai_system.py", "--server.port=5000", "--server.address=0.0.0.0", "--server.headless=true"]
```

#### Docker Compose for Production
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  ai-trading:
    build:
      context: .
      dockerfile: Dockerfile.production
    ports:
      - "5000:5000"
    environment:
      - ENVIRONMENT=production
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config:/app/config
    depends_on:
      - redis
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - ai-trading
    restart: unless-stopped

volumes:
  redis_data:
```

## Production Configuration

### 1. Environment Variables
```bash
# .env.production
ENVIRONMENT=production
DEBUG=false

# API Keys (encrypted storage recommended)
HUGGINGFACE_TOKEN=your_hf_token
ALPHA_VANTAGE_API_KEY=your_av_key
NEWSAPI_KEY=your_news_key
REDDIT_CLIENT_ID=your_reddit_id
REDDIT_CLIENT_SECRET=your_reddit_secret
TWITTER_BEARER_TOKEN=your_twitter_token

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/aitrading

# Security
SECRET_KEY=your_secret_key_here
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com

# Performance
MAX_WORKERS=4
CACHE_TTL=300
REDIS_URL=redis://localhost:6379

# Monitoring
SENTRY_DSN=your_sentry_dsn
LOG_LEVEL=INFO
```

### 2. Nginx Configuration
```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream ai_trading {
        server localhost:5000;
    }

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        location / {
            proxy_pass http://ai_trading;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
```

## Monitoring and Maintenance

### 1. System Monitoring
```bash
# monitoring_setup.sh
#!/bin/bash

# Install monitoring tools
sudo apt install -y htop iotop nethogs

# Create monitoring script
cat << 'EOF' > /opt/ai-trading-system/monitor.sh
#!/bin/bash

# System resources
echo "=== System Resources ==="
free -h
df -h
top -bn1 | head -20

# Service status
echo "=== Service Status ==="
systemctl status ai-trading.service

# Logs
echo "=== Recent Logs ==="
journalctl -u ai-trading.service --since "1 hour ago" | tail -20

# API Health
echo "=== API Health ==="
curl -s http://localhost:5000/_stcore/health || echo "Service not responding"
EOF

chmod +x /opt/ai-trading-system/monitor.sh

# Create cron job for monitoring
(crontab -l 2>/dev/null; echo "*/5 * * * * /opt/ai-trading-system/monitor.sh >> /opt/ai-trading-system/logs/monitor.log 2>&1") | crontab -
```

### 2. Backup Strategy
```bash
# backup_system.sh
#!/bin/bash

BACKUP_DIR="/backup/ai-trading-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup application data
cp -r /opt/ai-trading-system/data "$BACKUP_DIR/"
cp -r /opt/ai-trading-system/config "$BACKUP_DIR/"
cp -r /opt/ai-trading-system/ai_models.db "$BACKUP_DIR/" 2>/dev/null || true

# Backup logs (last 7 days)
find /opt/ai-trading-system/logs -name "*.log" -mtime -7 -exec cp {} "$BACKUP_DIR/" \;

# Compress backup
tar -czf "${BACKUP_DIR}.tar.gz" -C "$(dirname "$BACKUP_DIR")" "$(basename "$BACKUP_DIR")"
rm -rf "$BACKUP_DIR"

# Clean old backups (keep 30 days)
find /backup -name "ai-trading-*.tar.gz" -mtime +30 -delete

echo "Backup created: ${BACKUP_DIR}.tar.gz"
```

## Performance Optimization

### 1. System Tuning
```bash
# performance_tuning.sh
#!/bin/bash

# Increase file limits
echo "aitrader soft nofile 65536" >> /etc/security/limits.conf
echo "aitrader hard nofile 65536" >> /etc/security/limits.conf

# Optimize network settings
echo "net.core.rmem_max = 134217728" >> /etc/sysctl.conf
echo "net.core.wmem_max = 134217728" >> /etc/sysctl.conf
echo "net.ipv4.tcp_rmem = 4096 87380 134217728" >> /etc/sysctl.conf

# Apply settings
sysctl -p
```

### 2. Application Optimization
```python
# production_config.py
import os

class ProductionConfig:
    # Cache settings
    CACHE_TYPE = "redis"
    CACHE_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    CACHE_DEFAULT_TIMEOUT = 300
    
    # AI Model settings
    AI_MODEL_CACHE_SIZE = 1000  # MB
    MAX_CONCURRENT_MODELS = 4
    MODEL_INFERENCE_TIMEOUT = 30  # seconds
    
    # API rate limiting
    API_RATE_LIMIT = "1000/hour"
    BURST_LIMIT = 50
    
    # Performance
    MAX_WORKERS = os.cpu_count()
    THREAD_POOL_SIZE = 20
    ASYNC_TIMEOUT = 60
```

## Security Best Practices

### 1. API Key Management
```python
# secure_config.py
from cryptography.fernet import Fernet
import os

class SecureConfig:
    def __init__(self):
        self.key = os.getenv("ENCRYPTION_KEY") or Fernet.generate_key()
        self.cipher = Fernet(self.key)
    
    def encrypt_api_key(self, api_key: str) -> str:
        return self.cipher.encrypt(api_key.encode()).decode()
    
    def decrypt_api_key(self, encrypted_key: str) -> str:
        return self.cipher.decrypt(encrypted_key.encode()).decode()
```

### 2. Access Control
```python
# auth.py
import streamlit as st
import hashlib
import hmac

def check_authentication():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if verify_credentials(username, password):
                st.session_state.authenticated = True
                st.success("Authenticated successfully")
                st.rerun()
            else:
                st.error("Invalid credentials")
        return False
    
    return True

def verify_credentials(username: str, password: str) -> bool:
    # Use environment variables for credentials
    valid_username = os.getenv("ADMIN_USERNAME")
    valid_password_hash = os.getenv("ADMIN_PASSWORD_HASH")
    
    if username == valid_username:
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        return hmac.compare_digest(password_hash, valid_password_hash)
    
    return False
```

## Troubleshooting

### Common Issues and Solutions

1. **Service won't start**
   ```bash
   # Check logs
   journalctl -u ai-trading.service -f
   
   # Check permissions
   ls -la /opt/ai-trading-system/
   
   # Verify Python environment
   /opt/ai-trading-system/venv/bin/python --version
   ```

2. **High memory usage**
   ```bash
   # Monitor memory
   htop
   
   # Reduce AI model cache
   # Edit config: AI_MODEL_CACHE_SIZE = 500
   ```

3. **API timeouts**
   ```bash
   # Check network connectivity
   curl -I https://api.twitter.com
   
   # Increase timeout settings
   # Edit config: API_TIMEOUT = 60
   ```

## Update Procedure

```bash
# update_system.sh
#!/bin/bash

echo "🔄 Updating AI Trading System"

# Backup current version
./backup_system.sh

# Stop service
sudo systemctl stop ai-trading.service

# Update code
cd /opt/ai-trading-system
git pull origin main

# Activate environment
source venv/bin/activate

# Update dependencies
pip install --upgrade -r requirements.txt

# Run migrations (if any)
python migrate.py

# Start service
sudo systemctl start ai-trading.service

# Verify update
sleep 10
curl -f http://localhost:5000/_stcore/health && echo "✅ Update successful" || echo "❌ Update failed"
```

This production guide ensures enterprise-grade deployment with proper security, monitoring, and maintenance procedures across all major operating systems.