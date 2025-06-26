# AI Crypto Trading Bot - Remote Access & Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying and accessing the AI Crypto Trading Bot on remote servers, cloud instances, and local networks. Whether you're running on a VPS, dedicated server, or local machine, this guide covers all access methods and deployment scenarios.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Server Deployment](#server-deployment)
3. [Remote Access Methods](#remote-access-methods)
4. [Security Configurations](#security-configurations)
5. [Performance Optimization](#performance-optimization)
6. [Monitoring & Maintenance](#monitoring--maintenance)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Configurations](#advanced-configurations)

---

## Quick Start

### Immediate Access After Installation

```bash
# Start the bot (any system)
cd ~/ai-trading-bot
source venv/bin/activate  # Linux/macOS only
python advanced_ai_system.py

# Access via browser
http://localhost:5000
```

### For Remote Server Access

```bash
# SSH tunnel (secure method)
ssh -L 5000:localhost:5000 username@server-ip

# Then access: http://localhost:5000
```

---

## Server Deployment

### 1. Cloud Platform Deployment

#### AWS EC2 Instance
```bash
# Launch Ubuntu 20.04 LTS instance
# Security Group: Allow SSH (22) and HTTP (5000)

# Connect and install
ssh -i your-key.pem ubuntu@ec2-instance-ip
wget https://your-domain.com/tradingbot-installer-linux.sh
chmod +x tradingbot-installer-linux.sh
./tradingbot-installer-linux.sh

# Configure for remote access
sudo ufw allow 5000
sudo systemctl enable ai-trading-bot
sudo systemctl start ai-trading-bot
```

#### Google Cloud Platform
```bash
# Create Compute Engine instance
gcloud compute instances create ai-trading-bot \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --machine-type=e2-medium \
  --boot-disk-size=20GB

# Allow firewall traffic
gcloud compute firewall-rules create allow-trading-bot \
  --allow tcp:5000 \
  --source-ranges 0.0.0.0/0
```

#### DigitalOcean Droplet
```bash
# Create Ubuntu droplet via web interface
# SSH and run installer
ssh root@droplet-ip
wget https://your-domain.com/tradingbot-installer-linux.sh
bash tradingbot-installer-linux.sh
```

### 2. Local Network Deployment

#### Raspberry Pi Setup
```bash
# For Raspberry Pi 4 (recommended for 24/7 operations)
sudo apt update && sudo apt upgrade -y
curl -L https://your-domain.com/tradingbot-installer-linux.sh | bash

# Configure for headless operation
echo "STREAMLIT_HEADLESS=true" >> ~/.ai-trading-bot/.env
sudo systemctl enable ai-trading-bot
```

#### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "advanced_ai_system.py"]
```

```bash
# Build and run
docker build -t ai-trading-bot .
docker run -d -p 5000:5000 --name trading-bot ai-trading-bot
```

---

## Remote Access Methods

### 1. SSH Tunneling (Recommended - Most Secure)

#### Basic SSH Tunnel
```bash
# Create secure tunnel
ssh -L 5000:localhost:5000 username@server-ip

# For persistent connection
ssh -L 5000:localhost:5000 -N -f username@server-ip
```

#### Advanced SSH Configuration
Create `~/.ssh/config`:
```
Host trading-server
    HostName your-server-ip
    User username
    Port 22
    LocalForward 5000 localhost:5000
    KeepAlive yes
    ServerAliveInterval 60
```

Then connect with: `ssh trading-server`

#### SSH with Key Authentication
```bash
# Generate SSH key pair
ssh-keygen -t rsa -b 4096 -C "your-email@example.com"

# Copy public key to server
ssh-copy-id username@server-ip

# Connect without password
ssh -L 5000:localhost:5000 username@server-ip
```

### 2. Direct Network Access

#### Open Port Configuration
```bash
# Ubuntu/Debian (UFW)
sudo ufw allow 5000/tcp
sudo ufw enable

# CentOS/RHEL (FirewallD)
sudo firewall-cmd --permanent --add-port=5000/tcp
sudo firewall-cmd --reload

# iptables (manual)
sudo iptables -A INPUT -p tcp --dport 5000 -j ACCEPT
sudo iptables-save > /etc/iptables/rules.v4
```

#### Router Port Forwarding
For home networks:
1. Access router admin panel (usually 192.168.1.1)
2. Navigate to Port Forwarding
3. Forward external port 5000 → internal IP port 5000

### 3. Tunneling Services

#### Ngrok (Quick Setup)
```bash
# Install ngrok
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok

# Authenticate (get token from ngrok.com)
ngrok authtoken YOUR_AUTH_TOKEN

# Create tunnel
ngrok http 5000

# Access via provided HTTPS URL
```

#### Cloudflare Tunnel
```bash
# Install cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb

# Create tunnel
cloudflared tunnel --url http://localhost:5000

# For permanent setup
cloudflared tunnel login
cloudflared tunnel create ai-trading-bot
cloudflared tunnel route dns ai-trading-bot trading.yourdomain.com
```

#### LocalTunnel (Alternative)
```bash
# Install via npm
npm install -g localtunnel

# Create tunnel
lt --port 5000 --subdomain your-chosen-name

# Access via: https://your-chosen-name.loca.lt
```

### 4. VPN Access

#### WireGuard Setup
```bash
# Server setup
sudo apt install wireguard
wg genkey | tee privatekey | wg pubkey > publickey

# Configure /etc/wireguard/wg0.conf
[Interface]
PrivateKey = SERVER_PRIVATE_KEY
Address = 10.0.0.1/24
ListenPort = 51820

[Peer]
PublicKey = CLIENT_PUBLIC_KEY
AllowedIPs = 10.0.0.2/32
```

---

## Security Configurations

### 1. SSL/TLS Setup with Nginx

#### Install and Configure Nginx
```bash
sudo apt install nginx certbot python3-certbot-nginx

# Create Nginx configuration
sudo nano /etc/nginx/sites-available/trading-bot
```

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for Streamlit
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

```bash
# Enable site and get SSL certificate
sudo ln -s /etc/nginx/sites-available/trading-bot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
sudo certbot --nginx -d your-domain.com
```

### 2. Firewall Security

#### Advanced UFW Configuration
```bash
# Default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow essential services
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'

# Rate limiting for SSH
sudo ufw limit ssh

# Enable firewall
sudo ufw enable
```

#### Fail2ban for SSH Protection
```bash
sudo apt install fail2ban

# Configure /etc/fail2ban/jail.local
[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 3600
```

### 3. Authentication & Access Control

#### HTTP Basic Authentication
```bash
# Create password file
sudo apt install apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd admin

# Add to Nginx config
auth_basic "Trading Bot Access";
auth_basic_user_file /etc/nginx/.htpasswd;
```

#### IP Whitelisting
```nginx
# In Nginx server block
location / {
    allow 192.168.1.0/24;
    allow 10.0.0.0/8;
    deny all;
    
    proxy_pass http://localhost:5000;
}
```

---

## Performance Optimization

### 1. System Optimization

#### Kernel Parameters
```bash
# Edit /etc/sysctl.conf
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 65536 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
net.core.netdev_max_backlog = 5000

# Apply changes
sudo sysctl -p
```

#### Process Optimization
```bash
# Edit /etc/security/limits.conf
* soft nofile 65536
* hard nofile 65536
trading-bot soft nproc 4096
trading-bot hard nproc 4096

# CPU affinity for dedicated servers
sudo systemctl edit ai-trading-bot
```

Add to override:
```ini
[Service]
CPUAffinity=0-3
Nice=-10
IOSchedulingClass=1
```

### 2. Database Optimization

#### SQLite Optimization
```bash
# In trading bot directory
echo "PRAGMA journal_mode=WAL;" | sqlite3 data/trading_data.db
echo "PRAGMA synchronous=NORMAL;" | sqlite3 data/trading_data.db
echo "PRAGMA cache_size=10000;" | sqlite3 data/trading_data.db
```

#### Storage Optimization
```bash
# SSD optimization
echo 'noop' | sudo tee /sys/block/sda/queue/scheduler
echo 'deadline' | sudo tee /sys/block/sda/queue/scheduler

# For NVMe drives
echo 'none' | sudo tee /sys/block/nvme0n1/queue/scheduler
```

---

## Monitoring & Maintenance

### 1. Service Monitoring

#### Systemd Service Status
```bash
# Check service status
sudo systemctl status ai-trading-bot

# View recent logs
sudo journalctl -u ai-trading-bot -f

# Service performance
sudo systemctl show ai-trading-bot --property=MainPID
ps -p $(sudo systemctl show ai-trading-bot --property=MainPID --value) -o pid,ppid,cmd,%mem,%cpu
```

#### Log Monitoring
```bash
# Real-time log monitoring
tail -f ~/ai-trading-bot/logs/trading.log

# Log analysis
grep "ERROR" ~/ai-trading-bot/logs/*.log
grep "trade_executed" ~/ai-trading-bot/logs/*.log | tail -10
```

### 2. Automated Monitoring Scripts

#### Health Check Script
```bash
#!/bin/bash
# save as monitor_trading_bot.sh

BOT_DIR="$HOME/ai-trading-bot"
LOG_FILE="$BOT_DIR/logs/monitoring.log"

check_service() {
    if systemctl is-active --quiet ai-trading-bot; then
        echo "$(date): Service is running" >> $LOG_FILE
        return 0
    else
        echo "$(date): Service is down, restarting..." >> $LOG_FILE
        sudo systemctl restart ai-trading-bot
        return 1
    fi
}

check_web_interface() {
    if curl -f http://localhost:5000 >/dev/null 2>&1; then
        echo "$(date): Web interface responsive" >> $LOG_FILE
        return 0
    else
        echo "$(date): Web interface not responding" >> $LOG_FILE
        return 1
    fi
}

# Run checks
check_service
check_web_interface

# Add to crontab: */5 * * * * /path/to/monitor_trading_bot.sh
```

### 3. Backup Automation

#### Automated Backup Script
```bash
#!/bin/bash
# save as backup_trading_bot.sh

SOURCE_DIR="$HOME/ai-trading-bot"
BACKUP_DIR="$HOME/trading-bot-backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="trading-bot-backup-$DATE.tar.gz"

# Create backup directory
mkdir -p $BACKUP_DIR

# Create backup
tar -czf "$BACKUP_DIR/$BACKUP_NAME" \
    --exclude="$SOURCE_DIR/venv" \
    --exclude="$SOURCE_DIR/logs/*.log" \
    --exclude="$SOURCE_DIR/__pycache__" \
    -C $(dirname $SOURCE_DIR) $(basename $SOURCE_DIR)

# Keep only last 7 backups
cd $BACKUP_DIR
ls -t trading-bot-backup-*.tar.gz | tail -n +8 | xargs rm -f

echo "Backup created: $BACKUP_NAME"

# Add to crontab: 0 2 * * * /path/to/backup_trading_bot.sh
```

---

## Troubleshooting

### 1. Common Issues

#### Service Won't Start
```bash
# Check service status
sudo systemctl status ai-trading-bot

# Check logs for errors
sudo journalctl -u ai-trading-bot -n 50

# Check Python environment
cd ~/ai-trading-bot
source venv/bin/activate
python -c "import streamlit; print('Streamlit OK')"
```

#### Port Already in Use
```bash
# Find process using port 5000
sudo lsof -i :5000
sudo netstat -tulpn | grep 5000

# Kill process if needed
sudo kill -9 PID_NUMBER
```

#### Permission Issues
```bash
# Fix ownership
sudo chown -R $USER:$USER ~/ai-trading-bot

# Fix permissions
chmod +x ~/ai-trading-bot/tradingbot
chmod 600 ~/ai-trading-bot/.env
```

#### Memory Issues
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Restart service to clear memory
sudo systemctl restart ai-trading-bot
```

### 2. Network Troubleshooting

#### Connection Issues
```bash
# Test local connection
curl http://localhost:5000

# Test external connection
curl -I http://your-server-ip:5000

# Check firewall
sudo ufw status
sudo iptables -L
```

#### DNS Issues
```bash
# Test DNS resolution
nslookup your-domain.com
dig your-domain.com

# Flush DNS cache
sudo systemd-resolve --flush-caches
```

### 3. Performance Issues

#### High CPU Usage
```bash
# Monitor CPU usage
top -p $(pgrep -f "python.*advanced_ai_system")
htop

# CPU profiling
cd ~/ai-trading-bot
python -m cProfile advanced_ai_system.py
```

#### Memory Leaks
```bash
# Monitor memory over time
while true; do
    ps -p $(pgrep -f "python.*advanced_ai_system") -o pid,ppid,cmd,%mem,%cpu,etime
    sleep 60
done

# Memory profiling
pip install memory_profiler
python -m memory_profiler advanced_ai_system.py
```

---

## Advanced Configurations

### 1. Load Balancing (Multiple Instances)

#### HAProxy Configuration
```bash
# Install HAProxy
sudo apt install haproxy

# Configure /etc/haproxy/haproxy.cfg
frontend trading_frontend
    bind *:80
    default_backend trading_servers

backend trading_servers
    balance roundrobin
    server bot1 localhost:5001 check
    server bot2 localhost:5002 check
```

### 2. High Availability Setup

#### Multi-Server Configuration
```bash
# Server 1 (Primary)
python advanced_ai_system.py --port 5001 --mode primary

# Server 2 (Secondary)
python advanced_ai_system.py --port 5002 --mode secondary

# Keepalived for failover
sudo apt install keepalived
```

### 3. Container Orchestration

#### Docker Compose
```yaml
version: '3.8'
services:
  trading-bot:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - TRADING_MODE=live
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - trading-bot
```

#### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-trading-bot
spec:
  replicas: 2
  selector:
    matchLabels:
      app: trading-bot
  template:
    metadata:
      labels:
        app: trading-bot
    spec:
      containers:
      - name: trading-bot
        image: ai-trading-bot:latest
        ports:
        - containerPort: 5000
        env:
        - name: TRADING_MODE
          value: "live"
---
apiVersion: v1
kind: Service
metadata:
  name: trading-bot-service
spec:
  selector:
    app: trading-bot
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer
```

---

## Support & Resources

### Documentation
- Main README: `~/ai-trading-bot/README.md`
- Configuration Guide: `~/ai-trading-bot/config/README.md`
- API Documentation: `~/ai-trading-bot/docs/api.md`

### Diagnostic Tools
```bash
# Health check
cd ~/ai-trading-bot && python check_install.py

# Performance analysis
cd ~/ai-trading-bot && python PERFORMANCE_CALCULATOR.py

# System health
cd ~/ai-trading-bot && python system_health_check.py
```

### Emergency Procedures
```bash
# Emergency stop
sudo systemctl stop ai-trading-bot

# Emergency backup
tar -czf emergency-backup-$(date +%s).tar.gz ~/ai-trading-bot

# Factory reset
sudo systemctl stop ai-trading-bot
rm -rf ~/ai-trading-bot/data/*.db
rm -rf ~/ai-trading-bot/logs/*
sudo systemctl start ai-trading-bot
```

### Getting Help
1. Check logs: `~/ai-trading-bot/logs/`
2. Run diagnostics: `python check_install.py`
3. Review configuration: `.env` and `config/config.yaml`
4. Check system resources: `htop`, `df -h`, `free -h`
5. Restart service: `sudo systemctl restart ai-trading-bot`

---

*This guide covers the most common deployment scenarios and troubleshooting steps. For specific issues not covered here, consult the diagnostic tools and log files for detailed error information.*