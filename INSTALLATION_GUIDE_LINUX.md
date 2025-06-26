# AI Trading Bot - Guida Installazione Linux Server

## Installazione Completa su Server Linux

### Prerequisiti Sistema
```bash
# Ubuntu/Debian
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git curl wget

# CentOS/RHEL/Rocky Linux
sudo yum update -y
sudo yum install -y python3 python3-pip git curl wget
# oppure con dnf
sudo dnf install -y python3 python3-pip git curl wget
```

### Step 1: Download e Setup
```bash
# Crea directory progetto
mkdir ~/ai-trading-bot
cd ~/ai-trading-bot

# Download dei file (scegli uno dei metodi)

# Metodo 1: Da file ZIP scaricato
wget [URL_DEL_TUO_ZIP] -O AI_Trading_Bot_Linux.zip
unzip AI_Trading_Bot_Linux.zip
cd AI_Trading_Bot_Linux

# Metodo 2: Upload manuale via SCP/SFTP
# Carica il file ZIP sul server e poi:
unzip AI_Trading_Bot_Linux.zip
cd AI_Trading_Bot_Linux
```

### Step 2: Setup Environment Python
```bash
# Crea ambiente virtuale
python3 -m venv trading_env
source trading_env/bin/activate

# Installa dipendenze
pip install --upgrade pip
pip install -r requirements.txt

# Se non hai requirements.txt, installa manualmente:
pip install streamlit pandas numpy plotly scikit-learn asyncio
```

### Step 3: Configurazione per Accesso Remoto

#### Opzione A: Accesso Diretto (Raccomandato)
```bash
# Configura Streamlit per accesso remoto
mkdir -p ~/.streamlit
cat > ~/.streamlit/config.toml << EOF
[server]
headless = true
address = "0.0.0.0"
port = 5000
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
EOF
```

#### Opzione B: Con SSL/HTTPS (Sicurezza Extra)
```bash
# Genera certificati SSL self-signed
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Configura Streamlit con SSL
cat > ~/.streamlit/config.toml << EOF
[server]
headless = true
address = "0.0.0.0"
port = 5000
enableCORS = false
enableXsrfProtection = false
sslCertFile = "cert.pem"
sslKeyFile = "key.pem"

[browser]
gatherUsageStats = false
EOF
```

### Step 4: Configurazione Firewall
```bash
# Ubuntu/Debian (ufw)
sudo ufw allow 5000/tcp
sudo ufw enable

# CentOS/RHEL/Rocky (firewalld)
sudo firewall-cmd --permanent --add-port=5000/tcp
sudo firewall-cmd --reload

# Se usi iptables direttamente
sudo iptables -A INPUT -p tcp --dport 5000 -j ACCEPT
sudo iptables-save > /etc/iptables/rules.v4
```

### Step 5: Avvio del Sistema

#### Avvio Manuale (per test)
```bash
# Attiva ambiente virtuale
source trading_env/bin/activate

# Avvia l'applicazione
streamlit run advanced_ai_system.py --server.port 5000 --server.address 0.0.0.0
```

#### Avvio Automatico con Systemd (Production)
```bash
# Crea servizio systemd
sudo tee /etc/systemd/system/ai-trading-bot.service << EOF
[Unit]
Description=AI Trading Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/ai-trading-bot/AI_Trading_Bot_Linux
Environment=PATH=$HOME/ai-trading-bot/trading_env/bin
ExecStart=$HOME/ai-trading-bot/trading_env/bin/streamlit run advanced_ai_system.py --server.port 5000 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Abilita e avvia il servizio
sudo systemctl daemon-reload
sudo systemctl enable ai-trading-bot
sudo systemctl start ai-trading-bot

# Verifica status
sudo systemctl status ai-trading-bot
```

## Accesso Remoto alla Dashboard

### Da Windows
1. **Browser**: Apri browser e vai a `http://IP_DEL_SERVER:5000`
2. **Con SSL**: `https://IP_DEL_SERVER:5000`
3. **SSH Tunnel** (più sicuro):
   ```cmd
   # Da PowerShell/CMD
   ssh -L 5000:localhost:5000 user@IP_DEL_SERVER
   # Poi vai su http://localhost:5000
   ```

### Da macOS
1. **Browser**: `http://IP_DEL_SERVER:5000`
2. **SSH Tunnel**:
   ```bash
   ssh -L 5000:localhost:5000 user@IP_DEL_SERVER
   # Poi vai su http://localhost:5000
   ```

### Con Reverse Proxy (Nginx) - Opzionale
```bash
# Installa Nginx
sudo apt install nginx -y  # Ubuntu/Debian
sudo yum install nginx -y  # CentOS/RHEL

# Configura Nginx
sudo tee /etc/nginx/sites-available/trading-bot << EOF
server {
    listen 80;
    server_name YOUR_DOMAIN_OR_IP;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
    }
}
EOF

# Abilita sito
sudo ln -s /etc/nginx/sites-available/trading-bot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Gestione Sistema

### Comandi Utili
```bash
# Verifica status servizio
sudo systemctl status ai-trading-bot

# Riavvia servizio
sudo systemctl restart ai-trading-bot

# Visualizza logs
sudo journalctl -u ai-trading-bot -f

# Stop/Start manuale
sudo systemctl stop ai-trading-bot
sudo systemctl start ai-trading-bot

# Aggiorna applicazione
cd ~/ai-trading-bot/AI_Trading_Bot_Linux
git pull  # se usi git
# oppure sostituisci i file manualmente

# Riavvia dopo aggiornamento
sudo systemctl restart ai-trading-bot
```

### Monitoraggio
```bash
# Controlla utilizzo risorse
htop
top

# Controlla spazio disco
df -h

# Verifica connessioni di rete
netstat -tulpn | grep :5000
ss -tulpn | grep :5000

# Test connessione locale
curl http://localhost:5000
```

## Sicurezza Server

### Configurazione SSH (se non già fatto)
```bash
# Backup configurazione SSH
sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup

# Modifica configurazione SSH
sudo nano /etc/ssh/sshd_config

# Aggiungi/modifica:
# Port 22  (o cambia porta per sicurezza)
# PermitRootLogin no
# PasswordAuthentication no  (se usi chiavi SSH)
# AllowUsers tuo_username

# Riavvia SSH
sudo systemctl restart sshd
```

### Configurazione Fail2Ban
```bash
# Installa Fail2Ban
sudo apt install fail2ban -y  # Ubuntu/Debian
sudo yum install fail2ban -y  # CentOS/RHEL

# Configura per SSH
sudo tee /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
EOF

sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

### Aggiornamenti Automatici
```bash
# Ubuntu/Debian
sudo apt install unattended-upgrades -y
sudo dpkg-reconfigure unattended-upgrades

# CentOS/RHEL
sudo yum install yum-cron -y
sudo systemctl enable yum-cron
sudo systemctl start yum-cron
```

## Backup e Restore

### Script Backup Automatico
```bash
# Crea script backup
cat > ~/backup-trading-bot.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backup/trading-bot"
SOURCE_DIR="$HOME/ai-trading-bot"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR
tar -czf $BACKUP_DIR/trading-bot-backup-$DATE.tar.gz -C $SOURCE_DIR .

# Mantieni solo ultimi 7 backup
find $BACKUP_DIR -name "trading-bot-backup-*.tar.gz" -mtime +7 -delete

echo "Backup completed: trading-bot-backup-$DATE.tar.gz"
EOF

chmod +x ~/backup-trading-bot.sh

# Aggiungi a crontab per backup automatico
(crontab -l 2>/dev/null; echo "0 2 * * * $HOME/backup-trading-bot.sh") | crontab -
```

### Restore da Backup
```bash
# Stop servizio
sudo systemctl stop ai-trading-bot

# Restore
cd ~/ai-trading-bot
tar -xzf /backup/trading-bot/trading-bot-backup-YYYYMMDD_HHMMSS.tar.gz

# Riavvia servizio
sudo systemctl start ai-trading-bot
```

## Troubleshooting

### Problemi Comuni
```bash
# Porta occupata
sudo lsof -i :5000
sudo kill -9 PID_DEL_PROCESSO

# Permessi Python/Pip
sudo chown -R $USER:$USER ~/.local
sudo chown -R $USER:$USER ~/ai-trading-bot

# Problemi dipendenze
pip install --upgrade --force-reinstall streamlit

# Reset completo ambiente Python
rm -rf trading_env
python3 -m venv trading_env
source trading_env/bin/activate
pip install streamlit pandas numpy plotly scikit-learn
```

### Logs Debug
```bash
# Logs sistema
sudo journalctl -u ai-trading-bot -n 100

# Logs applicazione con debug
streamlit run advanced_ai_system.py --logger.level debug

# Logs Nginx (se usato)
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

## Performance Optimization

### Configurazione Server
```bash
# Aumenta limiti sistema
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Ottimizza memoria virtuale
echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### Configurazione Streamlit Avanzata
```bash
cat > ~/.streamlit/config.toml << EOF
[server]
headless = true
address = "0.0.0.0"
port = 5000
maxUploadSize = 200
maxMessageSize = 200
enableCORS = false
enableXsrfProtection = false

[runner]
magicEnabled = true
installTracer = false
fixMatplotlib = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
EOF
```

L'installazione è ora completa. Accedi alla dashboard da qualsiasi PC tramite browser all'indirizzo `http://IP_DEL_SERVER:5000`