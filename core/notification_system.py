import smtplib
import logging
import os
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class NotificationSystem:
    """Handles notifications for trading events"""
    
    def __init__(self):
        # Email configuration
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.email_user = os.getenv('EMAIL_USER', '')
        self.email_password = os.getenv('EMAIL_PASSWORD', '')
        self.notification_email = os.getenv('NOTIFICATION_EMAIL', self.email_user)
        
        # Notification settings
        self.email_enabled = bool(self.email_user and self.email_password)
        self.min_notification_interval = 300  # 5 minutes
        self.last_notification = {}
        
        logger.info(f"Notification System initialized - Email: {'Enabled' if self.email_enabled else 'Disabled'}")
    
    def send_notification(self, title: str, message: str, priority: str = 'normal'):
        """Send notification via available channels"""
        try:
            # Check rate limiting
            if not self._should_send_notification(title, priority):
                return
            
            # Send email notification
            if self.email_enabled:
                self._send_email(title, message, priority)
            
            # Log notification
            logger.info(f"Notification sent: {title}")
            
            # Update last notification time
            self.last_notification[title] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    def _should_send_notification(self, title: str, priority: str) -> bool:
        """Check if notification should be sent based on rate limiting"""
        try:
            # Always send high priority notifications
            if priority == 'high':
                return True
            
            # Check if we've sent this notification recently
            if title in self.last_notification:
                time_since_last = (datetime.now() - self.last_notification[title]).seconds
                return time_since_last >= self.min_notification_interval
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking notification rate limit: {e}")
            return False
    
    def _send_email(self, title: str, message: str, priority: str):
        """Send email notification"""
        try:
            if not self.email_enabled:
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = self.notification_email
            msg['Subject'] = f"AI Trader Alert: {title}"
            
            # Email body
            body = f"""
AI Crypto Trading Bot Alert

{title}

{message}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Priority: {priority.upper()}

---
This is an automated message from your AI Trading Bot.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
            
            logger.debug(f"Email sent successfully: {title}")
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
    
    def send_startup_notification(self, initial_capital: float):
        """Send notification when trading starts"""
        message = f"""
Your AI Trading Bot has started operating autonomously!

Initial Capital: ${initial_capital:,.2f}
Trading Pairs: KAS/USDT, BTC/USDT, ETH/USDT
Status: ACTIVE - Operating 24/7

The AI will now analyze markets and execute trades automatically.
You'll receive notifications for significant events only.
        """
        self.send_notification("🚀 AI Trader Started", message, 'high')
    
    def send_profit_notification(self, profit_amount: float, profit_pct: float):
        """Send notification for significant profits"""
        if profit_amount > 50 or profit_pct > 5:  # Only for significant profits
            message = f"""
Your AI has generated significant profits!

Profit: ${profit_amount:+,.2f} ({profit_pct:+.2f}%)
Status: AI continues trading autonomously

Keep letting the AI work - it's making money!
            """
            self.send_notification("💰 Significant Profit Generated", message, 'high')
    
    def send_risk_alert(self, alert_type: str, details: str):
        """Send risk management alert"""
        message = f"""
Risk Management Alert

Type: {alert_type}
Details: {details}
Action: AI has automatically adjusted positions

Your capital is being protected automatically.
        """
        self.send_notification("🚨 Risk Management Alert", message, 'high')
    
    def send_daily_summary(self, summary: Dict[str, Any]):
        """Send daily performance summary"""
        daily_pnl = summary.get('daily_pnl', 0)
        total_return = summary.get('total_return_pct', 0)
        trades_today = summary.get('trades_today', 0)
        win_rate = summary.get('win_rate', 0)
        
        message = f"""
Daily AI Trading Summary

Today's P&L: ${daily_pnl:+,.2f}
Total Return: {total_return:+.2f}%
Trades Executed: {trades_today}
Win Rate: {win_rate:.1f}%

Your AI continues working 24/7 to grow your portfolio.
        """
        
        title = "📊 Daily Trading Summary"
        if daily_pnl > 0:
            title = "📈 Daily Profit Summary"
        
        self.send_notification(title, message, 'normal')
    
    def get_notification_status(self) -> Dict[str, Any]:
        """Get notification system status"""
        return {
            'email_enabled': self.email_enabled,
            'email_configured': bool(self.email_user),
            'recent_notifications': len(self.last_notification),
            'last_notifications': {
                title: time.isoformat() 
                for title, time in list(self.last_notification.items())[-5:]
            }
        }
