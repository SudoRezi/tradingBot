"""
Sistema di Monitoraggio Completo - Real-time Performance & Security
Monitoraggio autonomo di performance, sicurezza, trading e sistema
"""

import time
import psutil
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import os
from utils.logger import setup_logger

logger = setup_logger('system_monitor')

class SystemMonitor:
    """Monitor completo per sistema trading autonomo"""
    
    def __init__(self):
        self.is_monitoring = False
        self.monitor_thread = None
        self.performance_data = []
        self.security_alerts = []
        self.trading_metrics = {}
        self.system_health = {}
        self.start_time = datetime.now()
        
        # Soglie di allarme
        self.alert_thresholds = {
            'cpu_usage': 85,  # %
            'memory_usage': 90,  # %
            'disk_usage': 95,  # %
            'response_time': 5000,  # ms
            'error_rate': 5,  # %
            'failed_trades': 3,  # consecutive
            'api_errors': 10,  # per hour
        }
        
        # Metriche trading
        self.trading_stats = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'consecutive_failures': 0,
            'total_volume': 0.0,
            'total_profit_loss': 0.0,
            'api_calls': 0,
            'api_errors': 0,
            'last_trade_time': None,
            'uptime_seconds': 0
        }
    
    def start_monitoring(self):
        """Avvia monitoraggio continuo"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🔍 Sistema di monitoraggio avviato")
    
    def stop_monitoring(self):
        """Ferma monitoraggio"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("🛑 Sistema di monitoraggio fermato")
    
    def _monitoring_loop(self):
        """Loop principale di monitoraggio"""
        while self.is_monitoring:
            try:
                # Aggiorna metriche sistema
                self._update_system_metrics()
                
                # Controlla sicurezza
                self._security_check()
                
                # Verifica performance
                self._performance_check()
                
                # Salva dati
                self._save_monitoring_data()
                
                # Aggiorna uptime
                self.trading_stats['uptime_seconds'] = (datetime.now() - self.start_time).total_seconds()
                
                time.sleep(30)  # Check ogni 30 secondi
                
            except Exception as e:
                logger.error(f"Errore monitoraggio: {e}")
                time.sleep(60)  # Retry dopo 1 minuto
    
    def _update_system_metrics(self):
        """Aggiorna metriche di sistema"""
        try:
            # CPU e memoria
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Processi
            process_count = len(psutil.pids())
            
            # Network
            network = psutil.net_io_counters()
            
            self.system_health = {
                'timestamp': datetime.now().isoformat(),
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_usage': (disk.used / disk.total) * 100,
                'disk_free_gb': disk.free / (1024**3),
                'process_count': process_count,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'uptime_hours': self.trading_stats['uptime_seconds'] / 3600
            }
            
            # Verifica soglie
            self._check_system_alerts()
            
        except Exception as e:
            logger.error(f"Errore metriche sistema: {e}")
    
    def _check_system_alerts(self):
        """Controlla soglie di allarme sistema"""
        alerts = []
        
        if self.system_health['cpu_usage'] > self.alert_thresholds['cpu_usage']:
            alerts.append({
                'type': 'SYSTEM_ALERT',
                'level': 'HIGH',
                'message': f"CPU usage alto: {self.system_health['cpu_usage']:.1f}%",
                'timestamp': datetime.now().isoformat()
            })
        
        if self.system_health['memory_usage'] > self.alert_thresholds['memory_usage']:
            alerts.append({
                'type': 'SYSTEM_ALERT',
                'level': 'HIGH',
                'message': f"Memoria usage alta: {self.system_health['memory_usage']:.1f}%",
                'timestamp': datetime.now().isoformat()
            })
        
        if self.system_health['disk_usage'] > self.alert_thresholds['disk_usage']:
            alerts.append({
                'type': 'SYSTEM_ALERT',
                'level': 'CRITICAL',
                'message': f"Spazio disco critico: {self.system_health['disk_usage']:.1f}%",
                'timestamp': datetime.now().isoformat()
            })
        
        if alerts:
            self.security_alerts.extend(alerts)
            for alert in alerts:
                logger.warning(f"🚨 {alert['message']}")
    
    def _security_check(self):
        """Controlli di sicurezza"""
        try:
            # Controlla file critici
            critical_files = [
                '.encryption_key',
                'config/settings.py',
                'utils/encryption.py'
            ]
            
            for file_path in critical_files:
                if os.path.exists(file_path):
                    # Verifica permessi
                    stat_info = os.stat(file_path)
                    if stat_info.st_mode & 0o077:  # Altri hanno accesso
                        self.security_alerts.append({
                            'type': 'SECURITY_ALERT',
                            'level': 'HIGH',
                            'message': f"Permessi file critici non sicuri: {file_path}",
                            'timestamp': datetime.now().isoformat()
                        })
            
            # Controlla log per pattern sospetti
            self._check_log_anomalies()
            
        except Exception as e:
            logger.error(f"Errore controllo sicurezza: {e}")
    
    def _check_log_anomalies(self):
        """Controlla log per anomalie"""
        try:
            log_file = f'logs/ai_trader_{datetime.now().strftime("%Y%m%d")}.log'
            if not os.path.exists(log_file):
                return
            
            # Pattern sospetti
            suspicious_patterns = [
                'Failed login',
                'Unauthorized access',
                'API key invalid',
                'Connection refused',
                'Too many requests'
            ]
            
            with open(log_file, 'r') as f:
                recent_lines = f.readlines()[-100:]  # Ultimi 100 righe
            
            for line in recent_lines:
                for pattern in suspicious_patterns:
                    if pattern.lower() in line.lower():
                        self.security_alerts.append({
                            'type': 'SECURITY_ALERT',
                            'level': 'MEDIUM',
                            'message': f"Pattern sospetto rilevato: {pattern}",
                            'timestamp': datetime.now().isoformat(),
                            'details': line.strip()
                        })
        except Exception as e:
            logger.error(f"Errore controllo log: {e}")
    
    def _performance_check(self):
        """Controlli di performance"""
        try:
            # Calcola error rate
            total_operations = self.trading_stats['api_calls']
            if total_operations > 0:
                error_rate = (self.trading_stats['api_errors'] / total_operations) * 100
                
                if error_rate > self.alert_thresholds['error_rate']:
                    self.security_alerts.append({
                        'type': 'PERFORMANCE_ALERT',
                        'level': 'HIGH',
                        'message': f"Error rate alto: {error_rate:.1f}%",
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Controlla trade consecutivi falliti
            if self.trading_stats['consecutive_failures'] >= self.alert_thresholds['failed_trades']:
                self.security_alerts.append({
                    'type': 'TRADING_ALERT',
                    'level': 'HIGH',
                    'message': f"Trade consecutivi falliti: {self.trading_stats['consecutive_failures']}",
                    'timestamp': datetime.now().isoformat()
                })
        
        except Exception as e:
            logger.error(f"Errore controllo performance: {e}")
    
    def _save_monitoring_data(self):
        """Salva dati di monitoraggio"""
        try:
            # Mantieni solo ultimi 1000 record
            if len(self.performance_data) > 1000:
                self.performance_data = self.performance_data[-1000:]
            
            self.performance_data.append({
                'timestamp': datetime.now().isoformat(),
                'system_health': self.system_health.copy(),
                'trading_stats': self.trading_stats.copy()
            })
            
            # Salva su file ogni 10 minuti
            current_time = datetime.now()
            if current_time.minute % 10 == 0:
                self._export_monitoring_data()
                
        except Exception as e:
            logger.error(f"Errore salvataggio dati: {e}")
    
    def _export_monitoring_data(self):
        """Esporta dati di monitoraggio"""
        try:
            os.makedirs('logs/monitoring', exist_ok=True)
            
            export_data = {
                'export_time': datetime.now().isoformat(),
                'system_health': self.system_health,
                'trading_stats': self.trading_stats,
                'recent_alerts': self.security_alerts[-50:],  # Ultimi 50 alert
                'performance_summary': self._calculate_performance_summary()
            }
            
            filename = f'logs/monitoring/monitor_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Errore esportazione: {e}")
    
    def _calculate_performance_summary(self) -> Dict[str, Any]:
        """Calcola summary performance"""
        try:
            if not self.performance_data:
                return {}
            
            # Ultimi 24 ore
            recent_data = [d for d in self.performance_data 
                          if datetime.fromisoformat(d['timestamp']) > datetime.now() - timedelta(hours=24)]
            
            if not recent_data:
                return {}
            
            # Calcola medie
            avg_cpu = sum(d['system_health']['cpu_usage'] for d in recent_data) / len(recent_data)
            avg_memory = sum(d['system_health']['memory_usage'] for d in recent_data) / len(recent_data)
            
            # Success rate
            total_trades = self.trading_stats['total_trades']
            success_rate = 0
            if total_trades > 0:
                success_rate = (self.trading_stats['successful_trades'] / total_trades) * 100
            
            return {
                'period_hours': 24,
                'data_points': len(recent_data),
                'avg_cpu_usage': round(avg_cpu, 2),
                'avg_memory_usage': round(avg_memory, 2),
                'total_trades': total_trades,
                'success_rate_percent': round(success_rate, 2),
                'total_profit_loss': self.trading_stats['total_profit_loss'],
                'uptime_hours': round(self.trading_stats['uptime_seconds'] / 3600, 2),
                'alert_count': len([a for a in self.security_alerts 
                                  if datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=24)])
            }
            
        except Exception as e:
            logger.error(f"Errore calcolo summary: {e}")
            return {}
    
    # Metodi pubblici per aggiornare statistiche
    def record_trade(self, success: bool, volume: float = 0, profit_loss: float = 0):
        """Registra trade eseguito"""
        self.trading_stats['total_trades'] += 1
        self.trading_stats['total_volume'] += volume
        self.trading_stats['total_profit_loss'] += profit_loss
        self.trading_stats['last_trade_time'] = datetime.now().isoformat()
        
        if success:
            self.trading_stats['successful_trades'] += 1
            self.trading_stats['consecutive_failures'] = 0
        else:
            self.trading_stats['failed_trades'] += 1
            self.trading_stats['consecutive_failures'] += 1
    
    def record_api_call(self, success: bool = True):
        """Registra chiamata API"""
        self.trading_stats['api_calls'] += 1
        if not success:
            self.trading_stats['api_errors'] += 1
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Dati per dashboard monitoraggio"""
        return {
            'system_health': self.system_health,
            'trading_stats': self.trading_stats,
            'recent_alerts': self.security_alerts[-10:],  # Ultimi 10 alert
            'performance_summary': self._calculate_performance_summary(),
            'is_monitoring': self.is_monitoring,
            'monitoring_since': self.start_time.isoformat()
        }
    
    def get_security_status(self) -> Dict[str, Any]:
        """Status sicurezza sistema"""
        recent_alerts = [a for a in self.security_alerts 
                        if datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=1)]
        
        critical_alerts = [a for a in recent_alerts if a['level'] == 'CRITICAL']
        high_alerts = [a for a in recent_alerts if a['level'] == 'HIGH']
        
        if critical_alerts:
            status = 'CRITICAL'
        elif high_alerts:
            status = 'WARNING'
        elif recent_alerts:
            status = 'MONITORING'
        else:
            status = 'SECURE'
        
        return {
            'status': status,
            'recent_alerts_count': len(recent_alerts),
            'critical_alerts': len(critical_alerts),
            'high_alerts': len(high_alerts),
            'last_check': datetime.now().isoformat()
        }

# Istanza globale
system_monitor = SystemMonitor()