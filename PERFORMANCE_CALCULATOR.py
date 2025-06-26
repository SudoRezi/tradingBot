"""
Performance Calculator - Calcola requisiti sistema in tempo reale
Analizza il carico attuale e stima consumi per configurazioni diverse
"""

import psutil
import time
import json
from datetime import datetime
from typing import Dict, List, Optional
import threading
import os

class SystemPerformanceCalculator:
    """Calcola performance e requisiti sistema in tempo reale"""
    
    def __init__(self):
        self.monitoring_active = False
        self.performance_data = []
        self.baseline_cpu = 0
        self.baseline_memory = 0
        self.baseline_network = 0
        self.monitoring_thread = None
        
    def get_current_system_specs(self) -> Dict:
        """Ottieni specifiche sistema corrente"""
        try:
            cpu_count = psutil.cpu_count(logical=True)
            cpu_freq = psutil.cpu_freq()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_cores": cpu_count,
                "cpu_freq_mhz": cpu_freq.current if cpu_freq else "Unknown",
                "total_ram_gb": round(memory.total / (1024**3), 1),
                "available_ram_gb": round(memory.available / (1024**3), 1),
                "total_disk_gb": round(disk.total / (1024**3), 1),
                "free_disk_gb": round(disk.free / (1024**3), 1),
                "cpu_usage_percent": psutil.cpu_percent(interval=1),
                "memory_usage_percent": memory.percent,
                "disk_usage_percent": round((disk.used / disk.total) * 100, 1)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def calculate_trading_bot_impact(self, config: Dict) -> Dict:
        """Calcola impatto del trading bot su sistema"""
        
        # Configurazioni predefinite per diversi scenari
        scenarios = {
            "casual": {
                "exchanges": 1,
                "ai_models": 3,
                "hft_enabled": False,
                "data_feeds": 2,
                "cpu_multiplier": 1.0,
                "memory_multiplier": 1.0
            },
            "active": {
                "exchanges": 3,
                "ai_models": 8,
                "hft_enabled": False,
                "data_feeds": 5,
                "cpu_multiplier": 1.5,
                "memory_multiplier": 1.3
            },
            "professional": {
                "exchanges": 5,
                "ai_models": 15,
                "hft_enabled": True,
                "data_feeds": 10,
                "cpu_multiplier": 2.5,
                "memory_multiplier": 2.0
            }
        }
        
        scenario = scenarios.get(config.get("mode", "casual"), scenarios["casual"])
        
        # Calcoli base per componenti
        base_cpu_usage = 5  # 5% CPU base
        base_memory_mb = 500  # 500 MB base
        
        # Calcolo per exchange
        exchange_cpu = scenario["exchanges"] * 3  # 3% per exchange
        exchange_memory = scenario["exchanges"] * 200  # 200 MB per exchange
        
        # Calcolo per AI models
        ai_cpu = scenario["ai_models"] * 2  # 2% per modello
        ai_memory = scenario["ai_models"] * 150  # 150 MB per modello
        
        # Calcolo per data feeds
        feed_cpu = scenario["data_feeds"] * 1.5  # 1.5% per feed
        feed_memory = scenario["data_feeds"] * 100  # 100 MB per feed
        
        # HFT overhead
        hft_cpu = 15 if scenario["hft_enabled"] else 0
        hft_memory = 1000 if scenario["hft_enabled"] else 0
        
        # Totali
        total_cpu = (base_cpu_usage + exchange_cpu + ai_cpu + feed_cpu + hft_cpu) * scenario["cpu_multiplier"]
        total_memory_mb = (base_memory_mb + exchange_memory + ai_memory + feed_memory + hft_memory) * scenario["memory_multiplier"]
        
        # Stima consumo energetico (Watts aggiuntivi)
        power_consumption = {
            "cpu_watts": total_cpu * 2,  # ~2W per % CPU
            "memory_watts": (total_memory_mb / 1024) * 3,  # ~3W per GB RAM
            "total_additional_watts": (total_cpu * 2) + ((total_memory_mb / 1024) * 3)
        }
        
        # Costo elettrico mensile (€0.25/kWh)
        monthly_cost_eur = (power_consumption["total_additional_watts"] * 24 * 30 * 0.25) / 1000
        
        return {
            "scenario": config.get("mode", "casual"),
            "resource_usage": {
                "cpu_percent": round(total_cpu, 1),
                "memory_mb": round(total_memory_mb),
                "memory_gb": round(total_memory_mb / 1024, 2)
            },
            "power_consumption": power_consumption,
            "monthly_cost_eur": round(monthly_cost_eur, 2),
            "components": {
                "base": {"cpu": base_cpu_usage, "memory_mb": base_memory_mb},
                "exchanges": {"cpu": exchange_cpu, "memory_mb": exchange_memory},
                "ai_models": {"cpu": ai_cpu, "memory_mb": ai_memory},
                "data_feeds": {"cpu": feed_cpu, "memory_mb": feed_memory},
                "hft": {"cpu": hft_cpu, "memory_mb": hft_memory}
            }
        }
    
    def check_system_compatibility(self, requirements: Dict) -> Dict:
        """Verifica compatibilità sistema con requisiti"""
        current_specs = self.get_current_system_specs()
        
        if "error" in current_specs:
            return {"compatible": False, "error": current_specs["error"]}
        
        compatibility = {
            "compatible": True,
            "warnings": [],
            "recommendations": []
        }
        
        # Controllo CPU
        if current_specs["cpu_cores"] < requirements.get("min_cpu_cores", 4):
            compatibility["compatible"] = False
            compatibility["warnings"].append(f"CPU insufficiente: {current_specs['cpu_cores']} core (minimo {requirements.get('min_cpu_cores', 4)})")
        
        # Controllo RAM
        required_ram = requirements.get("min_ram_gb", 8)
        if current_specs["total_ram_gb"] < required_ram:
            compatibility["compatible"] = False
            compatibility["warnings"].append(f"RAM insufficiente: {current_specs['total_ram_gb']}GB (minimo {required_ram}GB)")
        
        # Controllo spazio disco
        required_disk = requirements.get("min_disk_gb", 50)
        if current_specs["free_disk_gb"] < required_disk:
            compatibility["warnings"].append(f"Spazio disco limitato: {current_specs['free_disk_gb']}GB liberi (consigliato {required_disk}GB)")
        
        # Raccomandazioni
        if current_specs["cpu_usage_percent"] > 70:
            compatibility["recommendations"].append("CPU utilizzo alto, considera modalità Economy")
        
        if current_specs["memory_usage_percent"] > 80:
            compatibility["recommendations"].append("Memoria limitata, chiudi applicazioni non necessarie")
        
        return compatibility
    
    def get_optimization_suggestions(self, current_specs: Dict, target_performance: str) -> List[str]:
        """Suggerimenti per ottimizzazione sistema"""
        suggestions = []
        
        if target_performance == "professional":
            if current_specs["cpu_cores"] < 8:
                suggestions.append("Upgrade CPU a 8+ core per performance professionali")
            if current_specs["total_ram_gb"] < 32:
                suggestions.append("Upgrade RAM a 32GB+ per AI models avanzati")
        
        elif target_performance == "active":
            if current_specs["cpu_cores"] < 6:
                suggestions.append("Upgrade CPU a 6+ core per trading multi-exchange")
            if current_specs["total_ram_gb"] < 16:
                suggestions.append("Upgrade RAM a 16GB+ per performance ottimali")
        
        # Suggerimenti generali
        if current_specs["cpu_usage_percent"] > 80:
            suggestions.append("Abilita modalità Performance per distribuire carico")
        
        if current_specs["free_disk_gb"] < 20:
            suggestions.append("Libera spazio disco per logging e cache modelli")
        
        return suggestions
    
    def start_real_time_monitoring(self):
        """Avvia monitoraggio in tempo reale"""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def stop_real_time_monitoring(self):
        """Ferma monitoraggio"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
    
    def _monitor_loop(self):
        """Loop monitoraggio sistema"""
        while self.monitoring_active:
            try:
                data_point = {
                    "timestamp": datetime.now().isoformat(),
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "memory_used_mb": psutil.virtual_memory().used / (1024**2),
                    "network_bytes_sent": psutil.net_io_counters().bytes_sent,
                    "network_bytes_recv": psutil.net_io_counters().bytes_recv
                }
                
                self.performance_data.append(data_point)
                
                # Mantieni solo ultimi 1000 punti
                if len(self.performance_data) > 1000:
                    self.performance_data = self.performance_data[-1000:]
                
                time.sleep(5)  # Campiona ogni 5 secondi
                
            except Exception as e:
                print(f"Error in monitoring: {e}")
                time.sleep(10)
    
    def get_monitoring_data(self) -> Dict:
        """Ottieni dati monitoraggio"""
        if not self.performance_data:
            return {"error": "No monitoring data available"}
        
        recent_data = self.performance_data[-60:]  # Ultimi 5 minuti
        
        avg_cpu = sum(d["cpu_percent"] for d in recent_data) / len(recent_data)
        avg_memory = sum(d["memory_percent"] for d in recent_data) / len(recent_data)
        
        return {
            "current": self.performance_data[-1] if self.performance_data else {},
            "averages_5min": {
                "cpu_percent": round(avg_cpu, 1),
                "memory_percent": round(avg_memory, 1)
            },
            "data_points": len(self.performance_data),
            "monitoring_active": self.monitoring_active
        }
    
    def export_performance_report(self, filepath: str = None) -> str:
        """Esporta report performance"""
        if not filepath:
            filepath = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_specs": self.get_current_system_specs(),
            "monitoring_data": self.get_monitoring_data(),
            "scenarios": {
                "casual": self.calculate_trading_bot_impact({"mode": "casual"}),
                "active": self.calculate_trading_bot_impact({"mode": "active"}),
                "professional": self.calculate_trading_bot_impact({"mode": "professional"})
            }
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            return filepath
        except Exception as e:
            return f"Error exporting report: {e}"

def calculate_roi_vs_system_cost():
    """Calcola ROI vs costo sistema"""
    scenarios = {
        "Budget PC (€600)": {
            "hardware_cost": 600,
            "monthly_profit_min": 50,
            "monthly_profit_max": 200,
            "monthly_electricity": 3,
            "payback_months": 3
        },
        "Gaming PC (€1200)": {
            "hardware_cost": 1200,
            "monthly_profit_min": 200,
            "monthly_profit_max": 800,
            "monthly_electricity": 5,
            "payback_months": 2
        },
        "Workstation (€2500)": {
            "hardware_cost": 2500,
            "monthly_profit_min": 800,
            "monthly_profit_max": 3000,
            "monthly_electricity": 12,
            "payback_months": 1
        }
    }
    
    return scenarios

# Test del sistema
if __name__ == "__main__":
    calculator = SystemPerformanceCalculator()
    
    print("=== Sistema Corrente ===")
    specs = calculator.get_current_system_specs()
    for key, value in specs.items():
        print(f"{key}: {value}")
    
    print("\n=== Impatto Trading Bot ===")
    for mode in ["casual", "active", "professional"]:
        impact = calculator.calculate_trading_bot_impact({"mode": mode})
        print(f"\nModalità {mode.upper()}:")
        print(f"  CPU: {impact['resource_usage']['cpu_percent']}%")
        print(f"  RAM: {impact['resource_usage']['memory_gb']} GB")
        print(f"  Costo mensile: €{impact['monthly_cost_eur']}")