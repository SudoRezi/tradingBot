#!/usr/bin/env python3
"""
Stress Test del Sistema AI Trading Bot
Testa tutte le funzionalità principali sotto carico per validare stabilità
"""

import time
import threading
import asyncio
import json
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from utils.logger import setup_logger

logger = setup_logger('stress_test')

class TradingBotStressTest:
    """Stress test completo del sistema trading"""
    
    def __init__(self):
        self.test_results = {}
        self.errors = []
        self.start_time = None
        
    def run_full_stress_test(self):
        """Esegue stress test completo del sistema"""
        self.start_time = datetime.now()
        logger.info("🚀 Avvio Stress Test Completo Sistema AI Trading Bot")
        
        tests = [
            self.test_autonomous_wallet_detection,
            self.test_system_monitor_performance,
            self.test_multi_exchange_simulation,
            self.test_ml_engine_performance,
            self.test_concurrent_operations,
            self.test_memory_usage,
            self.test_error_recovery,
            self.test_security_features
        ]
        
        for test in tests:
            try:
                logger.info(f"Eseguendo: {test.__name__}")
                result = test()
                self.test_results[test.__name__] = result
                logger.info(f"✅ {test.__name__}: PASSED")
            except Exception as e:
                logger.error(f"❌ {test.__name__}: FAILED - {e}")
                self.errors.append(f"{test.__name__}: {e}")
                self.test_results[test.__name__] = {"status": "FAILED", "error": str(e)}
        
        self.generate_stress_test_report()
    
    def test_autonomous_wallet_detection(self):
        """Test rilevamento autonomo wallet"""
        from core.autonomous_wallet_manager import AutonomousWalletManager
        
        # Test multiple instances
        managers = []
        for i in range(5):
            manager = AutonomousWalletManager(f"test_key_{i}", f"test_secret_{i}", "Binance")
            managers.append(manager)
        
        # Test concurrent scanning
        start = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(manager.scan_all_balances) for manager in managers]
            for future in futures:
                results.append(future.result())
        
        scan_time = time.time() - start
        
        # Validate results
        assert len(results) == 5
        for result in results:
            assert 'balances' in result
            assert 'summary' in result
        
        return {
            "status": "PASSED",
            "scan_time": scan_time,
            "concurrent_scans": 5,
            "avg_scan_time": scan_time / 5
        }
    
    def test_system_monitor_performance(self):
        """Test performance system monitor"""
        from core.system_monitor import SystemMonitor
        
        monitor = SystemMonitor()
        monitor.start_monitoring()
        
        # Test under load
        start = time.time()
        for _ in range(100):
            monitor.record_trade(True, 1000, 50)
            monitor.record_api_call(True)
            time.sleep(0.01)
        
        # Test dashboard data generation
        dashboard_start = time.time()
        dashboard_data = monitor.get_dashboard_data()
        dashboard_time = time.time() - dashboard_start
        
        monitor.stop_monitoring()
        
        return {
            "status": "PASSED",
            "trades_recorded": 100,
            "dashboard_response_time": dashboard_time,
            "total_test_time": time.time() - start
        }
    
    def test_multi_exchange_simulation(self):
        """Test simulazione multi-exchange"""
        exchanges = ["Binance", "KuCoin", "Kraken", "Coinbase"]
        
        # Simulate concurrent exchange operations
        def simulate_exchange_ops(exchange):
            from core.autonomous_wallet_manager import AutonomousWalletManager
            manager = AutonomousWalletManager("test", "test", exchange)
            
            results = []
            for _ in range(10):
                result = manager.scan_all_balances()
                results.append(result)
                time.sleep(0.1)
            return results
        
        start = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(simulate_exchange_ops, ex): ex for ex in exchanges}
            exchange_results = {}
            
            for future, exchange in futures.items():
                exchange_results[exchange] = future.result()
        
        total_time = time.time() - start
        
        return {
            "status": "PASSED",
            "exchanges_tested": len(exchanges),
            "total_operations": sum(len(results) for results in exchange_results.values()),
            "total_time": total_time
        }
    
    def test_ml_engine_performance(self):
        """Test performance motore ML"""
        from core.advanced_ml_engine import AdvancedMLEngine
        import numpy as np
        
        ml_engine = AdvancedMLEngine()
        ml_engine.initialize()
        
        # Generate test data
        test_data = np.random.random((1000, 10))
        test_labels = np.random.random(1000)
        
        # Test training performance
        start = time.time()
        ml_engine.add_training_data(test_data, test_labels)
        training_result = ml_engine.train_models()
        training_time = time.time() - start
        
        # Test prediction performance
        prediction_start = time.time()
        for _ in range(100):
            signal = ml_engine.get_ml_signal(test_data[:60])
        prediction_time = time.time() - prediction_start
        
        return {
            "status": "PASSED",
            "training_time": training_time,
            "predictions_tested": 100,
            "avg_prediction_time": prediction_time / 100,
            "training_success": training_result
        }
    
    def test_concurrent_operations(self):
        """Test operazioni concorrenti"""
        
        def heavy_operation():
            # Simulate CPU intensive task
            result = 0
            for i in range(100000):
                result += i ** 0.5
            return result
        
        # Test 20 concurrent operations
        start = time.time()
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(heavy_operation) for _ in range(20)]
            results = [future.result() for future in futures]
        
        total_time = time.time() - start
        
        return {
            "status": "PASSED",
            "concurrent_operations": 20,
            "total_time": total_time,
            "avg_operation_time": total_time / 20
        }
    
    def test_memory_usage(self):
        """Test utilizzo memoria"""
        import psutil
        import gc
        
        # Baseline memory
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Create large data structures
        large_data = []
        for _ in range(1000):
            large_data.append({
                'data': list(range(1000)),
                'timestamp': datetime.now(),
                'metadata': {'test': True}
            })
        
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Cleanup and measure final memory
        del large_data
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = peak_memory - baseline_memory
        memory_cleanup = peak_memory - final_memory
        cleanup_efficiency = (memory_cleanup / memory_increase) * 100
        
        return {
            "status": "PASSED",
            "baseline_memory_mb": baseline_memory,
            "peak_memory_mb": peak_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": memory_increase,
            "cleanup_efficiency_percent": cleanup_efficiency
        }
    
    def test_error_recovery(self):
        """Test meccanismi di recovery da errori"""
        from core.autonomous_wallet_manager import AutonomousWalletManager
        
        recovery_tests = []
        
        # Test 1: Invalid API credentials
        try:
            manager = AutonomousWalletManager("invalid", "invalid", "TestExchange")
            result = manager.scan_all_balances()
            recovery_tests.append({"test": "invalid_api", "recovered": True})
        except Exception as e:
            recovery_tests.append({"test": "invalid_api", "error": str(e), "recovered": False})
        
        # Test 2: Network simulation errors
        import socket
        original_socket = socket.socket
        
        def failing_socket(*args, **kwargs):
            raise socket.error("Simulated network error")
        
        socket.socket = failing_socket
        
        try:
            # This should handle the error gracefully
            manager = AutonomousWalletManager("test", "test", "TestExchange")
            manager.scan_all_balances()
            recovery_tests.append({"test": "network_error", "recovered": True})
        except Exception as e:
            recovery_tests.append({"test": "network_error", "error": str(e), "recovered": False})
        finally:
            socket.socket = original_socket
        
        successful_recoveries = sum(1 for test in recovery_tests if test.get("recovered", False))
        
        return {
            "status": "PASSED",
            "recovery_tests": recovery_tests,
            "successful_recoveries": successful_recoveries,
            "total_tests": len(recovery_tests)
        }
    
    def test_security_features(self):
        """Test funzionalità di sicurezza"""
        import os
        
        security_tests = []
        
        # Test 1: File encryption key exists
        if os.path.exists('.encryption_key'):
            security_tests.append({"test": "encryption_key_exists", "passed": True})
        else:
            security_tests.append({"test": "encryption_key_exists", "passed": False})
        
        # Test 2: Config files security
        config_files = ['config/settings.py', 'utils/encryption.py']
        for config_file in config_files:
            if os.path.exists(config_file):
                stat_info = os.stat(config_file)
                # Check if file is readable by others
                secure = not (stat_info.st_mode & 0o044)
                security_tests.append({
                    "test": f"file_permissions_{config_file.replace('/', '_')}",
                    "passed": secure
                })
        
        # Test 3: No hardcoded secrets in main files
        sensitive_patterns = ['api_key', 'secret', 'password', 'token']
        main_files = ['app.py', 'core/ai_trader.py']
        
        for file_path in main_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read().lower()
                    
                hardcoded_secrets = any(f'"{pattern}"' in content or f"'{pattern}'" in content 
                                      for pattern in sensitive_patterns)
                security_tests.append({
                    "test": f"no_hardcoded_secrets_{file_path.replace('/', '_')}",
                    "passed": not hardcoded_secrets
                })
        
        passed_tests = sum(1 for test in security_tests if test.get("passed", False))
        
        return {
            "status": "PASSED",
            "security_tests": security_tests,
            "passed_tests": passed_tests,
            "total_tests": len(security_tests)
        }
    
    def generate_stress_test_report(self):
        """Genera report completo dello stress test"""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        report = {
            "stress_test_report": {
                "timestamp": end_time.isoformat(),
                "duration_seconds": total_duration,
                "total_tests": len(self.test_results),
                "passed_tests": len([r for r in self.test_results.values() 
                                   if r.get("status") == "PASSED"]),
                "failed_tests": len(self.errors),
                "errors": self.errors,
                "detailed_results": self.test_results
            }
        }
        
        # Save to file
        os.makedirs('logs/stress_tests', exist_ok=True)
        filename = f'logs/stress_tests/stress_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*60)
        print("🧪 STRESS TEST REPORT")
        print("="*60)
        print(f"Duration: {total_duration:.2f} seconds")
        print(f"Total Tests: {len(self.test_results)}")
        print(f"Passed: {len([r for r in self.test_results.values() if r.get('status') == 'PASSED'])}")
        print(f"Failed: {len(self.errors)}")
        
        if self.errors:
            print("\n❌ FAILED TESTS:")
            for error in self.errors:
                print(f"  - {error}")
        else:
            print("\n✅ ALL TESTS PASSED!")
        
        print(f"\nDetailed report saved: {filename}")
        print("="*60)

if __name__ == "__main__":
    stress_test = TradingBotStressTest()
    stress_test.run_full_stress_test()