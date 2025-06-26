#!/usr/bin/env python3
"""
AI Crypto Trading Bot - Installation Health Check & Diagnostic Tool
Comprehensive system verification and performance testing
"""

import os
import sys
import platform
import subprocess
import importlib
import sqlite3
import json
import time
from datetime import datetime
from pathlib import Path
import psutil

class InstallationChecker:
    """Comprehensive installation and health checker"""
    
    def __init__(self):
        self.results = {
            'system_info': {},
            'python_environment': {},
            'dependencies': {},
            'application_files': {},
            'configuration': {},
            'database': {},
            'performance': {},
            'security': {},
            'network': {},
            'recommendations': []
        }
        self.errors = []
        self.warnings = []
        
    def print_header(self, title):
        """Print formatted section header"""
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print('='*60)
        
    def print_status(self, item, status, details=None):
        """Print status with color coding"""
        status_symbols = {
            'pass': '✅',
            'fail': '❌', 
            'warning': '⚠️',
            'info': 'ℹ️'
        }
        
        symbol = status_symbols.get(status, '?')
        print(f"{symbol} {item}")
        if details:
            print(f"   {details}")
            
    def check_system_info(self):
        """Check system information and requirements"""
        self.print_header("SYSTEM INFORMATION")
        
        try:
            # Basic system info
            system_info = {
                'platform': platform.system(),
                'architecture': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'hostname': platform.node()
            }
            
            self.results['system_info'] = system_info
            
            # System resources
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            cpu_count = psutil.cpu_count()
            
            self.print_status("Operating System", "info", f"{system_info['platform']} {platform.release()}")
            self.print_status("Architecture", "info", system_info['architecture'])
            self.print_status("CPU Cores", "info", f"{cpu_count} cores")
            self.print_status("Total Memory", "info", f"{memory.total / (1024**3):.1f} GB")
            self.print_status("Available Memory", "pass" if memory.percent < 80 else "warning", 
                            f"{memory.available / (1024**3):.1f} GB ({100-memory.percent:.1f}% free)")
            self.print_status("Disk Space", "pass" if disk.percent < 90 else "warning",
                            f"{disk.free / (1024**3):.1f} GB free ({100-disk.percent:.1f}%)")
            
            # Minimum requirements check
            if memory.total < 2 * (1024**3):  # 2GB minimum
                self.warnings.append("Low system memory (< 2GB). Performance may be affected.")
                
            if disk.free < 1 * (1024**3):  # 1GB minimum
                self.warnings.append("Low disk space (< 1GB). Consider cleaning up files.")
                
        except Exception as e:
            self.errors.append(f"System info check failed: {e}")
            self.print_status("System Information", "fail", str(e))
            
    def check_python_environment(self):
        """Check Python installation and virtual environment"""
        self.print_header("PYTHON ENVIRONMENT")
        
        try:
            # Python version
            python_version = sys.version_info
            if python_version >= (3, 8):
                self.print_status("Python Version", "pass", f"{python_version.major}.{python_version.minor}.{python_version.micro}")
            else:
                self.print_status("Python Version", "fail", f"Version {python_version.major}.{python_version.minor} is too old (3.8+ required)")
                self.errors.append("Python version too old")
                
            # Virtual environment
            in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
            venv_path = os.environ.get('VIRTUAL_ENV', 'Not in virtual environment')
            
            if in_venv:
                self.print_status("Virtual Environment", "pass", venv_path)
            else:
                self.print_status("Virtual Environment", "warning", "Not using virtual environment")
                self.warnings.append("Not using virtual environment. Consider using venv for isolation.")
                
            # pip installation
            try:
                import pip
                self.print_status("pip", "pass", f"Version {pip.__version__}")
            except ImportError:
                self.print_status("pip", "fail", "pip not available")
                self.errors.append("pip not available")
                
            self.results['python_environment'] = {
                'version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                'virtual_env': in_venv,
                'venv_path': venv_path,
                'executable': sys.executable
            }
            
        except Exception as e:
            self.errors.append(f"Python environment check failed: {e}")
            self.print_status("Python Environment", "fail", str(e))
            
    def check_dependencies(self):
        """Check required Python packages"""
        self.print_header("PYTHON DEPENDENCIES")
        
        required_packages = {
            'streamlit': '1.28.0',
            'pandas': '2.0.0',
            'numpy': '1.24.0',
            'plotly': '5.15.0',
            'scikit-learn': '1.3.0',
            'requests': '2.31.0',
            'cryptography': '41.0.0',
            'apscheduler': '3.10.0',
            'yfinance': '0.2.0',
            'beautifulsoup4': '4.12.0',
            'feedparser': '6.0.0',
            'psutil': '5.9.0',
            'joblib': '1.3.0',
            'scipy': '1.11.0',
            'trafilatura': '1.6.0',
            'sendgrid': '6.10.0'
        }
        
        installed_packages = {}
        missing_packages = []
        
        for package, min_version in required_packages.items():
            try:
                module = importlib.import_module(package.replace('-', '_'))
                version = getattr(module, '__version__', 'Unknown')
                installed_packages[package] = version
                self.print_status(f"{package}", "pass", f"Version {version}")
            except ImportError:
                missing_packages.append(package)
                self.print_status(f"{package}", "fail", "Not installed")
                
        self.results['dependencies'] = {
            'installed': installed_packages,
            'missing': missing_packages,
            'total_required': len(required_packages),
            'total_installed': len(installed_packages)
        }
        
        if missing_packages:
            self.errors.append(f"Missing packages: {', '.join(missing_packages)}")
            
    def check_application_files(self):
        """Check core application files"""
        self.print_header("APPLICATION FILES")
        
        core_files = [
            'advanced_ai_system.py',
            'advanced_quant_engine.py',
            'advanced_order_system.py',
            'arctic_data_manager.py',
            'smart_performance_optimizer.py',
            'autonomous_ai_trader.py',
            'real_ai_integration.py',
            'multilayer_api_protection.py',
            'system_health_check.py',
            'PERFORMANCE_CALCULATOR.py'
        ]
        
        required_directories = [
            'config', 'data', 'logs', 'backups', 'models'
        ]
        
        existing_files = []
        missing_files = []
        
        # Check core files
        for file in core_files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                existing_files.append(file)
                self.print_status(f"Core file: {file}", "pass", f"{size:,} bytes")
            else:
                missing_files.append(file)
                self.print_status(f"Core file: {file}", "fail", "Missing")
                
        # Check directories
        existing_dirs = []
        missing_dirs = []
        
        for directory in required_directories:
            if os.path.exists(directory) and os.path.isdir(directory):
                existing_dirs.append(directory)
                self.print_status(f"Directory: {directory}", "pass", "Exists")
            else:
                missing_dirs.append(directory)
                self.print_status(f"Directory: {directory}", "warning", "Missing (will be created)")
                
        self.results['application_files'] = {
            'core_files': {
                'existing': existing_files,
                'missing': missing_files,
                'total_required': len(core_files)
            },
            'directories': {
                'existing': existing_dirs,
                'missing': missing_dirs,
                'total_required': len(required_directories)
            }
        }
        
        if missing_files:
            self.errors.append(f"Missing core files: {', '.join(missing_files)}")
            
    def check_configuration(self):
        """Check configuration files"""
        self.print_header("CONFIGURATION")
        
        config_files = {
            '.env': 'Environment variables',
            'config/config.yaml': 'Main configuration'
        }
        
        config_status = {}
        
        for file_path, description in config_files.items():
            if os.path.exists(file_path):
                try:
                    size = os.path.getsize(file_path)
                    if file_path.endswith('.env'):
                        # Check .env file content
                        with open(file_path, 'r') as f:
                            content = f.read()
                            lines = [line for line in content.split('\n') if line.strip() and not line.startswith('#')]
                            uncommented_lines = len(lines)
                        self.print_status(f"{description}", "pass", f"{size} bytes, {uncommented_lines} active settings")
                    else:
                        self.print_status(f"{description}", "pass", f"{size} bytes")
                    config_status[file_path] = True
                except Exception as e:
                    self.print_status(f"{description}", "warning", f"Exists but error reading: {e}")
                    config_status[file_path] = False
            else:
                self.print_status(f"{description}", "warning", "Not found (will use defaults)")
                config_status[file_path] = False
                
        self.results['configuration'] = config_status
        
    def check_database(self):
        """Check database connectivity and integrity"""
        self.print_header("DATABASE")
        
        try:
            # Check if AI models database exists
            ai_db_path = 'ai_models.db'
            if os.path.exists(ai_db_path):
                conn = sqlite3.connect(ai_db_path)
                cursor = conn.cursor()
                
                # Check tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                # Check AI models count
                try:
                    cursor.execute("SELECT COUNT(*) FROM ai_models")
                    models_count = cursor.fetchone()[0]
                    self.print_status("AI Models Database", "pass", f"{len(tables)} tables, {models_count} models")
                except:
                    self.print_status("AI Models Database", "warning", f"{len(tables)} tables, models table empty")
                    
                conn.close()
                
                self.results['database']['ai_models'] = {
                    'exists': True,
                    'tables': len(tables),
                    'models_count': models_count if 'models_count' in locals() else 0
                }
            else:
                self.print_status("AI Models Database", "info", "Will be created on first run")
                self.results['database']['ai_models'] = {'exists': False}
                
            # Check data directory and SQLite files
            data_dir = 'data'
            if os.path.exists(data_dir):
                sqlite_files = list(Path(data_dir).glob('*.db'))
                self.print_status("Data Directory", "pass", f"{len(sqlite_files)} database files")
                self.results['database']['data_files'] = len(sqlite_files)
            else:
                self.print_status("Data Directory", "info", "Will be created on first run")
                self.results['database']['data_files'] = 0
                
        except Exception as e:
            self.errors.append(f"Database check failed: {e}")
            self.print_status("Database Check", "fail", str(e))
            
    def check_performance(self):
        """Check system performance and optimization"""
        self.print_header("PERFORMANCE")
        
        try:
            # CPU performance test
            start_time = time.time()
            # Simple computation test
            result = sum(i**2 for i in range(10000))
            cpu_test_time = time.time() - start_time
            
            self.print_status("CPU Performance Test", "pass", f"{cpu_test_time:.3f} seconds")
            
            # Memory test
            try:
                import numpy as np
                start_time = time.time()
                arr = np.random.random((1000, 1000))
                result = np.sum(arr)
                memory_test_time = time.time() - start_time
                self.print_status("Memory Performance Test", "pass", f"{memory_test_time:.3f} seconds")
            except ImportError:
                self.print_status("Memory Performance Test", "warning", "NumPy not available")
                
            # Disk I/O test
            start_time = time.time()
            test_file = 'temp_test_file.tmp'
            with open(test_file, 'w') as f:
                f.write('test' * 10000)
            with open(test_file, 'r') as f:
                content = f.read()
            os.remove(test_file)
            disk_test_time = time.time() - start_time
            
            self.print_status("Disk I/O Test", "pass", f"{disk_test_time:.3f} seconds")
            
            self.results['performance'] = {
                'cpu_test_time': cpu_test_time,
                'memory_test_time': memory_test_time if 'memory_test_time' in locals() else None,
                'disk_test_time': disk_test_time
            }
            
        except Exception as e:
            self.errors.append(f"Performance test failed: {e}")
            self.print_status("Performance Test", "fail", str(e))
            
    def check_security(self):
        """Check security configuration"""
        self.print_header("SECURITY")
        
        try:
            # Check encryption key
            if os.path.exists('.encryption_key'):
                key_size = os.path.getsize('.encryption_key')
                self.print_status("Encryption Key", "pass", f"{key_size} bytes")
                self.results['security']['encryption_key'] = True
            else:
                self.print_status("Encryption Key", "warning", "Will be generated on first run")
                self.results['security']['encryption_key'] = False
                
            # Check file permissions (Unix-like systems)
            if platform.system() != 'Windows':
                try:
                    import stat
                    if os.path.exists('.env'):
                        env_perms = oct(os.stat('.env').st_mode)[-3:]
                        if env_perms in ['600', '644']:
                            self.print_status("Environment File Permissions", "pass", f"Permissions: {env_perms}")
                        else:
                            self.print_status("Environment File Permissions", "warning", f"Permissions: {env_perms} (consider 600)")
                except:
                    self.print_status("File Permissions", "info", "Could not check permissions")
            else:
                self.print_status("File Permissions", "info", "Windows - using NTFS permissions")
                
        except Exception as e:
            self.errors.append(f"Security check failed: {e}")
            self.print_status("Security Check", "fail", str(e))
            
    def check_network(self):
        """Check network connectivity"""
        self.print_header("NETWORK CONNECTIVITY")
        
        test_urls = {
            'GitHub API': 'https://api.github.com',
            'HuggingFace API': 'https://huggingface.co/api',
            'Alpha Vantage': 'https://www.alphavantage.co',
            'News API': 'https://newsapi.org',
            'Binance API': 'https://api.binance.com/api/v3/ping'
        }
        
        import requests
        
        for service, url in test_urls.items():
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    self.print_status(f"{service}", "pass", f"HTTP {response.status_code}")
                else:
                    self.print_status(f"{service}", "warning", f"HTTP {response.status_code}")
            except requests.RequestException as e:
                self.print_status(f"{service}", "fail", f"Connection failed: {str(e)[:50]}...")
                
        self.results['network'] = {'tested': len(test_urls)}
        
    def functional_test(self):
        """Run functional tests of core components"""
        self.print_header("FUNCTIONAL TESTS")
        
        try:
            # Test core module imports
            sys.path.insert(0, '.')
            
            tests = [
                ('advanced_ai_system', 'AdvancedAITradingSystem'),
                ('advanced_quant_engine', 'get_quant_module_manager'),
                ('advanced_order_system', 'get_order_system'),
                ('arctic_data_manager', 'get_arctic_manager')
            ]
            
            for module_name, class_or_function in tests:
                try:
                    module = importlib.import_module(module_name)
                    obj = getattr(module, class_or_function)
                    self.print_status(f"Module: {module_name}", "pass", f"Imported {class_or_function}")
                except Exception as e:
                    self.print_status(f"Module: {module_name}", "fail", f"Import failed: {str(e)[:50]}...")
                    
            # Test basic functionality
            try:
                from advanced_order_system import get_order_system, OrderSide
                order_system = get_order_system()
                test_order = order_system.create_limit_order('BTC/USD', OrderSide.BUY, 0.001, 45000)
                order_system.cancel_order(test_order)
                self.print_status("Order System Test", "pass", "Create/cancel order successful")
            except Exception as e:
                self.print_status("Order System Test", "fail", f"Test failed: {str(e)[:50]}...")
                
        except Exception as e:
            self.errors.append(f"Functional test failed: {e}")
            self.print_status("Functional Tests", "fail", str(e))
            
    def generate_recommendations(self):
        """Generate recommendations based on findings"""
        self.print_header("RECOMMENDATIONS")
        
        # Performance recommendations
        memory = psutil.virtual_memory()
        if memory.total < 4 * (1024**3):  # Less than 4GB
            self.recommendations.append("Consider upgrading to 8GB+ RAM for optimal performance")
            
        if psutil.cpu_count() < 4:
            self.recommendations.append("Consider using a multi-core processor for better performance")
            
        # Configuration recommendations
        if not os.path.exists('.env'):
            self.recommendations.append("Create .env file with API keys for live trading")
            
        if not os.path.exists('config/config.yaml'):
            self.recommendations.append("Create config.yaml for customized settings")
            
        # Security recommendations
        if not os.path.exists('.encryption_key'):
            self.recommendations.append("Encryption key will be auto-generated on first run")
            
        # Missing packages recommendations
        if self.results['dependencies'].get('missing'):
            missing = self.results['dependencies']['missing']
            self.recommendations.append(f"Install missing packages: pip install {' '.join(missing)}")
            
        # Display recommendations
        if self.recommendations:
            for i, rec in enumerate(self.recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("✅ No specific recommendations - system looks good!")
            
        self.results['recommendations'] = self.recommendations
        
    def generate_summary(self):
        """Generate installation summary"""
        self.print_header("INSTALLATION SUMMARY")
        
        total_deps = self.results['dependencies'].get('total_required', 0)
        installed_deps = self.results['dependencies'].get('total_installed', 0)
        
        total_files = self.results['application_files']['core_files'].get('total_required', 0)
        existing_files = len(self.results['application_files']['core_files'].get('existing', []))
        
        print(f"📊 System Overview:")
        print(f"   Platform: {self.results['system_info'].get('platform', 'Unknown')}")
        print(f"   Python: {self.results['python_environment'].get('version', 'Unknown')}")
        print(f"   Dependencies: {installed_deps}/{total_deps} installed")
        print(f"   Core Files: {existing_files}/{total_files} present")
        print(f"   Errors: {len(self.errors)}")
        print(f"   Warnings: {len(self.warnings)}")
        
        # Overall status
        if len(self.errors) == 0 and len(self.warnings) <= 2:
            print(f"\n🎉 Installation Status: EXCELLENT")
            print("   System is ready for trading operations!")
        elif len(self.errors) == 0:
            print(f"\n✅ Installation Status: GOOD")
            print("   System is functional with minor optimizations needed")
        elif len(self.errors) <= 2:
            print(f"\n⚠️ Installation Status: NEEDS ATTENTION")
            print("   System has issues that should be resolved")
        else:
            print(f"\n❌ Installation Status: REQUIRES FIXES")
            print("   System has serious issues that must be resolved")
            
        # Next steps
        print(f"\n📋 Next Steps:")
        if len(self.errors) > 0:
            print("   1. Fix errors listed above")
            print("   2. Re-run this diagnostic script")
        else:
            print("   1. Configure API keys in .env file")
            print("   2. Start the trading bot: python advanced_ai_system.py")
            print("   3. Access web interface at http://localhost:5000")
            
    def save_report(self):
        """Save detailed report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"installation_report_{timestamp}.json"
        
        full_report = {
            'timestamp': timestamp,
            'summary': {
                'errors': len(self.errors),
                'warnings': len(self.warnings),
                'recommendations': len(self.recommendations)
            },
            'errors': self.errors,
            'warnings': self.warnings,
            'results': self.results
        }
        
        try:
            with open(report_file, 'w') as f:
                json.dump(full_report, f, indent=2, default=str)
            print(f"\n📄 Detailed report saved: {report_file}")
        except Exception as e:
            print(f"\n❌ Failed to save report: {e}")
            
    def run_all_checks(self):
        """Run all diagnostic checks"""
        print("🔍 AI Crypto Trading Bot - Installation Diagnostic")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all checks
        self.check_system_info()
        self.check_python_environment()
        self.check_dependencies()
        self.check_application_files()
        self.check_configuration()
        self.check_database()
        self.check_performance()
        self.check_security()
        self.check_network()
        self.functional_test()
        
        # Generate results
        self.generate_recommendations()
        self.generate_summary()
        self.save_report()

def main():
    """Main function"""
    checker = InstallationChecker()
    checker.run_all_checks()

if __name__ == "__main__":
    main()