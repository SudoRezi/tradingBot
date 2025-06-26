#!/usr/bin/env python3
"""
QuantConnect Launcher
Lancia backtests tramite QuantConnect CLI e monitora esecuzione
"""

import os
import json
import subprocess
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable
from pathlib import Path

class QuantConnectLauncher:
    """Gestisce l'esecuzione di backtests QuantConnect"""
    
    def __init__(self, lean_cli_path="lean", project_path=".", results_path="results"):
        self.lean_cli_path = lean_cli_path
        self.project_path = Path(project_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(exist_ok=True)
        
        # Status tracking
        self.running_backtests = {}
        self.completed_backtests = {}
        
    def check_lean_installation(self) -> Dict[str, bool]:
        """Verifica installazione LEAN CLI e status connessione"""
        
        status = {
            "lean_installed": False,
            "lean_version": None,
            "docker_available": False,
            "logged_in": False,
            "local_data_available": False,
            "cloud_features_available": False
        }
        
        try:
            # Test LEAN CLI
            result = subprocess.run([self.lean_cli_path, "--version"], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                status["lean_installed"] = True
                status["lean_version"] = result.stdout.strip()
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            status["lean_installed"] = False
        
        try:
            # Test Docker
            subprocess.run(["docker", "--version"], 
                          capture_output=True, timeout=5)
            status["docker_available"] = True
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            status["docker_available"] = False
        
        try:
            # Test login status - prova diversi comandi
            test_commands = [
                [self.lean_cli_path, "cloud", "status"],
                [self.lean_cli_path, "login", "--check"],
                [self.lean_cli_path, "whoami"]
            ]
            
            for cmd in test_commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        output_lower = result.stdout.lower()
                        if any(keyword in output_lower for keyword in ["logged in", "authenticated", "connected", "organization"]):
                            status["logged_in"] = True
                            status["cloud_features_available"] = True
                            break
                except:
                    continue
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            status["logged_in"] = False
        
        # Check local data availability
        try:
            import os
            data_paths = ["./data", "./Data", "../data"]
            
            for path in data_paths:
                if os.path.exists(path) and os.listdir(path):
                    status["local_data_available"] = True
                    break
                    
        except Exception:
            status["local_data_available"] = False
        
        return status
    
    def install_lean_cli(self) -> Dict[str, str]:
        """Installa LEAN CLI automaticamente"""
        
        result = {
            "success": False,
            "message": "",
            "version": None
        }
        
        try:
            # Prova installazione via pip
            install_result = subprocess.run(
                ["pip", "install", "lean"], 
                capture_output=True, 
                text=True,
                timeout=120
            )
            
            if install_result.returncode == 0:
                # Verifica installazione
                version_result = subprocess.run(
                    ["lean", "--version"], 
                    capture_output=True, 
                    text=True,
                    timeout=10
                )
                
                if version_result.returncode == 0:
                    result["success"] = True
                    result["message"] = "LEAN CLI installed successfully"
                    result["version"] = version_result.stdout.strip()
                else:
                    result["message"] = "Installation completed but verification failed"
            else:
                result["message"] = f"Installation failed: {install_result.stderr}"
                
        except subprocess.TimeoutExpired:
            result["message"] = "Installation timed out"
        except Exception as e:
            result["message"] = f"Installation error: {str(e)}"
        
        return result
    
    def login_to_quantconnect(self, method: str = "interactive", org_id: str = None, api_token: str = None) -> Dict[str, str]:
        """Login a QuantConnect"""
        
        result = {
            "success": False,
            "message": "",
            "method_used": method
        }
        
        try:
            if method == "interactive":
                # Login interattivo
                login_result = subprocess.run(
                    [self.lean_cli_path, "login"], 
                    capture_output=True, 
                    text=True,
                    timeout=60
                )
                
            elif method == "api" and org_id and api_token:
                # Login con API credentials
                import os
                env = os.environ.copy()
                env['QC_ORGANIZATION_ID'] = org_id
                env['QC_API_TOKEN'] = api_token
                
                login_result = subprocess.run(
                    [self.lean_cli_path, "login", "--api"], 
                    capture_output=True, 
                    text=True,
                    timeout=30,
                    env=env
                )
                
            else:
                result["message"] = "Invalid login method or missing credentials"
                return result
            
            if login_result.returncode == 0:
                result["success"] = True
                result["message"] = "Successfully logged in to QuantConnect"
            else:
                result["message"] = f"Login failed: {login_result.stderr}"
                
        except subprocess.TimeoutExpired:
            result["message"] = "Login timed out"
        except Exception as e:
            result["message"] = f"Login error: {str(e)}"
        
        return result
    
    def logout_from_quantconnect(self) -> Dict[str, str]:
        """Logout da QuantConnect"""
        
        result = {
            "success": False,
            "message": ""
        }
        
        try:
            logout_result = subprocess.run(
                [self.lean_cli_path, "logout"], 
                capture_output=True, 
                text=True,
                timeout=10
            )
            
            # Logout può fallire ma non è critico
            result["success"] = True
            result["message"] = "Logged out from QuantConnect"
            
        except Exception as e:
            result["message"] = f"Logout error: {str(e)}"
        
        return result
    
    def setup_lean_project(self, strategy_file: str, project_name: str) -> Dict[str, str]:
        """Setup progetto LEAN per backtest"""
        
        project_dir = self.project_path / project_name
        project_dir.mkdir(exist_ok=True)
        
        # Copia strategia nel progetto
        strategy_path = project_dir / "main.py"
        
        with open(strategy_file, 'r', encoding='utf-8') as f:
            strategy_content = f.read()
        
        with open(strategy_path, 'w', encoding='utf-8') as f:
            f.write(strategy_content)
        
        # Crea config.json per il progetto
        config = {
            "algorithm-type-name": self._extract_class_name(strategy_content),
            "algorithm-language": "Python",
            "algorithm-location": "main.py",
            "parameters": {}
        }
        
        config_path = project_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        return {
            "project_dir": str(project_dir),
            "strategy_path": str(strategy_path),
            "config_path": str(config_path)
        }
    
    def _extract_class_name(self, strategy_content: str) -> str:
        """Estrae nome classe dalla strategia"""
        
        import re
        match = re.search(r'class\s+(\w+)\s*\(QCAlgorithm\)', strategy_content)
        
        return match.group(1) if match else "AITradingStrategy"
    
    def launch_backtest(self, 
                       strategy_file: str, 
                       strategy_name: str,
                       start_date: str = "2023-01-01",
                       end_date: str = "2024-01-01",
                       initial_cash: int = 100000,
                       callback: Optional[Callable] = None) -> str:
        """Lancia backtest asincrono"""
        
        backtest_id = f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Setup progetto
        project_setup = self.setup_lean_project(strategy_file, backtest_id)
        
        # Parametri backtest
        backtest_params = {
            "project_dir": project_setup["project_dir"],
            "start_date": start_date,
            "end_date": end_date,
            "initial_cash": initial_cash,
            "callback": callback
        }
        
        # Avvia backtest in thread separato
        thread = threading.Thread(
            target=self._run_backtest_thread,
            args=(backtest_id, backtest_params)
        )
        
        thread.start()
        
        self.running_backtests[backtest_id] = {
            "thread": thread,
            "start_time": datetime.now(),
            "status": "running",
            "params": backtest_params
        }
        
        return backtest_id
    
    def _run_backtest_thread(self, backtest_id: str, params: Dict):
        """Thread per esecuzione backtest"""
        
        try:
            result = self._execute_lean_backtest(backtest_id, params)
            
            self.running_backtests[backtest_id]["status"] = "completed"
            self.completed_backtests[backtest_id] = result
            
            # Callback se fornito
            if params.get("callback"):
                params["callback"](backtest_id, result)
                
        except Exception as e:
            self.running_backtests[backtest_id]["status"] = "failed"
            self.running_backtests[backtest_id]["error"] = str(e)
    
    def _execute_lean_backtest(self, backtest_id: str, params: Dict) -> Dict:
        """Esegue il backtest LEAN"""
        
        project_dir = params["project_dir"]
        start_time = datetime.now()
        
        # Costruisci comando LEAN
        cmd = [
            self.lean_cli_path,
            "backtest",
            project_dir,
            "--start", params["start_date"],
            "--end", params["end_date"],
            "--cash", str(params["initial_cash"])
        ]
        
        # Esegui backtest
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(self.project_path)
        )
        
        stdout, stderr = process.communicate()
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Risultati
        result = {
            "backtest_id": backtest_id,
            "start_time": start_time.isoformat(),
            "execution_time": execution_time,
            "return_code": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "success": process.returncode == 0
        }
        
        if result["success"]:
            # Cerca file di output
            result["output_files"] = self._find_output_files(project_dir)
            result["results_summary"] = self._parse_lean_output(stdout)
        
        # Salva risultati
        self._save_backtest_results(backtest_id, result)
        
        return result
    
    def _find_output_files(self, project_dir: str) -> List[str]:
        """Trova file di output del backtest"""
        
        output_files = []
        project_path = Path(project_dir)
        
        # Pattern comuni per output LEAN
        patterns = [
            "*.json",
            "*.csv",
            "*.html",
            "*backtest*",
            "*results*"
        ]
        
        for pattern in patterns:
            for file in project_path.glob(pattern):
                if file.is_file():
                    output_files.append(str(file))
        
        return output_files
    
    def _parse_lean_output(self, output: str) -> Dict:
        """Parse output LEAN per metriche base"""
        
        import re
        
        metrics = {}
        
        # Parse metriche comuni
        patterns = {
            "total_return": r"Total Return[:\s]+([\d\.\-%]+)",
            "sharpe_ratio": r"Sharpe Ratio[:\s]+([\d\.\-]+)",
            "max_drawdown": r"Max Drawdown[:\s]+([\d\.\-%]+)",
            "win_rate": r"Win Rate[:\s]+([\d\.\-%]+)",
            "total_trades": r"Total Trades[:\s]+(\d+)"
        }
        
        for metric, pattern in patterns.items():
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                value = match.group(1)
                try:
                    # Prova a convertire in numero
                    if '%' in value:
                        metrics[metric] = float(value.replace('%', '')) / 100
                    else:
                        metrics[metric] = float(value)
                except ValueError:
                    metrics[metric] = value
        
        return metrics
    
    def _save_backtest_results(self, backtest_id: str, result: Dict):
        """Salva risultati backtest"""
        
        results_file = self.results_path / f"{backtest_id}_results.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
    
    def get_backtest_status(self, backtest_id: str) -> Dict:
        """Ottieni status backtest"""
        
        if backtest_id in self.running_backtests:
            return self.running_backtests[backtest_id]
        elif backtest_id in self.completed_backtests:
            return {"status": "completed", "result": self.completed_backtests[backtest_id]}
        else:
            return {"status": "not_found"}
    
    def list_running_backtests(self) -> List[str]:
        """Lista backtests in esecuzione"""
        
        running = []
        for backtest_id, info in self.running_backtests.items():
            if info["status"] == "running" and info["thread"].is_alive():
                running.append(backtest_id)
        
        return running
    
    def list_completed_backtests(self) -> List[str]:
        """Lista backtests completati"""
        
        return list(self.completed_backtests.keys())
    
    def cancel_backtest(self, backtest_id: str) -> bool:
        """Cancella backtest in esecuzione"""
        
        if backtest_id in self.running_backtests:
            info = self.running_backtests[backtest_id]
            
            if info["status"] == "running":
                # Non possiamo fermare il processo LEAN facilmente
                # Ma possiamo marcare come cancellato
                info["status"] = "cancelled"
                return True
        
        return False
    
    def optimize_strategy(self, 
                         strategy_file: str,
                         strategy_name: str,
                         parameters_ranges: Dict) -> str:
        """Lancia ottimizzazione parametri (versione semplificata)"""
        
        optimization_id = f"{strategy_name}_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Per ora, genera multiple varianti e le testa
        variants = self._generate_parameter_variants(parameters_ranges)
        
        optimization_results = []
        
        for i, variant in enumerate(variants[:5]):  # Limita a 5 varianti per test
            variant_name = f"{strategy_name}_var_{i}"
            
            # Modifica strategia con nuovi parametri
            modified_strategy = self._modify_strategy_parameters(strategy_file, variant)
            
            # Salva strategia modificata
            variant_file = self.results_path / f"{variant_name}.py"
            with open(variant_file, 'w', encoding='utf-8') as f:
                f.write(modified_strategy)
            
            # Lancia backtest
            backtest_id = self.launch_backtest(str(variant_file), variant_name)
            
            optimization_results.append({
                "variant": i,
                "parameters": variant,
                "backtest_id": backtest_id
            })
        
        # Salva info ottimizzazione
        opt_info = {
            "optimization_id": optimization_id,
            "strategy_name": strategy_name,
            "parameter_ranges": parameters_ranges,
            "variants": optimization_results,
            "start_time": datetime.now().isoformat()
        }
        
        opt_file = self.results_path / f"{optimization_id}_optimization.json"
        with open(opt_file, 'w', encoding='utf-8') as f:
            json.dump(opt_info, f, indent=2, default=str)
        
        return optimization_id
    
    def _generate_parameter_variants(self, ranges: Dict) -> List[Dict]:
        """Genera varianti parametri per ottimizzazione"""
        
        import itertools
        
        # Crea griglia parametri
        param_grid = {}
        
        for param, range_def in ranges.items():
            if isinstance(range_def, dict):
                start = range_def.get("min", 10)
                end = range_def.get("max", 50)
                step = range_def.get("step", 5)
                param_grid[param] = list(range(start, end + 1, step))
            elif isinstance(range_def, list):
                param_grid[param] = range_def
        
        # Genera combinazioni (limita per evitare esplosione)
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = list(itertools.product(*values))[:20]  # Max 20 combinazioni
        
        variants = []
        for combo in combinations:
            variant = dict(zip(keys, combo))
            variants.append(variant)
        
        return variants
    
    def _modify_strategy_parameters(self, strategy_file: str, new_params: Dict) -> str:
        """Modifica parametri nella strategia"""
        
        with open(strategy_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Sostituisci parametri (implementazione semplice)
        for param, value in new_params.items():
            import re
            pattern = f"self\\.{param}\\s*=\\s*\\d+"
            replacement = f"self.{param} = {value}"
            content = re.sub(pattern, replacement, content)
        
        return content

def test_launcher():
    """Test del launcher"""
    
    launcher = QuantConnectLauncher()
    
    # Check installation
    status = launcher.check_lean_installation()
    print("LEAN Installation Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Se LEAN non è installato, simula risultati
    if not status["lean_installed"]:
        print("\\nLEAN CLI not found. Simulating backtest...")
        
        # Simula risultati backtest
        simulated_result = {
            "backtest_id": "test_backtest_20241225_120000",
            "success": True,
            "results_summary": {
                "total_return": 0.157,
                "sharpe_ratio": 1.23,
                "max_drawdown": -0.08,
                "win_rate": 0.62,
                "total_trades": 45
            },
            "execution_time": 12.5
        }
        
        print("Simulated Results:")
        for key, value in simulated_result["results_summary"].items():
            print(f"  {key}: {value}")
    
    return launcher

if __name__ == "__main__":
    test_launcher()