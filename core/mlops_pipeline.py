"""
Pipeline MLOps per il Trading Bot AI
Gestisce versioning, monitoring, retraining automatico e deployment dei modelli
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import pickle
import hashlib
import os
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelVersioning:
    """Sistema di versioning per i modelli ML"""
    
    def __init__(self, models_dir: str = "models_versioned"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.version_registry = {}
        self._load_registry()
    
    def _load_registry(self):
        """Carica registry delle versioni"""
        registry_path = self.models_dir / "registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                self.version_registry = json.load(f)
    
    def _save_registry(self):
        """Salva registry delle versioni"""
        registry_path = self.models_dir / "registry.json"
        with open(registry_path, 'w') as f:
            json.dump(self.version_registry, f, indent=2, default=str)
    
    def save_model_version(self, model: Any, model_name: str, 
                          performance_metrics: Dict[str, float],
                          metadata: Dict[str, Any] = None) -> str:
        """Salva una nuova versione del modello"""
        try:
            # Genera version ID
            timestamp = datetime.now()
            version_id = f"{model_name}_v{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Crea directory per questa versione
            version_dir = self.models_dir / version_id
            version_dir.mkdir(exist_ok=True)
            
            # Salva il modello
            model_path = version_dir / "model.joblib"
            joblib.dump(model, model_path)
            
            # Calcola hash del modello
            model_hash = self._calculate_model_hash(model_path)
            
            # Prepara metadati
            version_metadata = {
                'version_id': version_id,
                'model_name': model_name,
                'timestamp': timestamp.isoformat(),
                'model_hash': model_hash,
                'performance_metrics': performance_metrics,
                'file_size_mb': model_path.stat().st_size / (1024 * 1024),
                'metadata': metadata or {}
            }
            
            # Salva metadati
            metadata_path = version_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(version_metadata, f, indent=2, default=str)
            
            # Aggiorna registry
            if model_name not in self.version_registry:
                self.version_registry[model_name] = []
            
            self.version_registry[model_name].append(version_metadata)
            self._save_registry()
            
            logger.info(f"Saved model version: {version_id}")
            return version_id
            
        except Exception as e:
            logger.error(f"Error saving model version: {e}")
            return ""
    
    def load_model_version(self, version_id: str) -> Tuple[Any, Dict[str, Any]]:
        """Carica una versione specifica del modello"""
        try:
            version_dir = self.models_dir / version_id
            if not version_dir.exists():
                raise FileNotFoundError(f"Version {version_id} not found")
            
            # Carica modello
            model_path = version_dir / "model.joblib"
            model = joblib.load(model_path)
            
            # Carica metadati
            metadata_path = version_dir / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error loading model version {version_id}: {e}")
            return None, {}
    
    def get_best_model(self, model_name: str, metric: str = 'accuracy') -> Tuple[str, Any, Dict[str, Any]]:
        """Ottiene la migliore versione di un modello"""
        try:
            if model_name not in self.version_registry:
                return "", None, {}
            
            versions = self.version_registry[model_name]
            
            # Trova versione con miglior performance
            best_version = max(versions, 
                             key=lambda v: v['performance_metrics'].get(metric, 0))
            
            version_id = best_version['version_id']
            model, metadata = self.load_model_version(version_id)
            
            return version_id, model, metadata
            
        except Exception as e:
            logger.error(f"Error getting best model for {model_name}: {e}")
            return "", None, {}
    
    def _calculate_model_hash(self, model_path: Path) -> str:
        """Calcola hash del file modello"""
        hash_sha256 = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def list_model_versions(self, model_name: str = None) -> List[Dict[str, Any]]:
        """Lista versioni disponibili"""
        if model_name:
            return self.version_registry.get(model_name, [])
        return self.version_registry
    
    def cleanup_old_versions(self, model_name: str, keep_last: int = 5):
        """Rimuove versioni vecchie mantenendo solo le ultime N"""
        try:
            if model_name not in self.version_registry:
                return
            
            versions = sorted(self.version_registry[model_name], 
                            key=lambda v: v['timestamp'], reverse=True)
            
            versions_to_remove = versions[keep_last:]
            
            for version in versions_to_remove:
                version_dir = self.models_dir / version['version_id']
                if version_dir.exists():
                    import shutil
                    shutil.rmtree(version_dir)
                
                self.version_registry[model_name].remove(version)
            
            self._save_registry()
            logger.info(f"Cleaned up {len(versions_to_remove)} old versions of {model_name}")
            
        except Exception as e:
            logger.error(f"Error cleaning up versions: {e}")

class PerformanceMonitor:
    """Monitora performance dei modelli in produzione"""
    
    def __init__(self, monitoring_window: int = 1000):
        self.monitoring_window = monitoring_window
        self.prediction_history = []
        self.actual_results = []
        self.performance_history = []
        self.drift_detectors = {}
        
    def log_prediction(self, model_name: str, features: np.ndarray, 
                      prediction: Union[float, int], confidence: float,
                      timestamp: datetime = None):
        """Registra una predizione per monitoring"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            prediction_log = {
                'model_name': model_name,
                'timestamp': timestamp,
                'features': features.tolist() if isinstance(features, np.ndarray) else features,
                'prediction': prediction,
                'confidence': confidence,
                'feature_hash': self._hash_features(features)
            }
            
            self.prediction_history.append(prediction_log)
            
            # Mantieni solo finestra di monitoring
            if len(self.prediction_history) > self.monitoring_window:
                self.prediction_history = self.prediction_history[-self.monitoring_window:]
                
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")
    
    def log_actual_result(self, prediction_timestamp: datetime, actual_result: Union[float, int]):
        """Registra risultato effettivo per una predizione"""
        try:
            self.actual_results.append({
                'timestamp': prediction_timestamp,
                'actual_result': actual_result
            })
            
            # Mantieni finestra
            if len(self.actual_results) > self.monitoring_window:
                self.actual_results = self.actual_results[-self.monitoring_window:]
                
        except Exception as e:
            logger.error(f"Error logging actual result: {e}")
    
    def calculate_current_performance(self, model_name: str) -> Dict[str, float]:
        """Calcola performance corrente del modello"""
        try:
            # Filtra predizioni per questo modello
            model_predictions = [p for p in self.prediction_history 
                               if p['model_name'] == model_name]
            
            if not model_predictions:
                return {}
            
            # Match predizioni con risultati effettivi
            matched_pairs = []
            
            for pred in model_predictions:
                pred_time = pred['timestamp'] if isinstance(pred['timestamp'], datetime) else \
                           datetime.fromisoformat(pred['timestamp'])
                
                # Trova risultato corrispondente (entro 1 ora)
                for result in self.actual_results:
                    result_time = result['timestamp'] if isinstance(result['timestamp'], datetime) else \
                                datetime.fromisoformat(result['timestamp'])
                    
                    time_diff = abs((result_time - pred_time).total_seconds())
                    if time_diff <= 3600:  # Entro 1 ora
                        matched_pairs.append({
                            'prediction': pred['prediction'],
                            'actual': result['actual_result'],
                            'confidence': pred['confidence']
                        })
                        break
            
            if not matched_pairs:
                return {}
            
            # Calcola metriche
            predictions = [p['prediction'] for p in matched_pairs]
            actuals = [p['actual'] for p in matched_pairs]
            confidences = [p['confidence'] for p in matched_pairs]
            
            # Per classificazione binaria
            if all(isinstance(p, (int, bool)) for p in predictions):
                accuracy = accuracy_score(actuals, predictions)
                precision = precision_score(actuals, predictions, average='weighted', zero_division=0)
                recall = recall_score(actuals, predictions, average='weighted', zero_division=0)
                f1 = f1_score(actuals, predictions, average='weighted', zero_division=0)
                
                return {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'sample_count': len(matched_pairs),
                    'avg_confidence': np.mean(confidences)
                }
            
            # Per regressione
            else:
                mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
                mse = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
                rmse = np.sqrt(mse)
                
                # R-squared
                ss_res = np.sum((np.array(actuals) - np.array(predictions)) ** 2)
                ss_tot = np.sum((np.array(actuals) - np.mean(actuals)) ** 2)
                r2 = 1 - (ss_res / (ss_tot + 1e-8))
                
                return {
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2_score': r2,
                    'sample_count': len(matched_pairs),
                    'avg_confidence': np.mean(confidences)
                }
                
        except Exception as e:
            logger.error(f"Error calculating performance: {e}")
            return {}
    
    def detect_performance_drift(self, model_name: str, baseline_performance: Dict[str, float],
                                threshold: float = 0.1) -> Dict[str, Any]:
        """Detecta drift nelle performance"""
        try:
            current_performance = self.calculate_current_performance(model_name)
            
            if not current_performance or not baseline_performance:
                return {'drift_detected': False, 'reason': 'Insufficient data'}
            
            drift_alerts = []
            
            for metric, baseline_value in baseline_performance.items():
                if metric in current_performance:
                    current_value = current_performance[metric]
                    
                    # Calcola degradazione percentuale
                    if baseline_value != 0:
                        degradation = abs(current_value - baseline_value) / abs(baseline_value)
                        
                        if degradation > threshold:
                            drift_alerts.append({
                                'metric': metric,
                                'baseline': baseline_value,
                                'current': current_value,
                                'degradation': degradation
                            })
            
            drift_detected = len(drift_alerts) > 0
            
            return {
                'drift_detected': drift_detected,
                'drift_alerts': drift_alerts,
                'current_performance': current_performance,
                'baseline_performance': baseline_performance,
                'sample_count': current_performance.get('sample_count', 0)
            }
            
        except Exception as e:
            logger.error(f"Error detecting drift: {e}")
            return {'drift_detected': False, 'error': str(e)}
    
    def _hash_features(self, features) -> str:
        """Hash delle features per tracking"""
        if isinstance(features, np.ndarray):
            features_str = str(features.tolist())
        else:
            features_str = str(features)
        return hashlib.md5(features_str.encode()).hexdigest()[:8]

class AutoRetrainer:
    """Sistema di retraining automatico"""
    
    def __init__(self, model_versioning: ModelVersioning, performance_monitor: PerformanceMonitor):
        self.model_versioning = model_versioning
        self.performance_monitor = performance_monitor
        self.retraining_schedule = {}
        self.retraining_history = []
        
    def schedule_retraining(self, model_name: str, frequency_hours: int = 168,  # 1 settimana
                           performance_threshold: float = 0.1):
        """Programma retraining automatico"""
        self.retraining_schedule[model_name] = {
            'frequency_hours': frequency_hours,
            'performance_threshold': performance_threshold,
            'last_retraining': datetime.now(),
            'next_scheduled': datetime.now() + timedelta(hours=frequency_hours)
        }
        
        logger.info(f"Scheduled retraining for {model_name} every {frequency_hours} hours")
    
    def check_retraining_needed(self, model_name: str) -> Dict[str, Any]:
        """Controlla se è necessario retraining"""
        try:
            if model_name not in self.retraining_schedule:
                return {'needs_retraining': False, 'reason': 'No schedule configured'}
            
            schedule = self.retraining_schedule[model_name]
            now = datetime.now()
            
            # Controllo temporale
            time_based = now >= schedule['next_scheduled']
            
            # Controllo performance
            baseline_performance = self._get_baseline_performance(model_name)
            drift_analysis = self.performance_monitor.detect_performance_drift(
                model_name, baseline_performance, schedule['performance_threshold']
            )
            
            performance_based = drift_analysis['drift_detected']
            
            needs_retraining = time_based or performance_based
            
            reason = []
            if time_based:
                reason.append('Scheduled time reached')
            if performance_based:
                reason.append('Performance degradation detected')
            
            return {
                'needs_retraining': needs_retraining,
                'reason': ' and '.join(reason) if reason else 'No retraining needed',
                'time_based_trigger': time_based,
                'performance_based_trigger': performance_based,
                'drift_analysis': drift_analysis,
                'next_scheduled': schedule['next_scheduled'].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking retraining need: {e}")
            return {'needs_retraining': False, 'error': str(e)}
    
    def execute_retraining(self, model_name: str, training_data: pd.DataFrame,
                          training_labels: np.ndarray, model_factory_func) -> Dict[str, Any]:
        """Esegue retraining del modello"""
        try:
            logger.info(f"Starting retraining for {model_name}")
            
            # Crea nuovo modello
            new_model = model_factory_func()
            
            # Addestra
            new_model.fit(training_data, training_labels)
            
            # Valuta performance
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(new_model, training_data, training_labels, 
                                      cv=5, scoring='accuracy')
            
            performance_metrics = {
                'accuracy': cv_scores.mean(),
                'accuracy_std': cv_scores.std(),
                'training_samples': len(training_data),
                'retraining_date': datetime.now().isoformat()
            }
            
            # Salva nuova versione
            version_id = self.model_versioning.save_model_version(
                new_model, model_name, performance_metrics,
                {'retraining_trigger': 'automatic', 'training_data_size': len(training_data)}
            )
            
            # Aggiorna schedule
            if model_name in self.retraining_schedule:
                schedule = self.retraining_schedule[model_name]
                schedule['last_retraining'] = datetime.now()
                schedule['next_scheduled'] = datetime.now() + timedelta(hours=schedule['frequency_hours'])
            
            # Aggiungi alla storia
            self.retraining_history.append({
                'model_name': model_name,
                'version_id': version_id,
                'timestamp': datetime.now(),
                'performance_metrics': performance_metrics,
                'status': 'completed'
            })
            
            logger.info(f"Retraining completed for {model_name}, new version: {version_id}")
            
            return {
                'status': 'success',
                'version_id': version_id,
                'performance_metrics': performance_metrics,
                'training_samples': len(training_data)
            }
            
        except Exception as e:
            logger.error(f"Error in retraining {model_name}: {e}")
            
            # Registra fallimento
            self.retraining_history.append({
                'model_name': model_name,
                'timestamp': datetime.now(),
                'status': 'failed',
                'error': str(e)
            })
            
            return {'status': 'failed', 'error': str(e)}
    
    def _get_baseline_performance(self, model_name: str) -> Dict[str, float]:
        """Ottiene performance baseline per confronto"""
        versions = self.model_versioning.list_model_versions(model_name)
        if not versions:
            return {}
        
        # Usa performance della versione più recente come baseline
        latest_version = max(versions, key=lambda v: v['timestamp'])
        return latest_version.get('performance_metrics', {})

class MLOpsPipeline:
    """Pipeline MLOps principale"""
    
    def __init__(self, models_dir: str = "mlops_models"):
        self.model_versioning = ModelVersioning(models_dir)
        self.performance_monitor = PerformanceMonitor()
        self.auto_retrainer = AutoRetrainer(self.model_versioning, self.performance_monitor)
        self.pipeline_config = {
            'monitoring_enabled': True,
            'auto_retraining_enabled': True,
            'performance_alerts_enabled': True
        }
        
    def register_model(self, model: Any, model_name: str, 
                      performance_metrics: Dict[str, float],
                      retraining_frequency_hours: int = 168) -> str:
        """Registra nuovo modello nel pipeline"""
        try:
            # Salva versione iniziale
            version_id = self.model_versioning.save_model_version(
                model, model_name, performance_metrics,
                {'registration_date': datetime.now().isoformat()}
            )
            
            # Programma retraining
            self.auto_retrainer.schedule_retraining(
                model_name, retraining_frequency_hours
            )
            
            logger.info(f"Model {model_name} registered with version {version_id}")
            return version_id
            
        except Exception as e:
            logger.error(f"Error registering model {model_name}: {e}")
            return ""
    
    def get_production_model(self, model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """Ottiene modello per produzione"""
        version_id, model, metadata = self.model_versioning.get_best_model(model_name)
        return model, metadata
    
    def log_prediction_and_monitor(self, model_name: str, features: np.ndarray,
                                  prediction: Union[float, int], confidence: float):
        """Log predizione e monitoring"""
        if self.pipeline_config['monitoring_enabled']:
            self.performance_monitor.log_prediction(
                model_name, features, prediction, confidence
            )
    
    def update_with_actual_result(self, prediction_timestamp: datetime,
                                 actual_result: Union[float, int]):
        """Aggiorna con risultato effettivo"""
        if self.pipeline_config['monitoring_enabled']:
            self.performance_monitor.log_actual_result(prediction_timestamp, actual_result)
    
    def run_monitoring_cycle(self) -> Dict[str, Any]:
        """Esegue ciclo di monitoring completo"""
        try:
            monitoring_results = {}
            
            # Controlla tutti i modelli registrati
            for model_name in self.auto_retrainer.retraining_schedule.keys():
                
                # Controlla performance
                current_performance = self.performance_monitor.calculate_current_performance(model_name)
                
                # Controlla necessità retraining
                retraining_check = self.auto_retrainer.check_retraining_needed(model_name)
                
                monitoring_results[model_name] = {
                    'current_performance': current_performance,
                    'retraining_check': retraining_check,
                    'monitoring_timestamp': datetime.now().isoformat()
                }
                
                # Alert se performance degradata
                if (retraining_check.get('performance_based_trigger') and 
                    self.pipeline_config['performance_alerts_enabled']):
                    
                    logger.warning(f"Performance degradation detected for {model_name}")
            
            return {
                'monitoring_results': monitoring_results,
                'pipeline_status': 'healthy',
                'last_monitoring_cycle': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}")
            return {'pipeline_status': 'error', 'error': str(e)}
    
    def get_pipeline_dashboard(self) -> Dict[str, Any]:
        """Dashboard dello stato pipeline"""
        try:
            # Statistiche modelli
            model_stats = {}
            for model_name in self.auto_retrainer.retraining_schedule.keys():
                versions = self.model_versioning.list_model_versions(model_name)
                current_perf = self.performance_monitor.calculate_current_performance(model_name)
                
                model_stats[model_name] = {
                    'total_versions': len(versions),
                    'latest_version': versions[-1]['version_id'] if versions else None,
                    'current_performance': current_perf,
                    'monitoring_samples': current_perf.get('sample_count', 0)
                }
            
            # Storia retraining
            recent_retraining = [r for r in self.auto_retrainer.retraining_history
                               if (datetime.now() - r['timestamp']).days <= 7]
            
            return {
                'pipeline_config': self.pipeline_config,
                'model_statistics': model_stats,
                'recent_retraining_count': len(recent_retraining),
                'retraining_schedule': {
                    name: {
                        'frequency_hours': schedule['frequency_hours'],
                        'next_scheduled': schedule['next_scheduled'].isoformat(),
                        'last_retraining': schedule['last_retraining'].isoformat()
                    }
                    for name, schedule in self.auto_retrainer.retraining_schedule.items()
                },
                'system_health': 'operational',
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating pipeline dashboard: {e}")
            return {'system_health': 'error', 'error': str(e)}