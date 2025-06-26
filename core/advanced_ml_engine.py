"""
Motore ML Avanzato con LSTM, Transformer, DQN e Bayesian Optimization
Implementa hyperparameter tuning, ensemble pesato e reinforcement learning
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional, Tuple
import json
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class LSTMModel:
    """Modello LSTM per predizioni temporali"""
    
    def __init__(self, sequence_length: int = 60, features: int = 10):
        self.sequence_length = sequence_length
        self.features = features
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crea sequenze per LSTM"""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i, 0])  # Predice il prezzo di chiusura
        return np.array(X), np.array(y)
    
    def train(self, data: np.ndarray) -> bool:
        """Addestra il modello LSTM (simulato)"""
        try:
            if len(data) < self.sequence_length + 50:
                return False
            
            # Normalizza i dati
            data_scaled = self.scaler.fit_transform(data)
            
            # Crea sequenze
            X, y = self.create_sequences(data_scaled)
            
            # Simula addestramento LSTM
            # In un'implementazione reale, useremmo TensorFlow/Keras
            self.model = {
                'weights': np.random.randn(self.sequence_length, self.features),
                'bias': np.random.randn(1),
                'trained_on': len(data)
            }
            
            self.is_trained = True
            logger.info(f"LSTM model trained on {len(X)} sequences")
            return True
            
        except Exception as e:
            logger.error(f"Error training LSTM: {e}")
            return False
    
    def predict(self, data: np.ndarray) -> float:
        """Predice il prossimo valore"""
        try:
            if not self.is_trained or self.model is None:
                return 0.0
            
            if len(data) < self.sequence_length:
                return 0.0
            
            # Normalizza
            data_scaled = self.scaler.transform(data[-self.sequence_length:].reshape(-1, self.features))
            
            # Simulazione predizione LSTM
            weights = self.model['weights']
            bias = self.model['bias']
            
            # Operazione matriciale semplificata
            prediction = np.mean(data_scaled @ weights.T) + bias[0]
            
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Error in LSTM prediction: {e}")
            return 0.0

class TransformerModel:
    """Modello Transformer per analisi di sequenze"""
    
    def __init__(self, sequence_length: int = 100, d_model: int = 64):
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def attention_mechanism(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
        """Meccanismo di attenzione semplificato"""
        d_k = key.shape[-1]
        scores = np.dot(query, key.T) / np.sqrt(d_k)
        attention_weights = self.softmax(scores)
        return np.dot(attention_weights, value)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Funzione softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def train(self, data: np.ndarray) -> bool:
        """Addestra il modello Transformer (simulato)"""
        try:
            if len(data) < self.sequence_length + 50:
                return False
            
            # Normalizza
            data_scaled = self.scaler.fit_transform(data)
            
            # Simula l'addestramento del Transformer
            self.model = {
                'attention_weights': np.random.randn(self.d_model, self.d_model),
                'feed_forward': np.random.randn(self.d_model, data.shape[1]),
                'layer_norm': np.ones(self.d_model),
                'trained_on': len(data)
            }
            
            self.is_trained = True
            logger.info(f"Transformer model trained on {len(data)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error training Transformer: {e}")
            return False
    
    def predict(self, data: np.ndarray) -> float:
        """Predice usando il Transformer"""
        try:
            if not self.is_trained or self.model is None:
                return 0.0
            
            if len(data) < self.sequence_length:
                return 0.0
            
            # Normalizza
            data_scaled = self.scaler.transform(data[-self.sequence_length:])
            
            # Simulazione attenzione
            query = data_scaled[-1:].reshape(1, -1)
            key = data_scaled.reshape(self.sequence_length, -1)
            value = data_scaled.reshape(self.sequence_length, -1)
            
            # Applica attenzione (versione semplificata)
            attended = self.attention_mechanism(query, key, value)
            
            # Feed forward
            ff_weights = self.model['feed_forward']
            prediction = np.mean(attended @ ff_weights.T)
            
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Error in Transformer prediction: {e}")
            return 0.0

class DQNAgent:
    """Agente Deep Q-Network per trading decisions"""
    
    def __init__(self, state_size: int = 20, action_size: int = 3):
        self.state_size = state_size
        self.action_size = action_size  # 0: Hold, 1: Buy, 2: Sell
        self.memory = []
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = None
        self.target_model = None
        self.is_trained = False
        
    def build_model(self):
        """Costruisce la rete neurale (simulata)"""
        # In un'implementazione reale, useremmo TensorFlow/Keras
        self.model = {
            'layer1': np.random.randn(self.state_size, 64),
            'layer2': np.random.randn(64, 32),
            'output': np.random.randn(32, self.action_size),
            'bias1': np.random.randn(64),
            'bias2': np.random.randn(32),
            'bias_out': np.random.randn(self.action_size)
        }
        
        self.target_model = self.model.copy()
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Memorizza esperienza"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:  # Limite memoria
            self.memory.pop(0)
    
    def act(self, state: np.ndarray) -> int:
        """Sceglie un'azione basata sulla policy"""
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        if self.model is None:
            self.build_model()
        
        # Forward pass semplificato
        q_values = self.predict_q_values(state)
        return np.argmax(q_values)
    
    def predict_q_values(self, state: np.ndarray) -> np.ndarray:
        """Predice i Q-values"""
        if self.model is None:
            return np.random.randn(self.action_size)
        
        # Simulazione forward pass
        x = np.maximum(0, state @ self.model['layer1'] + self.model['bias1'])  # ReLU
        x = np.maximum(0, x @ self.model['layer2'] + self.model['bias2'])      # ReLU
        q_values = x @ self.model['output'] + self.model['bias_out']
        
        return q_values
    
    def replay(self, batch_size: int = 32):
        """Addestra il modello sui batch di memoria"""
        if len(self.memory) < batch_size:
            return
        
        # Simulazione training
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        
        # In un'implementazione reale, faremmo il vero training qui
        self.is_trained = True
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_action_recommendation(self, state: np.ndarray, current_position: str) -> Dict[str, Any]:
        """Ottiene raccomandazione di azione dal DQN"""
        action = self.act(state)
        q_values = self.predict_q_values(state)
        
        action_mapping = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        
        return {
            'action': action_mapping[action],
            'confidence': float(np.max(q_values) / (np.sum(np.abs(q_values)) + 1e-8)),
            'q_values': q_values.tolist()
        }

class BayesianOptimizer:
    """Ottimizzatore Bayesiano per hyperparameter tuning"""
    
    def __init__(self):
        self.optimization_history = []
        self.best_params = {}
        self.best_score = -np.inf
    
    def gaussian_process_surrogate(self, X: np.ndarray, y: np.ndarray, X_new: np.ndarray) -> Tuple[float, float]:
        """Modello surrogato semplificato (Gaussian Process)"""
        # Implementazione semplificata per dimostrazione
        if len(X) == 0:
            return 0.0, 1.0
        
        # Calcola media pesata basata sulla distanza
        distances = np.linalg.norm(X - X_new, axis=1)
        weights = np.exp(-distances)
        weights /= np.sum(weights)
        
        mean = np.sum(weights * y)
        variance = np.var(y) * (1 - np.max(weights))
        
        return mean, np.sqrt(variance)
    
    def acquisition_function(self, mean: float, std: float, best_y: float, xi: float = 0.01) -> float:
        """Expected Improvement acquisition function"""
        if std == 0:
            return 0
        
        z = (mean - best_y - xi) / std
        ei = (mean - best_y - xi) * self.normal_cdf(z) + std * self.normal_pdf(z)
        return ei
    
    def normal_cdf(self, x: float) -> float:
        """CDF della normale standard"""
        return 0.5 * (1 + np.tanh(x / np.sqrt(2)))
    
    def normal_pdf(self, x: float) -> float:
        """PDF della normale standard"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def optimize(self, objective_function, param_space: Dict[str, List], n_calls: int = 50) -> Dict[str, Any]:
        """Ottimizzazione Bayesiana"""
        X_tried = []
        y_tried = []
        
        # Random initialization
        for _ in range(min(10, n_calls)):
            params = {}
            for param, values in param_space.items():
                params[param] = np.random.choice(values)
            
            score = objective_function(params)
            
            X_tried.append(list(params.values()))
            y_tried.append(score)
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
        
        X_tried = np.array(X_tried)
        y_tried = np.array(y_tried)
        
        # Bayesian optimization iterations
        for i in range(n_calls - len(X_tried)):
            # Trova il prossimo punto da valutare
            best_ei = -np.inf
            next_params = None
            
            # Grid search semplificato per acquisition function
            for _ in range(100):
                candidate_params = {}
                candidate_x = []
                
                for param, values in param_space.items():
                    val = np.random.choice(values)
                    candidate_params[param] = val
                    candidate_x.append(val)
                
                candidate_x = np.array(candidate_x)
                
                # Predici con GP
                mean, std = self.gaussian_process_surrogate(X_tried, y_tried, candidate_x)
                
                # Calcola EI
                ei = self.acquisition_function(mean, std, np.max(y_tried))
                
                if ei > best_ei:
                    best_ei = ei
                    next_params = candidate_params.copy()
            
            if next_params is not None:
                score = objective_function(next_params)
                
                X_tried = np.vstack([X_tried, list(next_params.values())])
                y_tried = np.append(y_tried, score)
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = next_params.copy()
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': list(zip(X_tried.tolist(), y_tried.tolist()))
        }

class EnsembleEngine:
    """Motore ensemble con pesatura dinamica"""
    
    def __init__(self):
        self.models = {}
        self.model_weights = {}
        self.performance_history = {}
        self.lstm_model = LSTMModel()
        self.transformer_model = TransformerModel()
        self.dqn_agent = DQNAgent()
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.bayesian_optimizer = BayesianOptimizer()
        
        # Inizializza pesi uniformi
        self.model_weights = {
            'lstm': 0.25,
            'transformer': 0.25,
            'dqn': 0.25,
            'random_forest': 0.25
        }
    
    def train_all_models(self, data: np.ndarray, labels: np.ndarray) -> bool:
        """Addestra tutti i modelli dell'ensemble"""
        try:
            success = True
            
            # LSTM
            if self.lstm_model.train(data):
                logger.info("LSTM training completed")
            else:
                success = False
            
            # Transformer
            if self.transformer_model.train(data):
                logger.info("Transformer training completed")
            else:
                success = False
            
            # Random Forest
            try:
                self.rf_model.fit(data[:-1], labels)  # Usa tutti tranne l'ultimo per evitare shape mismatch
                logger.info("Random Forest training completed")
            except Exception as e:
                logger.error(f"Random Forest training failed: {e}")
                success = False
            
            # DQN (addestra su stati e ricompense simulate)
            for i in range(min(100, len(data) - 1)):
                state = data[i]
                next_state = data[i + 1]
                action = np.random.randint(0, 3)
                reward = np.random.normal(0, 0.1)  # Ricompensa simulata
                
                self.dqn_agent.remember(state, action, reward, next_state, False)
            
            self.dqn_agent.replay()
            logger.info("DQN training completed")
            
            return success
            
        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            return False
    
    def get_ensemble_prediction(self, data: np.ndarray) -> Dict[str, Any]:
        """Ottiene predizione pesata dall'ensemble"""
        try:
            predictions = {}
            confidences = {}
            
            # LSTM
            lstm_pred = self.lstm_model.predict(data)
            predictions['lstm'] = lstm_pred
            confidences['lstm'] = 0.8
            
            # Transformer
            transformer_pred = self.transformer_model.predict(data)
            predictions['transformer'] = transformer_pred
            confidences['transformer'] = 0.7
            
            # Random Forest
            try:
                if len(data) > 0:
                    rf_pred = self.rf_model.predict_proba(data[-1].reshape(1, -1))[0]
                    predictions['random_forest'] = rf_pred[1] - rf_pred[0]  # Buy - Sell probability
                    confidences['random_forest'] = max(rf_pred)
                else:
                    predictions['random_forest'] = 0.0
                    confidences['random_forest'] = 0.5
            except:
                predictions['random_forest'] = 0.0
                confidences['random_forest'] = 0.5
            
            # DQN
            if len(data) > 0:
                dqn_rec = self.dqn_agent.get_action_recommendation(data[-1], 'NEUTRAL')
                action_to_signal = {'BUY': 1.0, 'SELL': -1.0, 'HOLD': 0.0}
                predictions['dqn'] = action_to_signal.get(dqn_rec['action'], 0.0)
                confidences['dqn'] = dqn_rec['confidence']
            else:
                predictions['dqn'] = 0.0
                confidences['dqn'] = 0.5
            
            # Calcola predizione ensemble pesata
            total_weight = 0
            weighted_prediction = 0
            
            for model_name, pred in predictions.items():
                if model_name in self.model_weights:
                    weight = self.model_weights[model_name] * confidences[model_name]
                    weighted_prediction += pred * weight
                    total_weight += weight
            
            if total_weight > 0:
                final_prediction = weighted_prediction / total_weight
            else:
                final_prediction = 0.0
            
            return {
                'ensemble_prediction': final_prediction,
                'individual_predictions': predictions,
                'model_weights': self.model_weights.copy(),
                'confidences': confidences,
                'total_confidence': total_weight / sum(self.model_weights.values())
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return {
                'ensemble_prediction': 0.0,
                'individual_predictions': {},
                'model_weights': self.model_weights.copy(),
                'confidences': {},
                'total_confidence': 0.0
            }
    
    def update_model_weights(self, performance_data: Dict[str, float]):
        """Aggiorna i pesi dei modelli basandosi sulle performance"""
        try:
            # Normalizza le performance
            total_performance = sum(performance_data.values())
            if total_performance > 0:
                for model_name in self.model_weights:
                    if model_name in performance_data:
                        self.model_weights[model_name] = performance_data[model_name] / total_performance
            
            # Assicura che i pesi sommino a 1
            total_weight = sum(self.model_weights.values())
            if total_weight > 0:
                for model_name in self.model_weights:
                    self.model_weights[model_name] /= total_weight
            
            logger.info(f"Model weights updated: {self.model_weights}")
            
        except Exception as e:
            logger.error(f"Error updating model weights: {e}")
    
    def optimize_hyperparameters(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Ottimizza gli hyperparameters usando Bayesian Optimization"""
        
        def objective_function(params: Dict[str, Any]) -> float:
            """Funzione obiettivo per l'ottimizzazione"""
            try:
                # Simula training con parametri
                score = np.random.random() * params.get('learning_rate', 0.01) * 100
                score += params.get('n_estimators', 100) / 1000
                score += (1 - params.get('dropout', 0.2)) * 0.5
                
                return score
            except:
                return 0.0
        
        # Spazio parametri
        param_space = {
            'learning_rate': [0.001, 0.01, 0.1],
            'n_estimators': [50, 100, 200],
            'dropout': [0.1, 0.2, 0.3, 0.5]
        }
        
        result = self.bayesian_optimizer.optimize(objective_function, param_space, n_calls=30)
        
        logger.info(f"Hyperparameter optimization completed: {result['best_params']}")
        
        return result

class AdvancedMLEngine:
    """Motore ML principale che coordina tutti i componenti avanzati"""
    
    def __init__(self):
        self.ensemble_engine = EnsembleEngine()
        self.is_initialized = False
        self.training_data = []
        self.last_training = None
        
    def initialize(self):
        """Inizializza il motore ML"""
        try:
            self.is_initialized = True
            logger.info("Advanced ML Engine initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing ML engine: {e}")
            return False
    
    def add_training_data(self, market_data: np.ndarray, labels: np.ndarray):
        """Aggiunge dati di training"""
        try:
            self.training_data.append((market_data, labels))
            
            # Mantieni solo gli ultimi 1000 campioni
            if len(self.training_data) > 1000:
                self.training_data = self.training_data[-1000:]
                
        except Exception as e:
            logger.error(f"Error adding training data: {e}")
    
    def should_retrain(self) -> bool:
        """Determina se è necessario riaddestrare"""
        if self.last_training is None:
            return True
        
        # Riaddestra ogni 24 ore
        hours_since_training = (datetime.now() - self.last_training).total_seconds() / 3600
        return hours_since_training >= 24
    
    def train_models(self) -> bool:
        """Addestra tutti i modelli ML"""
        try:
            if not self.training_data:
                logger.warning("No training data available")
                return False
            
            # Combina tutti i dati di training
            all_data = []
            all_labels = []
            
            for data, labels in self.training_data:
                if len(data) > 0 and len(labels) > 0:
                    all_data.extend(data)
                    all_labels.extend(labels)
            
            if len(all_data) < 100:
                logger.warning("Insufficient training data")
                return False
            
            all_data = np.array(all_data)
            all_labels = np.array(all_labels)
            
            # Addestra ensemble
            success = self.ensemble_engine.train_all_models(all_data, all_labels)
            
            if success:
                self.last_training = datetime.now()
                logger.info("Model training completed successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return False
    
    def get_ml_signal(self, market_data: np.ndarray) -> Dict[str, Any]:
        """Ottiene segnale ML dall'ensemble"""
        try:
            if not self.is_initialized:
                self.initialize()
            
            # Ottieni predizione ensemble
            prediction = self.ensemble_engine.get_ensemble_prediction(market_data)
            
            # Converti in segnale di trading
            ensemble_pred = prediction.get('ensemble_prediction', 0.0)
            confidence = prediction.get('total_confidence', 0.0)
            
            if ensemble_pred > 0.3:
                action = 'BUY'
                strength = min(abs(ensemble_pred), 1.0)
            elif ensemble_pred < -0.3:
                action = 'SELL'
                strength = min(abs(ensemble_pred), 1.0)
            else:
                action = 'HOLD'
                strength = 0.0
            
            return {
                'action': action,
                'strength': strength,
                'confidence': confidence,
                'ensemble_details': prediction,
                'ml_engine': 'advanced_ensemble'
            }
            
        except Exception as e:
            logger.error(f"Error getting ML signal: {e}")
            return {
                'action': 'HOLD',
                'strength': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def optimize_hyperparameters(self) -> Dict[str, Any]:
        """Esegue ottimizzazione degli hyperparameters"""
        try:
            if not self.training_data:
                return {'error': 'No training data available'}
            
            # Usa gli ultimi dati per ottimizzazione
            latest_data, latest_labels = self.training_data[-1]
            
            result = self.ensemble_engine.optimize_hyperparameters(latest_data, latest_labels)
            
            logger.info("Hyperparameter optimization completed")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {e}")
            return {'error': str(e)}
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Ottiene metriche di performance dei modelli"""
        return {
            'model_weights': self.ensemble_engine.model_weights.copy(),
            'last_training': self.last_training.isoformat() if self.last_training else None,
            'training_samples': len(self.training_data),
            'is_initialized': self.is_initialized
        }