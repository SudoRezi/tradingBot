"""Advanced trading models and ML utilities for the AI Trading Bot"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

logger = logging.getLogger(__name__)

class AdvancedTradingModels:
    """Advanced ML models for trading signal generation"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'random_state': 42
                }
            },
            'neural_network': {
                'model': MLPClassifier,
                'params': {
                    'hidden_layer_sizes': (100, 50),
                    'activation': 'relu',
                    'solver': 'adam',
                    'max_iter': 1000,
                    'random_state': 42
                }
            },
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'C': 1.0,
                    'solver': 'liblinear',
                    'random_state': 42
                }
            }
        }
        
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all ML models"""
        try:
            os.makedirs('models', exist_ok=True)
            
            for model_name, config in self.model_configs.items():
                # Try to load existing model
                model_path = f'models/{model_name}_model.joblib'
                scaler_path = f'models/{model_name}_scaler.joblib'
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.models[model_name] = joblib.load(model_path)
                    self.scalers[model_name] = joblib.load(scaler_path)
                    logger.info(f"Loaded existing {model_name} model")
                else:
                    # Create new model
                    self.models[model_name] = config['model'](**config['params'])
                    self.scalers[model_name] = RobustScaler()
                    logger.info(f"Created new {model_name} model")
                    
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def create_ensemble_model(self, base_models: List[str] = None) -> VotingClassifier:
        """Create ensemble model from base models"""
        try:
            if base_models is None:
                base_models = ['random_forest', 'gradient_boosting', 'neural_network']
            
            estimators = []
            for model_name in base_models:
                if model_name in self.models:
                    estimators.append((model_name, self.models[model_name]))
            
            if len(estimators) < 2:
                logger.warning("Need at least 2 models for ensemble")
                return None
            
            ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft'  # Use probability voting
            )
            
            return ensemble
            
        except Exception as e:
            logger.error(f"Error creating ensemble model: {e}")
            return None
    
    def train_models(self, X: np.ndarray, y: np.ndarray, pair: str = 'default'):
        """Train all models with given data"""
        try:
            if len(X) < 100:  # Need minimum data for training
                logger.warning(f"Insufficient training data: {len(X)} samples")
                return False
            
            results = {}
            
            for model_name, model in self.models.items():
                try:
                    # Scale features
                    scaler = self.scalers[model_name]
                    X_scaled = scaler.fit_transform(X)
                    
                    # Train model
                    model.fit(X_scaled, y)
                    
                    # Evaluate with cross-validation
                    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
                    
                    # Store results
                    results[model_name] = {
                        'accuracy': cv_scores.mean(),
                        'accuracy_std': cv_scores.std(),
                        'trained_samples': len(X),
                        'training_date': datetime.now()
                    }
                    
                    # Save model
                    model_path = f'models/{model_name}_model_{pair}.joblib'
                    scaler_path = f'models/{model_name}_scaler_{pair}.joblib'
                    
                    joblib.dump(model, model_path)
                    joblib.dump(scaler, scaler_path)
                    
                    # Feature importance (if available)
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importance[model_name] = model.feature_importances_
                    
                    logger.info(f"{model_name} trained: {cv_scores.mean():.3f} ± {cv_scores.std():.3f} accuracy")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    results[model_name] = {'error': str(e)}
            
            self.model_performance[pair] = results
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return False
    
    def predict_ensemble(self, X: np.ndarray, pair: str = 'default') -> Dict[str, Any]:
        """Make predictions using ensemble of models"""
        try:
            predictions = {}
            probabilities = {}
            
            for model_name, model in self.models.items():
                try:
                    # Scale features
                    scaler = self.scalers[model_name]
                    X_scaled = scaler.transform(X.reshape(1, -1))
                    
                    # Make prediction
                    pred = model.predict(X_scaled)[0]
                    pred_proba = model.predict_proba(X_scaled)[0]
                    
                    predictions[model_name] = pred
                    probabilities[model_name] = pred_proba
                    
                except Exception as e:
                    logger.error(f"Error predicting with {model_name}: {e}")
                    continue
            
            if not predictions:
                return None
            
            # Ensemble prediction (majority vote)
            pred_values = list(predictions.values())
            ensemble_pred = max(set(pred_values), key=pred_values.count)
            
            # Average probabilities
            if probabilities:
                prob_arrays = list(probabilities.values())
                ensemble_proba = np.mean(prob_arrays, axis=0)
            else:
                ensemble_proba = [0.33, 0.33, 0.34]  # Default uniform
            
            return {
                'prediction': ensemble_pred,
                'probability': ensemble_proba,
                'individual_predictions': predictions,
                'individual_probabilities': probabilities,
                'confidence': max(ensemble_proba)
            }
            
        except Exception as e:
            logger.error(f"Error making ensemble prediction: {e}")
            return None
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, model_name: str):
        """Optimize hyperparameters for a specific model"""
        try:
            if model_name not in self.models:
                logger.error(f"Model {model_name} not found")
                return False
            
            # Define parameter grids
            param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'gradient_boosting': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 6, 9],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'neural_network': {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                },
                'logistic_regression': {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'solver': ['liblinear', 'lbfgs'],
                    'penalty': ['l1', 'l2']
                }
            }
            
            if model_name not in param_grids:
                logger.warning(f"No parameter grid defined for {model_name}")
                return False
            
            # Scale features
            scaler = self.scalers[model_name]
            X_scaled = scaler.fit_transform(X)
            
            # Grid search
            base_model = self.model_configs[model_name]['model']()
            grid_search = GridSearchCV(
                base_model,
                param_grids[model_name],
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_scaled, y)
            
            # Update model with best parameters
            self.models[model_name] = grid_search.best_estimator_
            
            logger.info(f"Optimized {model_name}: {grid_search.best_score_:.3f} accuracy")
            logger.info(f"Best parameters: {grid_search.best_params_}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing {model_name}: {e}")
            return False

class FeatureEngineer:
    """Advanced feature engineering for trading signals"""
    
    def __init__(self):
        self.feature_names = []
        self.feature_history = {}
    
    def create_technical_features(self, data: pd.DataFrame) -> np.ndarray:
        """Create comprehensive technical analysis features"""
        try:
            features = []
            feature_names = []
            
            if data is None or len(data) < 50:
                return np.array([]), []
            
            # Price-based features
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            volume = data['volume'].values
            
            # 1. Price ratios and changes
            price_change_1 = (close[-1] - close[-2]) / close[-2] if len(close) > 1 else 0
            price_change_5 = (close[-1] - close[-6]) / close[-6] if len(close) > 5 else 0
            price_change_10 = (close[-1] - close[-11]) / close[-11] if len(close) > 10 else 0
            
            features.extend([price_change_1, price_change_5, price_change_10])
            feature_names.extend(['price_change_1', 'price_change_5', 'price_change_10'])
            
            # 2. Moving average ratios
            if len(close) >= 50:
                sma_10 = np.mean(close[-10:])
                sma_20 = np.mean(close[-20:])
                sma_50 = np.mean(close[-50:])
                
                ma_ratio_10_20 = sma_10 / sma_20 - 1
                ma_ratio_20_50 = sma_20 / sma_50 - 1
                price_to_sma20 = close[-1] / sma_20 - 1
                
                features.extend([ma_ratio_10_20, ma_ratio_20_50, price_to_sma20])
                feature_names.extend(['ma_ratio_10_20', 'ma_ratio_20_50', 'price_to_sma20'])
            else:
                features.extend([0, 0, 0])
                feature_names.extend(['ma_ratio_10_20', 'ma_ratio_20_50', 'price_to_sma20'])
            
            # 3. Volatility features
            if len(close) >= 20:
                returns = np.diff(close[-20:]) / close[-20:-1]
                volatility = np.std(returns)
                avg_return = np.mean(returns)
                
                features.extend([volatility, avg_return])
                feature_names.extend(['volatility_20', 'avg_return_20'])
            else:
                features.extend([0, 0])
                feature_names.extend(['volatility_20', 'avg_return_20'])
            
            # 4. Volume features
            if len(volume) >= 20:
                volume_ratio = volume[-1] / np.mean(volume[-20:])
                volume_trend = np.polyfit(range(10), volume[-10:], 1)[0] if len(volume) >= 10 else 0
                
                features.extend([volume_ratio, volume_trend])
                feature_names.extend(['volume_ratio', 'volume_trend'])
            else:
                features.extend([1, 0])
                feature_names.extend(['volume_ratio', 'volume_trend'])
            
            # 5. High-Low features
            if len(high) >= 20:
                hl_ratio = (high[-1] - low[-1]) / close[-1]
                high_position = (close[-1] - low[-1]) / (high[-1] - low[-1]) if high[-1] != low[-1] else 0.5
                recent_high = np.max(high[-20:])
                recent_low = np.min(low[-20:])
                position_in_range = (close[-1] - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
                
                features.extend([hl_ratio, high_position, position_in_range])
                feature_names.extend(['hl_ratio', 'high_position', 'position_in_range'])
            else:
                features.extend([0.02, 0.5, 0.5])
                feature_names.extend(['hl_ratio', 'high_position', 'position_in_range'])
            
            # 6. Momentum features
            if len(close) >= 14:
                # Simple momentum
                momentum_5 = close[-1] / close[-6] - 1 if len(close) > 5 else 0
                momentum_14 = close[-1] / close[-15] - 1 if len(close) > 14 else 0
                
                # Rate of change
                roc_5 = (close[-1] - close[-6]) / close[-6] if len(close) > 5 else 0
                roc_14 = (close[-1] - close[-15]) / close[-15] if len(close) > 14 else 0
                
                features.extend([momentum_5, momentum_14, roc_5, roc_14])
                feature_names.extend(['momentum_5', 'momentum_14', 'roc_5', 'roc_14'])
            else:
                features.extend([0, 0, 0, 0])
                feature_names.extend(['momentum_5', 'momentum_14', 'roc_5', 'roc_14'])
            
            # 7. Pattern features
            if len(close) >= 5:
                # Consecutive movements
                consecutive_up = 0
                consecutive_down = 0
                
                for i in range(1, min(6, len(close))):
                    if close[-i] > close[-i-1]:
                        consecutive_up += 1
                    else:
                        break
                
                for i in range(1, min(6, len(close))):
                    if close[-i] < close[-i-1]:
                        consecutive_down += 1
                    else:
                        break
                
                # Gap detection
                gap_up = (low[-1] > high[-2]) if len(close) > 1 else False
                gap_down = (high[-1] < low[-2]) if len(close) > 1 else False
                
                features.extend([consecutive_up, consecutive_down, int(gap_up), int(gap_down)])
                feature_names.extend(['consecutive_up', 'consecutive_down', 'gap_up', 'gap_down'])
            else:
                features.extend([0, 0, 0, 0])
                feature_names.extend(['consecutive_up', 'consecutive_down', 'gap_up', 'gap_down'])
            
            self.feature_names = feature_names
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error creating technical features: {e}")
            return np.array([])
    
    def create_market_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """Create market regime-based features"""
        try:
            features = []
            
            if data is None or len(data) < 50:
                return np.array([0, 0, 0, 0, 0])
            
            close = data['close'].values
            volume = data['volume'].values
            
            # Trend strength
            if len(close) >= 20:
                trend_slope = np.polyfit(range(20), close[-20:], 1)[0]
                trend_r2 = np.corrcoef(range(20), close[-20:])[0, 1] ** 2
            else:
                trend_slope = 0
                trend_r2 = 0
            
            # Market volatility regime
            if len(close) >= 30:
                returns = np.diff(close[-30:]) / close[-30:-1]
                current_vol = np.std(returns[-10:])
                long_vol = np.std(returns)
                vol_regime = current_vol / long_vol if long_vol > 0 else 1
            else:
                vol_regime = 1
            
            # Volume regime
            if len(volume) >= 30:
                current_vol_avg = np.mean(volume[-10:])
                long_vol_avg = np.mean(volume[-30:])
                volume_regime = current_vol_avg / long_vol_avg if long_vol_avg > 0 else 1
            else:
                volume_regime = 1
            
            # Price level (support/resistance)
            if len(close) >= 50:
                price_percentile = np.percentile(close[-50:], 
                                               100 * np.sum(close[-50:] <= close[-1]) / 50)
            else:
                price_percentile = 50
            
            features = [trend_slope, trend_r2, vol_regime, volume_regime, price_percentile]
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error creating market regime features: {e}")
            return np.array([0, 0, 1, 1, 50])
    
    def create_labels(self, data: pd.DataFrame, future_periods: int = 5, 
                     threshold: float = 0.02) -> np.ndarray:
        """Create labels for supervised learning"""
        try:
            if data is None or len(data) < future_periods + 1:
                return np.array([])
            
            close = data['close'].values
            labels = []
            
            for i in range(len(close) - future_periods):
                current_price = close[i]
                future_price = close[i + future_periods]
                
                change_pct = (future_price - current_price) / current_price
                
                if change_pct > threshold:
                    label = 1  # Buy signal
                elif change_pct < -threshold:
                    label = -1  # Sell signal
                else:
                    label = 0  # Hold signal
                
                labels.append(label)
            
            return np.array(labels)
            
        except Exception as e:
            logger.error(f"Error creating labels: {e}")
            return np.array([])

class ModelEvaluator:
    """Evaluate and compare model performance"""
    
    def __init__(self):
        self.evaluation_history = {}
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Trading-specific metrics
            buy_signals = np.sum(y_pred == 1)
            sell_signals = np.sum(y_pred == -1)
            hold_signals = np.sum(y_pred == 0)
            
            # Signal distribution
            signal_ratio = buy_signals / len(y_pred) if len(y_pred) > 0 else 0
            
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'buy_signals': int(buy_signals),
                'sell_signals': int(sell_signals),
                'hold_signals': int(hold_signals),
                'signal_ratio': signal_ratio,
                'total_predictions': len(y_pred),
                'evaluation_date': datetime.now().isoformat()
            }
            
            # Store in history
            if model_name not in self.evaluation_history:
                self.evaluation_history[model_name] = []
            
            self.evaluation_history[model_name].append(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            return {}
    
    def compare_models(self, results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Compare multiple model results"""
        try:
            if not results:
                return {}
            
            # Find best model for each metric
            best_models = {}
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            
            for metric in metrics:
                best_score = -1
                best_model = None
                
                for model_name, model_results in results.items():
                    if metric in model_results and model_results[metric] > best_score:
                        best_score = model_results[metric]
                        best_model = model_name
                
                best_models[metric] = {
                    'model': best_model,
                    'score': best_score
                }
            
            # Overall ranking
            model_scores = {}
            for model_name, model_results in results.items():
                total_score = 0
                count = 0
                
                for metric in metrics:
                    if metric in model_results:
                        total_score += model_results[metric]
                        count += 1
                
                if count > 0:
                    model_scores[model_name] = total_score / count
            
            # Sort models by average score
            ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'best_models_by_metric': best_models,
                'model_ranking': ranked_models,
                'comparison_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return {}
    
    def get_model_summary(self, model_name: str) -> Dict[str, Any]:
        """Get summary statistics for a model"""
        try:
            if model_name not in self.evaluation_history:
                return {}
            
            history = self.evaluation_history[model_name]
            if not history:
                return {}
            
            # Calculate summary statistics
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            summary = {}
            
            for metric in metrics:
                values = [h[metric] for h in history if metric in h]
                if values:
                    summary[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'latest': values[-1]
                    }
            
            summary['evaluation_count'] = len(history)
            summary['first_evaluation'] = history[0]['evaluation_date']
            summary['latest_evaluation'] = history[-1]['evaluation_date']
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting model summary for {model_name}: {e}")
            return {}

# Global instances for easy access
trading_models = AdvancedTradingModels()
feature_engineer = FeatureEngineer()
model_evaluator = ModelEvaluator()
