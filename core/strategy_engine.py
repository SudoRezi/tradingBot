import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

logger = logging.getLogger(__name__)

class StrategyEngine:
    """Multi-strategy AI engine with adaptive learning"""
    
    def __init__(self, risk_level: str):
        self.risk_level = risk_level
        self.strategies = {}
        self.strategy_performance = {}
        self.ml_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        
        # Initialize strategies
        self._initialize_strategies()
        
        # Load or create ML model
        self._load_or_create_ml_model()
        
        logger.info(f"Strategy Engine initialized with {len(self.strategies)} strategies")
    
    def _initialize_strategies(self):
        """Initialize different trading strategies"""
        self.strategies = {
            'ema_crossover': {
                'name': 'EMA Crossover',
                'weight': 0.2,
                'performance': 0.5,
                'active': True
            },
            'rsi_mean_reversion': {
                'name': 'RSI Mean Reversion',
                'weight': 0.15,
                'performance': 0.5,
                'active': True
            },
            'macd_momentum': {
                'name': 'MACD Momentum',
                'weight': 0.15,
                'performance': 0.5,
                'active': True
            },
            'bollinger_squeeze': {
                'name': 'Bollinger Band Squeeze',
                'weight': 0.1,
                'performance': 0.5,
                'active': True
            },
            'volume_breakout': {
                'name': 'Volume Breakout',
                'weight': 0.1,
                'performance': 0.5,
                'active': True
            },
            'ml_prediction': {
                'name': 'ML Prediction',
                'weight': 0.3,
                'performance': 0.5,
                'active': True
            }
        }
    
    def generate_signal(self, market_analysis: Dict[str, Any], pair: str) -> Optional[Dict[str, Any]]:
        """Generate trading signal using multiple strategies"""
        try:
            if not market_analysis:
                return None
            
            signals = []
            total_weight = 0
            
            # Generate signals from each active strategy
            for strategy_id, strategy in self.strategies.items():
                if not strategy['active']:
                    continue
                
                strategy_signal = self._execute_strategy(strategy_id, market_analysis, pair)
                
                if strategy_signal:
                    # Weight the signal by strategy performance and base weight
                    weighted_strength = strategy_signal['strength'] * strategy['weight'] * strategy['performance']
                    signals.append({
                        'strategy': strategy_id,
                        'action': strategy_signal['action'],
                        'strength': strategy_signal['strength'],
                        'weighted_strength': weighted_strength,
                        'confidence': strategy_signal.get('confidence', 0.5)
                    })
                    total_weight += strategy['weight'] * strategy['performance']
            
            if not signals:
                return None
            
            # Combine signals
            combined_signal = self._combine_signals(signals, total_weight, market_analysis)
            
            if combined_signal and combined_signal['confidence'] > 0.3:
                return combined_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal for {pair}: {e}")
            return None
    
    def _execute_strategy(self, strategy_id: str, analysis: Dict[str, Any], pair: str) -> Optional[Dict[str, Any]]:
        """Execute specific strategy"""
        try:
            if strategy_id == 'ema_crossover':
                return self._ema_crossover_strategy(analysis)
            elif strategy_id == 'rsi_mean_reversion':
                return self._rsi_mean_reversion_strategy(analysis)
            elif strategy_id == 'macd_momentum':
                return self._macd_momentum_strategy(analysis)
            elif strategy_id == 'bollinger_squeeze':
                return self._bollinger_squeeze_strategy(analysis)
            elif strategy_id == 'volume_breakout':
                return self._volume_breakout_strategy(analysis)
            elif strategy_id == 'ml_prediction':
                return self._ml_prediction_strategy(analysis)
            
            return None
            
        except Exception as e:
            logger.error(f"Error executing strategy {strategy_id}: {e}")
            return None
    
    def _ema_crossover_strategy(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """EMA crossover strategy"""
        try:
            ema_9 = analysis.get('ema_9', 0)
            ema_21 = analysis.get('ema_21', 0)
            ema_50 = analysis.get('ema_50', 0)
            current_price = analysis.get('price', 0)
            
            if ema_9 == 0 or ema_21 == 0:
                return None
            
            # Bullish crossover
            if ema_9 > ema_21 and ema_21 > ema_50 and current_price > ema_9:
                strength = min((ema_9 - ema_21) / ema_21, 0.05) * 20  # Normalize to 0-1
                return {
                    'action': 'BUY',
                    'strength': strength,
                    'confidence': 0.7,
                    'price': current_price
                }
            
            # Bearish crossover
            elif ema_9 < ema_21 and ema_21 < ema_50 and current_price < ema_9:
                strength = min((ema_21 - ema_9) / ema_21, 0.05) * 20
                return {
                    'action': 'SELL',
                    'strength': strength,
                    'confidence': 0.7,
                    'price': current_price
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in EMA crossover strategy: {e}")
            return None
    
    def _rsi_mean_reversion_strategy(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """RSI mean reversion strategy"""
        try:
            rsi = analysis.get('rsi', 50)
            current_price = analysis.get('price', 0)
            trend_direction = analysis.get('trend_direction', 'neutral')
            
            # Oversold in uptrend
            if rsi < 30 and trend_direction in ['bullish', 'strong_bullish']:
                strength = (30 - rsi) / 30  # Stronger signal when more oversold
                return {
                    'action': 'BUY',
                    'strength': strength,
                    'confidence': 0.8,
                    'price': current_price
                }
            
            # Overbought in downtrend
            elif rsi > 70 and trend_direction in ['bearish', 'strong_bearish']:
                strength = (rsi - 70) / 30
                return {
                    'action': 'SELL',
                    'strength': strength,
                    'confidence': 0.8,
                    'price': current_price
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in RSI mean reversion strategy: {e}")
            return None
    
    def _macd_momentum_strategy(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """MACD momentum strategy"""
        try:
            macd = analysis.get('macd', 0)
            macd_signal = analysis.get('macd_signal', 0)
            macd_hist = analysis.get('macd_histogram', 0)
            current_price = analysis.get('price', 0)
            
            # Bullish momentum
            if macd > macd_signal and macd_hist > 0:
                strength = min(abs(macd_hist) / abs(macd) if macd != 0 else 0, 1)
                return {
                    'action': 'BUY',
                    'strength': strength,
                    'confidence': 0.6,
                    'price': current_price
                }
            
            # Bearish momentum
            elif macd < macd_signal and macd_hist < 0:
                strength = min(abs(macd_hist) / abs(macd) if macd != 0 else 0, 1)
                return {
                    'action': 'SELL',
                    'strength': strength,
                    'confidence': 0.6,
                    'price': current_price
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in MACD momentum strategy: {e}")
            return None
    
    def _bollinger_squeeze_strategy(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Bollinger Band squeeze/breakout strategy"""
        try:
            bb_upper = analysis.get('bb_upper', 0)
            bb_lower = analysis.get('bb_lower', 0)
            bb_position = analysis.get('bb_position', 0.5)
            current_price = analysis.get('price', 0)
            volatility = analysis.get('volatility_level', 0)
            
            # Low volatility squeeze followed by breakout
            if volatility < 0.02:  # Low volatility
                # Breakout above upper band
                if bb_position > 0.95:
                    return {
                        'action': 'BUY',
                        'strength': 0.8,
                        'confidence': 0.7,
                        'price': current_price
                    }
                # Breakdown below lower band
                elif bb_position < 0.05:
                    return {
                        'action': 'SELL',
                        'strength': 0.8,
                        'confidence': 0.7,
                        'price': current_price
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in Bollinger squeeze strategy: {e}")
            return None
    
    def _volume_breakout_strategy(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Volume breakout strategy"""
        try:
            volume_ratio = analysis.get('volume_ratio', 1)
            volume_strength = analysis.get('volume_strength', 'normal')
            trend_direction = analysis.get('trend_direction', 'neutral')
            current_price = analysis.get('price', 0)
            
            # High volume breakout
            if volume_ratio > 2 and volume_strength == 'high':
                if trend_direction in ['bullish', 'strong_bullish']:
                    strength = min(volume_ratio / 3, 1)  # Normalize
                    return {
                        'action': 'BUY',
                        'strength': strength,
                        'confidence': 0.75,
                        'price': current_price
                    }
                elif trend_direction in ['bearish', 'strong_bearish']:
                    strength = min(volume_ratio / 3, 1)
                    return {
                        'action': 'SELL',
                        'strength': strength,
                        'confidence': 0.75,
                        'price': current_price
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in volume breakout strategy: {e}")
            return None
    
    def _ml_prediction_strategy(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Machine learning prediction strategy"""
        try:
            if self.ml_model is None:
                return None
            
            # Prepare features
            features = self._extract_ml_features(analysis)
            if not features:
                return None
            
            # Make prediction
            feature_array = np.array(features).reshape(1, -1)
            feature_array = self.scaler.transform(feature_array)
            
            prediction_proba = self.ml_model.predict_proba(feature_array)[0]
            prediction = self.ml_model.predict(feature_array)[0]
            
            # Convert prediction to signal
            current_price = analysis.get('price', 0)
            
            if prediction == 1:  # Buy signal
                confidence = prediction_proba[1]
                return {
                    'action': 'BUY',
                    'strength': confidence,
                    'confidence': confidence,
                    'price': current_price
                }
            elif prediction == -1:  # Sell signal
                confidence = prediction_proba[0] if len(prediction_proba) > 2 else prediction_proba[0]
                return {
                    'action': 'SELL',
                    'strength': confidence,
                    'confidence': confidence,
                    'price': current_price
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in ML prediction strategy: {e}")
            return None
    
    def _extract_ml_features(self, analysis: Dict[str, Any]) -> List[float]:
        """Extract features for ML model"""
        try:
            features = [
                analysis.get('rsi', 50) / 100,  # Normalize to 0-1
                analysis.get('stoch_k', 50) / 100,
                analysis.get('macd', 0),
                analysis.get('trend_strength', 25) / 100,
                analysis.get('volume_ratio', 1),
                analysis.get('volatility_level', 0.02) * 50,  # Scale volatility
                1 if analysis.get('trend_direction') in ['bullish', 'strong_bullish'] else
                -1 if analysis.get('trend_direction') in ['bearish', 'strong_bearish'] else 0,
                analysis.get('bb_position', 0.5),
                analysis.get('support_distance', 0.05) * 20,
                analysis.get('resistance_distance', 0.05) * 20,
            ]
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting ML features: {e}")
            return []
    
    def _combine_signals(self, signals: List[Dict], total_weight: float, 
                        analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Combine multiple strategy signals"""
        try:
            if not signals or total_weight == 0:
                return None
            
            # Separate buy and sell signals
            buy_signals = [s for s in signals if s['action'] == 'BUY']
            sell_signals = [s for s in signals if s['action'] == 'SELL']
            
            buy_strength = sum(s['weighted_strength'] for s in buy_signals)
            sell_strength = sum(s['weighted_strength'] for s in sell_signals)
            
            # Determine final action
            if buy_strength > sell_strength and buy_strength > 0.1:
                action = 'BUY'
                strength = buy_strength / total_weight
                confidence = np.mean([s['confidence'] for s in buy_signals])
            elif sell_strength > buy_strength and sell_strength > 0.1:
                action = 'SELL'
                strength = sell_strength / total_weight
                confidence = np.mean([s['confidence'] for s in sell_signals])
            else:
                return None  # No clear signal
            
            # Apply risk level adjustments
            confidence *= self._get_risk_multiplier()
            
            return {
                'action': action,
                'strength': min(strength, 1.0),
                'confidence': min(confidence, 1.0),
                'price': analysis.get('price', 0),
                'volatility': analysis.get('volatility_level', 0.02),
                'strategies_used': len(signals),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            return None
    
    def _get_risk_multiplier(self) -> float:
        """Get confidence multiplier based on risk level"""
        multipliers = {
            'conservative': 0.8,
            'moderate': 1.0,
            'aggressive': 1.2
        }
        return multipliers.get(self.risk_level, 1.0)
    
    def _load_or_create_ml_model(self):
        """Load existing ML model or create new one"""
        try:
            model_path = 'models/trading_model.joblib'
            scaler_path = 'models/scaler.joblib'
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.ml_model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded existing ML model")
            else:
                # Create new model with default training
                self._create_default_ml_model()
                logger.info("Created new ML model")
                
        except Exception as e:
            logger.error(f"Error loading/creating ML model: {e}")
            self.ml_model = None
    
    def _create_default_ml_model(self):
        """Create a default ML model with synthetic training data"""
        try:
            # Generate synthetic training data
            n_samples = 1000
            features = []
            labels = []
            
            for _ in range(n_samples):
                # Generate random but realistic features
                rsi = np.random.uniform(0, 1)
                stoch = np.random.uniform(0, 1)
                macd = np.random.uniform(-0.1, 0.1)
                trend_strength = np.random.uniform(0, 1)
                volume_ratio = np.random.uniform(0.5, 3)
                volatility = np.random.uniform(0, 0.1)
                trend_dir = np.random.choice([-1, 0, 1])
                bb_pos = np.random.uniform(0, 1)
                support_dist = np.random.uniform(0, 1)
                resistance_dist = np.random.uniform(0, 1)
                
                feature_row = [rsi, stoch, macd, trend_strength, volume_ratio, 
                              volatility, trend_dir, bb_pos, support_dist, resistance_dist]
                features.append(feature_row)
                
                # Generate label based on simple rules
                if rsi < 0.3 and trend_dir >= 0:
                    label = 1  # Buy
                elif rsi > 0.7 and trend_dir <= 0:
                    label = -1  # Sell
                elif trend_dir > 0 and volume_ratio > 1.5:
                    label = 1  # Buy
                elif trend_dir < 0 and volume_ratio > 1.5:
                    label = -1  # Sell
                else:
                    label = 0  # Hold
                
                labels.append(label)
            
            # Train model
            X = np.array(features)
            y = np.array(labels)
            
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.ml_model.fit(X_scaled, y)
            
            # Save model
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.ml_model, 'models/trading_model.joblib')
            joblib.dump(self.scaler, 'models/scaler.joblib')
            
        except Exception as e:
            logger.error(f"Error creating default ML model: {e}")
            self.ml_model = None
    
    def optimize_from_performance(self, performance_metrics: Dict[str, Any]):
        """Optimize strategies based on performance"""
        try:
            win_rate = performance_metrics.get('win_rate', 50)
            total_return = performance_metrics.get('total_return_pct', 0)
            
            # Adjust strategy weights based on performance
            for strategy_id, strategy in self.strategies.items():
                # Increase performance score for good overall performance
                if win_rate > 60 and total_return > 5:
                    strategy['performance'] = min(strategy['performance'] * 1.1, 2.0)
                # Decrease for poor performance
                elif win_rate < 40 or total_return < -10:
                    strategy['performance'] = max(strategy['performance'] * 0.9, 0.1)
            
            # Rebalance weights
            total_performance = sum(s['performance'] for s in self.strategies.values())
            if total_performance > 0:
                for strategy in self.strategies.values():
                    strategy['weight'] = strategy['performance'] / total_performance
            
            logger.info(f"Strategy optimization completed - Win Rate: {win_rate:.1f}%")
            
        except Exception as e:
            logger.error(f"Error optimizing strategies: {e}")
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy status"""
        return {
            'total_strategies': len(self.strategies),
            'active_strategies': len([s for s in self.strategies.values() if s['active']]),
            'ml_model_active': self.ml_model is not None,
            'strategies': self.strategies
        }
