"""
Motore di Previsione Volatilità con Modelli GARCH e Position Sizing Dinamico
Implementa GARCH(1,1), EGARCH e volatility forecasting per risk management
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class GARCHModel:
    """Modello GARCH per previsione volatilità"""
    
    def __init__(self, p: int = 1, q: int = 1):
        self.p = p  # ordine ARCH
        self.q = q  # ordine GARCH
        self.params = None
        self.conditional_variance = None
        self.is_fitted = False
        
    def estimate_parameters(self, returns: np.ndarray) -> bool:
        """Stima parametri GARCH usando maximum likelihood"""
        try:
            if len(returns) < 100:
                return False
            
            # Inizializzazione parametri
            omega = np.var(returns) * 0.1
            alpha = 0.1
            beta = 0.85
            
            # Iterazioni per convergenza (simplified ML estimation)
            for iteration in range(50):
                # Calcola varianza condizionale
                conditional_var = self._calculate_conditional_variance(returns, omega, alpha, beta)
                
                # Log-likelihood
                log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * conditional_var) + 
                                              (returns**2) / conditional_var)
                
                # Aggiorna parametri (gradiente semplificato)
                gradient_omega = np.sum(1 / conditional_var - (returns**2) / (conditional_var**2))
                gradient_alpha = np.sum((returns[:-1]**2) * (1 / conditional_var[1:] - 
                                       (returns[1:]**2) / (conditional_var[1:]**2)))
                gradient_beta = np.sum(conditional_var[:-1] * (1 / conditional_var[1:] - 
                                      (returns[1:]**2) / (conditional_var[1:]**2)))
                
                # Update con learning rate
                lr = 0.0001
                omega += lr * gradient_omega / len(returns)
                alpha += lr * gradient_alpha / len(returns)
                beta += lr * gradient_beta / len(returns)
                
                # Vincoli
                omega = max(omega, 1e-6)
                alpha = np.clip(alpha, 0.01, 0.3)
                beta = np.clip(beta, 0.6, 0.95)
                
                # Controllo convergenza
                if iteration > 0 and abs(alpha + beta - 0.95) < 0.01:
                    break
            
            self.params = {'omega': omega, 'alpha': alpha, 'beta': beta}
            self.conditional_variance = conditional_var
            self.is_fitted = True
            
            logger.info(f"GARCH fitted: ω={omega:.6f}, α={alpha:.4f}, β={beta:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting GARCH model: {e}")
            return False
    
    def _calculate_conditional_variance(self, returns: np.ndarray, omega: float, 
                                      alpha: float, beta: float) -> np.ndarray:
        """Calcola varianza condizionale GARCH"""
        n = len(returns)
        conditional_var = np.zeros(n)
        
        # Inizializzazione
        conditional_var[0] = np.var(returns)
        
        for t in range(1, n):
            conditional_var[t] = (omega + 
                                alpha * returns[t-1]**2 + 
                                beta * conditional_var[t-1])
        
        return conditional_var
    
    def forecast_volatility(self, horizon: int = 1) -> np.ndarray:
        """Prevede volatilità futura"""
        try:
            if not self.is_fitted:
                return np.array([0.02] * horizon)  # Default 2% volatility
            
            omega, alpha, beta = self.params['omega'], self.params['alpha'], self.params['beta']
            
            # Long-run variance
            long_run_var = omega / (1 - alpha - beta)
            
            forecasts = np.zeros(horizon)
            current_var = self.conditional_variance[-1]
            
            for h in range(horizon):
                if h == 0:
                    forecasts[h] = omega + (alpha + beta) * current_var
                else:
                    forecasts[h] = long_run_var + (alpha + beta)**h * (current_var - long_run_var)
            
            return np.sqrt(forecasts)  # Ritorna volatilità (non varianza)
            
        except Exception as e:
            logger.error(f"Error forecasting volatility: {e}")
            return np.array([0.02] * horizon)

class EGARCHModel:
    """Exponential GARCH per effetti asimmetrici"""
    
    def __init__(self):
        self.params = None
        self.log_conditional_variance = None
        self.is_fitted = False
    
    def estimate_parameters(self, returns: np.ndarray) -> bool:
        """Stima parametri EGARCH"""
        try:
            if len(returns) < 100:
                return False
            
            # Parametri iniziali
            omega = -0.1
            alpha = 0.1
            gamma = -0.05  # Asymmetry parameter
            beta = 0.9
            
            # Standardizza returns
            std_returns = returns / np.std(returns)
            
            # Iterazioni ML (semplificato)
            for iteration in range(30):
                log_var = self._calculate_log_conditional_variance(std_returns, omega, alpha, gamma, beta)
                
                # Aggiorna parametri (gradiente approssimato)
                omega += 0.001 * np.mean(log_var)
                alpha += 0.001 * np.mean(np.abs(std_returns[:-1]) * np.diff(log_var))
                gamma += 0.001 * np.mean(std_returns[:-1] * np.diff(log_var))
                beta += 0.001 * np.mean(log_var[:-1] * np.diff(log_var))
                
                # Vincoli
                omega = np.clip(omega, -1.0, 0.5)
                alpha = np.clip(alpha, 0.01, 0.5)
                gamma = np.clip(gamma, -0.3, 0.1)
                beta = np.clip(beta, 0.8, 0.99)
            
            self.params = {'omega': omega, 'alpha': alpha, 'gamma': gamma, 'beta': beta}
            self.log_conditional_variance = log_var
            self.is_fitted = True
            
            logger.info(f"EGARCH fitted: ω={omega:.4f}, α={alpha:.4f}, γ={gamma:.4f}, β={beta:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting EGARCH model: {e}")
            return False
    
    def _calculate_log_conditional_variance(self, returns: np.ndarray, omega: float,
                                          alpha: float, gamma: float, beta: float) -> np.ndarray:
        """Calcola log varianza condizionale EGARCH"""
        n = len(returns)
        log_var = np.zeros(n)
        
        # Inizializzazione
        log_var[0] = np.log(np.var(returns))
        
        for t in range(1, n):
            z_t = returns[t-1]  # standardized residual
            
            log_var[t] = (omega + 
                         beta * log_var[t-1] + 
                         alpha * (np.abs(z_t) - np.sqrt(2/np.pi)) +
                         gamma * z_t)
        
        return log_var
    
    def forecast_volatility(self, horizon: int = 1) -> np.ndarray:
        """Prevede volatilità con EGARCH"""
        try:
            if not self.is_fitted:
                return np.array([0.02] * horizon)
            
            omega, alpha, gamma, beta = (self.params['omega'], self.params['alpha'],
                                       self.params['gamma'], self.params['beta'])
            
            forecasts = np.zeros(horizon)
            current_log_var = self.log_conditional_variance[-1]
            
            for h in range(horizon):
                if h == 0:
                    forecasts[h] = np.exp(omega + beta * current_log_var)
                else:
                    # Long-term forecast
                    forecasts[h] = np.exp(omega / (1 - beta) + beta**h * 
                                        (current_log_var - omega / (1 - beta)))
            
            return np.sqrt(forecasts)
            
        except Exception as e:
            logger.error(f"Error forecasting EGARCH volatility: {e}")
            return np.array([0.02] * horizon)

class VolatilityRegimeDetector:
    """Detectore di regime di volatilità"""
    
    def __init__(self, window: int = 252):
        self.window = window
        self.regime_history = []
        
    def detect_regime(self, returns: pd.Series) -> Dict[str, Any]:
        """Detecta regime di volatilità corrente"""
        try:
            if len(returns) < self.window:
                return {'regime': 'normal', 'confidence': 0.5}
            
            # Calcola volatilità rolling
            rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
            
            # Percentili storici
            vol_percentiles = rolling_vol.rolling(window=self.window).quantile([0.1, 0.25, 0.75, 0.9])
            
            current_vol = rolling_vol.iloc[-1]
            
            # Classifica regime
            if current_vol > vol_percentiles.iloc[-1, 3]:  # >90th percentile
                regime = 'high_volatility'
                confidence = 0.9
            elif current_vol > vol_percentiles.iloc[-1, 2]:  # >75th percentile
                regime = 'elevated_volatility'
                confidence = 0.7
            elif current_vol < vol_percentiles.iloc[-1, 0]:  # <10th percentile
                regime = 'low_volatility'
                confidence = 0.8
            elif current_vol < vol_percentiles.iloc[-1, 1]:  # <25th percentile
                regime = 'suppressed_volatility'
                confidence = 0.6
            else:
                regime = 'normal_volatility'
                confidence = 0.6
            
            # Volatility clustering detection
            recent_vols = rolling_vol.tail(10)
            vol_persistence = recent_vols.autocorr(lag=1)
            
            # Trend detection
            vol_trend = 'increasing' if rolling_vol.tail(5).is_monotonic_increasing else \
                       'decreasing' if rolling_vol.tail(5).is_monotonic_decreasing else 'stable'
            
            return {
                'regime': regime,
                'confidence': confidence,
                'current_volatility': current_vol,
                'volatility_percentile': rolling_vol.rank(pct=True).iloc[-1],
                'persistence': vol_persistence,
                'trend': vol_trend,
                'clustering_detected': abs(vol_persistence) > 0.3
            }
            
        except Exception as e:
            logger.error(f"Error detecting volatility regime: {e}")
            return {'regime': 'normal', 'confidence': 0.5}

class DynamicPositionSizer:
    """Sistema di sizing dinamico basato su volatilità"""
    
    def __init__(self, base_risk_per_trade: float = 0.02):
        self.base_risk_per_trade = base_risk_per_trade
        self.volatility_adjustments = {
            'low_volatility': 1.5,        # Aumenta size in bassa vol
            'suppressed_volatility': 1.3,
            'normal_volatility': 1.0,
            'elevated_volatility': 0.7,
            'high_volatility': 0.4         # Riduci size in alta vol
        }
        
    def calculate_position_size(self, account_balance: float, entry_price: float,
                              stop_loss_price: float, predicted_volatility: float,
                              volatility_regime: Dict[str, Any]) -> Dict[str, Any]:
        """Calcola size ottimale basato su volatilità"""
        try:
            # Risk per trade base
            base_risk_amount = account_balance * self.base_risk_per_trade
            
            # Aggiustamento per regime di volatilità
            regime = volatility_regime.get('regime', 'normal_volatility')
            vol_multiplier = self.volatility_adjustments.get(regime, 1.0)
            
            # Aggiustamento per volatilità predetta
            if predicted_volatility > 0:
                # Inversely proportional to volatility
                vol_adjustment = min(2.0, max(0.3, 0.02 / predicted_volatility))
            else:
                vol_adjustment = 1.0
            
            # Risk amount aggiustato
            adjusted_risk_amount = base_risk_amount * vol_multiplier * vol_adjustment
            
            # Calcola position size
            if stop_loss_price > 0 and entry_price > 0:
                price_risk = abs(entry_price - stop_loss_price) / entry_price
                if price_risk > 0:
                    position_value = adjusted_risk_amount / price_risk
                    position_size = position_value / entry_price
                else:
                    position_size = 0
            else:
                position_size = 0
            
            # Kelly Criterion adjustment
            win_rate = volatility_regime.get('confidence', 0.6)
            if win_rate > 0.5:
                kelly_multiplier = min(1.5, 2 * win_rate - 1)
                position_size *= kelly_multiplier
            
            # Limiti di sicurezza
            max_position_value = account_balance * 0.2  # Max 20% del capitale
            position_size = min(position_size, max_position_value / entry_price)
            
            return {
                'position_size': position_size,
                'position_value': position_size * entry_price,
                'risk_amount': adjusted_risk_amount,
                'volatility_multiplier': vol_multiplier,
                'predicted_vol_adjustment': vol_adjustment,
                'effective_risk_percentage': (adjusted_risk_amount / account_balance) * 100,
                'regime_based_sizing': regime,
                'confidence_score': volatility_regime.get('confidence', 0.5)
            }
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {
                'position_size': 0,
                'position_value': 0,
                'error': str(e)
            }
    
    def calculate_dynamic_stop_loss(self, entry_price: float, direction: str,
                                  predicted_volatility: float, time_horizon: int = 1) -> Dict[str, Any]:
        """Calcola stop loss dinamico basato su volatilità"""
        try:
            # Base stop loss (2 * volatility)
            base_stop_distance = 2 * predicted_volatility * entry_price
            
            # Time decay adjustment
            time_adjustment = np.sqrt(time_horizon)
            stop_distance = base_stop_distance * time_adjustment
            
            if direction.upper() == 'BUY':
                stop_loss_price = entry_price - stop_distance
                take_profit_price = entry_price + (2 * stop_distance)  # 1:2 risk/reward
            else:  # SELL
                stop_loss_price = entry_price + stop_distance
                take_profit_price = entry_price - (2 * stop_distance)
            
            return {
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price,
                'stop_distance': stop_distance,
                'stop_distance_percentage': (stop_distance / entry_price) * 100,
                'risk_reward_ratio': 2.0,
                'volatility_based': True
            }
            
        except Exception as e:
            logger.error(f"Error calculating dynamic stop loss: {e}")
            return {
                'stop_loss_price': entry_price * (0.95 if direction.upper() == 'BUY' else 1.05),
                'error': str(e)
            }

class VolatilityEngine:
    """Motore principale per gestione volatilità"""
    
    def __init__(self):
        self.garch_model = GARCHModel()
        self.egarch_model = EGARCHModel()
        self.regime_detector = VolatilityRegimeDetector()
        self.position_sizer = DynamicPositionSizer()
        self.volatility_history = {}
        self.last_update = None
        
    def update_volatility_models(self, price_data: pd.DataFrame, pair: str) -> bool:
        """Aggiorna modelli di volatilità con nuovi dati"""
        try:
            if len(price_data) < 100:
                return False
            
            # Calcola returns
            returns = price_data['close'].pct_change().dropna()
            
            # Fit modelli
            garch_success = self.garch_model.estimate_parameters(returns.values)
            egarch_success = self.egarch_model.estimate_parameters(returns.values)
            
            # Detecta regime
            regime = self.regime_detector.detect_regime(returns)
            
            # Salva nella storia
            self.volatility_history[pair] = {
                'returns': returns,
                'regime': regime,
                'garch_fitted': garch_success,
                'egarch_fitted': egarch_success,
                'last_update': datetime.now()
            }
            
            self.last_update = datetime.now()
            
            logger.info(f"Volatility models updated for {pair}: regime={regime['regime']}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating volatility models for {pair}: {e}")
            return False
    
    def get_volatility_forecast(self, pair: str, horizon: int = 5) -> Dict[str, Any]:
        """Ottiene previsioni di volatilità"""
        try:
            if pair not in self.volatility_history:
                return {
                    'garch_forecast': [0.02] * horizon,
                    'egarch_forecast': [0.02] * horizon,
                    'ensemble_forecast': [0.02] * horizon,
                    'confidence': 0.3
                }
            
            # Forecast dai modelli
            garch_forecast = self.garch_model.forecast_volatility(horizon)
            egarch_forecast = self.egarch_model.forecast_volatility(horizon)
            
            # Ensemble (media pesata)
            garch_weight = 0.6 if self.garch_model.is_fitted else 0.3
            egarch_weight = 0.4 if self.egarch_model.is_fitted else 0.3
            ensemble_weight = garch_weight + egarch_weight
            
            if ensemble_weight > 0:
                ensemble_forecast = ((garch_forecast * garch_weight + 
                                   egarch_forecast * egarch_weight) / ensemble_weight)
            else:
                ensemble_forecast = np.array([0.02] * horizon)
            
            return {
                'garch_forecast': garch_forecast.tolist(),
                'egarch_forecast': egarch_forecast.tolist(),
                'ensemble_forecast': ensemble_forecast.tolist(),
                'confidence': min(0.9, ensemble_weight),
                'horizon_days': horizon,
                'current_regime': self.volatility_history[pair]['regime']
            }
            
        except Exception as e:
            logger.error(f"Error getting volatility forecast for {pair}: {e}")
            return {
                'garch_forecast': [0.02] * horizon,
                'egarch_forecast': [0.02] * horizon,
                'ensemble_forecast': [0.02] * horizon,
                'confidence': 0.3,
                'error': str(e)
            }
    
    def get_position_sizing_recommendation(self, pair: str, account_balance: float,
                                         entry_price: float, direction: str,
                                         stop_loss_percentage: float = None) -> Dict[str, Any]:
        """Ottiene raccomandazione di position sizing"""
        try:
            # Ottieni forecast volatilità
            vol_forecast = self.get_volatility_forecast(pair, horizon=1)
            predicted_vol = vol_forecast['ensemble_forecast'][0]
            
            # Regime corrente
            regime = vol_forecast.get('current_regime', {'regime': 'normal', 'confidence': 0.6})
            
            # Calcola stop loss se non fornito
            if stop_loss_percentage is None:
                stop_info = self.position_sizer.calculate_dynamic_stop_loss(
                    entry_price, direction, predicted_vol
                )
                stop_loss_price = stop_info['stop_loss_price']
            else:
                if direction.upper() == 'BUY':
                    stop_loss_price = entry_price * (1 - stop_loss_percentage / 100)
                else:
                    stop_loss_price = entry_price * (1 + stop_loss_percentage / 100)
                stop_info = {'stop_loss_price': stop_loss_price}
            
            # Calcola position size
            sizing_info = self.position_sizer.calculate_position_size(
                account_balance, entry_price, stop_loss_price, predicted_vol, regime
            )
            
            return {
                'recommended_position_size': sizing_info['position_size'],
                'position_value': sizing_info['position_value'],
                'risk_amount': sizing_info['risk_amount'],
                'predicted_volatility': predicted_vol,
                'volatility_regime': regime['regime'],
                'stop_loss_info': stop_info,
                'volatility_adjustment': sizing_info.get('volatility_multiplier', 1.0),
                'confidence': vol_forecast['confidence']
            }
            
        except Exception as e:
            logger.error(f"Error getting position sizing recommendation: {e}")
            return {
                'recommended_position_size': account_balance * 0.01 / entry_price,
                'error': str(e)
            }
    
    def get_volatility_dashboard(self) -> Dict[str, Any]:
        """Dashboard volatilità"""
        try:
            dashboard = {
                'models_status': {
                    'garch_fitted': self.garch_model.is_fitted,
                    'egarch_fitted': self.egarch_model.is_fitted,
                    'last_update': self.last_update.isoformat() if self.last_update else None
                },
                'pair_analysis': {}
            }
            
            for pair, data in self.volatility_history.items():
                regime = data['regime']
                
                dashboard['pair_analysis'][pair] = {
                    'current_regime': regime['regime'],
                    'volatility_percentile': regime.get('volatility_percentile', 0.5),
                    'trend': regime.get('trend', 'stable'),
                    'clustering_detected': regime.get('clustering_detected', False),
                    'last_update': data['last_update'].isoformat()
                }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating volatility dashboard: {e}")
            return {}