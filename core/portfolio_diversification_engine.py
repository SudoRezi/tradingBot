"""
Portfolio Diversification Engine con Risk Parity e Correlazioni
Implementa ribilanciamento automatico e ottimizzazione Markowitz
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import logging
from scipy.optimize import minimize
from scipy.stats import pearsonr
import json

logger = logging.getLogger(__name__)

class CorrelationAnalyzer:
    """Analizzatore di correlazioni tra asset"""
    
    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days
        self.price_history = {}
        self.correlation_matrix = None
        self.last_update = None
        
    def update_price_data(self, symbol: str, prices: pd.Series):
        """Aggiorna dati di prezzo per calcolo correlazioni"""
        self.price_history[symbol] = prices.tail(self.lookback_days)
        
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calcola matrice di correlazione tra asset"""
        if len(self.price_history) < 2:
            return pd.DataFrame()
            
        # Allinea tutte le serie temporali
        price_df = pd.DataFrame(self.price_history)
        price_df = price_df.dropna()
        
        if len(price_df) < 10:  # Servono almeno 10 osservazioni
            return pd.DataFrame()
        
        # Calcola rendimenti
        returns = price_df.pct_change().dropna()
        
        # Calcola correlazioni
        self.correlation_matrix = returns.corr()
        self.last_update = datetime.now()
        
        return self.correlation_matrix
    
    def get_uncorrelated_pairs(self, threshold: float = 0.3) -> List[Tuple[str, str, float]]:
        """Trova coppie di asset poco correlate"""
        if self.correlation_matrix is None:
            self.calculate_correlation_matrix()
            
        if self.correlation_matrix.empty:
            return []
        
        uncorrelated_pairs = []
        symbols = list(self.correlation_matrix.columns)
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                corr = abs(self.correlation_matrix.iloc[i, j])
                if corr < threshold:
                    uncorrelated_pairs.append((symbols[i], symbols[j], corr))
        
        # Ordina per correlazione crescente
        return sorted(uncorrelated_pairs, key=lambda x: x[2])
    
    def get_highly_correlated_pairs(self, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """Trova coppie di asset altamente correlate (da evitare)"""
        if self.correlation_matrix is None:
            self.calculate_correlation_matrix()
            
        if self.correlation_matrix.empty:
            return []
        
        high_corr_pairs = []
        symbols = list(self.correlation_matrix.columns)
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                corr = abs(self.correlation_matrix.iloc[i, j])
                if corr > threshold:
                    high_corr_pairs.append((symbols[i], symbols[j], corr))
        
        return sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)

class RiskParityOptimizer:
    """Ottimizzatore Risk Parity per allocazione equa del rischio"""
    
    def __init__(self):
        self.min_weight = 0.01  # 1% peso minimo
        self.max_weight = 0.4   # 40% peso massimo per singolo asset
        
    def calculate_risk_parity_weights(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Calcola pesi risk parity"""
        if returns.empty or len(returns.columns) < 2:
            return {}
        
        # Calcola matrice di covarianza
        cov_matrix = returns.cov().values
        n_assets = len(cov_matrix)
        
        # Pesi iniziali equiponderati
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Constraint: somma pesi = 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        
        # Bounds: peso minimo e massimo
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        
        # Funzione obiettivo: minimizza la differenza tra contributi di rischio
        def objective(weights):
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            if portfolio_risk == 0:
                return 1e6  # Penalità alta se rischio è zero
            
            # Contributi di rischio marginali
            marginal_risk = np.dot(cov_matrix, weights) / portfolio_risk
            risk_contributions = weights * marginal_risk
            
            # Target: contributi equali
            target_contrib = portfolio_risk / n_assets
            deviations = (risk_contributions - target_contrib) ** 2
            
            return np.sum(deviations)
        
        # Ottimizzazione
        try:
            result = minimize(
                objective, 
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                weights = result.x
                # Normalizza per sicurezza
                weights = weights / np.sum(weights)
                
                return {
                    symbol: float(weight) 
                    for symbol, weight in zip(returns.columns, weights)
                }
            else:
                logger.warning("Risk parity optimization failed, using equal weights")
                equal_weight = 1.0 / n_assets
                return {symbol: equal_weight for symbol in returns.columns}
                
        except Exception as e:
            logger.error(f"Risk parity calculation error: {e}")
            equal_weight = 1.0 / len(returns.columns)
            return {symbol: equal_weight for symbol in returns.columns}

class MarkowitzOptimizer:
    """Ottimizzatore Markowitz per portfolio efficiente"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% risk-free rate
        
    def calculate_efficient_frontier(self, returns: pd.DataFrame, 
                                   target_returns: List[float]) -> List[Dict[str, Any]]:
        """Calcola frontiera efficiente"""
        if returns.empty:
            return []
        
        cov_matrix = returns.cov().values
        mean_returns = returns.mean().values
        n_assets = len(mean_returns)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            # Constraint: target return
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Somma = 1
                {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target_return}  # Target return
            ]
            
            bounds = [(0.0, 1.0) for _ in range(n_assets)]
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            # Minimizza varianza
            def objective(weights):
                return np.dot(weights, np.dot(cov_matrix, weights))
            
            try:
                result = minimize(
                    objective,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                
                if result.success:
                    weights = result.x
                    portfolio_return = np.dot(weights, mean_returns)
                    portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                    sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
                    
                    efficient_portfolios.append({
                        'target_return': target_return,
                        'actual_return': portfolio_return,
                        'risk': portfolio_risk,
                        'sharpe_ratio': sharpe_ratio,
                        'weights': {
                            symbol: float(weight) 
                            for symbol, weight in zip(returns.columns, weights)
                        }
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to optimize for target return {target_return}: {e}")
        
        return efficient_portfolios
    
    def find_max_sharpe_portfolio(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Trova portfolio con Sharpe ratio massimo"""
        if returns.empty:
            return {}
        
        cov_matrix = returns.cov().values
        mean_returns = returns.mean().values
        n_assets = len(mean_returns)
        
        # Constraint: somma pesi = 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Massimizza Sharpe ratio (minimizza -Sharpe)
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            if portfolio_risk == 0:
                return -1e6
            
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_risk
            return -sharpe
        
        try:
            result = minimize(
                negative_sharpe,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                weights = result.x
                portfolio_return = np.dot(weights, mean_returns)
                portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
                
                return {
                    'weights': {
                        symbol: float(weight) 
                        for symbol, weight in zip(returns.columns, weights)
                    },
                    'expected_return': portfolio_return,
                    'risk': portfolio_risk,
                    'sharpe_ratio': sharpe_ratio
                }
                
        except Exception as e:
            logger.error(f"Max Sharpe optimization failed: {e}")
        
        # Fallback: equal weights
        equal_weight = 1.0 / n_assets
        equal_weights = {symbol: equal_weight for symbol in returns.columns}
        
        portfolio_return = np.dot(list(equal_weights.values()), mean_returns)
        portfolio_risk = np.sqrt(np.dot(list(equal_weights.values()), np.dot(cov_matrix, list(equal_weights.values()))))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
        
        return {
            'weights': equal_weights,
            'expected_return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio
        }

class PortfolioDiversificationEngine:
    """Motore principale per diversificazione portfolio"""
    
    def __init__(self, rebalance_threshold: float = 0.05):
        self.correlation_analyzer = CorrelationAnalyzer()
        self.risk_parity_optimizer = RiskParityOptimizer()
        self.markowitz_optimizer = MarkowitzOptimizer()
        
        self.rebalance_threshold = rebalance_threshold  # 5% soglia per ribilanciamento
        self.target_weights = {}
        self.current_weights = {}
        self.last_rebalance = None
        
        # Asset universe
        self.asset_universe = {
            'crypto_major': ['BTC/USDT', 'ETH/USDT'],
            'crypto_alt': ['KAS/USDT', 'AVAX/USDT', 'SOL/USDT'],
            'stablecoins': ['USDC/USDT', 'DAI/USDT'],
            'defi_tokens': ['UNI/USDT', 'AAVE/USDT', 'COMP/USDT']
        }
        
    def update_market_data(self, price_data: Dict[str, pd.Series]):
        """Aggiorna dati di mercato per tutti gli asset"""
        for symbol, prices in price_data.items():
            self.correlation_analyzer.update_price_data(symbol, prices)
    
    def analyze_diversification_opportunity(self, current_portfolio: Dict[str, float]) -> Dict[str, Any]:
        """Analizza opportunità di diversificazione"""
        
        # Calcola correlazioni
        correlation_matrix = self.correlation_analyzer.calculate_correlation_matrix()
        
        if correlation_matrix.empty:
            return {
                'diversification_score': 0.0,
                'recommendations': ['Insufficient data for analysis'],
                'correlation_analysis': {}
            }
        
        # Calcola diversification score
        portfolio_symbols = list(current_portfolio.keys())
        weights = np.array([current_portfolio.get(s, 0) for s in portfolio_symbols])
        
        # Score basato su correlazioni ponderate
        diversification_score = self._calculate_diversification_score(
            portfolio_symbols, weights, correlation_matrix
        )
        
        # Trova asset correlati e non correlati
        uncorrelated_pairs = self.correlation_analyzer.get_uncorrelated_pairs()
        highly_correlated_pairs = self.correlation_analyzer.get_highly_correlated_pairs()
        
        # Genera raccomandazioni
        recommendations = self._generate_diversification_recommendations(
            current_portfolio, uncorrelated_pairs, highly_correlated_pairs
        )
        
        return {
            'diversification_score': diversification_score,
            'correlation_matrix': correlation_matrix.to_dict(),
            'uncorrelated_opportunities': uncorrelated_pairs[:5],
            'correlation_risks': highly_correlated_pairs[:5],
            'recommendations': recommendations,
            'portfolio_concentration': self._calculate_concentration_metrics(current_portfolio)
        }
    
    def calculate_optimal_allocation(self, price_data: Dict[str, pd.Series], 
                                   method: str = 'risk_parity') -> Dict[str, Any]:
        """Calcola allocazione ottimale"""
        
        if not price_data:
            return {'success': False, 'reason': 'no_data'}
        
        # Prepara dati rendimenti
        returns_data = {}
        for symbol, prices in price_data.items():
            if len(prices) > 1:
                returns = prices.pct_change().dropna()
                if len(returns) > 10:  # Almeno 10 osservazioni
                    returns_data[symbol] = returns
        
        if len(returns_data) < 2:
            return {'success': False, 'reason': 'insufficient_assets'}
        
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 10:
            return {'success': False, 'reason': 'insufficient_data'}
        
        # Calcola allocazione basata su metodo scelto
        if method == 'risk_parity':
            weights = self.risk_parity_optimizer.calculate_risk_parity_weights(returns_df)
            optimization_details = {'method': 'risk_parity'}
            
        elif method == 'max_sharpe':
            result = self.markowitz_optimizer.find_max_sharpe_portfolio(returns_df)
            weights = result.get('weights', {})
            optimization_details = {
                'method': 'max_sharpe',
                'expected_return': result.get('expected_return', 0),
                'risk': result.get('risk', 0),
                'sharpe_ratio': result.get('sharpe_ratio', 0)
            }
            
        elif method == 'equal_weight':
            n_assets = len(returns_df.columns)
            equal_weight = 1.0 / n_assets
            weights = {symbol: equal_weight for symbol in returns_df.columns}
            optimization_details = {'method': 'equal_weight'}
            
        else:
            return {'success': False, 'reason': 'unknown_method'}
        
        if not weights:
            return {'success': False, 'reason': 'optimization_failed'}
        
        # Salva target weights
        self.target_weights = weights
        
        return {
            'success': True,
            'target_weights': weights,
            'optimization_details': optimization_details,
            'diversification_metrics': self._calculate_portfolio_metrics(returns_df, weights)
        }
    
    def check_rebalancing_needed(self, current_portfolio_values: Dict[str, float]) -> Dict[str, Any]:
        """Controlla se serve ribilanciamento"""
        
        if not self.target_weights:
            return {'rebalancing_needed': False, 'reason': 'no_target_weights'}
        
        # Calcola pesi correnti
        total_value = sum(current_portfolio_values.values())
        if total_value == 0:
            return {'rebalancing_needed': False, 'reason': 'no_portfolio_value'}
        
        current_weights = {
            symbol: value / total_value 
            for symbol, value in current_portfolio_values.items()
        }
        
        self.current_weights = current_weights
        
        # Calcola deviazioni dai target
        deviations = {}
        max_deviation = 0
        
        for symbol in self.target_weights:
            target = self.target_weights[symbol]
            current = current_weights.get(symbol, 0)
            deviation = abs(current - target)
            deviations[symbol] = {
                'target': target,
                'current': current,
                'deviation': deviation,
                'deviation_pct': deviation / target if target > 0 else 0
            }
            max_deviation = max(max_deviation, deviation)
        
        # Determina se serve ribilanciamento
        needs_rebalancing = max_deviation > self.rebalance_threshold
        
        if needs_rebalancing:
            rebalancing_actions = self._calculate_rebalancing_actions(
                current_weights, self.target_weights, total_value
            )
        else:
            rebalancing_actions = {}
        
        return {
            'rebalancing_needed': needs_rebalancing,
            'max_deviation': max_deviation,
            'threshold': self.rebalance_threshold,
            'deviations': deviations,
            'rebalancing_actions': rebalancing_actions,
            'total_portfolio_value': total_value
        }
    
    def execute_rebalancing(self, current_portfolio_values: Dict[str, float]) -> Dict[str, Any]:
        """Esegue ribilanciamento portfolio"""
        
        rebalance_check = self.check_rebalancing_needed(current_portfolio_values)
        
        if not rebalance_check['rebalancing_needed']:
            return {
                'success': False,
                'reason': 'rebalancing_not_needed',
                'max_deviation': rebalance_check['max_deviation']
            }
        
        actions = rebalance_check['rebalancing_actions']
        self.last_rebalance = datetime.now()
        
        logger.info(f"Executing portfolio rebalancing with {len(actions)} actions")
        
        return {
            'success': True,
            'rebalancing_time': self.last_rebalance,
            'actions_executed': len(actions),
            'actions': actions,
            'previous_weights': self.current_weights,
            'new_target_weights': self.target_weights
        }
    
    def _calculate_diversification_score(self, symbols: List[str], weights: np.ndarray,
                                       correlation_matrix: pd.DataFrame) -> float:
        """Calcola score di diversificazione (0-1, 1 = massima diversificazione)"""
        if len(symbols) < 2:
            return 0.0
        
        # Correlazione media ponderata
        total_correlation = 0
        total_weight_pairs = 0
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i != j and symbol1 in correlation_matrix.index and symbol2 in correlation_matrix.columns:
                    corr = abs(correlation_matrix.loc[symbol1, symbol2])
                    weight_product = weights[i] * weights[j]
                    total_correlation += corr * weight_product
                    total_weight_pairs += weight_product
        
        if total_weight_pairs == 0:
            return 0.5  # Neutral score se non ci sono dati
        
        avg_correlation = total_correlation / total_weight_pairs
        
        # Score inverso alla correlazione (bassa correlazione = alta diversificazione)
        diversification_score = 1 - avg_correlation
        
        return max(0, min(1, diversification_score))
    
    def _generate_diversification_recommendations(self, portfolio: Dict[str, float],
                                                uncorrelated_pairs: List[Tuple[str, str, float]],
                                                correlated_pairs: List[Tuple[str, str, float]]) -> List[str]:
        """Genera raccomandazioni per migliorare diversificazione"""
        recommendations = []
        
        # Concentrazione eccessiva
        max_weight = max(portfolio.values()) if portfolio else 0
        if max_weight > 0.5:
            recommendations.append(f"Portfolio troppo concentrato: ridurre esposizione massima dal {max_weight:.1%}")
        
        # Asset correlati nel portfolio
        portfolio_symbols = set(portfolio.keys())
        for symbol1, symbol2, corr in correlated_pairs:
            if symbol1 in portfolio_symbols and symbol2 in portfolio_symbols:
                recommendations.append(f"Ridurre correlazione: {symbol1} e {symbol2} sono correlati al {corr:.1%}")
        
        # Opportunità di diversificazione
        all_symbols = set()
        for symbol1, symbol2, _ in uncorrelated_pairs:
            all_symbols.add(symbol1)
            all_symbols.add(symbol2)
        
        missing_uncorrelated = all_symbols - portfolio_symbols
        if missing_uncorrelated:
            top_missing = list(missing_uncorrelated)[:3]
            recommendations.append(f"Considerare aggiunta di: {', '.join(top_missing)} per diversificazione")
        
        # Asset class diversification
        crypto_weight = sum(w for s, w in portfolio.items() if any(crypto in s for crypto in ['BTC', 'ETH', 'KAS']))
        if crypto_weight > 0.8:
            recommendations.append("Considerare diversificazione oltre le criptovalute")
        
        return recommendations[:5]  # Top 5 raccomandazioni
    
    def _calculate_concentration_metrics(self, portfolio: Dict[str, float]) -> Dict[str, float]:
        """Calcola metriche di concentrazione portfolio"""
        if not portfolio:
            return {}
        
        weights = list(portfolio.values())
        
        # Herfindahl-Hirschman Index
        hhi = sum(w**2 for w in weights)
        
        # Effective number of assets
        effective_assets = 1 / hhi if hhi > 0 else 0
        
        # Max weight
        max_weight = max(weights)
        
        # Weight distribution
        sorted_weights = sorted(weights, reverse=True)
        top_3_concentration = sum(sorted_weights[:3]) if len(sorted_weights) >= 3 else sum(sorted_weights)
        
        return {
            'herfindahl_index': hhi,
            'effective_number_assets': effective_assets,
            'max_weight': max_weight,
            'top_3_concentration': top_3_concentration,
            'number_of_assets': len(portfolio)
        }
    
    def _calculate_portfolio_metrics(self, returns_df: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, float]:
        """Calcola metriche del portfolio ottimizzato"""
        if returns_df.empty or not weights:
            return {}
        
        # Allinea pesi con returns
        aligned_weights = np.array([weights.get(col, 0) for col in returns_df.columns])
        aligned_weights = aligned_weights / np.sum(aligned_weights)  # Normalizza
        
        # Rendimenti portfolio
        portfolio_returns = returns_df.dot(aligned_weights)
        
        # Metriche
        annual_return = portfolio_returns.mean() * 252  # Annualizzato
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Max drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': abs(max_drawdown),
            'total_return': (cumulative_returns.iloc[-1] - 1) if len(cumulative_returns) > 0 else 0
        }
    
    def _calculate_rebalancing_actions(self, current_weights: Dict[str, float],
                                     target_weights: Dict[str, float],
                                     total_value: float) -> Dict[str, Dict[str, float]]:
        """Calcola azioni di ribilanciamento necessarie"""
        actions = {}
        
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        
        for symbol in all_symbols:
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            
            difference = target - current
            value_difference = difference * total_value
            
            if abs(value_difference) > total_value * 0.001:  # Soglia minima 0.1%
                action_type = 'buy' if value_difference > 0 else 'sell'
                actions[symbol] = {
                    'action': action_type,
                    'current_weight': current,
                    'target_weight': target,
                    'weight_difference': difference,
                    'value_difference': abs(value_difference),
                    'percentage_change': (target / current - 1) if current > 0 else float('inf')
                }
        
        return actions
    
    def get_diversification_dashboard(self) -> Dict[str, Any]:
        """Dashboard per diversificazione portfolio"""
        
        correlation_matrix = self.correlation_analyzer.correlation_matrix
        
        return {
            'target_weights': self.target_weights,
            'current_weights': self.current_weights,
            'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None,
            'rebalance_threshold': self.rebalance_threshold,
            'correlation_analysis': {
                'matrix_available': correlation_matrix is not None and not correlation_matrix.empty,
                'last_correlation_update': self.correlation_analyzer.last_update.isoformat() if self.correlation_analyzer.last_update else None,
                'uncorrelated_pairs': len(self.correlation_analyzer.get_uncorrelated_pairs()),
                'highly_correlated_pairs': len(self.correlation_analyzer.get_highly_correlated_pairs())
            },
            'asset_universe': self.asset_universe,
            'concentration_metrics': self._calculate_concentration_metrics(self.current_weights) if self.current_weights else {}
        }