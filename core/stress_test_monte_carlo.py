"""
Sistema di Stress Testing e Simulazioni Monte Carlo
Testa resilienza del sistema in scenari estremi e valuta rischi di tail events
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional, Tuple
import json
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MarketScenarioGenerator:
    """Generatore di scenari di mercato per stress testing"""
    
    def __init__(self):
        self.scenario_templates = {
            'black_swan': {
                'description': 'Evento cigno nero - crash improvviso',
                'price_shock': -0.30,  # -30%
                'volatility_spike': 5.0,  # 5x normale
                'duration_days': 3,
                'recovery_days': 30
            },
            'flash_crash': {
                'description': 'Flash crash - crollo rapido e recupero',
                'price_shock': -0.15,
                'volatility_spike': 10.0,
                'duration_days': 1,
                'recovery_days': 5
            },
            'bear_market': {
                'description': 'Mercato orso prolungato',
                'price_shock': -0.50,
                'volatility_spike': 2.0,
                'duration_days': 180,
                'recovery_days': 365
            },
            'high_volatility': {
                'description': 'Periodo di alta volatilità',
                'price_shock': 0.0,
                'volatility_spike': 3.0,
                'duration_days': 30,
                'recovery_days': 60
            },
            'liquidity_crisis': {
                'description': 'Crisi di liquidità',
                'price_shock': -0.20,
                'volatility_spike': 4.0,
                'duration_days': 14,
                'recovery_days': 90,
                'liquidity_reduction': 0.8  # 80% riduzione liquidità
            },
            'regulatory_shock': {
                'description': 'Shock normativo',
                'price_shock': -0.25,
                'volatility_spike': 3.0,
                'duration_days': 7,
                'recovery_days': 120
            }
        }
    
    def generate_scenario_path(self, scenario_name: str, initial_price: float, 
                              trading_days: int = 252) -> pd.DataFrame:
        """Genera path di prezzo per scenario specifico"""
        try:
            scenario = self.scenario_templates.get(scenario_name)
            if not scenario:
                return self._generate_normal_path(initial_price, trading_days)
            
            dates = pd.date_range(start=datetime.now(), periods=trading_days, freq='D')
            prices = np.zeros(trading_days)
            volumes = np.zeros(trading_days)
            
            prices[0] = initial_price
            volumes[0] = 1000000  # Volume base
            
            # Parametri scenario
            shock_magnitude = scenario['price_shock']
            vol_spike = scenario['volatility_spike']
            shock_duration = scenario['duration_days']
            recovery_duration = scenario['recovery_days']
            
            base_volatility = 0.02  # 2% giornaliera
            
            for i in range(1, trading_days):
                # Determina fase dello scenario
                if i <= shock_duration:
                    # Fase di shock
                    trend = shock_magnitude / shock_duration
                    volatility = base_volatility * vol_spike
                    
                elif i <= shock_duration + recovery_duration:
                    # Fase di recupero
                    recovery_progress = (i - shock_duration) / recovery_duration
                    trend = -shock_magnitude * recovery_progress / recovery_duration
                    volatility = base_volatility * (vol_spike * (1 - recovery_progress) + 1)
                    
                else:
                    # Fase normale
                    trend = 0.0005  # Leggero trend positivo
                    volatility = base_volatility
                
                # Genera return con jump diffusion
                normal_return = np.random.normal(trend, volatility)
                
                # Possibili jump events
                if np.random.random() < 0.01:  # 1% probabilità jump
                    jump_size = np.random.normal(0, volatility * 3)
                    normal_return += jump_size
                
                prices[i] = prices[i-1] * (1 + normal_return)
                
                # Volume aumenta con volatilità
                volume_multiplier = 1 + (volatility / base_volatility - 1) * 0.5
                volumes[i] = volumes[0] * volume_multiplier * np.random.uniform(0.5, 1.5)
            
            return pd.DataFrame({
                'date': dates,
                'open': prices * np.random.uniform(0.99, 1.01, len(prices)),
                'high': prices * np.random.uniform(1.0, 1.02, len(prices)),
                'low': prices * np.random.uniform(0.98, 1.0, len(prices)),
                'close': prices,
                'volume': volumes,
                'scenario': scenario_name
            })
            
        except Exception as e:
            logger.error(f"Error generating scenario path: {e}")
            return self._generate_normal_path(initial_price, trading_days)
    
    def _generate_normal_path(self, initial_price: float, trading_days: int) -> pd.DataFrame:
        """Genera path normale senza shock"""
        dates = pd.date_range(start=datetime.now(), periods=trading_days, freq='D')
        returns = np.random.normal(0.0005, 0.02, trading_days)
        prices = initial_price * np.cumprod(1 + returns)
        
        return pd.DataFrame({
            'date': dates,
            'open': prices * np.random.uniform(0.99, 1.01, len(prices)),
            'high': prices * np.random.uniform(1.0, 1.02, len(prices)),
            'low': prices * np.random.uniform(0.98, 1.0, len(prices)),
            'close': prices,
            'volume': np.random.uniform(500000, 2000000, len(prices)),
            'scenario': 'normal'
        })

class MonteCarloSimulator:
    """Simulatore Monte Carlo per analisi di rischio"""
    
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations
        self.scenario_generator = MarketScenarioGenerator()
        
    def run_portfolio_simulations(self, initial_portfolio_value: float,
                                 trading_strategy_params: Dict[str, Any],
                                 simulation_days: int = 252) -> Dict[str, Any]:
        """Esegue simulazioni Monte Carlo del portafoglio"""
        try:
            results = {
                'final_values': [],
                'max_drawdowns': [],
                'sharpe_ratios': [],
                'win_rates': [],
                'max_consecutive_losses': [],
                'scenarios_tested': []
            }
            
            # Esegui simulazioni parallele
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                
                for i in range(self.n_simulations):
                    future = executor.submit(
                        self._single_simulation,
                        initial_portfolio_value,
                        trading_strategy_params,
                        simulation_days,
                        i
                    )
                    futures.append(future)
                
                # Raccogli risultati
                for future in futures:
                    sim_result = future.result()
                    if sim_result:
                        results['final_values'].append(sim_result['final_value'])
                        results['max_drawdowns'].append(sim_result['max_drawdown'])
                        results['sharpe_ratios'].append(sim_result['sharpe_ratio'])
                        results['win_rates'].append(sim_result['win_rate'])
                        results['max_consecutive_losses'].append(sim_result['max_consecutive_losses'])
                        results['scenarios_tested'].append(sim_result['scenario'])
            
            # Calcola statistiche aggregate
            final_values = np.array(results['final_values'])
            max_drawdowns = np.array(results['max_drawdowns'])
            
            aggregate_stats = {
                'portfolio_statistics': {
                    'mean_final_value': np.mean(final_values),
                    'median_final_value': np.median(final_values),
                    'std_final_value': np.std(final_values),
                    'min_final_value': np.min(final_values),
                    'max_final_value': np.max(final_values),
                    'probability_of_loss': np.mean(final_values < initial_portfolio_value),
                    'expected_return': np.mean((final_values / initial_portfolio_value - 1) * 100),
                    'return_volatility': np.std((final_values / initial_portfolio_value - 1) * 100)
                },
                'risk_metrics': {
                    'value_at_risk_95': np.percentile(final_values, 5),
                    'value_at_risk_99': np.percentile(final_values, 1),
                    'expected_shortfall_95': np.mean(final_values[final_values <= np.percentile(final_values, 5)]),
                    'maximum_drawdown_mean': np.mean(max_drawdowns),
                    'maximum_drawdown_worst': np.max(max_drawdowns),
                    'tail_ratio': np.percentile(final_values, 95) / np.percentile(final_values, 5)
                },
                'performance_metrics': {
                    'mean_sharpe_ratio': np.mean(results['sharpe_ratios']),
                    'mean_win_rate': np.mean(results['win_rates']),
                    'worst_consecutive_losses': np.max(results['max_consecutive_losses']),
                    'simulations_completed': len(results['final_values'])
                }
            }
            
            return {**results, **aggregate_stats}
            
        except Exception as e:
            logger.error(f"Error running Monte Carlo simulations: {e}")
            return {}
    
    def _single_simulation(self, initial_value: float, strategy_params: Dict[str, Any],
                          simulation_days: int, sim_number: int) -> Optional[Dict[str, Any]]:
        """Esegue singola simulazione"""
        try:
            # Scegli scenario casuale
            scenarios = list(self.scenario_generator.scenario_templates.keys()) + ['normal'] * 3
            scenario = np.random.choice(scenarios)
            
            # Genera path di prezzo
            price_data = self.scenario_generator.generate_scenario_path(
                scenario, 43000.0, simulation_days  # BTC base price
            )
            
            # Simula trading
            portfolio_values = [initial_value]
            trades = []
            consecutive_losses = 0
            max_consecutive_losses = 0
            
            for i in range(1, len(price_data)):
                current_price = price_data.iloc[i]['close']
                prev_price = price_data.iloc[i-1]['close']
                
                # Simula decisione di trading
                price_change = (current_price - prev_price) / prev_price
                
                # Strategia semplificata: momentum con stop loss
                if abs(price_change) > strategy_params.get('entry_threshold', 0.02):
                    trade_direction = 1 if price_change > 0 else -1
                    position_size = strategy_params.get('position_size', 0.1) * portfolio_values[-1]
                    
                    # Simula risultato trade
                    trade_return = np.random.normal(price_change * 0.7, 0.01)  # 70% capture con noise
                    trade_pnl = position_size * trade_return * trade_direction
                    
                    # Stop loss
                    if trade_pnl < -position_size * strategy_params.get('stop_loss', 0.05):
                        trade_pnl = -position_size * strategy_params.get('stop_loss', 0.05)
                    
                    # Take profit
                    if trade_pnl > position_size * strategy_params.get('take_profit', 0.10):
                        trade_pnl = position_size * strategy_params.get('take_profit', 0.10)
                    
                    portfolio_values.append(portfolio_values[-1] + trade_pnl)
                    
                    trades.append({
                        'day': i,
                        'pnl': trade_pnl,
                        'direction': trade_direction,
                        'success': trade_pnl > 0
                    })
                    
                    # Track consecutive losses
                    if trade_pnl < 0:
                        consecutive_losses += 1
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    else:
                        consecutive_losses = 0
                        
                else:
                    portfolio_values.append(portfolio_values[-1])
            
            # Calcola metriche
            portfolio_array = np.array(portfolio_values)
            returns = np.diff(portfolio_array) / portfolio_array[:-1]
            
            # Drawdown
            peak = np.maximum.accumulate(portfolio_array)
            drawdown = (portfolio_array - peak) / peak
            max_drawdown = np.min(drawdown)
            
            # Sharpe ratio
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Win rate
            winning_trades = [t for t in trades if t['success']]
            win_rate = len(winning_trades) / max(1, len(trades))
            
            return {
                'final_value': portfolio_values[-1],
                'max_drawdown': abs(max_drawdown),
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'max_consecutive_losses': max_consecutive_losses,
                'total_trades': len(trades),
                'scenario': scenario,
                'simulation_number': sim_number
            }
            
        except Exception as e:
            logger.error(f"Error in simulation {sim_number}: {e}")
            return None

class StressTestEngine:
    """Motore principale per stress testing"""
    
    def __init__(self):
        self.monte_carlo = MonteCarloSimulator()
        self.stress_scenarios = {
            'extreme_volatility': {
                'volatility_multiplier': 5.0,
                'correlation_breakdown': True,
                'liquidity_stress': 0.5
            },
            'market_crash': {
                'price_shock': -0.40,
                'recovery_time': 180,
                'contagion_effect': True
            },
            'liquidity_crisis': {
                'bid_ask_spread_multiplier': 10.0,
                'market_depth_reduction': 0.8,
                'execution_slippage': 0.05
            },
            'exchange_outage': {
                'outage_duration_hours': 24,
                'affected_exchanges': ['binance', 'kucoin'],
                'price_gaps': 0.02
            }
        }
    
    def run_comprehensive_stress_test(self, portfolio_state: Dict[str, Any],
                                    trading_params: Dict[str, Any]) -> Dict[str, Any]:
        """Esegue test di stress completo"""
        try:
            stress_results = {}
            
            # Test per ogni scenario
            for scenario_name, scenario_params in self.stress_scenarios.items():
                logger.info(f"Running stress test: {scenario_name}")
                
                scenario_result = self._test_scenario(
                    scenario_name, scenario_params, portfolio_state, trading_params
                )
                stress_results[scenario_name] = scenario_result
            
            # Monte Carlo generale
            logger.info("Running Monte Carlo simulations")
            mc_results = self.monte_carlo.run_portfolio_simulations(
                portfolio_state.get('total_value', 10000),
                trading_params,
                simulation_days=126  # 6 mesi
            )
            
            # Analisi aggregata
            aggregate_analysis = self._analyze_aggregate_risk(stress_results, mc_results)
            
            return {
                'stress_test_results': stress_results,
                'monte_carlo_results': mc_results,
                'aggregate_risk_analysis': aggregate_analysis,
                'test_timestamp': datetime.now().isoformat(),
                'recommendations': self._generate_recommendations(stress_results, mc_results)
            }
            
        except Exception as e:
            logger.error(f"Error running comprehensive stress test: {e}")
            return {}
    
    def _test_scenario(self, scenario_name: str, scenario_params: Dict[str, Any],
                      portfolio_state: Dict[str, Any], trading_params: Dict[str, Any]) -> Dict[str, Any]:
        """Testa scenario specifico"""
        try:
            initial_value = portfolio_state.get('total_value', 10000)
            
            # Simula scenario specifico
            if scenario_name == 'extreme_volatility':
                return self._test_volatility_shock(scenario_params, initial_value, trading_params)
            elif scenario_name == 'market_crash':
                return self._test_market_crash(scenario_params, initial_value, trading_params)
            elif scenario_name == 'liquidity_crisis':
                return self._test_liquidity_crisis(scenario_params, initial_value, trading_params)
            elif scenario_name == 'exchange_outage':
                return self._test_exchange_outage(scenario_params, initial_value, trading_params)
            
            return {'status': 'not_implemented'}
            
        except Exception as e:
            logger.error(f"Error testing scenario {scenario_name}: {e}")
            return {'error': str(e)}
    
    def _test_volatility_shock(self, params: Dict[str, Any], initial_value: float,
                              trading_params: Dict[str, Any]) -> Dict[str, Any]:
        """Test shock di volatilità"""
        vol_multiplier = params.get('volatility_multiplier', 5.0)
        
        # Simula 30 giorni di alta volatilità
        daily_returns = []
        portfolio_value = initial_value
        
        for day in range(30):
            # Volatilità aumentata
            daily_vol = 0.02 * vol_multiplier
            daily_return = np.random.normal(0, daily_vol)
            
            # Effetto su portfolio (con position sizing dinamico)
            position_adjustment = 1.0 / vol_multiplier  # Riduci posizioni
            portfolio_impact = daily_return * position_adjustment
            
            portfolio_value *= (1 + portfolio_impact)
            daily_returns.append(portfolio_impact)
        
        returns_array = np.array(daily_returns)
        max_daily_loss = np.min(returns_array)
        volatility = np.std(returns_array)
        
        return {
            'final_portfolio_value': portfolio_value,
            'total_return': (portfolio_value / initial_value - 1) * 100,
            'maximum_daily_loss': max_daily_loss * 100,
            'portfolio_volatility': volatility * 100,
            'days_tested': 30,
            'scenario_severity': 'HIGH' if abs(max_daily_loss) > 0.1 else 'MEDIUM'
        }
    
    def _test_market_crash(self, params: Dict[str, Any], initial_value: float,
                          trading_params: Dict[str, Any]) -> Dict[str, Any]:
        """Test crash di mercato"""
        price_shock = params.get('price_shock', -0.40)
        recovery_time = params.get('recovery_time', 180)
        
        # Simula crash immediato
        crash_impact = price_shock * trading_params.get('max_exposure', 0.8)
        portfolio_after_crash = initial_value * (1 + crash_impact)
        
        # Simula recupero graduale
        daily_recovery_rate = -crash_impact / recovery_time
        portfolio_value = portfolio_after_crash
        
        min_value = portfolio_after_crash
        
        for day in range(recovery_time):
            recovery_return = daily_recovery_rate * np.random.uniform(0.5, 1.5)
            portfolio_value *= (1 + recovery_return)
            min_value = min(min_value, portfolio_value)
        
        max_drawdown = (initial_value - min_value) / initial_value
        
        return {
            'immediate_crash_loss': abs(crash_impact) * 100,
            'minimum_portfolio_value': min_value,
            'maximum_drawdown': max_drawdown * 100,
            'recovery_portfolio_value': portfolio_value,
            'full_recovery_achieved': portfolio_value >= initial_value * 0.95,
            'recovery_time_days': recovery_time,
            'scenario_severity': 'CRITICAL' if max_drawdown > 0.5 else 'HIGH'
        }
    
    def _test_liquidity_crisis(self, params: Dict[str, Any], initial_value: float,
                              trading_params: Dict[str, Any]) -> Dict[str, Any]:
        """Test crisi di liquidità"""
        spread_multiplier = params.get('bid_ask_spread_multiplier', 10.0)
        depth_reduction = params.get('market_depth_reduction', 0.8)
        execution_slippage = params.get('execution_slippage', 0.05)
        
        # Simula trading in condizioni di bassa liquidità
        normal_spread = 0.001
        crisis_spread = normal_spread * spread_multiplier
        
        # Costo per trade aumentato
        trades_per_day = trading_params.get('daily_trades', 5)
        trading_days = 14  # 2 settimane di crisi
        
        total_trading_cost = 0
        portfolio_value = initial_value
        
        for day in range(trading_days):
            daily_trading_volume = portfolio_value * trading_params.get('daily_volume_ratio', 0.1)
            
            # Costi aggiuntivi
            spread_cost = daily_trading_volume * crisis_spread
            slippage_cost = daily_trading_volume * execution_slippage
            
            total_cost = spread_cost + slippage_cost
            portfolio_value -= total_cost
            total_trading_cost += total_cost
        
        liquidity_impact = total_trading_cost / initial_value
        
        return {
            'total_liquidity_cost': total_trading_cost,
            'liquidity_impact_percentage': liquidity_impact * 100,
            'final_portfolio_value': portfolio_value,
            'average_daily_cost': total_trading_cost / trading_days,
            'crisis_duration_days': trading_days,
            'scenario_severity': 'HIGH' if liquidity_impact > 0.1 else 'MEDIUM'
        }
    
    def _test_exchange_outage(self, params: Dict[str, Any], initial_value: float,
                             trading_params: Dict[str, Any]) -> Dict[str, Any]:
        """Test interruzione exchange"""
        outage_hours = params.get('outage_duration_hours', 24)
        affected_exchanges = params.get('affected_exchanges', ['binance'])
        price_gaps = params.get('price_gaps', 0.02)
        
        # Calcola esposizione su exchange interessati
        exchange_exposure = 0.6  # Assumiamo 60% su exchange principali
        
        # Perdite da gap di prezzo alla riapertura
        gap_loss = exchange_exposure * price_gaps * initial_value
        
        # Opportunità perse durante outage
        missed_opportunities = (outage_hours / 24) * trading_params.get('daily_expected_return', 0.001) * initial_value
        
        total_impact = gap_loss + missed_opportunities
        final_value = initial_value - total_impact
        
        return {
            'outage_duration_hours': outage_hours,
            'affected_exchange_exposure': exchange_exposure * 100,
            'price_gap_loss': gap_loss,
            'missed_opportunity_cost': missed_opportunities,
            'total_outage_impact': total_impact,
            'final_portfolio_value': final_value,
            'impact_percentage': (total_impact / initial_value) * 100,
            'scenario_severity': 'MEDIUM'
        }
    
    def _analyze_aggregate_risk(self, stress_results: Dict[str, Any], 
                               mc_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analizza rischio aggregato"""
        try:
            # Estrai worst case scenarios
            worst_case_losses = []
            
            for scenario_name, result in stress_results.items():
                if 'final_portfolio_value' in result:
                    initial_value = 10000  # Assumiamo valore iniziale
                    loss_pct = (1 - result['final_portfolio_value'] / initial_value) * 100
                    worst_case_losses.append(loss_pct)
            
            # Statistiche Monte Carlo
            mc_stats = mc_results.get('portfolio_statistics', {})
            mc_risk = mc_results.get('risk_metrics', {})
            
            return {
                'worst_case_scenario_loss': max(worst_case_losses) if worst_case_losses else 0,
                'average_stress_loss': np.mean(worst_case_losses) if worst_case_losses else 0,
                'monte_carlo_var_95': mc_risk.get('value_at_risk_95', 0),
                'monte_carlo_expected_shortfall': mc_risk.get('expected_shortfall_95', 0),
                'probability_of_loss': mc_stats.get('probability_of_loss', 0),
                'tail_risk_ratio': mc_risk.get('tail_ratio', 1),
                'overall_risk_rating': self._calculate_risk_rating(worst_case_losses, mc_risk),
                'diversification_benefit': self._calculate_diversification_benefit(stress_results)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing aggregate risk: {e}")
            return {}
    
    def _calculate_risk_rating(self, stress_losses: List[float], mc_risk: Dict[str, Any]) -> str:
        """Calcola rating di rischio complessivo"""
        if not stress_losses:
            return 'UNKNOWN'
        
        max_loss = max(stress_losses)
        var_95 = mc_risk.get('value_at_risk_95', 0)
        
        if max_loss > 50 or var_95 < 5000:  # >50% loss or VaR < 50% of initial
            return 'HIGH'
        elif max_loss > 25 or var_95 < 7500:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _calculate_diversification_benefit(self, stress_results: Dict[str, Any]) -> float:
        """Calcola beneficio della diversificazione"""
        # Semplificato: confronta performance tra scenari diversi
        performances = []
        for result in stress_results.values():
            if 'final_portfolio_value' in result:
                performances.append(result['final_portfolio_value'])
        
        if len(performances) > 1:
            return (1 - np.std(performances) / np.mean(performances)) * 100
        return 0
    
    def _generate_recommendations(self, stress_results: Dict[str, Any], 
                                 mc_results: Dict[str, Any]) -> List[str]:
        """Genera raccomandazioni basate sui test"""
        recommendations = []
        
        # Analizza risultati stress test
        for scenario, result in stress_results.items():
            if result.get('scenario_severity') == 'CRITICAL':
                recommendations.append(f"CRITICO: Ridurre esposizione per scenario {scenario}")
            elif result.get('scenario_severity') == 'HIGH':
                recommendations.append(f"ALTO RISCHIO: Implementare hedging per {scenario}")
        
        # Analizza Monte Carlo
        mc_risk = mc_results.get('risk_metrics', {})
        prob_loss = mc_results.get('portfolio_statistics', {}).get('probability_of_loss', 0)
        
        if prob_loss > 0.4:
            recommendations.append("Probabilità di perdita elevata - rivedere strategia")
        
        if mc_risk.get('maximum_drawdown_worst', 0) > 0.3:
            recommendations.append("Drawdown massimo eccessivo - implementare stop loss più rigidi")
        
        if not recommendations:
            recommendations.append("Portfolio mostra resilienza adeguata negli stress test")
        
        return recommendations