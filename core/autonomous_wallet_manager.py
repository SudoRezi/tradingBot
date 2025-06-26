"""
Autonomous Wallet Manager - Gestione AI completa dei fondi exchange
Rileva automaticamente tutti i saldi disponibili e gestisce l'allocazione ottimale
"""

import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from utils.currency_converter import CurrencyConverter
from utils.logger import setup_logger

logger = setup_logger('autonomous_wallet')

class AutonomousWalletManager:
    """Manager autonomo per rilevamento e gestione fondi exchange"""
    
    def __init__(self, api_key: str, api_secret: str, exchange_name: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange_name = exchange_name
        self.converter = CurrencyConverter()
        self.wallet_cache = {}
        self.last_scan = None
        self.min_balance_usd = 1.0  # Ignora saldi sotto $1
        
    def scan_all_balances(self) -> Dict[str, Any]:
        """Scansiona automaticamente tutti i saldi disponibili nell'exchange"""
        try:
            logger.info(f"🔍 Scanning all balances on {self.exchange_name}")
            
            # Simulated balance detection for multiple currencies
            # In produzione, questo userebbe le API reali dell'exchange
            detected_balances = self._simulate_balance_detection()
            
            # Filtra saldi significativi
            significant_balances = self._filter_significant_balances(detected_balances)
            
            # Calcola valori USD
            portfolio_summary = self._calculate_portfolio_summary(significant_balances)
            
            # Cache risultati
            self.wallet_cache = {
                'balances': significant_balances,
                'summary': portfolio_summary,
                'last_updated': datetime.now().isoformat(),
                'exchange': self.exchange_name
            }
            
            self.last_scan = datetime.now()
            
            logger.info(f"✅ Detected {len(significant_balances)} currencies with total value ${portfolio_summary['total_usd_value']:,.2f}")
            
            return self.wallet_cache
            
        except Exception as e:
            logger.error(f"Error scanning balances: {e}")
            return {}
    
    def _simulate_balance_detection(self) -> Dict[str, float]:
        """Simula rilevamento saldi reali (sostituire con API exchange)"""
        # Esempio di saldi che potrebbero essere trovati su un exchange reale
        possible_balances = {
            'USDT': 1250.75,
            'BTC': 0.15,
            'ETH': 2.34,
            'SOL': 45.2,
            'PEPE': 1500000.0,
            'DOGE': 850.0,
            'MATIC': 125.5,
            'ADA': 450.0,
            'LINK': 15.8,
            'UNI': 8.2,
            'AAVE': 2.1,
            'BNB': 3.5,
            'AVAX': 12.3,
            'DOT': 25.7,
            'SHIB': 2500000.0
        }
        
        # Restituisce saldi casuali per simulazione
        import random
        detected = {}
        for currency, max_amount in possible_balances.items():
            if random.random() > 0.6:  # 40% chance di avere questa moneta
                detected[currency] = round(random.uniform(0.1, max_amount), 8)
        
        return detected
    
    def _filter_significant_balances(self, balances: Dict[str, float]) -> Dict[str, float]:
        """Filtra solo i saldi che superano la soglia minima in USD"""
        significant = {}
        
        for currency, amount in balances.items():
            usd_value = self.converter.convert_amount(amount, currency, 'USD')
            if usd_value and usd_value >= self.min_balance_usd:
                significant[currency] = amount
                logger.info(f"  📊 {currency}: {amount:,.8f} (≈ ${usd_value:,.2f} USD)")
        
        return significant
    
    def _calculate_portfolio_summary(self, balances: Dict[str, float]) -> Dict[str, Any]:
        """Calcola summary completo del portafoglio"""
        total_usd = 0.0
        currency_breakdown = {}
        risk_analysis = {}
        
        # Classificazione rischi
        risk_categories = {
            'low': ['USDT', 'USDC', 'BUSD', 'DAI', 'FDUSD', 'TUSD'],
            'medium': ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'AVAX', 'DOT', 'MATIC', 'LINK'],
            'high': ['UNI', 'AAVE', 'COMP', 'MKR', 'CRV', 'SUSHI', 'YFI'],
            'very_high': ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF']
        }
        
        for currency, amount in balances.items():
            usd_value = self.converter.convert_amount(amount, currency, 'USD')
            if usd_value:
                total_usd += usd_value
                currency_breakdown[currency] = {
                    'amount': amount,
                    'usd_value': usd_value,
                    'percentage': 0  # Calcolato dopo
                }
                
                # Classifica rischio
                risk_level = 'medium'  # default
                for risk, currencies in risk_categories.items():
                    if currency in currencies:
                        risk_level = risk
                        break
                
                currency_breakdown[currency]['risk_level'] = risk_level
        
        # Calcola percentuali
        for currency in currency_breakdown:
            if total_usd > 0:
                currency_breakdown[currency]['percentage'] = (
                    currency_breakdown[currency]['usd_value'] / total_usd * 100
                )
        
        # Analisi rischio portafoglio
        risk_analysis = self._analyze_portfolio_risk(currency_breakdown, total_usd)
        
        return {
            'total_usd_value': total_usd,
            'currency_count': len(balances),
            'currency_breakdown': currency_breakdown,
            'risk_analysis': risk_analysis,
            'largest_holding': max(currency_breakdown.items(), 
                                 key=lambda x: x[1]['usd_value']) if currency_breakdown else None,
            'diversification_score': self._calculate_diversification_score(currency_breakdown)
        }
    
    def _analyze_portfolio_risk(self, breakdown: Dict[str, Dict], total_usd: float) -> Dict[str, Any]:
        """Analizza il profilo di rischio del portafoglio"""
        risk_weights = {'low': 1, 'medium': 3, 'high': 6, 'very_high': 9}
        weighted_risk = 0.0
        risk_distribution = {'low': 0, 'medium': 0, 'high': 0, 'very_high': 0}
        
        for currency, data in breakdown.items():
            risk_level = data['risk_level']
            risk_contribution = (data['usd_value'] / total_usd) * risk_weights[risk_level]
            weighted_risk += risk_contribution
            risk_distribution[risk_level] += data['percentage']
        
        # Classifica rischio generale
        if weighted_risk <= 2:
            overall_risk = 'Conservative'
        elif weighted_risk <= 4:
            overall_risk = 'Moderate'
        elif weighted_risk <= 6:
            overall_risk = 'Aggressive'
        else:
            overall_risk = 'Very Aggressive'
        
        return {
            'overall_risk': overall_risk,
            'risk_score': round(weighted_risk, 2),
            'risk_distribution': risk_distribution,
            'recommendations': self._get_risk_recommendations(overall_risk, risk_distribution)
        }
    
    def _calculate_diversification_score(self, breakdown: Dict[str, Dict]) -> float:
        """Calcola score di diversificazione (0-100)"""
        if len(breakdown) <= 1:
            return 0.0
        
        # Herfindahl Index inverso per diversificazione
        hhi = sum((data['percentage'] / 100) ** 2 for data in breakdown.values())
        diversification = (1 - hhi) * 100
        
        # Bonus per avere diverse categorie di asset
        categories = set(data['risk_level'] for data in breakdown.values())
        category_bonus = len(categories) * 5
        
        return min(100, diversification + category_bonus)
    
    def _get_risk_recommendations(self, overall_risk: str, distribution: Dict[str, float]) -> List[str]:
        """Genera raccomandazioni basate sul profilo di rischio"""
        recommendations = []
        
        if overall_risk == 'Very Aggressive':
            recommendations.append("Considera di aumentare la percentuale di stablecoin per stabilità")
            recommendations.append("Riduci esposizione ai meme coin se supera il 20%")
        elif overall_risk == 'Conservative':
            recommendations.append("Potresti considerare una piccola allocazione in crypto major per crescita")
        
        if distribution['very_high'] > 30:
            recommendations.append("Attenzione: alta concentrazione in asset ad alto rischio")
        
        if distribution['low'] < 10:
            recommendations.append("Aggiungi stablecoin per emergency fund")
        
        return recommendations
    
    def get_optimal_allocation_strategy(self) -> Dict[str, Any]:
        """Genera strategia di allocazione ottimale basata sui fondi disponibili"""
        if not self.wallet_cache:
            self.scan_all_balances()
        
        portfolio = self.wallet_cache.get('summary', {})
        balances = self.wallet_cache.get('balances', {})
        
        strategy = {
            'current_portfolio': portfolio,
            'rebalancing_suggestions': self._generate_rebalancing_suggestions(balances, portfolio),
            'trading_allocation': self._calculate_trading_allocation(balances, portfolio),
            'risk_management': self._generate_risk_management_rules(portfolio),
            'auto_actions': self._plan_autonomous_actions(balances, portfolio)
        }
        
        return strategy
    
    def _generate_rebalancing_suggestions(self, balances: Dict[str, float], 
                                        portfolio: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera suggerimenti per ribilanciamento portafoglio"""
        suggestions = []
        breakdown = portfolio.get('currency_breakdown', {})
        
        # Identifica concentrazioni eccessive
        for currency, data in breakdown.items():
            if data['percentage'] > 40:
                suggestions.append({
                    'action': 'reduce_position',
                    'currency': currency,
                    'current_percentage': data['percentage'],
                    'suggested_percentage': 25,
                    'reason': 'Ridurre concentrazione eccessiva'
                })
        
        # Suggerisci diversificazione se troppo concentrato
        if portfolio.get('diversification_score', 0) < 30:
            suggestions.append({
                'action': 'diversify',
                'recommendation': 'Aumentare diversificazione con asset correlati negativamente',
                'target_currencies': ['BTC', 'ETH', 'USDT']
            })
        
        return suggestions
    
    def _calculate_trading_allocation(self, balances: Dict[str, float], 
                                    portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Calcola quanto capitale destinare al trading attivo"""
        total_usd = portfolio.get('total_usd_value', 0)
        risk_level = portfolio.get('risk_analysis', {}).get('overall_risk', 'Moderate')
        
        # Percentuali di trading basate sul rischio
        trading_percentages = {
            'Conservative': 15,
            'Moderate': 25,
            'Aggressive': 35,
            'Very Aggressive': 45
        }
        
        trading_percentage = trading_percentages.get(risk_level, 25)
        trading_amount_usd = total_usd * (trading_percentage / 100)
        
        # Identifica quali asset usare per trading
        stable_coins = {k: v for k, v in balances.items() 
                       if k in ['USDT', 'USDC', 'BUSD', 'DAI']}
        major_crypto = {k: v for k, v in balances.items() 
                       if k in ['BTC', 'ETH', 'BNB', 'SOL']}
        
        return {
            'trading_percentage': trading_percentage,
            'trading_amount_usd': trading_amount_usd,
            'recommended_base_currencies': list(stable_coins.keys()),
            'recommended_trading_pairs': self._suggest_trading_pairs(balances),
            'reserved_amount_usd': total_usd - trading_amount_usd,
            'emergency_fund_usd': max(total_usd * 0.1, 100)  # 10% o min $100
        }
    
    def _suggest_trading_pairs(self, balances: Dict[str, float]) -> List[str]:
        """Suggerisce coppie di trading ottimali basate sui fondi disponibili"""
        available_majors = [k for k in balances.keys() 
                          if k in ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'AVAX']]
        available_stables = [k for k in balances.keys() 
                           if k in ['USDT', 'USDC', 'BUSD']]
        
        suggested_pairs = []
        
        # Coppie major/stable
        for major in available_majors[:3]:  # Top 3 major disponibili
            for stable in available_stables[:2]:  # Top 2 stable disponibili
                suggested_pairs.append(f"{major}/{stable}")
        
        # Coppie major/major se sufficienti fondi
        if len(available_majors) >= 2:
            suggested_pairs.append(f"{available_majors[0]}/{available_majors[1]}")
        
        return suggested_pairs[:5]  # Max 5 coppie
    
    def _generate_risk_management_rules(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Genera regole di risk management personalizzate"""
        risk_level = portfolio.get('risk_analysis', {}).get('overall_risk', 'Moderate')
        total_value = portfolio.get('total_usd_value', 0)
        
        rules = {
            'max_position_size_percent': self._get_max_position_size(risk_level),
            'stop_loss_percent': self._get_stop_loss_percent(risk_level),
            'take_profit_percent': self._get_take_profit_percent(risk_level),
            'max_daily_loss_usd': max(total_value * 0.02, 10),  # 2% o min $10
            'correlation_limit': 0.7,  # Max correlazione tra posizioni
            'emergency_stop_loss_percent': 15,  # Stop loss di emergenza
            'rebalancing_threshold_percent': 5  # Quando ribilanciare
        }
        
        return rules
    
    def _get_max_position_size(self, risk_level: str) -> float:
        """Calcola max position size basata sul rischio"""
        sizes = {
            'Conservative': 10,
            'Moderate': 15,
            'Aggressive': 25,
            'Very Aggressive': 35
        }
        return sizes.get(risk_level, 15)
    
    def _get_stop_loss_percent(self, risk_level: str) -> float:
        """Calcola stop loss basato sul rischio"""
        stops = {
            'Conservative': 3,
            'Moderate': 5,
            'Aggressive': 8,
            'Very Aggressive': 12
        }
        return stops.get(risk_level, 5)
    
    def _get_take_profit_percent(self, risk_level: str) -> float:
        """Calcola take profit basato sul rischio"""
        profits = {
            'Conservative': 5,
            'Moderate': 8,
            'Aggressive': 12,
            'Very Aggressive': 20
        }
        return profits.get(risk_level, 8)
    
    def _plan_autonomous_actions(self, balances: Dict[str, float], 
                               portfolio: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Pianifica azioni autonome che l'AI dovrebbe eseguire"""
        actions = []
        
        # Auto-rebalancing se necessario
        diversification = portfolio.get('diversification_score', 0)
        if diversification < 20:
            actions.append({
                'type': 'auto_rebalance',
                'priority': 'high',
                'description': 'Ribilanciamento automatico per migliorare diversificazione',
                'trigger': 'low_diversification'
            })
        
        # Conversione dust amounts
        breakdown = portfolio.get('currency_breakdown', {})
        for currency, data in breakdown.items():
            if data['usd_value'] < 5:  # Meno di $5
                actions.append({
                    'type': 'convert_dust',
                    'priority': 'low',
                    'currency': currency,
                    'amount': data['amount'],
                    'description': f'Converti dust {currency} in currency principale',
                    'trigger': 'dust_amount'
                })
        
        # Emergency stop se rischio troppo alto
        risk_score = portfolio.get('risk_analysis', {}).get('risk_score', 0)
        if risk_score > 7:
            actions.append({
                'type': 'emergency_risk_reduction',
                'priority': 'critical',
                'description': 'Riduzione automatica esposizione ad alto rischio',
                'trigger': 'high_risk_score'
            })
        
        return actions
    
    def should_refresh_scan(self) -> bool:
        """Determina se è necessario aggiornare la scansione"""
        if not self.last_scan:
            return True
        
        time_diff = datetime.now() - self.last_scan
        return time_diff.total_seconds() > 300  # 5 minuti
    
    def get_autonomous_summary(self) -> Dict[str, Any]:
        """Restituisce summary per dashboard autonoma"""
        if self.should_refresh_scan():
            self.scan_all_balances()
        
        return {
            'wallet_status': self.wallet_cache,
            'trading_strategy': self.get_optimal_allocation_strategy(),
            'ai_recommendations': self._get_ai_recommendations(),
            'autonomous_mode': True,
            'last_updated': self.last_scan.isoformat() if self.last_scan else None
        }
    
    def _get_ai_recommendations(self) -> List[str]:
        """Genera raccomandazioni AI specifiche per il portafoglio"""
        recommendations = []
        
        if not self.wallet_cache:
            return ["Esegui scansione iniziale dei fondi"]
        
        portfolio = self.wallet_cache.get('summary', {})
        risk_analysis = portfolio.get('risk_analysis', {})
        
        recommendations.extend(risk_analysis.get('recommendations', []))
        
        # Raccomandazioni specifiche per trading
        total_value = portfolio.get('total_usd_value', 0)
        if total_value > 1000:
            recommendations.append("Portfolio idoneo per trading multi-pair avanzato")
        elif total_value > 500:
            recommendations.append("Portfolio adatto per trading conservativo")
        else:
            recommendations.append("Considera accumulo fondi prima di trading intensivo")
        
        return recommendations