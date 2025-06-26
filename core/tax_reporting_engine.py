"""
Tax Reporting Engine per Report Fiscali Exportabili
Genera report dettagliati per audit e dichiarazioni fiscali
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import logging
import json
import csv
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TaxEvent(Enum):
    BUY = "buy"
    SELL = "sell"
    STAKE = "stake"
    UNSTAKE = "unstake"
    DIVIDEND = "dividend"
    MINING = "mining"
    FORK = "fork"
    AIRDROP = "airdrop"
    FEE = "fee"

class TaxMethod(Enum):
    FIFO = "fifo"  # First In First Out
    LIFO = "lifo"  # Last In First Out
    HIFO = "hifo"  # Highest In First Out
    SPECIFIC = "specific"  # Specific Identification

@dataclass
class TaxableTransaction:
    timestamp: datetime
    symbol: str
    event_type: TaxEvent
    quantity: float
    price_usd: float
    value_usd: float
    fee_usd: float
    exchange: str
    transaction_id: str
    notes: str = ""

@dataclass
class TaxLot:
    """Lotto fiscale per calcolo FIFO/LIFO"""
    symbol: str
    quantity: float
    cost_basis: float  # USD per unit
    acquisition_date: datetime
    transaction_id: str

class TaxCalculator:
    """Calcolatore per tasse su capital gains"""
    
    def __init__(self, tax_method: TaxMethod = TaxMethod.FIFO):
        self.tax_method = tax_method
        self.tax_lots = {}  # symbol -> List[TaxLot]
        
    def add_acquisition(self, transaction: TaxableTransaction):
        """Aggiunge acquisizione di asset"""
        symbol = transaction.symbol
        if symbol not in self.tax_lots:
            self.tax_lots[symbol] = []
            
        tax_lot = TaxLot(
            symbol=symbol,
            quantity=transaction.quantity,
            cost_basis=transaction.price_usd,
            acquisition_date=transaction.timestamp,
            transaction_id=transaction.transaction_id
        )
        
        self.tax_lots[symbol].append(tax_lot)
        logger.debug(f"Added tax lot for {symbol}: {transaction.quantity} @ ${transaction.price_usd}")
    
    def calculate_disposal(self, transaction: TaxableTransaction) -> List[Dict[str, Any]]:
        """Calcola capital gains per vendita"""
        symbol = transaction.symbol
        quantity_to_sell = transaction.quantity
        disposal_price = transaction.price_usd
        
        if symbol not in self.tax_lots or not self.tax_lots[symbol]:
            logger.warning(f"No tax lots available for {symbol} disposal")
            return []
        
        # Ordina lotti secondo metodo fiscale
        if self.tax_method == TaxMethod.FIFO:
            lots = sorted(self.tax_lots[symbol], key=lambda x: x.acquisition_date)
        elif self.tax_method == TaxMethod.LIFO:
            lots = sorted(self.tax_lots[symbol], key=lambda x: x.acquisition_date, reverse=True)
        elif self.tax_method == TaxMethod.HIFO:
            lots = sorted(self.tax_lots[symbol], key=lambda x: x.cost_basis, reverse=True)
        else:  # SPECIFIC
            lots = self.tax_lots[symbol]
        
        gains_losses = []
        remaining_quantity = quantity_to_sell
        
        for lot in lots:
            if remaining_quantity <= 0:
                break
                
            if lot.quantity <= 0:
                continue
                
            # Quantità da questo lotto
            lot_quantity = min(remaining_quantity, lot.quantity)
            
            # Calcola gain/loss
            proceeds = lot_quantity * disposal_price
            cost_basis_total = lot_quantity * lot.cost_basis
            gain_loss = proceeds - cost_basis_total
            
            # Determina se short/long term
            holding_period = transaction.timestamp - lot.acquisition_date
            is_long_term = holding_period.days >= 365
            
            gains_losses.append({
                'symbol': symbol,
                'quantity': lot_quantity,
                'acquisition_date': lot.acquisition_date,
                'disposal_date': transaction.timestamp,
                'holding_period_days': holding_period.days,
                'is_long_term': is_long_term,
                'cost_basis': lot.cost_basis,
                'disposal_price': disposal_price,
                'proceeds': proceeds,
                'cost_basis_total': cost_basis_total,
                'gain_loss': gain_loss,
                'transaction_id': transaction.transaction_id,
                'lot_transaction_id': lot.transaction_id
            })
            
            # Aggiorna lotto
            lot.quantity -= lot_quantity
            remaining_quantity -= lot_quantity
        
        # Rimuovi lotti vuoti
        self.tax_lots[symbol] = [lot for lot in self.tax_lots[symbol] if lot.quantity > 0]
        
        if remaining_quantity > 0:
            logger.warning(f"Insufficient tax lots for {symbol}: {remaining_quantity} units not matched")
        
        return gains_losses

class TaxReportingEngine:
    """Motore principale per reporting fiscale"""
    
    def __init__(self, tax_method: TaxMethod = TaxMethod.FIFO):
        self.tax_calculator = TaxCalculator(tax_method)
        self.transactions = []
        self.gains_losses = []
        self.income_events = []
        
        # Configurazione fiscale
        self.tax_config = {
            'short_term_rate': 0.37,  # 37% aliquota short-term
            'long_term_rate': 0.20,   # 20% aliquota long-term
            'ordinary_income_rate': 0.24,  # 24% reddito ordinario
            'mining_rate': 0.24,      # 24% mining income
            'staking_rate': 0.24      # 24% staking rewards
        }
        
    def add_transaction(self, transaction: TaxableTransaction):
        """Aggiunge transazione per elaborazione fiscale"""
        self.transactions.append(transaction)
        
        # Processa in base al tipo
        if transaction.event_type in [TaxEvent.BUY, TaxEvent.STAKE]:
            self.tax_calculator.add_acquisition(transaction)
            
        elif transaction.event_type in [TaxEvent.SELL, TaxEvent.UNSTAKE]:
            gains = self.tax_calculator.calculate_disposal(transaction)
            self.gains_losses.extend(gains)
            
        elif transaction.event_type in [TaxEvent.DIVIDEND, TaxEvent.MINING, 
                                       TaxEvent.AIRDROP, TaxEvent.FORK]:
            self.income_events.append({
                'timestamp': transaction.timestamp,
                'symbol': transaction.symbol,
                'event_type': transaction.event_type.value,
                'quantity': transaction.quantity,
                'fair_market_value': transaction.price_usd,
                'income_usd': transaction.value_usd,
                'taxable_income': transaction.value_usd
            })
        
        logger.debug(f"Processed tax transaction: {transaction.event_type.value} {transaction.quantity} {transaction.symbol}")
    
    def generate_capital_gains_report(self, tax_year: int) -> Dict[str, Any]:
        """Genera report capital gains per anno fiscale"""
        year_gains = [
            gain for gain in self.gains_losses 
            if gain['disposal_date'].year == tax_year
        ]
        
        if not year_gains:
            return {
                'tax_year': tax_year,
                'total_transactions': 0,
                'short_term_gains': 0,
                'long_term_gains': 0,
                'total_gains': 0,
                'estimated_tax': 0,
                'transactions': []
            }
        
        # Separa short/long term
        short_term = [g for g in year_gains if not g['is_long_term']]
        long_term = [g for g in year_gains if g['is_long_term']]
        
        short_term_total = sum(g['gain_loss'] for g in short_term)
        long_term_total = sum(g['gain_loss'] for g in long_term)
        total_gains = short_term_total + long_term_total
        
        # Calcola tasse stimate
        short_term_tax = max(0, short_term_total * self.tax_config['short_term_rate'])
        long_term_tax = max(0, long_term_total * self.tax_config['long_term_rate'])
        estimated_tax = short_term_tax + long_term_tax
        
        return {
            'tax_year': tax_year,
            'total_transactions': len(year_gains),
            'short_term_transactions': len(short_term),
            'long_term_transactions': len(long_term),
            'short_term_gains': short_term_total,
            'long_term_gains': long_term_total,
            'total_gains': total_gains,
            'short_term_tax': short_term_tax,
            'long_term_tax': long_term_tax,
            'estimated_tax': estimated_tax,
            'transactions': year_gains,
            'method_used': self.tax_calculator.tax_method.value
        }
    
    def generate_income_report(self, tax_year: int) -> Dict[str, Any]:
        """Genera report redditi (mining, staking, airdrops)"""
        year_income = [
            income for income in self.income_events
            if income['timestamp'].year == tax_year
        ]
        
        if not year_income:
            return {
                'tax_year': tax_year,
                'total_income': 0,
                'estimated_tax': 0,
                'events': []
            }
        
        # Aggrega per tipo
        income_by_type = {}
        for income in year_income:
            event_type = income['event_type']
            if event_type not in income_by_type:
                income_by_type[event_type] = []
            income_by_type[event_type].append(income)
        
        # Calcola totali
        total_income = sum(income['taxable_income'] for income in year_income)
        estimated_tax = total_income * self.tax_config['ordinary_income_rate']
        
        return {
            'tax_year': tax_year,
            'total_income': total_income,
            'estimated_tax': estimated_tax,
            'income_by_type': {
                event_type: {
                    'count': len(events),
                    'total_income': sum(e['taxable_income'] for e in events),
                    'events': events
                }
                for event_type, events in income_by_type.items()
            },
            'all_events': year_income
        }
    
    def export_to_csv(self, tax_year: int, filepath: str) -> bool:
        """Esporta report in formato CSV"""
        try:
            # Capital gains
            gains_report = self.generate_capital_gains_report(tax_year)
            gains_df = pd.DataFrame(gains_report['transactions'])
            
            # Income events
            income_report = self.generate_income_report(tax_year)
            income_df = pd.DataFrame(income_report['all_events'])
            
            # Export con multiple sheets (simulated con files separati)
            if not gains_df.empty:
                gains_file = filepath.replace('.csv', '_capital_gains.csv')
                gains_df.to_csv(gains_file, index=False)
                logger.info(f"Exported capital gains to {gains_file}")
            
            if not income_df.empty:
                income_file = filepath.replace('.csv', '_income.csv')
                income_df.to_csv(income_file, index=False)
                logger.info(f"Exported income to {income_file}")
            
            # Summary report
            summary = {
                'tax_year': tax_year,
                'capital_gains_summary': {
                    'total_gains': gains_report['total_gains'],
                    'short_term_gains': gains_report['short_term_gains'],
                    'long_term_gains': gains_report['long_term_gains'],
                    'estimated_tax': gains_report['estimated_tax']
                },
                'income_summary': {
                    'total_income': income_report['total_income'],
                    'estimated_tax': income_report['estimated_tax']
                },
                'total_estimated_tax': gains_report['estimated_tax'] + income_report['estimated_tax']
            }
            
            summary_file = filepath.replace('.csv', '_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Tax reports exported successfully for year {tax_year}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export tax reports: {e}")
            return False
    
    def generate_audit_trail(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Genera audit trail per verifica fiscale"""
        filtered_transactions = self.transactions
        
        if symbol:
            filtered_transactions = [t for t in self.transactions if t.symbol == symbol]
        
        audit_trail = []
        for tx in filtered_transactions:
            audit_trail.append({
                'timestamp': tx.timestamp.isoformat(),
                'symbol': tx.symbol,
                'event_type': tx.event_type.value,
                'quantity': tx.quantity,
                'price_usd': tx.price_usd,
                'value_usd': tx.value_usd,
                'fee_usd': tx.fee_usd,
                'exchange': tx.exchange,
                'transaction_id': tx.transaction_id,
                'notes': tx.notes
            })
        
        return sorted(audit_trail, key=lambda x: x['timestamp'])
    
    def calculate_unrealized_gains(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Calcola guadagni non realizzati per planning fiscale"""
        unrealized = {}
        
        for symbol, lots in self.tax_calculator.tax_lots.items():
            if not lots or symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            symbol_unrealized = []
            
            for lot in lots:
                if lot.quantity <= 0:
                    continue
                    
                current_value = lot.quantity * current_price
                cost_basis_total = lot.quantity * lot.cost_basis
                unrealized_gain = current_value - cost_basis_total
                
                holding_period = datetime.now() - lot.acquisition_date
                is_long_term = holding_period.days >= 365
                
                symbol_unrealized.append({
                    'quantity': lot.quantity,
                    'cost_basis': lot.cost_basis,
                    'current_price': current_price,
                    'current_value': current_value,
                    'cost_basis_total': cost_basis_total,
                    'unrealized_gain': unrealized_gain,
                    'unrealized_gain_pct': (unrealized_gain / cost_basis_total * 100) if cost_basis_total > 0 else 0,
                    'holding_period_days': holding_period.days,
                    'is_long_term': is_long_term,
                    'acquisition_date': lot.acquisition_date
                })
            
            if symbol_unrealized:
                total_quantity = sum(lot['quantity'] for lot in symbol_unrealized)
                total_cost_basis = sum(lot['cost_basis_total'] for lot in symbol_unrealized)
                total_current_value = sum(lot['current_value'] for lot in symbol_unrealized)
                total_unrealized = total_current_value - total_cost_basis
                
                unrealized[symbol] = {
                    'total_quantity': total_quantity,
                    'total_cost_basis': total_cost_basis,
                    'total_current_value': total_current_value,
                    'total_unrealized_gain': total_unrealized,
                    'average_cost_basis': total_cost_basis / total_quantity if total_quantity > 0 else 0,
                    'lots': symbol_unrealized
                }
        
        return unrealized
    
    def tax_loss_harvesting_opportunities(self, current_prices: Dict[str, float],
                                        min_loss_threshold: float = 1000) -> List[Dict[str, Any]]:
        """Identifica opportunità di tax loss harvesting"""
        unrealized = self.calculate_unrealized_gains(current_prices)
        opportunities = []
        
        for symbol, data in unrealized.items():
            if data['total_unrealized_gain'] < -min_loss_threshold:
                # Perdita significativa che può essere harvested
                
                # Analizza timing (wash sale rule - 30 giorni)
                recent_purchases = [
                    lot for lot in data['lots']
                    if (datetime.now() - lot['acquisition_date']).days <= 30
                ]
                
                wash_sale_risk = len(recent_purchases) > 0
                
                opportunities.append({
                    'symbol': symbol,
                    'total_unrealized_loss': abs(data['total_unrealized_gain']),
                    'tax_savings_estimate': abs(data['total_unrealized_gain']) * self.tax_config['short_term_rate'],
                    'wash_sale_risk': wash_sale_risk,
                    'recent_purchases': len(recent_purchases),
                    'recommendation': 'harvest' if not wash_sale_risk else 'wait_wash_sale',
                    'quantity_to_sell': data['total_quantity'],
                    'current_price': current_prices[symbol]
                })
        
        return sorted(opportunities, key=lambda x: x['total_unrealized_loss'], reverse=True)
    
    def get_tax_dashboard(self, current_year: int = None) -> Dict[str, Any]:
        """Dashboard fiscale completo"""
        if current_year is None:
            current_year = datetime.now().year
        
        # Report anno corrente
        gains_report = self.generate_capital_gains_report(current_year)
        income_report = self.generate_income_report(current_year)
        
        # Posizioni correnti (simulato)
        current_prices = {'BTC/USDT': 50000, 'ETH/USDT': 3000}  # Mock data
        unrealized = self.calculate_unrealized_gains(current_prices)
        harvesting_opps = self.tax_loss_harvesting_opportunities(current_prices)
        
        return {
            'current_year': current_year,
            'ytd_capital_gains': gains_report['total_gains'],
            'ytd_income': income_report['total_income'],
            'ytd_estimated_tax': gains_report['estimated_tax'] + income_report['estimated_tax'],
            'unrealized_positions': len(unrealized),
            'total_unrealized_gain': sum(pos['total_unrealized_gain'] for pos in unrealized.values()),
            'tax_loss_opportunities': len(harvesting_opps),
            'potential_tax_savings': sum(opp['tax_savings_estimate'] for opp in harvesting_opps),
            'total_transactions': len(self.transactions),
            'method_used': self.tax_calculator.tax_method.value,
            'tax_rates': self.tax_config
        }