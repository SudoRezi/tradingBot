"""
Currency Converter per supporto multi-currency nel capitale
Gestisce conversioni tra crypto, stablecoins, fiat e token DeFi
"""

import requests
import time
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import json

class CurrencyConverter:
    """Convertitore universal per tutte le tipologie di valute"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minuti
        self.last_update = {}
        
        # Exchange rate sources
        self.crypto_api_url = "https://api.coingecko.com/api/v3/simple/price"
        self.fiat_api_url = "https://api.exchangerate-api.com/v4/latest"
        
        # Currency mappings for CoinGecko
        self.crypto_mapping = {
            'BTC': 'bitcoin', 'ETH': 'ethereum', 'BNB': 'binancecoin',
            'SOL': 'solana', 'ADA': 'cardano', 'AVAX': 'avalanche-2',
            'DOT': 'polkadot', 'MATIC': 'matic-network', 'LINK': 'chainlink',
            'UNI': 'uniswap', 'ESDC': 'esdc', 'KAS': 'kaspa',
            'XRP': 'ripple', 'DOGE': 'dogecoin', 'SHIB': 'shiba-inu',
            'LTC': 'litecoin', 'BCH': 'bitcoin-cash', 'ETC': 'ethereum-classic',
            'ATOM': 'cosmos', 'NEAR': 'near', 'AAVE': 'aave',
            'COMP': 'compound-governance-token', 'MKR': 'maker',
            'CRV': 'curve-dao-token', 'SUSHI': 'sushi', 'YFI': 'yearn-finance',
            'BAL': 'balancer', 'SNX': 'havven', '1INCH': '1inch',
            'UMA': 'uma', 'PEPE': 'pepe', 'FLOKI': 'floki',
            'BONK': 'bonk', 'WIF': 'dogwifcoin', 'LUNA': 'terra-luna-2',
            'ALGO': 'algorand', 'TEZOS': 'tezos', 'FLOW': 'flow',
            'EGLD': 'elrond-erd-2', 'FTM': 'fantom', 'HARMONY': 'harmony',
            'CELO': 'celo', 'THETA': 'theta-token'
        }
        
        # Stablecoin values (sempre ~1 USD)
        self.stablecoins = {
            'USDT': 1.0, 'USDC': 1.0, 'BUSD': 1.0, 'DAI': 1.0,
            'FDUSD': 1.0, 'TUSD': 1.0, 'FRAX': 1.0
        }
    
    def get_exchange_rate(self, from_currency: str, to_currency: str = 'USD') -> Optional[float]:
        """Ottiene tasso di cambio tra due valute"""
        try:
            # Se entrambe sono stablecoin
            if from_currency in self.stablecoins and to_currency in self.stablecoins:
                return 1.0
            
            # Se from è stablecoin e to è USD
            if from_currency in self.stablecoins and to_currency == 'USD':
                return self.stablecoins[from_currency]
            
            # Se to è stablecoin e from è USD
            if to_currency in self.stablecoins and from_currency == 'USD':
                return 1.0 / self.stablecoins[to_currency]
            
            # Cache check
            cache_key = f"{from_currency}_{to_currency}"
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]
            
            rate = None
            
            # Crypto to USD/Stablecoin
            if from_currency in self.crypto_mapping:
                rate = self._get_crypto_rate(from_currency, to_currency)
            
            # Fiat currencies
            elif from_currency in ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY', 'KRW', 'SGD']:
                rate = self._get_fiat_rate(from_currency, to_currency)
            
            # Unknown currency - try as crypto
            else:
                rate = self._get_crypto_rate(from_currency, to_currency)
            
            if rate:
                self._update_cache(cache_key, rate)
            
            return rate
            
        except Exception as e:
            print(f"Error getting exchange rate {from_currency}/{to_currency}: {e}")
            return None
    
    def _get_crypto_rate(self, from_currency: str, to_currency: str = 'USD') -> Optional[float]:
        """Ottiene tasso crypto tramite CoinGecko"""
        try:
            crypto_id = self.crypto_mapping.get(from_currency, from_currency.lower())
            
            # Convert to_currency for API
            vs_currency = 'usd'
            if to_currency in self.stablecoins:
                vs_currency = 'usd'  # Stablecoins = USD
            elif to_currency in self.crypto_mapping:
                vs_currency = self.crypto_mapping[to_currency]
            elif to_currency.lower() in ['eur', 'gbp', 'jpy', 'cad', 'aud', 'chf']:
                vs_currency = to_currency.lower()
            
            url = f"{self.crypto_api_url}?ids={crypto_id}&vs_currencies={vs_currency}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if crypto_id in data and vs_currency in data[crypto_id]:
                    return float(data[crypto_id][vs_currency])
            
            return None
            
        except Exception as e:
            print(f"Error getting crypto rate: {e}")
            return None
    
    def _get_fiat_rate(self, from_currency: str, to_currency: str = 'USD') -> Optional[float]:
        """Ottiene tasso fiat tramite Exchange Rate API"""
        try:
            # Se to_currency è crypto, convertiamo al contrario
            if to_currency in self.crypto_mapping or to_currency in self.stablecoins:
                # Prima converti fiat to USD, poi USD to crypto
                if from_currency != 'USD':
                    fiat_to_usd = self._get_fiat_to_usd(from_currency)
                    if not fiat_to_usd:
                        return None
                else:
                    fiat_to_usd = 1.0
                
                usd_to_crypto = self.get_exchange_rate('USD', to_currency)
                if usd_to_crypto:
                    return fiat_to_usd * usd_to_crypto
                return None
            
            # Fiat to fiat
            return self._get_fiat_to_usd(from_currency, to_currency)
            
        except Exception as e:
            print(f"Error getting fiat rate: {e}")
            return None
    
    def _get_fiat_to_usd(self, from_currency: str, to_currency: str = 'USD') -> Optional[float]:
        """Helper per conversioni fiat"""
        try:
            url = f"{self.fiat_api_url}/{from_currency}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'rates' in data and to_currency in data['rates']:
                    return float(data['rates'][to_currency])
            
            return None
            
        except Exception as e:
            print(f"Error getting fiat rate: {e}")
            return None
    
    def convert_amount(self, amount: float, from_currency: str, to_currency: str = 'USD') -> Optional[float]:
        """Converte un importo da una valuta all'altra"""
        if from_currency == to_currency:
            return amount
        
        rate = self.get_exchange_rate(from_currency, to_currency)
        if rate:
            return amount * rate
        
        return None
    
    def get_portfolio_value_usd(self, holdings: Dict[str, float]) -> float:
        """Calcola valore totale portfolio in USD"""
        total_usd = 0.0
        
        for currency, amount in holdings.items():
            if amount <= 0:
                continue
                
            usd_value = self.convert_amount(amount, currency, 'USD')
            if usd_value:
                total_usd += usd_value
        
        return total_usd
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Controlla se cache è ancora valida"""
        if cache_key not in self.cache or cache_key not in self.last_update:
            return False
        
        time_diff = time.time() - self.last_update[cache_key]
        return time_diff < self.cache_duration
    
    def _update_cache(self, cache_key: str, value: float):
        """Aggiorna cache"""
        self.cache[cache_key] = value
        self.last_update[cache_key] = time.time()
    
    def get_supported_currencies(self) -> Dict[str, list]:
        """Ritorna lista di tutte le valute supportate"""
        return {
            "stablecoins": list(self.stablecoins.keys()),
            "cryptocurrencies": list(self.crypto_mapping.keys()),
            "fiat": ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY', 'KRW', 'SGD']
        }
    
    def get_conversion_summary(self, base_currency: str, amount: float) -> Dict[str, Any]:
        """Genera summary delle conversioni per una valuta base"""
        summary = {
            'base_currency': base_currency,
            'base_amount': amount,
            'conversions': {},
            'total_categories': {},
            'last_updated': datetime.now().isoformat()
        }
        
        # Test conversions to major currencies
        test_currencies = ['USD', 'BTC', 'ETH', 'USDT']
        
        for currency in test_currencies:
            if currency != base_currency:
                converted = self.convert_amount(amount, base_currency, currency)
                if converted:
                    summary['conversions'][currency] = {
                        'amount': converted,
                        'rate': converted / amount if amount > 0 else 0
                    }
        
        return summary