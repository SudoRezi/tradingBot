import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional
from utils.technical_indicators import *

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    """AI-powered market analysis engine"""
    
    def __init__(self):
        self.analysis_cache = {}
        self.trend_memory = {}  # Remember market trends for learning
        
    def analyze_market(self, market_data: pd.DataFrame, pair: str) -> Dict[str, Any]:
        """Comprehensive market analysis using multiple indicators"""
        try:
            if market_data is None or len(market_data) < 50:
                return self._get_default_analysis()
            
            # Prepare OHLCV data
            high = market_data['high'].values
            low = market_data['low'].values
            close = market_data['close'].values
            volume = market_data['volume'].values
            
            # Technical indicators
            analysis = {
                'pair': pair,
                'timestamp': datetime.now(),
                'price': close[-1],
                'volume': volume[-1],
            }
            
            # Trend analysis
            analysis.update(self._analyze_trend(close, high, low))
            
            # Momentum indicators
            analysis.update(self._analyze_momentum(close, high, low, volume))
            
            # Volatility analysis
            analysis.update(self._analyze_volatility(close))
            
            # Support/Resistance levels
            analysis.update(self._find_support_resistance(close, high, low))
            
            # Volume analysis
            analysis.update(self._analyze_volume(close, volume))
            
            # Market regime detection
            analysis.update(self._detect_market_regime(close, volume))
            
            # Overall market sentiment
            analysis['sentiment'] = self._calculate_sentiment(analysis)
            analysis['strength'] = self._calculate_signal_strength(analysis)
            
            # Cache analysis for learning
            self.analysis_cache[pair] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market for {pair}: {e}")
            return self._get_default_analysis()
    
    def _analyze_trend(self, close, high, low):
        """Analyze trend direction and strength"""
        try:
            # Moving averages
            ema_9 = calculate_ema(close, 9)
            ema_21 = calculate_ema(close, 21)
            ema_50 = calculate_ema(close, 50)
            sma_200 = calculate_sma(close, 200)
            
            # MACD
            macd, macd_signal, macd_hist = calculate_macd(close)
            
            # ADX for trend strength
            adx = calculate_adx(high, low, close, 14)
            
            # Parabolic SAR
            sar = calculate_sar(high, low)
            
            current_price = close[-1]
            
            # Trend direction
            trend_direction = 'neutral'
            if current_price > ema_21[-1] > ema_50[-1] > sma_200[-1]:
                trend_direction = 'strong_bullish'
            elif current_price > ema_21[-1] > ema_50[-1]:
                trend_direction = 'bullish'
            elif current_price < ema_21[-1] < ema_50[-1] < sma_200[-1]:
                trend_direction = 'strong_bearish'
            elif current_price < ema_21[-1] < ema_50[-1]:
                trend_direction = 'bearish'
            
            return {
                'trend_direction': trend_direction,
                'trend_strength': adx[-1] if not np.isnan(adx[-1]) else 25,
                'ema_9': ema_9[-1],
                'ema_21': ema_21[-1],
                'ema_50': ema_50[-1],
                'sma_200': sma_200[-1],
                'macd': macd[-1] if not np.isnan(macd[-1]) else 0,
                'macd_signal': macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0,
                'macd_histogram': macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0,
                'parabolic_sar': sar[-1] if not np.isnan(sar[-1]) else current_price,
            }
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return {'trend_direction': 'neutral', 'trend_strength': 25}
    
    def _analyze_momentum(self, close, high, low, volume):
        """Analyze momentum indicators"""
        try:
            # RSI
            rsi = calculate_rsi(close, 14)
            
            # Stochastic
            stoch_k, stoch_d = calculate_stochastic(high, low, close)
            
            # Williams %R
            willr = calculate_williams_r(high, low, close, 14)
            
            # CCI
            cci = calculate_cci(high, low, close, 20)
            
            # Money Flow Index
            mfi = calculate_mfi(high, low, close, volume, 14)
            
            return {
                'rsi': rsi[-1] if not np.isnan(rsi[-1]) else 50,
                'stoch_k': stoch_k[-1] if not np.isnan(stoch_k[-1]) else 50,
                'stoch_d': stoch_d[-1] if not np.isnan(stoch_d[-1]) else 50,
                'williams_r': willr[-1] if not np.isnan(willr[-1]) else -50,
                'cci': cci[-1] if not np.isnan(cci[-1]) else 0,
                'mfi': mfi[-1] if not np.isnan(mfi[-1]) else 50,
            }
            
        except Exception as e:
            logger.error(f"Error in momentum analysis: {e}")
            return {'rsi': 50, 'stoch_k': 50, 'stoch_d': 50}
    
    def _analyze_volatility(self, close):
        """Analyze price volatility"""
        try:
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close)
            
            # Average True Range
            atr = calculate_atr(close, close, close, 14)  # Using close as high/low approximation
            
            # Standard deviation
            std = calculate_stddev(close, 20)
            
            current_price = close[-1]
            bb_position = (current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
            
            return {
                'bb_upper': bb_upper[-1] if not np.isnan(bb_upper[-1]) else current_price * 1.02,
                'bb_middle': bb_middle[-1] if not np.isnan(bb_middle[-1]) else current_price,
                'bb_lower': bb_lower[-1] if not np.isnan(bb_lower[-1]) else current_price * 0.98,
                'bb_position': bb_position if not np.isnan(bb_position) else 0.5,
                'atr': atr[-1] if not np.isnan(atr[-1]) else current_price * 0.02,
                'volatility': std[-1] if not np.isnan(std[-1]) else current_price * 0.02,
            }
            
        except Exception as e:
            logger.error(f"Error in volatility analysis: {e}")
            return {'volatility': 0.02}
    
    def _find_support_resistance(self, close, high, low):
        """Find key support and resistance levels"""
        try:
            # Recent highs and lows
            recent_period = min(50, len(close))
            recent_close = close[-recent_period:]
            recent_high = high[-recent_period:]
            recent_low = low[-recent_period:]
            
            # Support levels (recent lows)
            support_levels = []
            for i in range(2, len(recent_low) - 2):
                if (recent_low[i] < recent_low[i-1] and recent_low[i] < recent_low[i+1] and
                    recent_low[i] < recent_low[i-2] and recent_low[i] < recent_low[i+2]):
                    support_levels.append(recent_low[i])
            
            # Resistance levels (recent highs)
            resistance_levels = []
            for i in range(2, len(recent_high) - 2):
                if (recent_high[i] > recent_high[i-1] and recent_high[i] > recent_high[i+1] and
                    recent_high[i] > recent_high[i-2] and recent_high[i] > recent_high[i+2]):
                    resistance_levels.append(recent_high[i])
            
            current_price = close[-1]
            
            # Find nearest levels
            nearest_support = max([s for s in support_levels if s < current_price], default=current_price * 0.95)
            nearest_resistance = min([r for r in resistance_levels if r > current_price], default=current_price * 1.05)
            
            return {
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'support_distance': (current_price - nearest_support) / current_price,
                'resistance_distance': (nearest_resistance - current_price) / current_price,
            }
            
        except Exception as e:
            logger.error(f"Error finding support/resistance: {e}")
            current_price = close[-1]
            return {
                'nearest_support': current_price * 0.95,
                'nearest_resistance': current_price * 1.05,
            }
    
    def _analyze_volume(self, close, volume):
        """Analyze volume patterns"""
        try:
            # Volume SMA
            volume_sma = calculate_sma(volume, 20)
            
            # On Balance Volume
            obv = calculate_obv(close, volume)
            
            # Volume Rate of Change
            volume_roc = calculate_roc(volume, 10)
            
            current_volume = volume[-1]
            avg_volume = volume_sma[-1] if not np.isnan(volume_sma[-1]) else current_volume
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            return {
                'volume_ratio': volume_ratio,
                'obv': obv[-1] if not np.isnan(obv[-1]) else 0,
                'volume_trend': 'increasing' if volume_roc[-1] > 5 else 'decreasing' if volume_roc[-1] < -5 else 'stable',
                'volume_strength': 'high' if volume_ratio > 1.5 else 'low' if volume_ratio < 0.5 else 'normal'
            }
            
        except Exception as e:
            logger.error(f"Error in volume analysis: {e}")
            return {'volume_ratio': 1, 'volume_strength': 'normal'}
    
    def _detect_market_regime(self, close, volume):
        """Detect current market regime (trending, ranging, volatile)"""
        try:
            # Calculate price changes
            returns = np.diff(close) / close[:-1]
            
            # Volatility
            volatility = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
            
            # Trend consistency
            recent_returns = returns[-10:] if len(returns) >= 10 else returns
            trend_consistency = np.mean(np.sign(recent_returns)) if len(recent_returns) > 0 else 0
            
            # Market regime classification
            if abs(trend_consistency) > 0.6 and volatility < 0.03:
                regime = 'trending'
            elif volatility > 0.05:
                regime = 'volatile'
            else:
                regime = 'ranging'
            
            return {
                'market_regime': regime,
                'volatility_level': volatility,
                'trend_consistency': trend_consistency,
            }
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return {'market_regime': 'ranging', 'volatility_level': 0.02}
    
    def _calculate_sentiment(self, analysis):
        """Calculate overall market sentiment"""
        try:
            bullish_signals = 0
            bearish_signals = 0
            
            # Trend signals
            if analysis.get('trend_direction') in ['bullish', 'strong_bullish']:
                bullish_signals += 2
            elif analysis.get('trend_direction') in ['bearish', 'strong_bearish']:
                bearish_signals += 2
            
            # RSI signals
            rsi = analysis.get('rsi', 50)
            if rsi < 30:
                bullish_signals += 1
            elif rsi > 70:
                bearish_signals += 1
            
            # MACD signals
            if analysis.get('macd', 0) > analysis.get('macd_signal', 0):
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # Volume confirmation
            if analysis.get('volume_strength') == 'high':
                if analysis.get('trend_direction') in ['bullish', 'strong_bullish']:
                    bullish_signals += 1
                elif analysis.get('trend_direction') in ['bearish', 'strong_bearish']:
                    bearish_signals += 1
            
            # Calculate sentiment score
            total_signals = bullish_signals + bearish_signals
            if total_signals == 0:
                return 'neutral'
            
            bullish_ratio = bullish_signals / total_signals
            
            if bullish_ratio > 0.7:
                return 'strongly_bullish'
            elif bullish_ratio > 0.6:
                return 'bullish'
            elif bullish_ratio < 0.3:
                return 'strongly_bearish'
            elif bullish_ratio < 0.4:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error calculating sentiment: {e}")
            return 'neutral'
    
    def _calculate_signal_strength(self, analysis):
        """Calculate signal strength (0-1)"""
        try:
            strength_factors = []
            
            # Trend strength
            trend_strength = analysis.get('trend_strength', 25) / 100
            strength_factors.append(min(trend_strength, 1.0))
            
            # Volume confirmation
            volume_ratio = analysis.get('volume_ratio', 1)
            volume_strength = min(volume_ratio / 2, 1.0)
            strength_factors.append(volume_strength)
            
            # Momentum alignment
            rsi = analysis.get('rsi', 50)
            momentum_strength = abs(rsi - 50) / 50
            strength_factors.append(momentum_strength)
            
            # Volatility factor
            volatility = analysis.get('volatility_level', 0.02)
            volatility_factor = min(volatility * 10, 1.0)
            strength_factors.append(volatility_factor)
            
            # Average strength
            return np.mean(strength_factors)
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0.5
    
    def _get_default_analysis(self):
        """Return default analysis when data is insufficient"""
        return {
            'sentiment': 'neutral',
            'strength': 0.3,
            'trend_direction': 'neutral',
            'market_regime': 'ranging',
            'rsi': 50,
            'volatility_level': 0.02,
        }
