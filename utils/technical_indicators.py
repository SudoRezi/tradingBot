"""
Implementazione personalizzata degli indicatori tecnici per il trading AI
Sostituisce TA-Lib con implementazioni native Python
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union

def calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average"""
    result = np.full_like(data, np.nan)
    if len(data) < period:
        return result
    
    for i in range(period - 1, len(data)):
        result[i] = np.mean(data[i - period + 1:i + 1])
    
    return result

def calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average"""
    result = np.full_like(data, np.nan)
    if len(data) < period:
        return result
    
    multiplier = 2 / (period + 1)
    result[period - 1] = np.mean(data[:period])
    
    for i in range(period, len(data)):
        result[i] = (data[i] * multiplier) + (result[i - 1] * (1 - multiplier))
    
    return result

def calculate_rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index"""
    if len(data) < period + 1:
        return np.full_like(data, 50.0)
    
    changes = np.diff(data)
    gains = np.where(changes > 0, changes, 0)
    losses = np.where(changes < 0, -changes, 0)
    
    avg_gains = calculate_ema(gains, period)[period:]
    avg_losses = calculate_ema(losses, period)[period:]
    
    rs = avg_gains / np.where(avg_losses == 0, 1e-10, avg_losses)
    rsi = 100 - (100 / (1 + rs))
    
    result = np.full_like(data, 50.0)
    result[period:] = rsi
    
    return result

def calculate_macd(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD (Moving Average Convergence Divergence)"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line[~np.isnan(macd_line)], signal)
    
    # Expand signal line to match original length
    full_signal = np.full_like(data, np.nan)
    start_idx = len(data) - len(signal_line)
    full_signal[start_idx:] = signal_line
    
    histogram = macd_line - full_signal
    
    return macd_line, full_signal, histogram

def calculate_bollinger_bands(data: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands"""
    sma = calculate_sma(data, period)
    
    result_upper = np.full_like(data, np.nan)
    result_lower = np.full_like(data, np.nan)
    
    for i in range(period - 1, len(data)):
        std = np.std(data[i - period + 1:i + 1])
        result_upper[i] = sma[i] + (std * std_dev)
        result_lower[i] = sma[i] - (std * std_dev)
    
    return result_upper, sma, result_lower

def calculate_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                        k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Stochastic Oscillator"""
    result_k = np.full_like(close, 50.0)
    
    for i in range(k_period - 1, len(close)):
        period_high = np.max(high[i - k_period + 1:i + 1])
        period_low = np.min(low[i - k_period + 1:i + 1])
        
        if period_high != period_low:
            result_k[i] = ((close[i] - period_low) / (period_high - period_low)) * 100
        else:
            result_k[i] = 50.0
    
    result_d = calculate_sma(result_k, d_period)
    
    return result_k, result_d

def calculate_williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Williams %R"""
    result = np.full_like(close, -50.0)
    
    for i in range(period - 1, len(close)):
        period_high = np.max(high[i - period + 1:i + 1])
        period_low = np.min(low[i - period + 1:i + 1])
        
        if period_high != period_low:
            result[i] = ((period_high - close[i]) / (period_high - period_low)) * -100
        else:
            result[i] = -50.0
    
    return result

def calculate_cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> np.ndarray:
    """Commodity Channel Index"""
    typical_price = (high + low + close) / 3
    sma_tp = calculate_sma(typical_price, period)
    
    result = np.full_like(close, 0.0)
    
    for i in range(period - 1, len(close)):
        mean_deviation = np.mean(np.abs(typical_price[i - period + 1:i + 1] - sma_tp[i]))
        if mean_deviation != 0:
            result[i] = (typical_price[i] - sma_tp[i]) / (0.015 * mean_deviation)
    
    return result

def calculate_mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 14) -> np.ndarray:
    """Money Flow Index"""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    result = np.full_like(close, 50.0)
    
    for i in range(period, len(close)):
        positive_flow = 0
        negative_flow = 0
        
        for j in range(i - period + 1, i + 1):
            if j > 0:
                if typical_price[j] > typical_price[j - 1]:
                    positive_flow += money_flow[j]
                elif typical_price[j] < typical_price[j - 1]:
                    negative_flow += money_flow[j]
        
        if negative_flow != 0:
            money_ratio = positive_flow / negative_flow
            result[i] = 100 - (100 / (1 + money_ratio))
    
    return result

def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range"""
    true_ranges = []
    
    for i in range(1, len(close)):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        true_ranges.append(max(tr1, tr2, tr3))
    
    tr_array = np.array(true_ranges)
    atr = calculate_ema(tr_array, period)
    
    result = np.full_like(close, np.nan)
    result[1:] = np.concatenate([[tr_array[0]], atr])
    
    return result

def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average Directional Index"""
    if len(close) < period + 1:
        return np.full_like(close, 25.0)
    
    # Calculate True Range
    tr = calculate_atr(high, low, close, 1)
    
    # Calculate Directional Movement
    dm_plus = np.zeros_like(close)
    dm_minus = np.zeros_like(close)
    
    for i in range(1, len(close)):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
    
    # Smooth the values
    tr_smooth = calculate_ema(tr[1:], period)
    dm_plus_smooth = calculate_ema(dm_plus[1:], period)
    dm_minus_smooth = calculate_ema(dm_minus[1:], period)
    
    # Calculate DI+ and DI-
    di_plus = (dm_plus_smooth / tr_smooth) * 100
    di_minus = (dm_minus_smooth / tr_smooth) * 100
    
    # Calculate DX and ADX
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    dx = np.where(np.isnan(dx), 0, dx)
    
    adx = calculate_ema(dx, period)
    
    result = np.full_like(close, 25.0)
    start_idx = len(close) - len(adx)
    result[start_idx:] = adx
    
    return result

def calculate_sar(high: np.ndarray, low: np.ndarray, acceleration: float = 0.02, maximum: float = 0.2) -> np.ndarray:
    """Parabolic SAR"""
    if len(high) < 2:
        return np.full_like(high, high[0] if len(high) > 0 else 0)
    
    result = np.full_like(high, np.nan)
    
    # Initialize
    af = acceleration
    uptrend = True
    sar = low[0]
    ep = high[0]
    
    result[0] = sar
    
    for i in range(1, len(high)):
        if uptrend:
            sar = sar + af * (ep - sar)
            
            if high[i] > ep:
                ep = high[i]
                af = min(af + acceleration, maximum)
            
            if low[i] <= sar:
                uptrend = False
                sar = ep
                ep = low[i]
                af = acceleration
        else:
            sar = sar + af * (ep - sar)
            
            if low[i] < ep:
                ep = low[i]
                af = min(af + acceleration, maximum)
            
            if high[i] >= sar:
                uptrend = True
                sar = ep
                ep = high[i]
                af = acceleration
        
        result[i] = sar
    
    return result

def calculate_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On Balance Volume"""
    obv = np.zeros_like(close)
    
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    
    return obv

def calculate_roc(data: np.ndarray, period: int = 10) -> np.ndarray:
    """Rate of Change"""
    result = np.full_like(data, 0.0)
    
    for i in range(period, len(data)):
        if data[i - period] != 0:
            result[i] = ((data[i] - data[i - period]) / data[i - period]) * 100
    
    return result

def calculate_stddev(data: np.ndarray, period: int = 20) -> np.ndarray:
    """Standard Deviation"""
    result = np.full_like(data, np.nan)
    
    for i in range(period - 1, len(data)):
        result[i] = np.std(data[i - period + 1:i + 1])
    
    return result