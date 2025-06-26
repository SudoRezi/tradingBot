"""Configuration settings for the AI Trading Bot"""

# Trading pairs to operate on
TRADING_PAIRS = [
    'KAS/USDT',
    'BTC/USDT', 
    'ETH/USDT',
    'AVAX/USDT',
    'SOL/USDT',
    'BTC/ETH',
    'BTC/SOL',
    'ETH/SOL'
]

# Risk level configurations
RISK_LEVELS = {
    'conservative': {
        'position_size_pct': 0.02,  # 2% of capital per trade
        'max_daily_loss': 0.05,     # 5% max daily loss
        'max_drawdown': 0.15,       # 15% max drawdown
        'max_correlation': 0.7,     # Max correlation between positions
        'max_volatility': 0.03,     # Max volatility threshold
        'min_signal_confidence': 0.8, # Min signal confidence to trade
        'max_position_size': 0.1,   # Max 10% in single position
    },
    'moderate': {
        'position_size_pct': 0.03,  # 3% of capital per trade
        'max_daily_loss': 0.08,     # 8% max daily loss
        'max_drawdown': 0.20,       # 20% max drawdown
        'max_correlation': 0.8,     # Max correlation between positions
        'max_volatility': 0.05,     # Max volatility threshold
        'min_signal_confidence': 0.6, # Min signal confidence to trade
        'max_position_size': 0.15,  # Max 15% in single position
    },
    'aggressive': {
        'position_size_pct': 0.05,  # 5% of capital per trade
        'max_daily_loss': 0.12,     # 12% max daily loss
        'max_drawdown': 0.25,       # 25% max drawdown
        'max_correlation': 0.9,     # Max correlation between positions
        'max_volatility': 0.08,     # Max volatility threshold
        'min_signal_confidence': 0.4, # Min signal confidence to trade
        'max_position_size': 0.25,  # Max 25% in single position
    }
}

# Analysis and execution intervals
ANALYSIS_INTERVAL = 30  # seconds between market analysis
REBALANCE_INTERVAL = 300  # seconds between portfolio rebalancing
RISK_CHECK_INTERVAL = 60  # seconds between risk assessments

# Technical analysis parameters
TA_PERIODS = {
    'rsi': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'ema_short': 9,
    'ema_medium': 21,
    'ema_long': 50,
    'sma_long': 200,
    'bb_period': 20,
    'bb_std': 2,
    'stoch_k': 14,
    'stoch_d': 3,
    'adx_period': 14,
    'atr_period': 14
}

# Machine Learning parameters
ML_CONFIG = {
    'features': 10,
    'lookback_periods': 50,
    'training_size': 1000,
    'retrain_interval': 24,  # hours
    'model_type': 'random_forest'
}

# Notification settings
NOTIFICATION_CONFIG = {
    'min_profit_notification': 50,    # Min $ profit for notification
    'min_profit_pct_notification': 5, # Min % profit for notification
    'daily_summary_enabled': True,
    'risk_alerts_enabled': True,
    'trade_confirmations_enabled': False  # Only for large trades
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'good_win_rate': 60,      # % win rate considered good
    'poor_win_rate': 40,      # % win rate considered poor
    'good_return': 10,        # % monthly return considered good
    'poor_return': -5,        # % monthly return considered poor
    'high_sharpe': 1.5,       # Sharpe ratio considered high
    'low_sharpe': 0.5         # Sharpe ratio considered low
}

# Exchange settings (for real exchange integration)
EXCHANGE_CONFIG = {
    'name': 'binance',  # Default exchange
    'testnet': True,    # Use testnet by default
    'rate_limit': 1200, # Requests per minute
    'timeout': 30,      # Request timeout in seconds
    'retry_attempts': 3 # Number of retry attempts
}

# Database settings
DATABASE_CONFIG = {
    'type': 'sqlite',
    'path': 'data/trading_bot.db',
    'backup_interval': 3600,  # Backup every hour
    'max_backups': 24         # Keep 24 hourly backups
}

# Security settings
SECURITY_CONFIG = {
    'api_key_encryption': True,
    'log_api_calls': False,    # Don't log API calls for security
    'max_session_time': 86400, # 24 hours
    'require_confirmation_for_large_trades': True,
    'large_trade_threshold': 1000  # $ threshold for large trades
}
