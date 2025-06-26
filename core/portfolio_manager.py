import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional
import json

logger = logging.getLogger(__name__)

class PortfolioManager:
    """Advanced portfolio management with automatic rebalancing"""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.start_time = datetime.now()
        
        # Portfolio state
        self.positions = {}  # Active positions
        self.cash_balance = initial_capital
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Performance tracking
        self.performance_history = []
        self.trade_history = []
        self.daily_pnl_history = []
        
        # Rebalancing parameters
        self.target_allocation = {
            'KAS/USDT': 0.4,
            'BTC/USDT': 0.35,
            'ETH/USDT': 0.25
        }
        self.rebalance_threshold = 0.05  # 5% deviation triggers rebalance
        
        logger.info(f"Portfolio Manager initialized with ${initial_capital:,.2f}")
    
    def execute_trade(self, pair: str, side: str, size: float, price: float) -> bool:
        """Execute a trade and update portfolio"""
        try:
            trade_value = size * price
            
            if side.upper() == 'BUY':
                # Check if we have enough cash
                if trade_value > self.cash_balance:
                    logger.warning(f"Insufficient cash for {pair} buy: need ${trade_value:.2f}, have ${self.cash_balance:.2f}")
                    return False
                
                # Execute buy
                if pair in self.positions:
                    # Add to existing position
                    current_size = self.positions[pair]['size']
                    current_value = current_size * self.positions[pair]['entry_price']
                    new_total_value = current_value + trade_value
                    new_total_size = current_size + size
                    new_avg_price = new_total_value / new_total_size
                    
                    self.positions[pair].update({
                        'size': new_total_size,
                        'entry_price': new_avg_price,
                        'current_price': price,
                        'last_update': datetime.now()
                    })
                else:
                    # Create new position
                    self.positions[pair] = {
                        'side': 'LONG',
                        'size': size,
                        'entry_price': price,
                        'current_price': price,
                        'entry_time': datetime.now(),
                        'last_update': datetime.now(),
                        'unrealized_pnl': 0,
                        'unrealized_pnl_pct': 0,
                        'stop_loss': price * 0.95,  # 5% stop loss
                        'take_profit': price * 1.10  # 10% take profit
                    }
                
                self.cash_balance -= trade_value
                
            elif side.upper() == 'SELL':
                # Check if we have the position
                if pair not in self.positions:
                    logger.warning(f"No position to sell for {pair}")
                    return False
                
                current_position = self.positions[pair]
                if size > current_position['size']:
                    logger.warning(f"Cannot sell {size} {pair}, only have {current_position['size']}")
                    return False
                
                # Calculate P&L
                entry_price = current_position['entry_price']
                pnl = (price - entry_price) * size
                pnl_pct = (price / entry_price - 1) * 100
                
                # Execute sell
                if size == current_position['size']:
                    # Close entire position
                    del self.positions[pair]
                else:
                    # Partial close
                    remaining_size = current_position['size'] - size
                    self.positions[pair]['size'] = remaining_size
                    self.positions[pair]['last_update'] = datetime.now()
                
                self.cash_balance += trade_value
                
                # Record trade
                self._record_trade(pair, side, size, price, entry_price, pnl, pnl_pct)
            
            self.total_trades += 1
            self._update_performance_history()
            
            logger.info(f"Trade executed: {side} {size:.6f} {pair} @ ${price:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def update_positions(self, market_data: Dict[str, pd.DataFrame]):
        """Update all positions with current market prices"""
        try:
            for pair in self.positions:
                if pair in market_data and not market_data[pair].empty:
                    current_price = market_data[pair]['close'].iloc[-1]
                    position = self.positions[pair]
                    
                    # Update current price and P&L
                    position['current_price'] = current_price
                    entry_price = position['entry_price']
                    size = position['size']
                    
                    unrealized_pnl = (current_price - entry_price) * size
                    unrealized_pnl_pct = (current_price / entry_price - 1) * 100
                    
                    position['unrealized_pnl'] = unrealized_pnl
                    position['unrealized_pnl_pct'] = unrealized_pnl_pct
                    position['last_update'] = datetime.now()
                    
                    # Check stop loss and take profit
                    self._check_stop_loss_take_profit(pair, position)
            
            self._update_performance_history()
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def _check_stop_loss_take_profit(self, pair: str, position: Dict[str, Any]):
        """Check and execute stop loss or take profit"""
        try:
            current_price = position['current_price']
            stop_loss = position.get('stop_loss')
            take_profit = position.get('take_profit')
            
            if stop_loss and current_price <= stop_loss:
                logger.warning(f"Stop loss triggered for {pair} at ${current_price:.6f}")
                self.execute_trade(pair, 'SELL', position['size'], current_price)
                
            elif take_profit and current_price >= take_profit:
                logger.info(f"Take profit triggered for {pair} at ${current_price:.6f}")
                self.execute_trade(pair, 'SELL', position['size'], current_price)
                
        except Exception as e:
            logger.error(f"Error checking stop loss/take profit: {e}")
    
    def update_stop_losses(self):
        """Update stop losses using trailing stop logic"""
        try:
            for pair, position in self.positions.items():
                current_price = position['current_price']
                entry_price = position['entry_price']
                current_stop = position.get('stop_loss', entry_price * 0.95)
                
                # Calculate trailing stop (5% below current price or break-even, whichever is higher)
                trailing_stop = max(current_price * 0.95, entry_price)
                
                # Only update if trailing stop is higher than current stop
                if trailing_stop > current_stop:
                    position['stop_loss'] = trailing_stop
                    logger.info(f"Updated trailing stop for {pair}: ${trailing_stop:.6f}")
                    
        except Exception as e:
            logger.error(f"Error updating stop losses: {e}")
    
    def close_position(self, pair: str):
        """Close a specific position"""
        try:
            if pair in self.positions:
                position = self.positions[pair]
                current_price = position['current_price']
                size = position['size']
                
                self.execute_trade(pair, 'SELL', size, current_price)
                logger.info(f"Position closed for {pair}")
                return True
            else:
                logger.warning(f"No position to close for {pair}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position for {pair}: {e}")
            return False
    
    def get_total_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        try:
            total_value = self.cash_balance
            
            for pair, position in self.positions.items():
                position_value = position['size'] * position['current_price']
                total_value += position_value
            
            return total_value
            
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return self.cash_balance
    
    def get_available_balance(self) -> float:
        """Get available cash balance for trading"""
        return self.cash_balance
    
    def get_daily_pnl(self) -> float:
        """Calculate daily P&L"""
        try:
            current_value = self.get_total_portfolio_value()
            
            # Get yesterday's value
            yesterday = datetime.now() - timedelta(days=1)
            yesterday_value = self.initial_capital
            
            for record in reversed(self.performance_history):
                if record['timestamp'].date() <= yesterday.date():
                    yesterday_value = record['portfolio_value']
                    break
            
            return current_value - yesterday_value
            
        except Exception as e:
            logger.error(f"Error calculating daily P&L: {e}")
            return 0
    
    def get_win_rate(self) -> float:
        """Calculate win rate percentage"""
        try:
            if self.total_trades == 0:
                return 0
            return (self.winning_trades / self.total_trades) * 100
        except:
            return 0
    
    def needs_rebalancing(self) -> bool:
        """Check if portfolio needs rebalancing"""
        try:
            total_value = self.get_total_portfolio_value()
            if total_value == 0:
                return False
            
            for pair, target_pct in self.target_allocation.items():
                if pair in self.positions:
                    position_value = self.positions[pair]['size'] * self.positions[pair]['current_price']
                    current_pct = position_value / total_value
                    
                    # Check if deviation exceeds threshold
                    if abs(current_pct - target_pct) > self.rebalance_threshold:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking rebalancing need: {e}")
            return False
    
    def calculate_rebalancing(self) -> List[Dict[str, Any]]:
        """Calculate rebalancing actions needed"""
        try:
            actions = []
            total_value = self.get_total_portfolio_value()
            
            if total_value == 0:
                return actions
            
            for pair, target_pct in self.target_allocation.items():
                target_value = total_value * target_pct
                
                if pair in self.positions:
                    current_value = self.positions[pair]['size'] * self.positions[pair]['current_price']
                else:
                    current_value = 0
                
                difference = target_value - current_value
                
                if abs(difference) > total_value * self.rebalance_threshold:
                    current_price = self.positions[pair]['current_price'] if pair in self.positions else 1
                    size = abs(difference) / current_price
                    side = 'BUY' if difference > 0 else 'SELL'
                    
                    actions.append({
                        'pair': pair,
                        'side': side,
                        'size': size,
                        'price': current_price,
                        'reason': f'Rebalance to {target_pct*100:.1f}%'
                    })
            
            return actions
            
        except Exception as e:
            logger.error(f"Error calculating rebalancing: {e}")
            return []
    
    def _record_trade(self, pair: str, side: str, size: float, price: float, 
                     entry_price: float, pnl: float, pnl_pct: float):
        """Record completed trade"""
        try:
            trade_record = {
                'timestamp': datetime.now(),
                'pair': pair,
                'side': side,
                'size': size,
                'price': price,
                'entry_price': entry_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            }
            
            self.trade_history.append(trade_record)
            
            # Update win/loss counters
            if pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Keep only recent trades
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
                
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def _update_performance_history(self):
        """Update performance tracking"""
        try:
            portfolio_value = self.get_total_portfolio_value()
            
            performance_record = {
                'timestamp': datetime.now(),
                'portfolio_value': portfolio_value,
                'cash_balance': self.cash_balance,
                'total_pnl': portfolio_value - self.initial_capital,
                'total_pnl_pct': ((portfolio_value / self.initial_capital) - 1) * 100,
                'active_positions': len(self.positions)
            }
            
            self.performance_history.append(performance_record)
            
            # Keep only recent history
            if len(self.performance_history) > 10000:
                self.performance_history = self.performance_history[-10000:]
                
        except Exception as e:
            logger.error(f"Error updating performance history: {e}")
    
    def get_performance_history(self) -> pd.DataFrame:
        """Get performance history as DataFrame"""
        try:
            if not self.performance_history:
                return pd.DataFrame()
            
            df = pd.DataFrame(self.performance_history)
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Error getting performance history: {e}")
            return pd.DataFrame()
    
    def get_recent_trades(self, limit: int = 10) -> pd.DataFrame:
        """Get recent trades as DataFrame"""
        try:
            if not self.trade_history:
                return pd.DataFrame()
            
            recent_trades = self.trade_history[-limit:]
            df = pd.DataFrame(recent_trades)
            
            # Format for display
            if not df.empty:
                df['Time'] = df['timestamp'].dt.strftime('%H:%M:%S')
                df['P&L $'] = df['pnl'].apply(lambda x: f"${x:+.2f}")
                df['P&L %'] = df['pnl_pct'].apply(lambda x: f"{x:+.2f}%")
                df = df[['Time', 'pair', 'side', 'size', 'price', 'P&L $', 'P&L %']]
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return pd.DataFrame()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        try:
            current_value = self.get_total_portfolio_value()
            total_return = (current_value / self.initial_capital - 1) * 100
            
            # Calculate Sharpe ratio (simplified)
            if len(self.performance_history) > 1:
                returns = []
                for i in range(1, len(self.performance_history)):
                    prev_val = self.performance_history[i-1]['portfolio_value']
                    curr_val = self.performance_history[i]['portfolio_value']
                    ret = (curr_val / prev_val - 1) if prev_val > 0 else 0
                    returns.append(ret)
                
                if returns:
                    avg_return = np.mean(returns)
                    std_return = np.std(returns) if len(returns) > 1 else 0
                    sharpe_ratio = avg_return / std_return if std_return > 0 else 0
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
            
            # Calculate max drawdown
            peak_value = self.initial_capital
            max_drawdown = 0
            
            for record in self.performance_history:
                value = record['portfolio_value']
                if value > peak_value:
                    peak_value = value
                drawdown = (peak_value - value) / peak_value
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            return {
                'total_return_pct': total_return,
                'total_trades': self.total_trades,
                'win_rate': self.get_win_rate(),
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown * 100,
                'current_positions': len(self.positions),
                'days_active': (datetime.now() - self.start_time).days,
                'portfolio_value': current_value
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
