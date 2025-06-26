#!/usr/bin/env python3
"""
Backtest Results Parser
Legge e analizza risultati dei backtest QuantConnect
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

class BacktestResultsParser:
    """Parser per risultati backtest QuantConnect"""
    
    def __init__(self, results_path="results"):
        self.results_path = Path(results_path)
        self.results_path.mkdir(exist_ok=True)
        
    def parse_lean_json_results(self, json_file: str) -> Dict[str, Any]:
        """Parse file JSON risultati LEAN"""
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Estrai metriche principali
            metrics = self._extract_performance_metrics(data)
            
            # Estrai trades
            trades = self._extract_trades_data(data)
            
            # Estrai equity curve
            equity_curve = self._extract_equity_curve(data)
            
            # Estrai drawdown
            drawdown_data = self._extract_drawdown_data(data)
            
            return {
                "metrics": metrics,
                "trades": trades,
                "equity_curve": equity_curve,
                "drawdown": drawdown_data,
                "raw_data": data
            }
            
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def _extract_performance_metrics(self, data: Dict) -> Dict[str, float]:
        """Estrae metriche performance dai risultati"""
        
        metrics = {}
        
        # Metriche da Statistics section
        if "Statistics" in data:
            stats = data["Statistics"]
            
            # Mapping delle metriche LEAN -> nostro formato
            metric_mappings = {
                "Total Return": "total_return",
                "Annual Return": "annual_return", 
                "Sharpe Ratio": "sharpe_ratio",
                "Sortino Ratio": "sortino_ratio",
                "Maximum Drawdown": "max_drawdown",
                "Compounding Annual Return": "cagr",
                "Win Rate": "win_rate",
                "Loss Rate": "loss_rate",
                "Profit-Loss Ratio": "profit_loss_ratio",
                "Alpha": "alpha",
                "Beta": "beta",
                "Annual Standard Deviation": "volatility",
                "Information Ratio": "information_ratio",
                "Tracking Error": "tracking_error",
                "Treynor Ratio": "treynor_ratio",
                "Total Fees": "total_fees",
                "Estimated Strategy Capacity": "strategy_capacity",
                "Portfolio Turnover": "portfolio_turnover"
            }
            
            for lean_name, our_name in metric_mappings.items():
                if lean_name in stats:
                    value = stats[lean_name]
                    metrics[our_name] = self._parse_metric_value(value)
        
        # Metriche da RuntimeStatistics
        if "RuntimeStatistics" in data:
            runtime_stats = data["RuntimeStatistics"]
            
            runtime_mappings = {
                "Unrealized": "unrealized_pnl",
                "Fees": "total_fees_runtime",
                "Net Profit": "net_profit",
                "Return": "total_return_runtime",
                "Equity": "final_equity"
            }
            
            for lean_name, our_name in runtime_mappings.items():
                if lean_name in runtime_stats:
                    value = runtime_stats[lean_name]
                    metrics[our_name] = self._parse_metric_value(value)
        
        # Calcola metriche aggiuntive
        metrics.update(self._calculate_additional_metrics(data))
        
        return metrics
    
    def _parse_metric_value(self, value: Any) -> float:
        """Parse valore metrica in float"""
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Rimuovi simboli e converti
            clean_value = value.replace('$', '').replace(',', '').replace('%', '')
            
            try:
                parsed = float(clean_value)
                
                # Se era una percentuale, converti in decimale
                if '%' in value:
                    return parsed / 100
                
                return parsed
                
            except ValueError:
                return 0.0
        
        return 0.0
    
    def _extract_trades_data(self, data: Dict) -> List[Dict]:
        """Estrae dati dei singoli trade"""
        
        trades = []
        
        # Trades da Orders section
        if "Orders" in data:
            orders = data["Orders"]
            
            for order in orders:
                if order.get("Status") == "Filled":
                    trade = {
                        "time": order.get("Time"),
                        "symbol": order.get("Symbol", {}).get("Value", ""),
                        "direction": order.get("Direction"),
                        "quantity": order.get("Quantity", 0),
                        "fill_price": order.get("Price", 0),
                        "value": order.get("Value", 0),
                        "tag": order.get("Tag", ""),
                        "order_id": order.get("Id", 0)
                    }
                    trades.append(trade)
        
        # Se non ci sono ordini, prova da ProfitLoss
        elif "ProfitLoss" in data:
            profit_loss = data["ProfitLoss"]
            
            for symbol, pnl_data in profit_loss.items():
                trade = {
                    "symbol": symbol,
                    "pnl": pnl_data,
                    "type": "summary"
                }
                trades.append(trade)
        
        return trades
    
    def _extract_equity_curve(self, data: Dict) -> List[Dict]:
        """Estrae curva equity dal backtest"""
        
        equity_points = []
        
        # Da Charts section
        if "Charts" in data:
            charts = data["Charts"]
            
            # Cerca chart Benchmark o Strategy Equity
            for chart_name, chart_data in charts.items():
                if "equity" in chart_name.lower() or "strategy" in chart_name.lower():
                    
                    if "Series" in chart_data:
                        for series_name, series_data in chart_data["Series"].items():
                            if "Values" in series_data:
                                
                                for point in series_data["Values"]:
                                    equity_point = {
                                        "time": point.get("x"),
                                        "value": point.get("y", 0),
                                        "series": series_name
                                    }
                                    equity_points.append(equity_point)
        
        # Se non trovato, usa RuntimeStatistics
        if not equity_points and "RuntimeStatistics" in data:
            runtime_stats = data["RuntimeStatistics"]
            
            if "Equity" in runtime_stats:
                equity_points.append({
                    "time": datetime.now().isoformat(),
                    "value": self._parse_metric_value(runtime_stats["Equity"]),
                    "series": "final_equity"
                })
        
        return sorted(equity_points, key=lambda x: x.get("time", ""))
    
    def _extract_drawdown_data(self, data: Dict) -> Dict[str, Any]:
        """Estrae dati drawdown"""
        
        drawdown_info = {
            "max_drawdown": 0.0,
            "max_drawdown_duration": 0,
            "drawdown_periods": []
        }
        
        # Da Statistics
        if "Statistics" in data and "Maximum Drawdown" in data["Statistics"]:
            drawdown_info["max_drawdown"] = self._parse_metric_value(
                data["Statistics"]["Maximum Drawdown"]
            )
        
        # Da Charts - cerca drawdown chart
        if "Charts" in data:
            charts = data["Charts"]
            
            for chart_name, chart_data in charts.items():
                if "drawdown" in chart_name.lower():
                    
                    if "Series" in chart_data:
                        for series_name, series_data in chart_data["Series"].items():
                            if "Values" in series_data:
                                
                                drawdown_points = []
                                for point in series_data["Values"]:
                                    drawdown_points.append({
                                        "time": point.get("x"),
                                        "drawdown": point.get("y", 0)
                                    })
                                
                                drawdown_info["drawdown_periods"] = drawdown_points
        
        return drawdown_info
    
    def _calculate_additional_metrics(self, data: Dict) -> Dict[str, float]:
        """Calcola metriche aggiuntive"""
        
        additional = {}
        
        # Profit Factor
        if "Statistics" in data:
            stats = data["Statistics"]
            
            gross_profit = self._parse_metric_value(stats.get("Total Profit", 0))
            gross_loss = abs(self._parse_metric_value(stats.get("Total Loss", 0)))
            
            if gross_loss > 0:
                additional["profit_factor"] = gross_profit / gross_loss
            else:
                additional["profit_factor"] = float('inf') if gross_profit > 0 else 0
        
        # Trade Frequency (trades per month)
        if "Orders" in data:
            filled_orders = [o for o in data["Orders"] if o.get("Status") == "Filled"]
            
            if filled_orders and len(filled_orders) > 1:
                first_time = filled_orders[0].get("Time")
                last_time = filled_orders[-1].get("Time")
                
                if first_time and last_time:
                    try:
                        first_dt = datetime.fromisoformat(first_time.replace('Z', '+00:00'))
                        last_dt = datetime.fromisoformat(last_time.replace('Z', '+00:00'))
                        
                        days_diff = (last_dt - first_dt).days
                        months_diff = days_diff / 30.44  # Giorni medi per mese
                        
                        if months_diff > 0:
                            additional["trades_per_month"] = len(filled_orders) / months_diff
                        
                    except Exception:
                        pass
        
        return additional
    
    def analyze_backtest_performance(self, backtest_id: str) -> Dict[str, Any]:
        """Analizza performance completa di un backtest"""
        
        # Cerca file risultati
        results_file = self.results_path / f"{backtest_id}_results.json"
        
        if not results_file.exists():
            return {"error": f"Results file not found for {backtest_id}"}
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                backtest_data = json.load(f)
            
            analysis = {
                "backtest_id": backtest_id,
                "analysis_time": datetime.now().isoformat(),
                "success": backtest_data.get("success", False)
            }
            
            if not analysis["success"]:
                analysis["error"] = backtest_data.get("stderr", "Unknown error")
                return analysis
            
            # Parse output files se presenti
            output_files = backtest_data.get("output_files", [])
            
            for file_path in output_files:
                if file_path.endswith('.json'):
                    parsed_results = self.parse_lean_json_results(file_path)
                    
                    if "error" not in parsed_results:
                        analysis["detailed_metrics"] = parsed_results["metrics"]
                        analysis["trades"] = parsed_results["trades"]
                        analysis["equity_curve"] = parsed_results["equity_curve"]
                        analysis["drawdown"] = parsed_results["drawdown"]
            
            # Se non ci sono file dettagliati, usa summary dai logs
            if "detailed_metrics" not in analysis:
                analysis["basic_metrics"] = backtest_data.get("results_summary", {})
            
            # Calcola score performance
            analysis["performance_score"] = self._calculate_performance_score(analysis)
            
            # Genera raccomandazioni
            analysis["recommendations"] = self._generate_recommendations(analysis)
            
            return analysis
            
        except Exception as e:
            return {"error": f"Error analyzing backtest: {str(e)}"}
    
    def _calculate_performance_score(self, analysis: Dict) -> float:
        """Calcola score performance (0-100)"""
        
        score = 50.0  # Base score
        
        metrics = analysis.get("detailed_metrics", analysis.get("basic_metrics", {}))
        
        if not metrics:
            return score
        
        # Sharpe Ratio component (30 points max)
        sharpe = metrics.get("sharpe_ratio", 0)
        if sharpe > 2.0:
            score += 30
        elif sharpe > 1.0:
            score += 20
        elif sharpe > 0.5:
            score += 10
        elif sharpe < 0:
            score -= 20
        
        # Return component (25 points max)
        total_return = metrics.get("total_return", 0)
        if total_return > 0.3:  # 30%+
            score += 25
        elif total_return > 0.15:  # 15%+
            score += 15
        elif total_return > 0.05:  # 5%+
            score += 5
        elif total_return < 0:
            score -= 15
        
        # Drawdown component (20 points max)
        max_drawdown = abs(metrics.get("max_drawdown", 0))
        if max_drawdown < 0.05:  # <5%
            score += 20
        elif max_drawdown < 0.10:  # <10%
            score += 10
        elif max_drawdown < 0.20:  # <20%
            score += 0
        else:  # >20%
            score -= 15
        
        # Win Rate component (15 points max)
        win_rate = metrics.get("win_rate", 0.5)
        if win_rate > 0.6:
            score += 15
        elif win_rate > 0.5:
            score += 10
        elif win_rate < 0.4:
            score -= 10
        
        # Profit Factor component (10 points max)
        profit_factor = metrics.get("profit_factor", 1.0)
        if profit_factor > 2.0:
            score += 10
        elif profit_factor > 1.5:
            score += 5
        elif profit_factor < 1.0:
            score -= 10
        
        return max(0, min(100, score))
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Genera raccomandazioni per miglioramento"""
        
        recommendations = []
        metrics = analysis.get("detailed_metrics", analysis.get("basic_metrics", {}))
        
        if not metrics:
            return ["Insufficient data for recommendations"]
        
        # Sharpe Ratio
        sharpe = metrics.get("sharpe_ratio", 0)
        if sharpe < 1.0:
            recommendations.append("Low Sharpe ratio - consider improving risk-adjusted returns")
        
        # Drawdown
        max_drawdown = abs(metrics.get("max_drawdown", 0))
        if max_drawdown > 0.15:
            recommendations.append("High drawdown - implement stricter risk management")
        
        # Win Rate
        win_rate = metrics.get("win_rate", 0.5)
        if win_rate < 0.45:
            recommendations.append("Low win rate - review entry/exit criteria")
        
        # Profit Factor
        profit_factor = metrics.get("profit_factor", 1.0)
        if profit_factor < 1.2:
            recommendations.append("Low profit factor - optimize trade sizing or filtering")
        
        # Trading Frequency
        trades_per_month = metrics.get("trades_per_month", 0)
        if trades_per_month > 100:
            recommendations.append("High trading frequency - consider reducing overtrading")
        elif trades_per_month < 2:
            recommendations.append("Low trading frequency - consider more opportunities")
        
        # Return vs Risk
        total_return = metrics.get("total_return", 0)
        volatility = metrics.get("volatility", 0)
        
        if total_return > 0 and volatility > 0:
            if total_return / volatility < 0.5:
                recommendations.append("Return too low for risk taken - review strategy logic")
        
        if not recommendations:
            recommendations.append("Strategy performance is well-balanced")
        
        return recommendations
    
    def compare_strategies(self, backtest_ids: List[str]) -> Dict[str, Any]:
        """Confronta multiple strategie"""
        
        comparison = {
            "strategies": [],
            "best_overall": None,
            "best_by_metric": {},
            "comparison_time": datetime.now().isoformat()
        }
        
        for backtest_id in backtest_ids:
            analysis = self.analyze_backtest_performance(backtest_id)
            
            if analysis.get("success"):
                comparison["strategies"].append(analysis)
        
        if not comparison["strategies"]:
            return comparison
        
        # Trova migliore per score complessivo
        best_score = max(comparison["strategies"], 
                        key=lambda x: x.get("performance_score", 0))
        comparison["best_overall"] = best_score["backtest_id"]
        
        # Migliori per metrica specifica
        metrics_to_compare = ["sharpe_ratio", "total_return", "max_drawdown", "win_rate"]
        
        for metric in metrics_to_compare:
            values = []
            for strategy in comparison["strategies"]:
                metrics = strategy.get("detailed_metrics", strategy.get("basic_metrics", {}))
                value = metrics.get(metric)
                
                if value is not None:
                    values.append((strategy["backtest_id"], value))
            
            if values:
                if metric == "max_drawdown":
                    # Per drawdown, il migliore è il più basso (in valore assoluto)
                    best = min(values, key=lambda x: abs(x[1]))
                else:
                    # Per altri, il migliore è il più alto
                    best = max(values, key=lambda x: x[1])
                
                comparison["best_by_metric"][metric] = best[0]
        
        return comparison

def test_parser():
    """Test del parser"""
    
    parser = BacktestResultsParser()
    
    # Simula risultati se non ci sono file reali
    print("Testing backtest results parser...")
    
    # Crea dati di test
    test_data = {
        "Statistics": {
            "Total Return": "15.7%",
            "Sharpe Ratio": "1.23",
            "Maximum Drawdown": "-8.2%",
            "Win Rate": "62.3%"
        },
        "RuntimeStatistics": {
            "Equity": "$115,700",
            "Net Profit": "$15,700"
        },
        "Orders": [
            {
                "Time": "2023-01-15T10:30:00Z",
                "Symbol": {"Value": "SPY"},
                "Direction": "Buy",
                "Quantity": 100,
                "Price": 385.50,
                "Status": "Filled"
            }
        ]
    }
    
    # Test parsing
    parsed = parser._extract_performance_metrics(test_data)
    print("Parsed metrics:")
    for key, value in parsed.items():
        print(f"  {key}: {value}")
    
    return parser

if __name__ == "__main__":
    test_parser()