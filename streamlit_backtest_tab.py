#!/usr/bin/env python3
"""
Streamlit Backtest Tab
Tab Streamlit per gestione backtests QuantConnect
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List

# Import moduli locali
from quantconnect_generator import QuantConnectGenerator
from quantconnect_launcher import QuantConnectLauncher
from backtest_results_parser import BacktestResultsParser

class StreamlitBacktestTab:
    """Tab Streamlit per backtesting QuantConnect"""
    
    def __init__(self):
        self.generator = QuantConnectGenerator()
        self.launcher = QuantConnectLauncher()
        self.parser = BacktestResultsParser()
        
        # Initialize session state
        if 'backtest_results' not in st.session_state:
            st.session_state.backtest_results = {}
        
        if 'running_backtests' not in st.session_state:
            st.session_state.running_backtests = []
    
    def render_backtest_tab(self):
        """Renderizza tab completa backtesting"""
        
        st.header("🔬 QuantConnect Backtesting")
        
        # Check system status
        self._render_system_status()
        
        st.divider()
        
        # Tabs secondari
        tab1, tab2, tab3, tab4 = st.tabs([
            "🚀 New Backtest", 
            "📊 Results", 
            "🔄 Optimization", 
            "📈 Comparison"
        ])
        
        with tab1:
            self._render_new_backtest_tab()
        
        with tab2:
            self._render_results_tab()
        
        with tab3:
            self._render_optimization_tab()
        
        with tab4:
            self._render_comparison_tab()
    
    def _render_system_status(self):
        """Mostra status sistema QuantConnect"""
        
        # Status overview
        col1, col2, col3, col4 = st.columns(4)
        
        # Check LEAN installation
        lean_status = self.launcher.check_lean_installation()
        
        with col1:
            if lean_status["lean_installed"]:
                st.success("✅ LEAN CLI")
                if lean_status["lean_version"]:
                    st.caption(f"v{lean_status['lean_version']}")
            else:
                st.error("❌ LEAN CLI")
                st.caption("Not installed")
        
        with col2:
            if lean_status["docker_available"]:
                st.success("✅ Docker")
            else:
                st.warning("⚠️ Docker")
                st.caption("Optional")
        
        with col3:
            if lean_status["logged_in"]:
                st.success("✅ Cloud Connected")
            else:
                st.warning("⚠️ Local Only")
                st.caption("Cloud features limited")
        
        with col4:
            running = len(self.launcher.list_running_backtests())
            if running > 0:
                st.info(f"🔄 {running} Running")
            else:
                st.success("✅ Ready")
        
        st.divider()
        
        # Connection Management
        st.markdown("**🔗 Connection Setup**")
        
        connection_col1, connection_col2 = st.columns(2)
        
        with connection_col1:
            st.markdown("**Local LEAN Installation**")
            
            if not lean_status["lean_installed"]:
                st.warning("LEAN CLI not installed")
                
                if st.button("🚀 Install LEAN CLI", type="primary"):
                    self._install_lean_cli()
            else:
                st.success("LEAN CLI ready for local backtesting")
                
                # Local configuration options
                with st.expander("⚙️ Local Configuration"):
                    data_path = st.text_input("Data Directory", value="./data")
                    output_path = st.text_input("Output Directory", value="./results")
                    
                    if st.button("📁 Create Directories"):
                        import os
                        os.makedirs(data_path, exist_ok=True)
                        os.makedirs(output_path, exist_ok=True)
                        st.success("Directories created successfully")
        
        with connection_col2:
            st.markdown("**QuantConnect Cloud Account**")
            
            if lean_status["logged_in"]:
                st.success("Connected to QuantConnect Cloud")
                
                if st.button("🔄 Refresh Cloud Status"):
                    st.rerun()
                    
                if st.button("🚪 Logout from Cloud"):
                    self._logout_quantconnect()
            else:
                st.info("Connect to access cloud features and data")
                
                # Cloud login form
                with st.expander("🌐 Connect to QuantConnect"):
                    st.markdown("**Login Options:**")
                    
                    login_method = st.radio(
                        "Choose login method:",
                        ["Interactive Login", "API Credentials"],
                        help="Interactive login opens browser, API uses organization ID and token"
                    )
                    
                    if login_method == "Interactive Login":
                        if st.button("🔐 Login via Browser", type="primary"):
                            self._login_quantconnect_interactive()
                    
                    else:  # API Credentials
                        org_id = st.text_input("Organization ID", help="Found in QuantConnect account settings")
                        api_token = st.text_input("API Token", type="password", help="Generate in QuantConnect account")
                        
                        if st.button("🔐 Login with API", type="primary"):
                            if org_id and api_token:
                                self._login_quantconnect_api(org_id, api_token)
                            else:
                                st.error("Please provide both Organization ID and API Token")
                    
                    st.markdown("**Don't have an account?**")
                    st.markdown("[Create QuantConnect Account](https://www.quantconnect.com/signup)")
        
        # Advanced options
        with st.expander("🔧 Advanced Setup Options"):
            col_adv1, col_adv2 = st.columns(2)
            
            with col_adv1:
                st.markdown("**Local Development**")
                
                if st.button("📦 Download Sample Data"):
                    self._download_sample_data()
                
                if st.button("🧪 Test Local Setup"):
                    self._test_local_setup()
            
            with col_adv2:
                st.markdown("**Cloud Features**")
                
                if st.button("☁️ Sync Cloud Projects"):
                    self._sync_cloud_projects()
                
                if st.button("📊 Access Cloud Data"):
                    self._access_cloud_data()
        
        # Installation help
        if not lean_status["lean_installed"]:
            with st.expander("📥 Manual Installation Instructions"):
                st.code("""
# Option 1: Install via pip
pip install lean

# Option 2: Install via conda
conda install -c conda-forge lean

# Option 3: Download from GitHub
git clone https://github.com/QuantConnect/Lean.git

# Verify installation
lean --version

# Login to QuantConnect (optional)
lean login
                """)
    
    def _render_new_backtest_tab(self):
        """Tab per nuovo backtest"""
        
        st.subheader("🚀 Create New Backtest")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Strategy selection
            st.markdown("**Strategy Configuration**")
            
            strategy_type = st.selectbox(
                "Strategy Type",
                ["AI Generated", "Technical Indicators", "Mean Reversion", "Momentum", "Custom"]
            )
            
            strategy_name = st.text_input(
                "Strategy Name",
                value=f"AI_Strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Symbols
            symbols_input = st.text_input(
                "Trading Symbols (comma separated)",
                value="SPY,QQQ,BTC-USD,ETH-USD"
            )
            symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]
            
            # Date range
            col_start, col_end = st.columns(2)
            
            with col_start:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=365)
                )
            
            with col_end:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now() - timedelta(days=30)
                )
            
            # Initial cash
            initial_cash = st.number_input(
                "Initial Cash ($)",
                min_value=1000,
                max_value=10000000,
                value=100000,
                step=10000
            )
        
        with col2:
            st.markdown("**Strategy Parameters**")
            
            # Parameters based on strategy type
            if strategy_type == "Technical Indicators":
                rsi_period = st.slider("RSI Period", 5, 30, 14)
                rsi_overbought = st.slider("RSI Overbought", 60, 90, 70)
                rsi_oversold = st.slider("RSI Oversold", 10, 40, 30)
                ma_fast = st.slider("Fast MA", 5, 30, 10)
                ma_slow = st.slider("Slow MA", 20, 100, 30)
                
                strategy_params = {
                    "rsi_period": rsi_period,
                    "rsi_overbought": rsi_overbought,
                    "rsi_oversold": rsi_oversold,
                    "ma_fast": ma_fast,
                    "ma_slow": ma_slow
                }
                
            elif strategy_type == "AI Generated":
                # AI strategy parameters
                ai_model_weight = st.slider("AI Model Weight", 0.0, 1.0, 0.6, 0.1)
                technical_weight = st.slider("Technical Weight", 0.0, 1.0, 0.4, 0.1)
                confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.9, 0.7, 0.05)
                
                strategy_params = {
                    "ai_model_weight": ai_model_weight,
                    "technical_weight": technical_weight,
                    "confidence_threshold": confidence_threshold
                }
            
            else:
                # Generic parameters
                risk_per_trade = st.slider("Risk per Trade (%)", 1, 10, 2)
                max_positions = st.slider("Max Positions", 1, 20, 5)
                
                strategy_params = {
                    "risk_per_trade": risk_per_trade / 100,
                    "max_positions": max_positions
                }
            
            # Risk management
            st.markdown("**Risk Management**")
            
            max_drawdown = st.slider("Max Drawdown (%)", 5, 50, 15)
            stop_loss = st.slider("Stop Loss (%)", 1, 20, 5)
            
            risk_params = {
                "max_drawdown": max_drawdown / 100,
                "stop_loss": stop_loss / 100
            }
        
        st.divider()
        
        # Generate and launch
        col_gen, col_launch = st.columns(2)
        
        with col_gen:
            if st.button("🔧 Generate Strategy", type="primary"):
                self._generate_strategy(
                    strategy_name, strategy_type, symbols, 
                    strategy_params, risk_params
                )
        
        with col_launch:
            if st.button("🚀 Launch Backtest", type="primary"):
                self._launch_backtest(
                    strategy_name, strategy_type, symbols,
                    start_date, end_date, initial_cash,
                    strategy_params, risk_params
                )
    
    def _generate_strategy(self, name: str, type_: str, symbols: List[str], 
                          params: Dict, risk_params: Dict):
        """Genera strategia"""
        
        with st.spinner("Generating strategy..."):
            
            # Crea configurazione strategia
            strategy_config = {
                "description": f"{type_} strategy generated by AI Trading Bot",
                "symbols": symbols,
                "logic_type": type_.lower().replace(" ", "_"),
                "parameters": params,
                "risk_management": risk_params,
                "indicators": ["RSI", "SMA", "EMA", "MACD"]
            }
            
            # Genera codice LEAN
            lean_code = self.generator.generate_lean_strategy(strategy_config, name)
            
            # Salva strategia
            filepath = self.generator.save_strategy(lean_code, name)
            
            st.success(f"✅ Strategy generated: {filepath}")
            
            # Mostra anteprima codice
            with st.expander("📝 Strategy Code Preview"):
                st.code(lean_code[:1000] + "..." if len(lean_code) > 1000 else lean_code, 
                       language="python")
    
    def _launch_backtest(self, name: str, type_: str, symbols: List[str],
                        start_date, end_date, initial_cash: int,
                        params: Dict, risk_params: Dict):
        """Lancia backtest"""
        
        with st.spinner("Launching backtest..."):
            
            # Prima genera la strategia se non esiste
            strategy_config = {
                "description": f"{type_} strategy for backtesting",
                "symbols": symbols,
                "logic_type": type_.lower().replace(" ", "_"),
                "parameters": params,
                "risk_management": risk_params,
                "start_year": start_date.year,
                "start_month": start_date.month,
                "start_day": start_date.day,
                "end_year": end_date.year,
                "end_month": end_date.month,
                "end_day": end_date.day,
                "initial_cash": initial_cash
            }
            
            lean_code = self.generator.generate_lean_strategy(strategy_config, name)
            filepath = self.generator.save_strategy(lean_code, name)
            
            # Lancia backtest
            def backtest_callback(backtest_id, result):
                """Callback quando backtest completa"""
                st.session_state.backtest_results[backtest_id] = result
                if backtest_id in st.session_state.running_backtests:
                    st.session_state.running_backtests.remove(backtest_id)
            
            backtest_id = self.launcher.launch_backtest(
                filepath, name,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                initial_cash,
                callback=backtest_callback
            )
            
            st.session_state.running_backtests.append(backtest_id)
            
            st.success(f"✅ Backtest launched: {backtest_id}")
            
            # Progress tracking
            self._show_backtest_progress(backtest_id)
    
    def _show_backtest_progress(self, backtest_id: str):
        """Mostra progress backtest"""
        
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Monitor progress (simplified)
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"Running backtest... {i+1}%")
                
                # Check if completed
                if backtest_id in st.session_state.backtest_results:
                    progress_bar.progress(100)
                    status_text.success("✅ Backtest completed!")
                    break
                
                # Simulated progress
                import time
                time.sleep(0.1)
    
    def _render_results_tab(self):
        """Tab risultati backtests"""
        
        st.subheader("📊 Backtest Results")
        
        # Running backtests
        if st.session_state.running_backtests:
            st.markdown("**🔄 Running Backtests:**")
            
            for backtest_id in st.session_state.running_backtests:
                status = self.launcher.get_backtest_status(backtest_id)
                
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.text(backtest_id)
                
                with col2:
                    st.info(status.get("status", "unknown"))
                
                with col3:
                    if st.button("🛑 Cancel", key=f"cancel_{backtest_id}"):
                        self.launcher.cancel_backtest(backtest_id)
                        st.session_state.running_backtests.remove(backtest_id)
                        st.rerun()
        
        st.divider()
        
        # Completed backtests
        st.markdown("**✅ Completed Backtests:**")
        
        completed_backtests = list(st.session_state.backtest_results.keys())
        completed_backtests.extend(self.launcher.list_completed_backtests())
        completed_backtests = list(set(completed_backtests))  # Remove duplicates
        
        if not completed_backtests:
            st.info("No completed backtests yet. Create a new backtest to get started.")
            return
        
        # Results selection
        selected_backtest = st.selectbox(
            "Select backtest to analyze:",
            completed_backtests
        )
        
        if selected_backtest:
            self._show_backtest_results(selected_backtest)
    
    def _show_backtest_results(self, backtest_id: str):
        """Mostra risultati dettagliati backtest"""
        
        # Analizza risultati
        analysis = self.parser.analyze_backtest_performance(backtest_id)
        
        if "error" in analysis:
            st.error(f"Error analyzing results: {analysis['error']}")
            return
        
        if not analysis.get("success"):
            st.error("Backtest failed or incomplete")
            return
        
        # Performance metrics
        st.markdown("**📈 Performance Metrics:**")
        
        metrics = analysis.get("detailed_metrics", analysis.get("basic_metrics", {}))
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_return = metrics.get("total_return", 0) * 100
            st.metric("Total Return", f"{total_return:.1f}%")
        
        with col2:
            sharpe = metrics.get("sharpe_ratio", 0)
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        with col3:
            max_dd = abs(metrics.get("max_drawdown", 0)) * 100
            st.metric("Max Drawdown", f"{max_dd:.1f}%")
        
        with col4:
            win_rate = metrics.get("win_rate", 0) * 100
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Performance score
        score = analysis.get("performance_score", 50)
        st.markdown(f"**Overall Performance Score: {score:.0f}/100**")
        
        # Progress bar per score
        score_color = "green" if score >= 70 else "orange" if score >= 50 else "red"
        st.markdown(f"""
        <div style="background-color: #f0f0f0; border-radius: 10px; padding: 10px;">
            <div style="background-color: {score_color}; height: 20px; width: {score}%; border-radius: 10px;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            self._render_equity_curve(analysis)
        
        with col_chart2:
            self._render_drawdown_chart(analysis)
        
        # Detailed metrics table
        st.markdown("**📋 Detailed Metrics:**")
        
        if metrics:
            metrics_df = pd.DataFrame([
                {"Metric": k.replace("_", " ").title(), "Value": v}
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            ])
            
            st.dataframe(metrics_df, use_container_width=True)
        
        # Recommendations
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            st.markdown("**💡 Recommendations:**")
            for rec in recommendations:
                st.write(f"• {rec}")
        
        # Download options
        st.markdown("**📥 Downloads:**")
        
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        with col_dl1:
            if st.button("📊 Download Results JSON"):
                self._download_results_json(backtest_id, analysis)
        
        with col_dl2:
            if st.button("📈 Download Performance CSV"):
                self._download_performance_csv(backtest_id, metrics)
        
        with col_dl3:
            if st.button("📋 Download Full Report"):
                self._download_full_report(backtest_id, analysis)
    
    def _render_equity_curve(self, analysis: Dict):
        """Renderizza curva equity"""
        
        st.markdown("**💹 Equity Curve**")
        
        equity_data = analysis.get("equity_curve", [])
        
        if equity_data:
            df = pd.DataFrame(equity_data)
            
            if not df.empty and "time" in df.columns and "value" in df.columns:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df["time"],
                    y=df["value"],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig.update_layout(
                    height=300,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Equity curve data not available")
        else:
            # Simulated equity curve
            dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
            values = [100000 * (1 + i * 0.01 + (i % 7 - 3) * 0.005) for i in range(30)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines',
                name='Portfolio Value (Simulated)'
            ))
            
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_drawdown_chart(self, analysis: Dict):
        """Renderizza grafico drawdown"""
        
        st.markdown("**📉 Drawdown**")
        
        drawdown_data = analysis.get("drawdown", {}).get("drawdown_periods", [])
        
        if drawdown_data:
            df = pd.DataFrame(drawdown_data)
            
            if not df.empty:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df["time"],
                    y=df["drawdown"],
                    mode='lines',
                    name='Drawdown',
                    fill='tonexty',
                    line=dict(color='red', width=1)
                ))
                
                fig.update_layout(
                    height=300,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Drawdown chart data not available")
        else:
            # Simulated drawdown
            dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
            drawdowns = [-(i % 10) * 0.01 for i in range(30)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=drawdowns,
                mode='lines',
                fill='tonexty',
                name='Drawdown (Simulated)'
            ))
            
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_optimization_tab(self):
        """Tab ottimizzazione strategie"""
        
        st.subheader("🔄 Strategy Optimization")
        
        st.markdown("""
        Optimize your trading strategy by testing different parameter combinations
        to find the best performing configuration.
        """)
        
        # Strategy selection for optimization
        strategies_folder = "strategies"
        if os.path.exists(strategies_folder):
            strategy_files = [f for f in os.listdir(strategies_folder) if f.endswith('.py')]
        else:
            strategy_files = []
        
        if not strategy_files:
            st.warning("No strategies found. Create a strategy first.")
            return
        
        selected_strategy = st.selectbox(
            "Select strategy to optimize:",
            strategy_files
        )
        
        # Parameter ranges
        st.markdown("**Parameter Ranges for Optimization:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Technical Parameters:**")
            
            # RSI parameters
            rsi_min = st.number_input("RSI Period Min", value=10, min_value=5, max_value=30)
            rsi_max = st.number_input("RSI Period Max", value=20, min_value=rsi_min, max_value=50)
            
            # Moving average parameters
            ma_fast_min = st.number_input("Fast MA Min", value=5, min_value=2, max_value=20)
            ma_fast_max = st.number_input("Fast MA Max", value=15, min_value=ma_fast_min, max_value=30)
            
            ma_slow_min = st.number_input("Slow MA Min", value=20, min_value=15, max_value=50)
            ma_slow_max = st.number_input("Slow MA Max", value=40, min_value=ma_slow_min, max_value=100)
        
        with col2:
            st.markdown("**Risk Parameters:**")
            
            # Risk per trade
            risk_min = st.slider("Risk per Trade Min (%)", 0.5, 5.0, 1.0, 0.5)
            risk_max = st.slider("Risk per Trade Max (%)", risk_min, 10.0, 3.0, 0.5)
            
            # Stop loss
            stop_min = st.slider("Stop Loss Min (%)", 1.0, 10.0, 2.0, 0.5)
            stop_max = st.slider("Stop Loss Max (%)", stop_min, 20.0, 8.0, 0.5)
        
        # Optimization settings
        st.markdown("**Optimization Settings:**")
        
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            optimization_metric = st.selectbox(
                "Optimize for:",
                ["Sharpe Ratio", "Total Return", "Profit Factor", "Win Rate"]
            )
        
        with col_opt2:
            max_combinations = st.slider("Max Parameter Combinations", 5, 50, 20)
        
        # Launch optimization
        if st.button("🚀 Start Optimization", type="primary"):
            self._launch_optimization(
                selected_strategy,
                {
                    "rsi_period": {"min": rsi_min, "max": rsi_max, "step": 2},
                    "ma_fast": {"min": ma_fast_min, "max": ma_fast_max, "step": 2},
                    "ma_slow": {"min": ma_slow_min, "max": ma_slow_max, "step": 5},
                    "risk_per_trade": {"min": risk_min/100, "max": risk_max/100, "step": 0.005},
                    "stop_loss": {"min": stop_min/100, "max": stop_max/100, "step": 0.01}
                },
                optimization_metric,
                max_combinations
            )
    
    def _launch_optimization(self, strategy_file: str, param_ranges: Dict, 
                           metric: str, max_combinations: int):
        """Lancia ottimizzazione strategia"""
        
        with st.spinner("Starting optimization..."):
            
            strategy_path = os.path.join("strategies", strategy_file)
            strategy_name = strategy_file.replace('.py', '')
            
            optimization_id = self.launcher.optimize_strategy(
                strategy_path, strategy_name, param_ranges
            )
            
            st.success(f"✅ Optimization started: {optimization_id}")
            
            # Monitor optimization progress
            st.markdown("**Optimization Progress:**")
            
            progress_bar = st.progress(0)
            results_container = st.container()
            
            # This would be updated in real-time in a production version
            with results_container:
                st.info("Optimization running... Results will appear here.")
    
    def _render_comparison_tab(self):
        """Tab confronto strategie"""
        
        st.subheader("📈 Strategy Comparison")
        
        completed_backtests = list(st.session_state.backtest_results.keys())
        completed_backtests.extend(self.launcher.list_completed_backtests())
        completed_backtests = list(set(completed_backtests))
        
        if len(completed_backtests) < 2:
            st.info("Need at least 2 completed backtests for comparison.")
            return
        
        # Strategy selection
        selected_strategies = st.multiselect(
            "Select strategies to compare:",
            completed_backtests,
            default=completed_backtests[:min(3, len(completed_backtests))]
        )
        
        if len(selected_strategies) < 2:
            st.warning("Select at least 2 strategies to compare.")
            return
        
        # Comparison analysis
        comparison = self.parser.compare_strategies(selected_strategies)
        
        # Best overall
        if comparison.get("best_overall"):
            st.success(f"🏆 Best Overall Strategy: {comparison['best_overall']}")
        
        # Metrics comparison table
        st.markdown("**📊 Metrics Comparison:**")
        
        comparison_data = []
        
        for strategy in comparison["strategies"]:
            strategy_id = strategy["backtest_id"]
            metrics = strategy.get("detailed_metrics", strategy.get("basic_metrics", {}))
            
            row = {"Strategy": strategy_id}
            
            for metric in ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]:
                value = metrics.get(metric, 0)
                if metric == "max_drawdown":
                    row[metric.replace("_", " ").title()] = f"{abs(value)*100:.1f}%"
                elif metric in ["total_return", "win_rate"]:
                    row[metric.replace("_", " ").title()] = f"{value*100:.1f}%"
                else:
                    row[metric.replace("_", " ").title()] = f"{value:.2f}"
            
            row["Performance Score"] = f"{strategy.get('performance_score', 0):.0f}"
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visual comparison
        self._render_comparison_charts(comparison_data)
        
        # Best by metric
        st.markdown("**🎯 Best by Metric:**")
        
        best_by_metric = comparison.get("best_by_metric", {})
        
        for metric, best_strategy in best_by_metric.items():
            st.write(f"• **{metric.replace('_', ' ').title()}**: {best_strategy}")
    
    def _render_comparison_charts(self, comparison_data: List[Dict]):
        """Renderizza grafici confronto"""
        
        if not comparison_data:
            return
        
        df = pd.DataFrame(comparison_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Return vs Risk scatter plot
            st.markdown("**Return vs Risk**")
            
            try:
                returns = [float(row["Total Return"].replace('%', '')) for row in comparison_data]
                drawdowns = [float(row["Max Drawdown"].replace('%', '')) for row in comparison_data]
                strategies = [row["Strategy"] for row in comparison_data]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=drawdowns,
                    y=returns,
                    mode='markers+text',
                    text=strategies,
                    textposition="top center",
                    marker=dict(size=10)
                ))
                
                fig.update_layout(
                    height=400,
                    xaxis_title="Max Drawdown (%)",
                    yaxis_title="Total Return (%)",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception:
                st.error("Could not create return vs risk chart")
        
        with col2:
            # Performance scores bar chart
            st.markdown("**Performance Scores**")
            
            try:
                scores = [float(row["Performance Score"]) for row in comparison_data]
                strategies = [row["Strategy"] for row in comparison_data]
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=strategies,
                    y=scores,
                    text=scores,
                    textposition='auto'
                ))
                
                fig.update_layout(
                    height=400,
                    yaxis_title="Performance Score",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception:
                st.error("Could not create performance scores chart")
    
    def _download_results_json(self, backtest_id: str, analysis: Dict):
        """Download risultati in JSON"""
        
        json_data = json.dumps(analysis, indent=2, default=str)
        
        st.download_button(
            label="📊 Download JSON",
            data=json_data,
            file_name=f"{backtest_id}_results.json",
            mime="application/json"
        )
    
    def _download_performance_csv(self, backtest_id: str, metrics: Dict):
        """Download metriche in CSV"""
        
        df = pd.DataFrame([
            {"Metric": k.replace("_", " ").title(), "Value": v}
            for k, v in metrics.items()
        ])
        
        csv_data = df.to_csv(index=False)
        
        st.download_button(
            label="📈 Download CSV",
            data=csv_data,
            file_name=f"{backtest_id}_metrics.csv",
            mime="text/csv"
        )
    
    def _download_full_report(self, backtest_id: str, analysis: Dict):
        """Download report completo"""
        
        report = f"""
# Backtest Report: {backtest_id}

## Summary
- Backtest ID: {backtest_id}
- Analysis Date: {analysis.get('analysis_time', 'Unknown')}
- Performance Score: {analysis.get('performance_score', 0):.0f}/100

## Performance Metrics
"""
        
        metrics = analysis.get("detailed_metrics", analysis.get("basic_metrics", {}))
        
        for metric, value in metrics.items():
            report += f"- {metric.replace('_', ' ').title()}: {value}\n"
        
        report += "\n## Recommendations\n"
        
        for rec in analysis.get("recommendations", []):
            report += f"- {rec}\n"
        
        st.download_button(
            label="📋 Download Report",
            data=report,
            file_name=f"{backtest_id}_report.md",
            mime="text/markdown"
        )

def create_backtest_tab():
    """Crea e renderizza tab backtest"""
    
    tab = StreamlitBacktestTab()
    tab.render_backtest_tab()

# Per test standalone
if __name__ == "__main__":
    # Test della tab
    st.set_page_config(page_title="QuantConnect Backtesting", layout="wide")
    create_backtest_tab()