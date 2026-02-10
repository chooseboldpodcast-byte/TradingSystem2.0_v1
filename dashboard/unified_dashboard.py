# dashboard/unified_dashboard.py
"""
Unified Trading Dashboard
=========================

Combined Live Trading + Backtest Analysis Dashboard

Run: streamlit run dashboard/unified_dashboard.py

Features:
- Mode selector (Live vs Backtest)
- Dynamic sidebar
- Live trading: signals, positions, performance, heatmap, settings
- Backtest analysis: portfolio, trades, universe, heatmap, charts
- Clear visual separation between modes
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="MB Trading System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR VISUAL DIFFERENTIATION
# ============================================================================

def apply_live_theme():
    """Apply red/orange theme for live trading"""
    st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] {
            background-color: #e60624;
        }
        .stTabs [data-baseweb="tab"] {
            color: #ffffff;
        }
        .stTabs [aria-selected="true"] {
            background-color: #fa3f58;
            color: white;
        }
        .mode-indicator-live {
            background-color: #e60624;
            color: white;
            padding: 5px 10px;
            border-radius: 1px;
            text-align: center;
            font-weight: bold;
            margin-bottom: 2px;
        }
    </style>
    """, unsafe_allow_html=True)

def apply_backtest_theme():
    """Apply blue/gray theme for backtest"""
    st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] {
            background-color: #1e3a8a;
        }
        .stTabs [data-baseweb="tab"] {
            color: #93c5fd;
        }
        .stTabs [aria-selected="true"] {
            background-color: #3b82f6;
            color: white;
        }
        .mode-indicator-backtest {
            background-color: #3b82f6;
            color: white;
            padding: 5px 10px;
            border-radius: 1px;
            text-align: center;
            font-weight: bold;
            margin-bottom: 2px;
        }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATABASE CONNECTIONS
# ============================================================================

LIVE_DB_PATH = "database/live_trading.db"
BACKTEST_DB_PATH = "database/weinstein.db"

@st.cache_resource
def get_backtest_db():
    return DatabaseManager(BACKTEST_DB_PATH)

# ============================================================================
# HELPER FUNCTIONS - LIVE TRADING
# ============================================================================

def load_model_allocations():
    """Load model allocation percentages from config"""
    import yaml
    try:
        with open('config/models_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        # Get allocations from active config (C is default for live)
        allocations = config.get('configurations', {}).get('C', {}).get('allocations', {})
        # Convert to model name format
        return {
            'Weinstein_Core': allocations.get('weinstein_core', 0.35),
            '52W_High_Momentum': allocations.get('momentum_52w_high', 0.20),
            'Consolidation_Breakout': allocations.get('consolidation_breakout', 0.10),
            'Enhanced_Mean_Reversion': allocations.get('enhanced_mean_reversion', 0.15),
            'VCP': allocations.get('vcp', 0.10),
            'Pocket_Pivot': allocations.get('pocket_pivot', 0.10),
            'RS_Breakout': allocations.get('rs_breakout', 0.10),
            'RSI_Mean_Reversion': allocations.get('rsi_mean_reversion', 0.10),
            'High_Tight_Flag': allocations.get('high_tight_flag', 0.10),
        }
    except:
        return {}


def calculate_position_size(entry_price, confidence, model_name, unallocated_capital, model_allocations):
    """
    Calculate position size based on:
    1. Model's allocation percentage
    2. Confidence scaling
    3. 30% cap on any single trade
    """
    # Get model's allocation percentage (default 10%)
    allocation_pct = model_allocations.get(model_name, 0.10)

    # Base position from allocation
    base_position = unallocated_capital * allocation_pct

    # Scale by confidence (0.75-0.95 typically)
    scaled_position = base_position * confidence

    # Apply 30% cap
    max_position = unallocated_capital * 0.30
    final_position = min(scaled_position, max_position)

    # Calculate shares
    shares = int(final_position / entry_price) if entry_price > 0 else 0

    return shares, final_position


def get_today_signals():
    """Get today's live signals - checks DB first, then JSON files"""
    import json
    import glob

    # Try database first
    try:
        with sqlite3.connect(LIVE_DB_PATH) as conn:
            df = pd.read_sql_query("""
                SELECT * FROM daily_signals
                WHERE signal_date = ?
                ORDER BY signal_type, symbol
            """, conn, params=(str(date.today()),))
        if len(df) > 0:
            return df
    except:
        pass

    # Fall back to JSON files
    try:
        # Look for most recent signal file
        signal_files = sorted(glob.glob('logs/signals/signals_*.json'), reverse=True)
        if signal_files:
            with open(signal_files[0], 'r') as f:
                data = json.load(f)

            # Load model allocations and calculate unallocated capital
            model_allocations = load_model_allocations()
            initial_capital = 50000
            # TODO: Subtract existing positions' value from initial_capital
            unallocated_capital = initial_capital  # Simplified for now

            # Convert to DataFrame format expected by dashboard
            signals = []
            for sig in data.get('entry_signals', []):
                entry_price = sig.get('entry_price', sig.get('price', 100))
                confidence = sig.get('confidence', 0.8)
                model_name = sig.get('model', 'Unknown')

                shares, position_value = calculate_position_size(
                    entry_price, confidence, model_name,
                    unallocated_capital, model_allocations
                )

                signals.append({
                    'id': len(signals) + 1,
                    'signal_date': data.get('summary', {}).get('date', str(date.today())),
                    'signal_type': 'ENTRY',
                    'symbol': sig.get('symbol'),
                    'entry_price': entry_price,
                    'stop_loss': sig.get('stop_loss'),
                    'shares': shares,
                    'position_value': position_value,
                    'confidence': confidence,
                    'reason': sig.get('method', 'UNKNOWN'),
                    'model': model_name,
                    'rs': sig.get('rs', 0),
                    'volume_ratio': sig.get('volume_ratio', 1.0),
                    'css': sig.get('css', 0),
                    'grade': sig.get('grade', 'D'),
                    'css_components': sig.get('css_components', {}),
                    'approved': 0,
                    'executed': 0
                })

            # Include summary info
            summary = data.get('summary', {})

            result_df = pd.DataFrame(signals) if signals else pd.DataFrame()

            # Store summary in session state for sidebar access
            if 'signal_summary' not in st.session_state:
                st.session_state['signal_summary'] = summary

            return result_df
    except Exception as e:
        pass

    return pd.DataFrame()

def get_open_positions():
    """Get current open positions"""
    try:
        with sqlite3.connect(LIVE_DB_PATH) as conn:
            df = pd.read_sql_query("""
                SELECT * FROM live_positions 
                WHERE status = 'OPEN'
                ORDER BY entry_date DESC
            """, conn)
        return df
    except:
        return pd.DataFrame()

def get_closed_trades_live():
    """Get all closed live trades"""
    try:
        with sqlite3.connect(LIVE_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='live_trades'
            """)
            if cursor.fetchone() is None:
                return pd.DataFrame()
            
            df = pd.read_sql_query("""
                SELECT * FROM live_trades 
                ORDER BY exit_date DESC
            """, conn)
        return df
    except:
        return pd.DataFrame()

def approve_signal(signal_id):
    """Mark signal as approved"""
    with sqlite3.connect(LIVE_DB_PATH) as conn:
        conn.execute("""
            UPDATE daily_signals 
            SET approved = 1
            WHERE id = ?
        """, (signal_id,))
        conn.commit()

def reject_signal(signal_id):
    """Mark signal as rejected"""
    with sqlite3.connect(LIVE_DB_PATH) as conn:
        conn.execute("""
            DELETE FROM daily_signals 
            WHERE id = ?
        """, (signal_id,))
        conn.commit()

# ============================================================================
# MODE SELECTOR
# ============================================================================

st.title("📊 MB Trading System")

# Mode selection with radio buttons
mode = st.radio(
    "Select Mode:",
    options=["🔴 Live Trading", "📈 Backtest Analysis"],
    horizontal=True,
    label_visibility="collapsed"
)

is_live_mode = (mode == "🔴 Live Trading")

# Apply theme based on mode
if is_live_mode:
    apply_live_theme()
    st.markdown('<div class="mode-indicator-live">⚠️ LIVE TRADING MODE - Real Money</div>', 
                unsafe_allow_html=True)
else:
    apply_backtest_theme()
    st.markdown('<div class="mode-indicator-backtest">ℹ️ BACKTEST ANALYSIS - Historical Simulation</div>', 
                unsafe_allow_html=True)

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S PT')}")
#st.markdown("---")

# ============================================================================
# DYNAMIC SIDEBAR
# ============================================================================

if is_live_mode:
    # LIVE MODE SIDEBAR
    st.sidebar.header("🔴 Live Trading")
    
    # Config selector for live trading
    st.sidebar.subheader("⚙️ Configuration")
    live_config = st.sidebar.selectbox(
        "Active Config",
        ["A", "B", "C"],
        index=0,
        help="A: 4 models | B: 8 models | C: Scanner + 8 models"
    )
    config_descriptions = {
        "A": "126 stocks, 4 original models",
        "B": "126 stocks, 8 models (4+4)",
        "C": "Scanner + 126, 8 models"
    }
    st.sidebar.caption(f"📌 {config_descriptions[live_config]}")
    
    st.sidebar.markdown("---")
    
    # Quick stats
    st.sidebar.subheader("📊 Quick Stats")
    
    positions = get_open_positions()
    signals = get_today_signals()
    trades_live = get_closed_trades_live()
    
    # Calculate metrics
    initial_capital = 50000  # Updated budget
    total_pnl = trades_live['pnl'].sum() if len(trades_live) > 0 else 0
    portfolio_value = initial_capital + total_pnl
    total_return_pct = (total_pnl / initial_capital) * 100
    
    total_cost = (positions['shares'] * positions['entry_price']).sum() if len(positions) > 0 else 0
    deployment_pct = (total_cost / initial_capital) * 100
    
    entry_signals = signals[signals['signal_type'] == 'ENTRY'] if len(signals) > 0 else pd.DataFrame()
    exit_signals = signals[signals['signal_type'] == 'EXIT'] if len(signals) > 0 else pd.DataFrame()

    st.sidebar.metric("Open Positions", len(positions))
    st.sidebar.metric("Today's Signals", f"{len(entry_signals)}E / {len(exit_signals)}X")
    st.sidebar.metric("Portfolio Value", f"${portfolio_value:,.0f}", f"{total_return_pct:+.1f}%")
    st.sidebar.metric("Capital Deployed", f"${total_cost:,.0f}", f"{deployment_pct:.0f}%")

    # Signal Grade Summary
    if len(entry_signals) > 0 and 'grade' in entry_signals.columns:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Signal Grades")
        grade_a_plus = len(entry_signals[entry_signals['grade'] == 'A+'])
        grade_a = len(entry_signals[entry_signals['grade'] == 'A'])
        grade_b = len(entry_signals[entry_signals['grade'] == 'B'])
        grade_cd = len(entry_signals[entry_signals['grade'].isin(['C', 'D'])])

        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.markdown(f"**A+:** {grade_a_plus}")
            st.markdown(f"**A:** {grade_a}")
        with col2:
            st.markdown(f"**B:** {grade_b}")
            st.markdown(f"**C/D:** {grade_cd}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Position Sizing")
    st.sidebar.markdown("**Method:** Model Allocation %")
    st.sidebar.markdown("**Scaling:** By Confidence")
    st.sidebar.markdown("**Max Single Trade:** 30%")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last signal: {date.today()}")

else:
    # BACKTEST MODE SIDEBAR
    st.sidebar.header("📈 Backtest Analysis")
    
    # Initialize database
    db = get_backtest_db()
    
    # ========== FILTERS ==========
    st.sidebar.subheader("🔍 Filters")
    
    # Run Type filter
    run_type_filter = st.sidebar.selectbox(
        "Run Type",
        ["All", "production", "development"],
        index=0,
        help="Filter by production or development runs"
    )
    
    # Config Type filter
    config_type_filter = st.sidebar.selectbox(
        "Config",
        ["All", "A", "B", "C"],
        index=0,
        help="A: 126 stocks, 4 models | B: 126 stocks, 8 models | C: Scanner + 8 models"
    )
    
    # Config descriptions
    config_desc = {
        "A": "126 stocks, 4 original models",
        "B": "126 stocks, 8 models (4+4)",
        "C": "Scanner + 126, 8 models"
    }
    if config_type_filter != "All":
        st.sidebar.caption(f"📌 {config_desc.get(config_type_filter, '')}")
    
    st.sidebar.markdown("---")
    
    # Get filtered runs
    rt_filter = None if run_type_filter == "All" else run_type_filter
    ct_filter = None if config_type_filter == "All" else config_type_filter
    all_runs = db.get_all_runs(run_type=rt_filter, config_type=ct_filter)
    
    if len(all_runs) == 0:
        st.sidebar.warning("No backtest runs found with selected filters")
        st.stop()
    
    # Run selector with run_type and config_type info
    def format_run_option(x):
        rt = x.get('run_type', 'dev')[:4] if pd.notna(x.get('run_type')) else 'dev'
        ct = x.get('config_type', '?') if pd.notna(x.get('config_type')) else '?'
        desc = x['description'][:15] if x['description'] else 'No desc'
        return f"#{x['id']} [{rt.upper()}/{ct}] {desc}... ({x['total_trades']} trades)"
    
    run_options = all_runs.apply(format_run_option, axis=1)
    selected_run = st.sidebar.selectbox("Select Run", run_options)
    run_id = int(selected_run.split('#')[1].split(' ')[0])
    
    # Load run details
    run_summary, trades_df = db.get_run_details(run_id)
    
    if len(trades_df) == 0:
        st.sidebar.error("No trades in this run")
        st.stop()
    
    # Summary metrics
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Summary")
    
    # Show run type and config
    rt = run_summary.get('run_type', 'unknown') if pd.notna(run_summary.get('run_type')) else 'unknown'
    ct = run_summary.get('config_type', '?') if pd.notna(run_summary.get('config_type')) else '?'
    st.sidebar.markdown(f"**Type:** {rt.title()} / Config {ct}")
    
    st.sidebar.metric("Total Return", f"{run_summary['total_return_pct']:.1f}%")
    st.sidebar.metric("CAGR", f"{run_summary['cagr']:.1f}%")
    st.sidebar.metric("Win Rate", f"{run_summary['win_rate']:.1f}%")
    st.sidebar.metric("Profit Factor", f"{run_summary['profit_factor']:.2f}")
    st.sidebar.metric("Total Trades", f"{run_summary['total_trades']}")
    
    st.sidebar.markdown("---")
    
    # Check for model data
    if 'model' in trades_df.columns:
        models = sorted(trades_df['model'].unique())
        st.sidebar.subheader("🎯 Models")
        for model in models:
            st.sidebar.markdown(f"✅ {model}")

# ============================================================================
# MAIN CONTENT - LIVE MODE
# ============================================================================

if is_live_mode:
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Today's Signals",
        "💼 Open Positions",
        "📈 Performance",
        "📅 Monthly Heatmap",
        "⚙️ Settings",
        "📈 Recent Price Action",
        "🔔 Stage 1→2 Breakouts"
    ])
    
    # ========================================================================
    # TAB 1: TODAY'S SIGNALS
    # ========================================================================
    with tab1:
        st.caption(f"Generated: {date.today().strftime('%Y-%m-%d')}")

        signals = get_today_signals()

        if len(signals) == 0:
            st.info("No signals today - HOLD current positions")
        else:
            # Entry signals
            entry_signals = signals[signals['signal_type'] == 'ENTRY'].copy()

            if len(entry_signals) > 0:
                # Grade filter
                col_header, col_filter = st.columns([3, 1])
                with col_header:
                    st.subheader("ENTRY SIGNALS")
                with col_filter:
                    grade_filter = st.selectbox(
                        "Grade Filter",
                        ["A+, A, B (Default)", "C, D only"],
                        index=0,
                        label_visibility="collapsed"
                    )

                # Apply grade filter
                if grade_filter == "A+, A, B (Default)":
                    filtered_signals = entry_signals[entry_signals['grade'].isin(['A+', 'A', 'B'])]
                else:
                    filtered_signals = entry_signals[entry_signals['grade'].isin(['C', 'D'])]

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Signals", len(entry_signals))
                with col2:
                    st.metric("Displayed", len(filtered_signals))
                with col3:
                    grade_a_plus = len(entry_signals[entry_signals['grade'] == 'A+'])
                    grade_a = len(entry_signals[entry_signals['grade'] == 'A'])
                    st.metric("A+ / A", f"{grade_a_plus} / {grade_a}")
                with col4:
                    grade_b = len(entry_signals[entry_signals['grade'] == 'B'])
                    grade_cd = len(entry_signals[entry_signals['grade'].isin(['C', 'D'])])
                    st.metric("B / C,D", f"{grade_b} / {grade_cd}")

                st.markdown("---")

                if len(filtered_signals) > 0:
                    # Prepare display dataframe
                    display_df = filtered_signals[[
                        'grade', 'css', 'symbol', 'model', 'entry_price', 'stop_loss',
                        'rs', 'volume_ratio', 'confidence', 'shares', 'position_value', 'reason'
                    ]].copy()

                    # Format columns
                    display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}")
                    display_df['stop_loss'] = display_df['stop_loss'].apply(lambda x: f"${x:.2f}")
                    display_df['rs'] = display_df['rs'].apply(lambda x: f"{x:.1f}")
                    display_df['volume_ratio'] = display_df['volume_ratio'].apply(lambda x: f"{x:.1f}x")
                    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.0%}")
                    display_df['position_value'] = display_df['position_value'].apply(lambda x: f"${x:,.0f}")
                    display_df['css'] = display_df['css'].apply(lambda x: f"{x:.1f}")

                    # Rename columns for display
                    display_df.columns = [
                        'Grade', 'CSS', 'Symbol', 'Model', 'Entry', 'Stop',
                        'RS', 'Vol', 'Conf', 'Shares', 'Value', 'Method'
                    ]

                    # Color-code by grade using custom styling
                    def highlight_grade(row):
                        grade = row['Grade']
                        if grade == 'A+':
                            return ['background-color: #c6efce; color: #000000'] * len(row)
                        elif grade == 'A':
                            return ['background-color: #d4edda; color: #000000'] * len(row)
                        elif grade == 'B':
                            return ['background-color: #fff3cd; color: #000000'] * len(row)
                        else:
                            return ['background-color: #f8d7da; color: #000000'] * len(row)

                    styled_df = display_df.style.apply(highlight_grade, axis=1)

                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        height=min(400, 35 * len(display_df) + 38),
                        hide_index=True
                    )

                    # Entry execution checklist
                    st.markdown("---")
                    with st.expander("Entry Execution Checklist"):
                        st.markdown("""
                        **Before executing:**
                        1. Verify Grade A+ or A signals first
                        2. Check available capital
                        3. Confirm position count < max positions
                        4. Set limit order at entry price + 1%
                        5. Place stop loss order immediately after fill
                        6. Export fill from broker and update portfolio
                        """)
                else:
                    st.info("No signals match the selected grade filter")

            # Exit signals
            exit_signals = signals[signals['signal_type'] == 'EXIT']

            if len(exit_signals) > 0:
                st.markdown("---")
                st.subheader("EXIT SIGNALS (Close Positions)")

                for _, signal in exit_signals.iterrows():
                    with st.container(border=True):
                        col1, col2, col3 = st.columns([2, 3, 1])

                        with col1:
                            st.markdown(f"### {signal['symbol']}")
                            st.caption(f"Reason: {signal['reason']}")

                        with col2:
                            st.markdown(f"**Exit Price:** ${signal['entry_price']:.2f}")
                            st.markdown(f"**Shares:** {signal['shares']}")

                        with col3:
                            if signal['approved']:
                                st.success("Approved")
                            elif signal['executed']:
                                st.info("Executed")
                            else:
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    if st.button("Approve", key=f"approve_exit_{signal['id']}"):
                                        approve_signal(signal['id'])
                                        st.rerun()
                                with col_b:
                                    if st.button("Reject", key=f"reject_exit_{signal['id']}"):
                                        reject_signal(signal['id'])
                                        st.rerun()

                # Exit execution checklist
                with st.expander("Exit Execution Checklist"):
                    st.markdown("""
                    **Before executing:**
                    1. Use market order at open OR limit at yesterday's close
                    2. Cancel stop loss order after fill
                    3. Export fill from broker and update portfolio
                    """)
    
    # ========================================================================
    # TAB 2: OPEN POSITIONS
    # ========================================================================
    with tab2:
        # st.header("💼 Open Positions")
        
        positions = get_open_positions()
        
        if len(positions) == 0:
            st.info("No open positions")
        else:
            # Calculate totals
            data_dir = "data"
            total_cost = (positions['shares'] * positions['entry_price']).sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Open Positions", len(positions))
            with col2:
                st.metric("Capital Deployed", f"${total_cost:,.0f}")
            with col3:
                deployment_pct = (total_cost / 50000) * 100
                st.metric("Deployment %", f"{deployment_pct:.1f}%")
            
            st.markdown("---")
            
            # Position details
            for _, pos in positions.iterrows():
                # Try to load current price
                try:
                    df = pd.read_csv(f"{data_dir}/{pos['symbol']}.csv", index_col=0, parse_dates=True)
                    current_price = df['close'].iloc[-1]
                    last_date = df.index[-1].strftime('%Y-%m-%d')
                except:
                    current_price = pos['entry_price']
                    last_date = "Unknown"
                
                # Calculate P&L
                unrealized_pnl = (current_price - pos['entry_price']) * pos['shares']
                unrealized_pnl_pct = (current_price / pos['entry_price'] - 1) * 100
                
                # Calculate days held
                entry_date = pd.to_datetime(pos['entry_date']).date()
                days_held = (date.today() - entry_date).days
                
                with st.container(border=True):
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                    
                    with col1:
                        st.markdown(f"### {pos['symbol']}")
                        st.caption(f"Entry: {entry_date}")
                    
                    with col2:
                        st.markdown(f"**Entry:** ${pos['entry_price']:.2f}")
                        st.markdown(f"**Current:** ${current_price:.2f}")
                        st.caption(f"As of {last_date}")
                    
                    with col3:
                        st.markdown(f"**Shares:** {pos['shares']}")
                        st.markdown(f"**Stop:** ${pos['stop_loss']:.2f}")
                        st.markdown(f"**Days Held:** {days_held}")
                    
                    with col4:
                        st.markdown(f"**P&L:** :{'green' if unrealized_pnl > 0 else 'red'}[${unrealized_pnl:,.0f}]")
                        st.markdown(f"**P&L %:** :{'green' if unrealized_pnl > 0 else 'red'}[{unrealized_pnl_pct:+.1f}%]")
    
    # ========================================================================
    # TAB 3: PERFORMANCE
    # ========================================================================
    with tab3:
        # st.header("📈 Performance")
        
        trades = get_closed_trades_live()
        positions = get_open_positions()
        
        # Calculate metrics
        initial_capital = 50000  # Updated budget

        if len(trades) > 0:
            total_pnl = trades['pnl'].sum()
            total_return_pct = (total_pnl / initial_capital) * 100
            winning_trades = len(trades[trades['pnl'] > 0])
            losing_trades = len(trades[trades['pnl'] <= 0])
            win_rate = (winning_trades / len(trades)) * 100
            
            # Profit factor
            wins = trades[trades['pnl'] > 0]['pnl'].sum()
            losses = abs(trades[trades['pnl'] < 0]['pnl'].sum())
            profit_factor = wins / losses if losses > 0 else 0
            
            avg_hold = trades['hold_days'].mean()
        else:
            total_pnl = 0
            total_return_pct = 0
            winning_trades = 0
            losing_trades = 0
            win_rate = 0
            profit_factor = 0
            avg_hold = 0
        
        portfolio_value = initial_capital + total_pnl
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Portfolio Value",
                f"${portfolio_value:,.0f}",
                f"${total_pnl:,.0f}"
            )
        
        with col2:
            st.metric(
                "Total Return",
                f"{total_return_pct:+.2f}%"
            )
        
        with col3:
            st.metric(
                "Win Rate",
                f"{win_rate:.1f}%",
                f"{winning_trades}W - {losing_trades}L"
            )
        
        with col4:
            st.metric(
                "Profit Factor",
                f"{profit_factor:.2f}"
            )
        
        st.markdown("---")
        
        # Equity curve
        if len(trades) > 0:
            st.subheader("💰 Equity Curve")
            
            trades_sorted = trades.sort_values('exit_date').copy()
            trades_sorted['exit_date'] = pd.to_datetime(trades_sorted['exit_date'])
            trades_sorted['cumulative_pnl'] = trades_sorted['pnl'].cumsum()
            trades_sorted['equity'] = initial_capital + trades_sorted['cumulative_pnl']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trades_sorted['exit_date'],
                y=trades_sorted['equity'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#dc2626', width=2),
                fill='tonexty',
                fillcolor='rgba(220, 38, 38, 0.1)'
            ))
            
            fig.add_hline(
                y=initial_capital,
                line_dash="dash",
                line_color="gray",
                annotation_text="Initial Capital"
            )
            
            fig.update_layout(
                title="Portfolio Equity Over Time",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade log
            st.markdown("---")
            st.subheader("📋 Trade Log")
            
            display_trades = trades[[
                'symbol', 'entry_date', 'exit_date', 'entry_price', 'exit_price',
                'shares', 'pnl', 'pnl_pct', 'hold_days', 'exit_reason'
            ]].copy()
            
            display_trades['pnl'] = display_trades['pnl'].apply(lambda x: f"${x:,.0f}")
            display_trades['pnl_pct'] = display_trades['pnl_pct'].apply(lambda x: f"{x:+.1f}%")
            
            st.dataframe(display_trades, use_container_width=True, height=300)
        else:
            st.info("No closed trades yet")
    
    # ========================================================================
    # TAB 4: MONTHLY HEATMAP (NEW!)
    # ========================================================================
    with tab4:
        # st.header("📅 Monthly Performance Heatmap")
        st.markdown("**Green** = profitable months, **Red** = losing months. Shows monthly return % and average capital deployed.")
        
        trades = get_closed_trades_live()
        
        if len(trades) == 0:
            st.info("No closed trades yet - Heatmap will populate as you trade")
        else:
            # Calculate monthly capital allocation
            def calculate_monthly_capital_allocation(trades_df, initial_capital):
                """Calculate average daily capital allocated for each month"""
                trades_calc = trades_df.copy()
                trades_calc['entry_date'] = pd.to_datetime(trades_calc['entry_date'])
                trades_calc['exit_date'] = pd.to_datetime(trades_calc['exit_date'])
                trades_calc['position_size'] = trades_calc['shares'] * trades_calc['entry_price']
                
                trades_sorted = trades_calc.sort_values('exit_date')
                trades_sorted['cumulative_pnl'] = trades_sorted['pnl'].cumsum()
                
                min_date = trades_calc['entry_date'].min()
                max_date = trades_calc['exit_date'].max()
                all_dates = pd.date_range(start=min_date, end=max_date, freq='B')
                
                monthly_allocation = {}
                
                for date in all_dates:
                    year = date.year
                    month = date.month
                    
                    completed_trades = trades_sorted[trades_sorted['exit_date'] <= date]
                    cumulative_pnl = completed_trades['pnl'].sum() if len(completed_trades) > 0 else 0
                    portfolio_value = initial_capital + cumulative_pnl
                    
                    active_trades = trades_calc[
                        (trades_calc['entry_date'] <= date) & 
                        (trades_calc['exit_date'] >= date)
                    ]
                    
                    daily_deployed_capital = active_trades['position_size'].sum()
                    
                    if portfolio_value > 0:
                        daily_allocation_pct = (daily_deployed_capital / portfolio_value) * 100
                    else:
                        daily_allocation_pct = 0
                    
                    key = (year, month)
                    if key not in monthly_allocation:
                        monthly_allocation[key] = []
                    monthly_allocation[key].append(daily_allocation_pct)
                
                monthly_avg_allocation = {}
                for key, daily_pcts in monthly_allocation.items():
                    avg_allocation_pct = sum(daily_pcts) / len(daily_pcts)
                    monthly_avg_allocation[key] = avg_allocation_pct
                
                return monthly_avg_allocation
            
            # Calculate monthly allocation
            monthly_allocation = calculate_monthly_capital_allocation(trades, 50000)
            
            # Prepare monthly returns
            trades_monthly = trades.copy()
            trades_monthly['exit_date'] = pd.to_datetime(trades_monthly['exit_date'])
            trades_monthly['year'] = trades_monthly['exit_date'].dt.year
            trades_monthly['month'] = trades_monthly['exit_date'].dt.month
            
            monthly_pnl = trades_monthly.groupby(['year', 'month'])['pnl'].sum().reset_index()
            monthly_pnl['return_pct'] = (monthly_pnl['pnl'] / 50000) * 100
            monthly_pnl['capital_alloc_pct'] = monthly_pnl.apply(
                lambda row: monthly_allocation.get((row['year'], row['month']), 0),
                axis=1
            )
            
            # Pivot for heatmap
            heatmap_data = monthly_pnl.pivot(index='year', columns='month', values='return_pct')
            capital_alloc_data = monthly_pnl.pivot(index='year', columns='month', values='capital_alloc_pct')
            
            heatmap_data = heatmap_data.sort_index(ascending=False)
            capital_alloc_data = capital_alloc_data.sort_index(ascending=False)
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            heatmap_data.columns = [month_names[int(m)-1] for m in heatmap_data.columns]
            capital_alloc_data.columns = [month_names[int(m)-1] for m in capital_alloc_data.columns]
            
            # Prepare text display
            text_display = []
            for year_idx in range(len(heatmap_data)):
                row_text = []
                for month_col in heatmap_data.columns:
                    return_val = heatmap_data.iloc[year_idx][month_col]
                    capital_val = capital_alloc_data.iloc[year_idx][month_col]
                    
                    if pd.isna(return_val):
                        row_text.append('')
                    else:
                        row_text.append(f'{return_val:.1f}%<br>{capital_val:.0f}% cap')
                text_display.append(row_text)
            
            # Color scale
            custom_colorscale = [
                [0.0, '#8B0000'],
                [0.2, '#B22222'],
                [0.35, '#DC143C'],
                [0.45, '#E67E7E'],
                [0.5, '#F5F5F5'],
                [0.55, '#A8D5A8'],
                [0.7, '#4CAF50'],
                [0.85, '#2E7D32'],
                [1.0, '#006400']
            ]
            
            z_values = heatmap_data.fillna(0).values
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=z_values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale=custom_colorscale,
                zmid=0,
                text=text_display,
                texttemplate='%{text}',
                textfont={"size": 13, "color": "black"},
                colorbar=dict(title="Return %"),
                hoverongaps=False,
                xgap=3,
                ygap=3
            ))
            
            fig_heatmap.update_layout(
                height=max(400, len(heatmap_data) * 85),
                xaxis=dict(title="Month", side='top'),
                yaxis=dict(title="Year")
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Monthly statistics
            st.markdown("---")
            st.markdown("**Monthly Statistics:**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                best_month = monthly_pnl['return_pct'].max() if len(monthly_pnl) > 0 else 0
                st.metric("Best Month", f"{best_month:.1f}%")
            
            with col2:
                worst_month = monthly_pnl['return_pct'].min() if len(monthly_pnl) > 0 else 0
                st.metric("Worst Month", f"{worst_month:.1f}%")
            
            with col3:
                avg_return = monthly_pnl['return_pct'].mean() if len(monthly_pnl) > 0 else 0
                st.metric("Avg Monthly", f"{avg_return:.1f}%")
            
            with col4:
                avg_capital = monthly_pnl['capital_alloc_pct'].mean() if len(monthly_pnl) > 0 else 0
                st.metric("Avg Capital", f"{avg_capital:.0f}%")
    
    # ========================================================================
    # TAB 5: SETTINGS
    # ========================================================================
    with tab5:
        st.header("Settings & Configuration")

        # Create sub-tabs for different settings sections
        settings_tab1, settings_tab2, settings_tab3, settings_tab4 = st.tabs([
            "Capital & Risk", "Signal Scoring (CSS)", "Confidence Levels", "System"
        ])

        # ---- Capital & Risk Settings ----
        with settings_tab1:
            st.subheader("Capital Allocation")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Capital Settings:**")
                st.markdown("- Total Capital: $50,000")
                st.markdown("- Max Deployment: 90% ($45,000)")
                st.markdown("- Cash Reserve: 10% ($5,000)")

                st.markdown("**Position Sizing:**")
                st.markdown("- Method: Model Allocation %")
                st.markdown("- Scaling: Confidence Score")
                st.markdown("- Max Single Trade: 30% of capital")

            with col2:
                st.markdown("**Risk Limits:**")
                st.markdown("- Daily Loss Limit: -$1,000 (-2%)")
                st.markdown("- Weekly Loss Limit: -$2,500 (-5%)")
                st.markdown("- Max Drawdown: -$7,500 (-15%)")

                st.markdown("**Entry Criteria:**")
                st.markdown("- Min RS Rating: 80")
                st.markdown("- Min Volume: 500K shares/day")

        # ---- CSS (Composite Signal Strength) Settings ----
        with settings_tab2:
            st.subheader("Composite Signal Strength (CSS)")

            st.markdown("""
            **CSS Formula:**
            ```
            CSS = (0.35 × RS_Score) + (0.25 × Momentum_Score) + (0.20 × Volume_Score)
                + (0.10 × Tightness_Score) + (0.10 × Model_Quality_Score)
            ```
            """)

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Component Weights:**")
                st.markdown("- RS Score: 35% (Relative Strength)")
                st.markdown("- Momentum Score: 25% (Multi-timeframe)")
                st.markdown("- Volume Score: 20% (Institutional interest)")
                st.markdown("- Tightness Score: 10% (Distance from MA)")
                st.markdown("- Model Quality: 10% (Historical performance)")

            with col2:
                st.markdown("**Grade Ranges:**")
                st.markdown("- **A+ (85-100):** Priority buy - highest conviction")
                st.markdown("- **A (75-84):** Strong candidate - execute with confidence")
                st.markdown("- **B (65-74):** Good candidate - solid setup")
                st.markdown("- **C (55-64):** Average - consider if capital permits")
                st.markdown("- **D (<55):** Low priority - typically skip")

            st.markdown("---")

            st.markdown("**Model Quality Score (MQS):**")
            st.markdown("""
            MQS measures historical model effectiveness based on backtest performance.
            Formula: `Avg_Return × Win_Rate_Factor × Sample_Size_Factor`

            | Model | MQS Score | Based On |
            |-------|-----------|----------|
            | Weinstein_Core | 100 | 21.81% avg return, 4.51 PF |
            | 52W_High_Momentum | 62 | 12.25% avg return, 2.43 PF |
            | Consolidation_Breakout | 35 | 7.54% avg return, 2.47 PF |

            *MQS blends backtest data with live trading results over time.*
            """)

        # ---- Confidence Levels ----
        with settings_tab3:
            st.subheader("Confidence Score Definitions")

            st.markdown("""
            Confidence scores indicate the model's conviction level for each signal.
            Higher confidence = larger position size (scaled).
            """)

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Exceptional (90-95%):**")
                st.markdown("""
                - All entry criteria strongly met
                - Multi-timeframe momentum alignment
                - Strong market context (SPY in uptrend)
                - Volume surge confirming breakout
                """)

                st.markdown("**High (80-89%):**")
                st.markdown("""
                - All entry criteria met
                - Good momentum score
                - Positive relative strength
                - Adequate volume confirmation
                """)

            with col2:
                st.markdown("**Standard (75-79%):**")
                st.markdown("""
                - Meets minimum entry criteria
                - Basic momentum requirements
                - Above-average relative strength
                """)

                st.markdown("**Reduced (<75%):**")
                st.markdown("""
                - Borderline setup
                - One or more weak criteria
                - Shown for reference only
                - Consider skipping
                """)

            st.markdown("---")

            st.markdown("**Position Sizing by Confidence:**")
            st.markdown("""
            ```
            Position Size = Unallocated_Capital × Model_Allocation% × Confidence
            Max Position = 30% of Unallocated Capital (hard cap)
            ```

            Example: $50K capital, 20% model allocation, 85% confidence
            - Base: $50,000 × 0.20 = $10,000
            - Scaled: $10,000 × 0.85 = $8,500
            - If < 30% cap ($15,000): Use $8,500
            """)

        # ---- System Settings ----
        with settings_tab4:
            st.subheader("System Configuration")

            st.markdown("**Database Locations:**")
            st.code("database/live_trading.db - Live trading data")
            st.code("database/weinstein.db - Backtest data")

            st.markdown("**Data Directories:**")
            st.code("data/ - Stock price data (CSV)")
            st.code("logs/signals/ - Daily signal JSON files")
            st.code("config/ - Configuration files")

            st.markdown("**Key Config Files:**")
            st.code("config/models_config.yaml - Model settings & allocations")
            st.code("config/model_quality.yaml - MQS scores & CSS settings")

            st.markdown("---")

            st.subheader("Refresh Dashboard")
            if st.button("Refresh Data"):
                st.rerun()
    # ========================================================================
    # TAB 5: RECENT PRICE ACTION
    # ========================================================================
    with tab6:
        col_header, col_refresh = st.columns([5, 1])
        with col_header:
            st.caption("Price Action - Last 5 Trading Days")
        with col_refresh:
            if st.button("🔄 Refresh Data", help="Download latest data from Yahoo Finance"):
                with st.spinner("Downloading latest data..."):
                    import sys
                    import os
                    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from scripts.download_data import download_historical_data
                    
                    WATCHLIST_TICKERS = [
                        'MSFT', 'COST', 'NFLX', 'NVDA', 'AVGO', 'AMD', 'UNH', 'META', 
                        'GOOG', 'NVO', 'PLTR', 'HOOD', 'AMZN', 'TSLA', 'RKLB', 'CRWD', 'CVX'
                    ]
                    
                    try:
                        download_historical_data(WATCHLIST_TICKERS)
                        st.success("✅ Data refreshed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error refreshing data: {e}")
        
        st.markdown("---")
        
        WATCHLIST_TICKERS = [
            'MSFT', 'COST', 'NFLX', 'NVDA', 'AVGO', 'AMD', 'UNH', 'META', 
            'GOOG', 'NVO', 'PLTR', 'HOOD', 'AMZN', 'TSLA', 'RKLB', 'CRWD', 'CVX'
        ]
        
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        
        performance_data = []
        ticker_data_cache = {}
        ticker_full_data_cache = {}
        
        for ticker in WATCHLIST_TICKERS:
            try:
                ticker_file = os.path.join(data_dir, f"{ticker}.csv")
                if not os.path.exists(ticker_file):
                    continue
                
                df = pd.read_csv(ticker_file, index_col=0, parse_dates=True)
                df_recent = df.tail(21).copy()
                
                if len(df_recent) == 0:
                    continue
                
                ticker_data_cache[ticker] = df_recent
                ticker_full_data_cache[ticker] = df
                
                first_close = df_recent['close'].iloc[0]
                last_close = df_recent['close'].iloc[-1]
                change_pct = ((last_close - first_close) / first_close) * 100
                
                df_52w = df.tail(252)
                week_52_high = df_52w['high'].max()
                week_52_low = df_52w['low'].min()
                
                performance_data.append({
                    'Ticker': ticker,
                    'Last Price': f"${last_close:.2f}",
                    '21-Day Change (%)': f"{change_pct:+.2f}%",
                    '52W High': f"${week_52_high:.2f}",
                    '52W Low': f"${week_52_low:.2f}",
                    'change_pct_raw': change_pct
                })
            except Exception as e:
                continue
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            perf_df = perf_df.sort_values('change_pct_raw', ascending=True)
            
            sorted_tickers = perf_df['Ticker'].tolist()
            
            perf_df_display = perf_df[['Ticker', 'Last Price', '21-Day Change (%)', '52W High', '52W Low']]
            
            st.markdown("**21-Day Performance Summary**")
            st.dataframe(perf_df_display, use_container_width=True, hide_index=True)
            st.markdown("---")
        else:
            sorted_tickers = WATCHLIST_TICKERS
        
        # Create grid layout (2 columns)
        cols_per_row = 2
        
        for i in range(0, len(sorted_tickers), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, col in enumerate(cols):
                ticker_idx = i + j
                if ticker_idx >= len(sorted_tickers):
                    break
                    
                ticker = sorted_tickers[ticker_idx]
                
                with col:
                    with st.container(border=True):
                        try:
                            if ticker not in ticker_data_cache:
                                st.warning(f"⚠️ {ticker}: No data")
                                continue
                            
                            df_recent = ticker_data_cache[ticker]
                            
                            first_close = df_recent['close'].iloc[0]
                            last_close = df_recent['close'].iloc[-1]
                            change_pct = ((last_close - first_close) / first_close) * 100
                            
                            color = "green" if change_pct >= 0 else "red"
                            
                            date_labels = [d.strftime('%m/%d') for d in df_recent.index]
                            x_indices = list(range(len(df_recent)))
                            
                            # Candlestick chart
                            fig = go.Figure()
                            
                            fig.add_trace(go.Candlestick(
                                x=x_indices,
                                open=df_recent['open'],
                                high=df_recent['high'],
                                low=df_recent['low'],
                                close=df_recent['close'],
                                name=ticker,
                                increasing_line_color='green',
                                decreasing_line_color='red',
                                customdata=date_labels,
                                hovertext=[f'<b>{date}</b><br>O: {o:.2f}<br>H: {h:.2f}<br>L: {l:.2f}<br>C: {c:.2f}' 
                                for date, o, h, l, c in zip(date_labels, df_recent['open'], df_recent['high'], df_recent['low'], df_recent['close'])],
                                hoverinfo='text'
                            ))
                            
                            if 'ema_21' in df_recent.columns:
                                fig.add_trace(go.Scatter(
                                    x=x_indices,
                                    y=df_recent['ema_21'],
                                    mode='lines',
                                    name='21-day EMA',
                                    line=dict(color='blue', width=1.5),
                                    showlegend=True
                                ))
                                
                            if 'sma_200' in df_recent.columns:
                                fig.add_trace(go.Scatter(
                                    x=x_indices,
                                    y=df_recent['sma_200'],
                                    mode='lines',
                                    name='200-day SMA',
                                    line=dict(color='orange', width=1.5),
                                    showlegend=True
                                ))
                            
                            fig.update_layout(
                                title=dict(
                                    text=f"<b>{ticker}</b> ({change_pct:+.1f}%)",
                                    font=dict(size=14, color=color)
                                ),
                                xaxis=dict(
                                    title="",
                                    showgrid=True,
                                    showticklabels=False,
                                    type='linear',
                                    range=[-0.5, len(df_recent) - 0.5]
                                ),
                                yaxis=dict(
                                    title="",
                                    showgrid=True
                                ),
                                height=250,
                                margin=dict(l=0, r=10, t=40, b=0),
                                showlegend=False,
                                hovermode='x unified',
                                xaxis_rangeslider_visible=False
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Volume chart
                            fig_vol = go.Figure()
                            
                            colors = ['green' if df_recent['close'].iloc[i] >= df_recent['open'].iloc[i] 
                                     else 'red' for i in range(len(df_recent))]
                            
                            fig_vol.add_trace(go.Bar(
                                x=x_indices,
                                y=df_recent['volume'],
                                marker_color=colors,
                                name='Volume',
                                showlegend=False,
                                width=0.8
                            ))
                            
                            fig_vol.update_layout(
                                title=dict(text="Volume", font=dict(size=10)),
                                xaxis=dict(
                                    title="",
                                    showgrid=False,
                                    tickangle=-45,
                                    type='linear',
                                    tickmode='array',
                                    tickvals=x_indices,
                                    ticktext=date_labels,
                                    tickfont=dict(size=9),
                                    range=[-1, len(df_recent) - 0.5]
                                ),
                                yaxis=dict(
                                    title="",
                                    showgrid=False,
                                    showticklabels=False
                                ),
                                height=120,
                                margin=dict(l=10, r=10, t=15, b=30),
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig_vol, use_container_width=True)
                            
                            last_date = df_recent.index[-1].strftime('%Y-%m-%d')
                            st.caption(f"Last: ${last_close:.2f} ({last_date})")
                            
                        except Exception as e:
                            st.error(f"❌ {ticker}: {str(e)[:30]}...")

    # ========================================================================
    # TAB 7: STAGE 1→2 BREAKOUT ALERTS
    # ========================================================================
    with tab7:
        st.caption("Weinstein Stage 1→2 Breakout Detection")

        col_header, col_actions = st.columns([3, 2])
        with col_header:
            st.markdown("### Stage 1→2 Breakout Monitor")
        with col_actions:
            col_scan, col_refresh = st.columns(2)
            with col_scan:
                run_scan = st.button("🔍 Run Scan", help="Scan watchlist for Stage 1→2 setups")
            with col_refresh:
                refresh_alerts = st.button("🔄 Refresh", help="Reload alerts from database")

        st.markdown("---")

        # Info box explaining the methodology
        with st.expander("ℹ️ About Stage 1→2 Breakout Detection", expanded=False):
            st.markdown("""
            **Stan Weinstein's Stage Analysis:**

            - **Stage 1 (Basing):** Stock consolidates after decline, building a base
            - **Stage 2 (Advancing):** Breakout above resistance with volume confirmation

            **Our Criteria:**
            - ✅ Medium/Long base (6-12+ months)
            - ✅ Volume surge 2x+ average on breakout
            - ✅ Price above resistance (Stage 1 high)
            - ✅ Improving Relative Strength (RS line making higher highs)
            - ✅ Profitable or near-profitable company

            **The longer the base, the bigger the breakout!**
            """)

        # Load alerts from database
        def load_stage1_to2_alerts():
            """Load Stage 1→2 alerts from database"""
            try:
                conn = sqlite3.connect(LIVE_DB_PATH)
                df = pd.read_sql_query('''
                    SELECT * FROM stage1_to2_alerts
                    WHERE status IN ('WATCHING', 'BREAKOUT_PENDING', 'CONFIRMED')
                    ORDER BY
                        CASE status
                            WHEN 'CONFIRMED' THEN 1
                            WHEN 'BREAKOUT_PENDING' THEN 2
                            ELSE 3
                        END,
                        confidence DESC
                ''', conn)
                conn.close()
                return df
            except Exception as e:
                return pd.DataFrame()

        # Run scan if button pressed
        if run_scan:
            with st.spinner("Scanning for Stage 1→2 breakout candidates..."):
                try:
                    import subprocess
                    result = subprocess.run(
                        ['python3', 'scripts/stage1_to2_breakout_monitor.py', '--scan', '--save'],
                        capture_output=True,
                        text=True,
                        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    )
                    if result.returncode == 0:
                        st.success("✅ Scan completed! Results saved to database.")
                        st.code(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
                    else:
                        st.error(f"Scan failed: {result.stderr}")
                except Exception as e:
                    st.error(f"Error running scan: {e}")

        # Load and display alerts
        alerts_df = load_stage1_to2_alerts()

        if len(alerts_df) == 0:
            st.info("No active Stage 1→2 alerts. Click 'Run Scan' to search for candidates.")

            # Show default watchlist
            st.markdown("### 📋 Current Watchlist")
            default_watchlist = [
                # Drone ecosystem
                'FEIM', 'COHU', 'TTMI', 'DCO', 'MTSI',
                # AI Energy / Nuclear
                'LEU', 'MOD', 'POWL', 'BWXT', 'UEC',
                'DNN', 'AMBA', 'ACLS', 'UUUU', 'NVT',
                # Natural Gas
                'TLN', 'EQT', 'AR', 'KMI', 'KGS'
            ]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Drones:**")
                st.write(", ".join(['FEIM', 'COHU', 'TTMI', 'DCO', 'MTSI']))
            with col2:
                st.markdown("**AI Energy/Nuclear:**")
                st.write(", ".join(['LEU', 'MOD', 'POWL', 'BWXT', 'UEC', 'DNN', 'AMBA', 'ACLS', 'UUUU']))
            with col3:
                st.markdown("**Natural Gas:**")
                st.write(", ".join(['TLN', 'EQT', 'AR', 'KMI', 'KGS']))
        else:
            # Summary metrics
            confirmed = len(alerts_df[alerts_df['status'] == 'CONFIRMED'])
            pending = len(alerts_df[alerts_df['status'] == 'BREAKOUT_PENDING'])
            watching = len(alerts_df[alerts_df['status'] == 'WATCHING'])

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🚨 Confirmed", confirmed)
            with col2:
                st.metric("⏳ Pending", pending)
            with col3:
                st.metric("👀 Watching", watching)
            with col4:
                st.metric("Total Alerts", len(alerts_df))

            st.markdown("---")

            # Filter by status
            status_filter = st.selectbox(
                "Filter by Status",
                ["All", "CONFIRMED", "BREAKOUT_PENDING", "WATCHING"],
                index=0
            )

            if status_filter != "All":
                display_df = alerts_df[alerts_df['status'] == status_filter].copy()
            else:
                display_df = alerts_df.copy()

            if len(display_df) > 0:
                # Format for display
                display_cols = ['symbol', 'status', 'entry_price', 'stop_loss', 'stage1_duration_days',
                               'volume_ratio', 'rs_rating', 'rs_improving', 'confidence', 'updated_at']

                available_cols = [c for c in display_cols if c in display_df.columns]
                display_df = display_df[available_cols].copy()

                # Format columns
                if 'entry_price' in display_df.columns:
                    display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "-")
                if 'stop_loss' in display_df.columns:
                    display_df['stop_loss'] = display_df['stop_loss'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "-")
                if 'volume_ratio' in display_df.columns:
                    display_df['volume_ratio'] = display_df['volume_ratio'].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else "-")
                if 'rs_rating' in display_df.columns:
                    display_df['rs_rating'] = display_df['rs_rating'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
                if 'confidence' in display_df.columns:
                    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "-")
                if 'rs_improving' in display_df.columns:
                    display_df['rs_improving'] = display_df['rs_improving'].apply(lambda x: "✓" if x else "✗")

                # Rename columns
                display_df.columns = ['Symbol', 'Status', 'Entry', 'Stop', 'Base Days',
                                     'Volume', 'RS', 'RS↑', 'Conf', 'Updated'][:len(available_cols)]

                # Style based on status
                def highlight_status(row):
                    if row.get('Status') == 'CONFIRMED':
                        return ['background-color: #d4edda'] * len(row)
                    elif row.get('Status') == 'BREAKOUT_PENDING':
                        return ['background-color: #fff3cd'] * len(row)
                    else:
                        return [''] * len(row)

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

                # Show details for confirmed breakouts
                confirmed_alerts = alerts_df[alerts_df['status'] == 'CONFIRMED']
                if len(confirmed_alerts) > 0:
                    st.markdown("---")
                    st.markdown("### 🚨 Confirmed Breakout Details")

                    for _, alert in confirmed_alerts.iterrows():
                        with st.container(border=True):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"**{alert['symbol']}**")
                                st.write(f"Entry: ${alert['entry_price']:.2f}")
                                st.write(f"Stop: ${alert['stop_loss']:.2f}")
                            with col2:
                                risk_pct = ((alert['entry_price'] - alert['stop_loss']) / alert['entry_price']) * 100
                                st.write(f"Risk: {risk_pct:.1f}%")
                                st.write(f"Base: {alert['stage1_duration_days']} days")
                            with col3:
                                st.write(f"Volume: {alert['volume_ratio']:.1f}x")
                                st.write(f"RS: {alert['rs_rating']:.1f}")
                                st.write(f"Confidence: {alert['confidence']:.0%}")
            else:
                st.info("No alerts match the selected filter.")

        # Manual watchlist management
        st.markdown("---")
        with st.expander("📝 Manage Watchlist", expanded=False):
            st.markdown("Add symbols to scan (one per line):")
            custom_symbols = st.text_area(
                "Custom Symbols",
                value="FEIM\nCOHU\nDNN\nAMBA\nACLS\nUUUU",
                height=150,
                label_visibility="collapsed"
            )

            if st.button("Save Watchlist"):
                # TODO: Save to config
                st.success("Watchlist updated! Run scan to check these symbols.")

# ============================================================================
# MAIN CONTENT - BACKTEST MODE
# ============================================================================

else:  # Backtest mode
    
    # Note: run_summary and trades_df are already loaded in sidebar
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Portfolio Performance",
        "📋 View Executed Trades",
        "🏢 My Stocks Universe",
        "📅 Monthly Returns Heatmap"
    ])
    
    # ========================================================================
    # TAB 1: PORTFOLIO PERFORMANCE
    # ========================================================================
    with tab1:
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Final Value",
                f"${run_summary['final_value']:,.0f}",
                f"${run_summary['total_pnl']:,.0f}"
            )

        with col2:
            st.metric(
                "Winning Trades",
                run_summary['winning_trades'],
                f"{run_summary['win_rate']:.1f}% win rate"
            )

        with col3:
            st.metric(
                "Losing Trades",
                run_summary['losing_trades']
            )

        with col4:
            st.metric(
                "Avg Hold Time",
                f"{run_summary['avg_hold_days']:.0f} days"
            )

        st.markdown("---")

        # Continue with model performance cards from app.py...
        # (I'll include the rest of the backtest tabs in the next section)

        # MODEL PERFORMANCE CARDS
        st.subheader("🎯 Performance by Model")
        
        if 'model' in trades_df.columns:
            models = sorted(trades_df['model'].unique())
            model_stats = {}
            
            for model_name in models:
                model_trades = trades_df[trades_df['model'] == model_name]
                
                if len(model_trades) > 0:
                    total_pnl = model_trades['pnl'].sum()
                    total_trades = len(model_trades)
                    winning_trades = len(model_trades[model_trades['pnl'] > 0])
                    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                    avg_hold_days = model_trades['hold_days'].mean()
                    avg_return_pct = model_trades['pnl_pct'].mean()
                    
                    wins = model_trades[model_trades['pnl'] > 0]['pnl']
                    losses = model_trades[model_trades['pnl'] <= 0]['pnl']
                    total_wins = wins.sum() if len(wins) > 0 else 0
                    total_losses = abs(losses.sum()) if len(losses) > 0 else 0
                    profit_factor = (total_wins / total_losses) if total_losses > 0 else 0
                    
                    model_stats[model_name] = {
                        'total_pnl': total_pnl,
                        'total_trades': total_trades,
                        'win_rate': win_rate,
                        'avg_hold_days': avg_hold_days,
                        'avg_return_pct': avg_return_pct,
                        'profit_factor': profit_factor
                    }
            
            num_models = len(model_stats)
            if num_models > 0:
                cols_per_row = min(3, num_models)
                
                model_display_names = {
                    'Weinstein_Core': '🎯 Weinstein Core',
                    'RSI_Mean_Reversion': '📊 RSI Mean Reversion',
                    '52W_High_Momentum': '🚀 52-Week High',
                    'Consolidation_Breakout': '📈 Consolidation Breakout',
                    'UNKNOWN': '❓ Unknown Model'
                }
                
                for i in range(0, num_models, cols_per_row):
                    cols = st.columns(cols_per_row)
                    
                    for j, col in enumerate(cols):
                        model_idx = i + j
                        if model_idx >= num_models:
                            break
                        
                        model_name = list(model_stats.keys())[model_idx]
                        stats = model_stats[model_name]
                        
                        with col:
                            with st.container(border=True):
                                display_name = model_display_names.get(model_name, model_name)
                                st.markdown(f"### {display_name}")
                                
                                m1, m2 = st.columns(2)
                                
                                with m1:
                                    st.metric("Total P&L", f"${stats['total_pnl']:,.0f}")
                                    st.metric("Trades", f"{stats['total_trades']}")
                                with m2:
                                    st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
                                    st.metric("Avg Hold", f"{stats['avg_hold_days']:.0f}d")
                                
                                st.markdown("---")
                                st.caption(f"Profit Factor: {stats['profit_factor']:.2f}")
                                st.caption(f"Avg Return: {stats['avg_return_pct']:.1f}%")
        
        st.markdown("---")
        
        # Equity Curve
        st.subheader("💰 Equity Curve")
        
        period_options = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "ALL"]
        selected_period = st.radio(
            "Select Time Period:",
            options=period_options,
            index=len(period_options) - 1,
            horizontal=True,
            key="equity_period"
        )
        
        trades_sorted_all = trades_df.sort_values('exit_date').copy()
        trades_sorted_all['exit_date_dt'] = pd.to_datetime(trades_sorted_all['exit_date'], utc=True).dt.tz_localize(None)
        trades_sorted_all['cumulative_pnl'] = trades_sorted_all['pnl'].cumsum()
        trades_sorted_all['equity'] = run_summary['initial_capital'] + trades_sorted_all['cumulative_pnl']
        
        if selected_period != "ALL":
            latest_date = trades_sorted_all['exit_date_dt'].max()
            
            period_days = {
                "1M": 30, "3M": 90, "6M": 180, "1Y": 365,
                "2Y": 730, "3Y": 1095, "5Y": 1825
            }
            
            days_back = period_days[selected_period]
            cutoff_date = latest_date - pd.Timedelta(days=days_back)
            
            trades_sorted = trades_sorted_all[trades_sorted_all['exit_date_dt'] >= cutoff_date].copy()
            
            if len(trades_sorted) > 0:
                trades_before_cutoff = trades_sorted_all[trades_sorted_all['exit_date_dt'] < cutoff_date]
                if len(trades_before_cutoff) > 0:
                    starting_equity = trades_before_cutoff['equity'].iloc[-1]
                else:
                    starting_equity = run_summary['initial_capital']
                
                trades_sorted['period_cumulative_pnl'] = trades_sorted['pnl'].cumsum()
                trades_sorted['equity'] = starting_equity + trades_sorted['period_cumulative_pnl']
                has_period_data = True
            else:
                has_period_data = False
        else:
            trades_sorted = trades_sorted_all.copy()
            has_period_data = True

        if has_period_data and len(trades_sorted) > 0:
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(
                x=trades_sorted['exit_date_dt'],
                y=trades_sorted['equity'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#3b82f6', width=2),
                fill='tonexty',
                fillcolor='rgba(59, 130, 246, 0.1)'
            ))

            fig_equity.add_hline(
                y=run_summary['initial_capital'],
                line_dash="dash",
                line_color="gray",
                annotation_text="Initial Capital"
            )
            
            period_start_equity = trades_sorted['equity'].iloc[0] - trades_sorted['pnl'].iloc[0]
            period_end_equity = trades_sorted['equity'].iloc[-1]
            period_return = ((period_end_equity - period_start_equity) / period_start_equity) * 100
            period_pnl = trades_sorted['pnl'].sum()
            title_text = f"Portfolio Equity Over Time ({selected_period}) | Return: {period_return:+.1f}% | P&L: ${period_pnl:,.0f}"

            fig_equity.update_layout(
                title=title_text,
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                hovermode='x unified',
                height=400
            )

            st.plotly_chart(fig_equity, use_container_width=True)
        else:
            st.info(f"No trades found in the selected period ({selected_period}).")

        # Drawdown Analysis
        st.markdown("---")
        st.subheader("📉 Drawdown Analysis")
        
        trades_dd = trades_sorted_all.copy()
        trades_dd['running_max'] = trades_dd['equity'].cummax()
        trades_dd['drawdown_pct'] = ((trades_dd['equity'] - trades_dd['running_max']) 
                                      / trades_dd['running_max'] * 100)
        
        max_dd = trades_dd['drawdown_pct'].min()
        max_dd_date = trades_dd[trades_dd['drawdown_pct'] == max_dd]['exit_date_dt'].iloc[0]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_dd = go.Figure()
            
            fig_dd.add_trace(go.Scatter(
                x=trades_dd['exit_date_dt'],
                y=trades_dd['drawdown_pct'],
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.2)',
                line=dict(color='#3b82f6', width=2),
                name='Drawdown %'
            ))
            
            fig_dd.update_layout(
                title="Portfolio Drawdown Over Time",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                hovermode='x unified',
                height=350
            )
            
            st.plotly_chart(fig_dd, use_container_width=True)
        
        with col2:
            st.markdown("**Risk Metrics**")
            st.metric("Max Drawdown", f"{max_dd:.1f}%")
            st.metric("Max DD Date", max_dd_date.strftime('%Y-%m-%d'))
            
            max_dd_amount = abs(max_dd / 100 * run_summary['initial_capital'])
            recovery_factor = run_summary['total_pnl'] / max_dd_amount if max_dd_amount > 0 else 0
            st.metric("Recovery Factor", f"{recovery_factor:.2f}")
            
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
            avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else 1
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            st.metric("Win/Loss Ratio", f"{win_loss_ratio:.2f}")
    
    # ========================================================================
    # TAB 2: VIEW EXECUTED TRADES - UPDATED VERSION
    # ========================================================================
    # Replace your existing tab2 content (in the backtest mode section) with this:
    with tab2:
        st.subheader("📋 All Executed Trades")

        trades_df_dates = trades_df.copy()
        trades_df_dates['entry_date_dt'] = pd.to_datetime(trades_df_dates['entry_date'], utc=True).dt.tz_localize(None)
        trades_df_dates['exit_date_dt'] = pd.to_datetime(trades_df_dates['exit_date'], utc=True).dt.tz_localize(None)
        
        all_entry_years = sorted(trades_df_dates['entry_date_dt'].dt.year.unique())
        all_exit_years = sorted(trades_df_dates['exit_date_dt'].dt.year.unique())
        month_options = ["All", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        st.markdown("**Basic Filters**")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            filter_symbol = st.multiselect(
                "Filter by Symbol",
                options=sorted(trades_df['symbol'].unique()),
                default=[]
            )

        with col2:
            filter_result = st.selectbox(
                "Filter by Result",
                options=["All", "Winners", "Losers"]
            )

        with col3:
            # Get available models (handle missing column gracefully)
            if 'model' in trades_df.columns:
                model_options = sorted(trades_df['model'].dropna().unique())
            else:
                model_options = []
            filter_model = st.multiselect(
                "Filter by Model",
                options=model_options,
                default=[]
            )

        with col4:
            filter_method = st.multiselect(
                "Filter by Entry Method",
                options=sorted(trades_df['entry_method'].unique()),
                default=[]
            )

        st.markdown("**Date Filters**")
        date_col1, date_col2 = st.columns(2)
        
        with date_col1:
            st.markdown("*Entry Date*")
            entry_date_cols = st.columns(2)
            with entry_date_cols[0]:
                entry_year_filter = st.selectbox(
                    "Year",
                    options=["All"] + [str(y) for y in all_entry_years],
                    key="entry_year"
                )
            with entry_date_cols[1]:
                entry_month_filter = st.selectbox(
                    "Month",
                    options=month_options,
                    key="entry_month"
                )
        
        with date_col2:
            st.markdown("*Exit Date*")
            exit_date_cols = st.columns(2)
            with exit_date_cols[0]:
                exit_year_filter = st.selectbox(
                    "Year",
                    options=["All"] + [str(y) for y in all_exit_years],
                    key="exit_year"
                )
            with exit_date_cols[1]:
                exit_month_filter = st.selectbox(
                    "Month",
                    options=month_options,
                    key="exit_month"
                )

        st.markdown("---")

        # Apply filters
        filtered_df = trades_df_dates.copy()
        
        if entry_year_filter != "All":
            filtered_df = filtered_df[filtered_df['entry_date_dt'].dt.year == int(entry_year_filter)]
        if entry_month_filter != "All":
            month_num = month_options.index(entry_month_filter)
            filtered_df = filtered_df[filtered_df['entry_date_dt'].dt.month == month_num]
        
        if exit_year_filter != "All":
            filtered_df = filtered_df[filtered_df['exit_date_dt'].dt.year == int(exit_year_filter)]
        if exit_month_filter != "All":
            month_num = month_options.index(exit_month_filter)
            filtered_df = filtered_df[filtered_df['exit_date_dt'].dt.month == month_num]

        if filter_symbol:
            filtered_df = filtered_df[filtered_df['symbol'].isin(filter_symbol)]

        if filter_result == "Winners":
            filtered_df = filtered_df[filtered_df['pnl'] > 0]
        elif filter_result == "Losers":
            filtered_df = filtered_df[filtered_df['pnl'] < 0]

        if filter_method:
            filtered_df = filtered_df[filtered_df['entry_method'].isin(filter_method)]

        # Filter by model
        if filter_model:
            filtered_df = filtered_df[filtered_df['model'].isin(filter_model)]

        # Calculate position_size
        filtered_df = filtered_df.copy()
        if 'shares' in filtered_df.columns:
            filtered_df['position_size'] = filtered_df['shares'] * filtered_df['entry_price']
        else:
            filtered_df['position_size'] = 0

        # Display table - include model, shares, and position_size
        display_columns = ['symbol', 'model', 'entry_date', 'exit_date', 'entry_method',
                          'shares', 'position_size', 'entry_price', 'exit_price', 
                          'pnl', 'pnl_pct', 'hold_days', 'exit_reason']
        
        # Only include columns that exist
        available_cols = [col for col in display_columns if col in filtered_df.columns]
        display_df = filtered_df[available_cols].copy()

        try:
            display_df['entry_date'] = pd.to_datetime(display_df['entry_date'], utc=True).dt.tz_localize(None).dt.strftime('%Y-%m-%d')
        except:
            display_df['entry_date'] = display_df['entry_date'].apply(lambda x: str(x).split()[0] if pd.notna(x) else '')

        try:
            display_df['exit_date'] = pd.to_datetime(display_df['exit_date'], utc=True).dt.tz_localize(None).dt.strftime('%Y-%m-%d')
        except:
            display_df['exit_date'] = display_df['exit_date'].apply(lambda x: str(x).split()[0] if pd.notna(x) else '')

        display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:,.0f}")
        display_df['pnl_pct'] = display_df['pnl_pct'].apply(lambda x: f"{x:+.1f}%")
        
        # Format position_size
        if 'position_size' in display_df.columns:
            display_df['position_size'] = display_df['position_size'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "$0")
        
        # Format entry_price and exit_price for better readability
        if 'entry_price' in display_df.columns:
            display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "")
        if 'exit_price' in display_df.columns:
            display_df['exit_price'] = display_df['exit_price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "")

        st.dataframe(
            display_df,
            use_container_width=True,
            height=500
        )

        # Summary stats
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Filtered Trades", len(filtered_df))
        
        with col2:
            if len(filtered_df) > 0:
                filtered_pnl = filtered_df['pnl'].sum()
                st.metric("Total P&L", f"${filtered_pnl:,.0f}")
            else:
                st.metric("Total P&L", "$0")
        
        with col3:
            if len(filtered_df) > 0:
                filtered_win_rate = (len(filtered_df[filtered_df['pnl'] > 0]) / len(filtered_df)) * 100
                st.metric("Win Rate", f"{filtered_win_rate:.1f}%")
            else:
                st.metric("Win Rate", "N/A")
        
        with col4:
            if len(filtered_df) > 0:
                st.metric("Avg Hold", f"{filtered_df['hold_days'].mean():.0f} days")
            else:
                st.metric("Avg Hold", "N/A")

        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Trades as CSV",
            data=csv,
            file_name=f"backtest_run_{run_id}_trades.csv",
            mime="text/csv"
        ) 
    # ========================================================================
    # TAB 3: MY STOCKS UNIVERSE
    # ========================================================================
    with tab3:
        st.subheader("🏢 My Stocks Universe")
        
        universe_symbols = sorted(trades_df['symbol'].unique())

        trades_df['position_size'] = trades_df['shares'] * trades_df['entry_price']

        universe_stats = trades_df.groupby('symbol').agg({
            'pnl': ['sum', 'count'],
            'position_size': 'mean',
            'pnl_pct': 'mean',
            'hold_days': 'mean'
        }).round(2)
        
        universe_stats.columns = ['Total P&L', 'Trade Count', 'Avg Position Size', 'Avg Return %', 'Avg Hold Days']
        universe_stats = universe_stats.sort_values('Total P&L', ascending=False)
        
        win_rates = trades_df.groupby('symbol').apply(
            lambda x: (x['pnl'] > 0).sum() / len(x) * 100
        ).round(1)
        universe_stats['Win Rate %'] = win_rates
        
        st.markdown(f"**Total Symbols Traded:** {len(universe_symbols)}")
        st.markdown(f"**Total Trades Across Universe:** {len(trades_df)}")
        
        st.markdown("---")
        
        st.dataframe(
            universe_stats,
            use_container_width=True,
            height=500
        )
        
        st.markdown("---")
        st.subheader("📊 Trade Activity by Symbol")
        
        trade_counts = trades_df['symbol'].value_counts().sort_values(ascending=True)
        
        fig_universe = go.Figure()
        fig_universe.add_trace(go.Bar(
            x=trade_counts.values,
            y=trade_counts.index,
            orientation='h',
            marker=dict(color='#3b82f6')
        ))
        
        fig_universe.update_layout(
            title="Number of Trades per Symbol",
            xaxis_title="Trade Count",
            yaxis_title="Symbol",
            height=max(400, len(trade_counts) * 20),
            showlegend=False
        )
        
        st.plotly_chart(fig_universe, use_container_width=True)
    
    # ========================================================================
    # TAB 4: MONTHLY RETURNS HEATMAP
    # ========================================================================
    with tab4:
        st.subheader("📅 Monthly Returns & Capital Allocation Heatmap")
        st.markdown("**Green cells** = profitable months, **Red cells** = losing months. Shows monthly return % and average capital deployed.")
        
        # Calculate monthly capital allocation
        def calculate_monthly_capital_allocation(trades_df, initial_capital):
            trades_calc = trades_df.copy()
            trades_calc['entry_date'] = pd.to_datetime(trades_calc['entry_date'], utc=True).dt.tz_localize(None)
            trades_calc['exit_date'] = pd.to_datetime(trades_calc['exit_date'], utc=True).dt.tz_localize(None)
            
            trades_calc['position_size'] = trades_calc['shares'] * trades_calc['entry_price']
            
            trades_sorted = trades_calc.sort_values('exit_date')
            trades_sorted['cumulative_pnl'] = trades_sorted['pnl'].cumsum()
            
            min_date = trades_calc['entry_date'].min()
            max_date = trades_calc['exit_date'].max()
            
            all_dates = pd.date_range(start=min_date, end=max_date, freq='B')
            
            monthly_allocation = {}
            
            for dt in all_dates:
                year = dt.year
                month = dt.month
                
                completed_trades = trades_sorted[trades_sorted['exit_date'] <= dt]
                cumulative_pnl = completed_trades['pnl'].sum() if len(completed_trades) > 0 else 0
                portfolio_value = initial_capital + cumulative_pnl
                
                active_trades = trades_calc[
                    (trades_calc['entry_date'] <= dt) & 
                    (trades_calc['exit_date'] >= dt)
                ]
                
                daily_deployed_capital = active_trades['position_size'].sum()
                
                if portfolio_value > 0:
                    daily_allocation_pct = (daily_deployed_capital / portfolio_value) * 100
                else:
                    daily_allocation_pct = 0
                
                key = (year, month)
                if key not in monthly_allocation:
                    monthly_allocation[key] = []
                monthly_allocation[key].append(daily_allocation_pct)
            
            monthly_avg_allocation = {}
            for key, daily_pcts in monthly_allocation.items():
                avg_allocation_pct = sum(daily_pcts) / len(daily_pcts)
                monthly_avg_allocation[key] = avg_allocation_pct
            
            return monthly_avg_allocation
        
        monthly_allocation = calculate_monthly_capital_allocation(trades_df, run_summary['initial_capital'])
        
        trades_monthly = trades_df.copy()
        trades_monthly['exit_date'] = pd.to_datetime(trades_monthly['exit_date'], utc=True).dt.tz_localize(None)
        trades_monthly['year'] = trades_monthly['exit_date'].dt.year
        trades_monthly['month'] = trades_monthly['exit_date'].dt.month
        
        monthly_pnl = trades_monthly.groupby(['year', 'month'])['pnl'].sum().reset_index()
        monthly_pnl['return_pct'] = (monthly_pnl['pnl'] / run_summary['initial_capital']) * 100
        
        monthly_pnl['capital_alloc_pct'] = monthly_pnl.apply(
            lambda row: monthly_allocation.get((row['year'], row['month']), 0),
            axis=1
        )
        
        # Year range selector
        available_years = sorted(monthly_pnl['year'].unique())
        if len(available_years) > 0:
            st.markdown("**Select Year Range:**")
            col1, col2 = st.columns(2)
            with col1:
                start_year = st.selectbox(
                    "Start Year",
                    options=available_years,
                    index=0,
                    key='heatmap_start_year'
                )
            with col2:
                end_year = st.selectbox(
                    "End Year",
                    options=available_years,
                    index=len(available_years) - 1,
                    key='heatmap_end_year'
                )
            
            monthly_pnl_filtered = monthly_pnl[
                (monthly_pnl['year'] >= start_year) &
                (monthly_pnl['year'] <= end_year)
            ]
        else:
            monthly_pnl_filtered = monthly_pnl
        
        # Pivot for heatmap
        heatmap_data = monthly_pnl_filtered.pivot(index='year', columns='month', values='return_pct')
        capital_alloc_data = monthly_pnl_filtered.pivot(index='year', columns='month', values='capital_alloc_pct')
        
        heatmap_data = heatmap_data.sort_index(ascending=False)
        capital_alloc_data = capital_alloc_data.sort_index(ascending=False)
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        heatmap_data.columns = [month_names[int(m)-1] for m in heatmap_data.columns]
        capital_alloc_data.columns = [month_names[int(m)-1] for m in capital_alloc_data.columns]
        
        # Prepare text display
        text_display = []
        for year_idx in range(len(heatmap_data)):
            row_text = []
            for month_col in heatmap_data.columns:
                return_val = heatmap_data.iloc[year_idx][month_col]
                capital_val = capital_alloc_data.iloc[year_idx][month_col]
                
                if pd.isna(return_val):
                    row_text.append('')
                else:
                    row_text.append(f'{return_val:.1f}%<br>{capital_val:.0f}% cap')
            text_display.append(row_text)
        
        # Custom colorscale
        custom_colorscale = [
            [0.0, '#8B0000'], [0.2, '#B22222'], [0.35, '#DC143C'],
            [0.45, '#E67E7E'], [0.5, '#F5F5F5'], [0.55, '#A8D5A8'],
            [0.7, '#4CAF50'], [0.85, '#2E7D32'], [1.0, '#006400']
        ]
        
        z_values = heatmap_data.fillna(0).values
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=z_values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale=custom_colorscale,
            zmid=0,
            text=text_display,
            texttemplate='%{text}',
            textfont={"size": 13, "color": "black"},
            colorbar=dict(title="Return %"),
            hoverongaps=False,
            xgap=3,
            ygap=3
        ))
        
        fig_heatmap.update_layout(
            height=max(400, len(heatmap_data) * 85),
            xaxis=dict(title="Month", side='top'),
            yaxis=dict(title="Year")
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Monthly statistics
        st.markdown("---")
        st.markdown("**Performance Metrics for Selected Period:**")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            best_month_return = monthly_pnl_filtered['return_pct'].max() if len(monthly_pnl_filtered) > 0 else 0
            st.metric("Best Month", f"{best_month_return:.1f}%")
        
        with col2:
            worst_month_return = monthly_pnl_filtered['return_pct'].min() if len(monthly_pnl_filtered) > 0 else 0
            st.metric("Worst Month", f"{worst_month_return:.1f}%")
        
        with col3:
            avg_month_return = monthly_pnl_filtered['return_pct'].mean() if len(monthly_pnl_filtered) > 0 else 0
            st.metric("Avg Monthly Return", f"{avg_month_return:.1f}%")
        
        with col4:
            if len(monthly_pnl_filtered) > 0:
                winning_months = (monthly_pnl_filtered['return_pct'] > 0).sum()
                total_months = len(monthly_pnl_filtered)
                month_win_rate = (winning_months / total_months * 100) if total_months > 0 else 0
                st.metric("Winning Months", f"{month_win_rate:.0f}%")
            else:
                st.metric("Winning Months", "N/A")
        
        with col5:
            avg_capital_alloc = monthly_pnl_filtered['capital_alloc_pct'].mean() if len(monthly_pnl_filtered) > 0 else 0
            st.metric("Avg Capital", f"{avg_capital_alloc:.0f}%")

# Footer
st.markdown("---")
if is_live_mode:
    st.caption(f"Live Trading Dashboard | {datetime.now().strftime('%Y-%m-%d %H:%M:%S PT')}")
else:
    st.caption(f"Backtest Run #{run_id} | {run_summary['description']} | Run Date: {run_summary['run_date']}")