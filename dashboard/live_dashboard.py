# dashboard/live_dashboard.py
"""
Live Trading Dashboard
======================

Streamlit dashboard for reviewing and managing live trades.

Run: streamlit run dashboard/live_dashboard.py

Features:
- View today's entry/exit signals
- Approve/reject signals
- Monitor open positions
- Track performance
- Risk metrics
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta

# Page config
st.set_page_config(
    page_title="Live Trading Dashboard",
    page_icon="🎯",
    layout="wide"
)

# Database path
DB_PATH = "database/live_trading.db"

# Initialize database connection
@st.cache_resource
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

# Helper functions
def get_today_signals():
    """Get today's signals"""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("""
            SELECT * FROM daily_signals 
            WHERE signal_date = ?
            ORDER BY signal_type, symbol
        """, conn, params=(str(date.today()),))
    return df

def get_open_positions():
    """Get current open positions"""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("""
            SELECT * FROM live_positions 
            WHERE status = 'OPEN'
            ORDER BY entry_date DESC
        """, conn)
    return df

def get_closed_trades():
    """Get all closed trades"""
    with sqlite3.connect(DB_PATH) as conn:
        # Check if table exists
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

def approve_signal(signal_id):
    """Mark signal as approved"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            UPDATE daily_signals 
            SET approved = 1
            WHERE id = ?
        """, (signal_id,))
        conn.commit()
    st.success("✅ Signal approved!")

def reject_signal(signal_id):
    """Mark signal as rejected"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            DELETE FROM daily_signals 
            WHERE id = ?
        """, (signal_id,))
        conn.commit()
    st.warning("❌ Signal rejected!")

# Title
st.title("🎯 Live Trading Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Today's Signals",
    "💼 Open Positions", 
    "📈 Performance",
    "⚙️ Settings"
])

# ============================================================================
# TAB 1: TODAY'S SIGNALS
# ============================================================================
with tab1:
    st.header("📊 Today's Signals")
    st.caption(f"Generated: {date.today().strftime('%Y-%m-%d')}")
    
    signals = get_today_signals()
    
    if len(signals) == 0:
        st.info("✅ No signals today - HOLD current positions")
    else:
        # Entry signals
        entry_signals = signals[signals['signal_type'] == 'ENTRY']
        
        if len(entry_signals) > 0:
            st.subheader("🟢 ENTRY SIGNALS (New Positions)")
            
            for _, signal in entry_signals.iterrows():
                with st.container(border=True):
                    col1, col2, col3 = st.columns([2, 3, 1])
                    
                    with col1:
                        st.markdown(f"### {signal['symbol']}")
                        st.caption(f"Confidence: {signal['confidence']:.1%}")
                    
                    with col2:
                        st.markdown(f"**Entry:** ${signal['entry_price']:.2f}")
                        st.markdown(f"**Stop Loss:** ${signal['stop_loss']:.2f}")
                        st.markdown(f"**Shares:** {signal['shares']}")
                        st.markdown(f"**Position Value:** ${signal['position_value']:.2f}")
                        st.caption(f"Reason: {signal['reason']}")
                    
                    with col3:
                        if signal['approved']:
                            st.success("✅ Approved")
                        elif signal['executed']:
                            st.info("✓ Executed")
                        else:
                            col_a, col_b = st.columns(2)
                            with col_a:
                                if st.button("✅", key=f"approve_{signal['id']}", help="Approve"):
                                    approve_signal(signal['id'])
                                    st.rerun()
                            with col_b:
                                if st.button("❌", key=f"reject_{signal['id']}", help="Reject"):
                                    reject_signal(signal['id'])
                                    st.rerun()
            
            # Entry execution checklist
            st.markdown("---")
            st.markdown("### ✅ Entry Execution Checklist")
            st.markdown("""
            **Before executing:**
            1. ✅ Verify signal is approved above
            2. ✅ Check available capital
            3. ✅ Confirm position count < 4
            4. ✅ Set limit order at entry price + 1%
            5. ✅ Place stop loss order immediately after fill
            6. ✅ Export fill from broker and update portfolio
            """)
        
        # Exit signals
        exit_signals = signals[signals['signal_type'] == 'EXIT']
        
        if len(exit_signals) > 0:
            st.markdown("---")
            st.subheader("🔴 EXIT SIGNALS (Close Positions)")
            
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
                            st.success("✅ Approved")
                        elif signal['executed']:
                            st.info("✓ Executed")
                        else:
                            col_a, col_b = st.columns(2)
                            with col_a:
                                if st.button("✅", key=f"approve_exit_{signal['id']}", help="Approve"):
                                    approve_signal(signal['id'])
                                    st.rerun()
                            with col_b:
                                if st.button("❌", key=f"reject_exit_{signal['id']}", help="Reject"):
                                    reject_signal(signal['id'])
                                    st.rerun()
            
            # Exit execution checklist
            st.markdown("---")
            st.markdown("### ✅ Exit Execution Checklist")
            st.markdown("""
            **Before executing:**
            1. ✅ Verify signal is approved above
            2. ✅ Use market order at open OR limit at yesterday's close
            3. ✅ Cancel stop loss order after fill
            4. ✅ Export fill from broker and update portfolio
            """)

# ============================================================================
# TAB 2: OPEN POSITIONS
# ============================================================================
with tab2:
    st.header("💼 Open Positions")
    
    positions = get_open_positions()
    
    if len(positions) == 0:
        st.info("No open positions")
    else:
        st.caption(f"Total: {len(positions)} positions")
        
        # Load current prices for P&L calculation
        import os
        data_dir = "data"
        
        # Summary metrics
        total_cost = positions['position_value'].sum()
        
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
            
            # Color based on P&L
            pnl_color = "green" if unrealized_pnl > 0 else "red"
            
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

# ============================================================================
# TAB 3: PERFORMANCE
# ============================================================================
with tab3:
    st.header("📈 Performance")
    
    trades = get_closed_trades()
    positions = get_open_positions()
    
    # Calculate metrics
    initial_capital = 50000
    
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
        
        # Average metrics
        avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades[trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        avg_hold = trades['hold_days'].mean()
    else:
        total_pnl = 0
        total_return_pct = 0
        winning_trades = 0
        losing_trades = 0
        win_rate = 0
        profit_factor = 0
        avg_win = 0
        avg_loss = 0
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
            line=dict(color='#1f77b4', width=2),
            fill='tonexty',
            fillcolor='rgba(31, 119, 180, 0.1)'
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

# ============================================================================
# TAB 4: SETTINGS
# ============================================================================
with tab4:
    st.header("⚙️ Settings & Configuration")
    
    st.subheader("📊 Pilot Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Capital Allocation:**")
        st.markdown("- Total Capital: $50,000")
        st.markdown("- Deployed: 90% ($45,000)")
        st.markdown("- Cash Reserve: 10% ($5,000)")

        st.markdown("**Position Sizing:**")
        st.markdown("- Per Position: 5% ($2,500)")
        st.markdown("- Max Positions: 8")
        st.markdown("- Max Deployment: 90% (~$45,000)")

    with col2:
        st.markdown("**Risk Limits:**")
        st.markdown("- Daily Loss Limit: -$1,000 (-2%)")
        st.markdown("- Weekly Loss Limit: -$2,500 (-5%)")
        st.markdown("- Max Drawdown: -$7,500 (-15%)")
        
        st.markdown("**Entry Criteria:**")
        st.markdown("- Min Confidence: 0.75")
        st.markdown("- Min Volume: 500K shares/day")
        st.markdown("- Max Spread: 0.5%")
    
    st.markdown("---")
    
    st.subheader("🗂️ Database Location")
    st.code(f"database/live_trading.db")
    
    st.subheader("📁 Data Directories")
    st.code("data/ - Stock price data")
    st.code("data/broker_fills/ - Daily trade fills from broker")
    st.code("logs/ - System logs")
    
    st.markdown("---")
    
    st.subheader("🔄 Refresh Dashboard")
    if st.button("🔄 Refresh Data"):
        st.rerun()

# Footer
st.markdown("---")
st.caption(f"Live Trading Dashboard | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")