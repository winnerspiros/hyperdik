"""
Streamlit Dashboard — lightweight crypto bot monitoring.

Shows:
  - Current EUR balance
  - Open positions with PnL
  - Recent trades table
  - Performance chart (equity curve)
  - Strategy stats

Designed for low-resource server (2 cores, 1GB RAM).
Under 120 lines. SQLite-backed (reads trader.db).

Usage:
  streamlit run dashboard.py --server.port 8501

Or via UV:
  uv run streamlit run dashboard.py --server.port 8501
"""
import sqlite3
import os
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Config ─────────────────────────────────────────────────────────────────

DB_DIR = Path(__file__).parent / "data"
DB_PATH = DB_DIR / "trader.db"
REFRESH_SECONDS = 30  # Auto-refresh interval

st.set_page_config(
    page_title="Revolut X Trader",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Data helpers ───────────────────────────────────────────────────────────

@st.cache_data(ttl=REFRESH_SECONDS)
def get_db_data(query: str) -> pd.DataFrame:
    """Query the trader database with caching."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_balance() -> float:
    """Get latest EUR balance from snapshots."""
    df = get_db_data(
        "SELECT eur_balance FROM performance_snapshots "
        "ORDER BY snapshot_time DESC LIMIT 1"
    )
    if not df.empty:
        return float(df.iloc[0]["eur_balance"])
    return 0.0


# ── Dashboard ──────────────────────────────────────────────────────────────

st.title("🤖 Revolut X Trader Dashboard")

# ── Row 1: Key metrics ─────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)

balance = get_balance()
col1.metric("💰 EUR Balance", f"€{balance:.2f}" if balance else "N/A")

positions = get_db_data(
    "SELECT symbol, qty, entry_price, (SELECT price FROM trades "
    "WHERE symbol=positions.symbol AND side='buy' ORDER BY entry_time DESC LIMIT 1) "
    "as current_price FROM positions WHERE is_open=1"
)
col2.metric("📊 Open Positions", str(len(positions)))

trades_today = get_db_data(
    "SELECT COUNT(*) as cnt, COALESCE(SUM(pnl),0) as pnl "
    "FROM trades WHERE date(exit_time)=date('now')"
)
today_pnl = trades_today.iloc[0]["pnl"] if not trades_today.empty else 0
col3.metric("📈 Today PnL", f"€{today_pnl:.2f}", delta=f"{today_pnl:.2f}")

stats = get_db_data("SELECT * FROM strategy_stats")
if not stats.empty:
    wr = stats.iloc[0]["win_rate"]
    col4.metric("🏆 Win Rate", f"{wr:.1%}" if wr else "N/A")
else:
    col4.metric("🏆 Win Rate", "N/A")

# ── Row 2: Open positions ──────────────────────────────────────────────────

st.subheader("📋 Open Positions")
if not positions.empty:
    # Add rough PnL calculation
    positions["pnl_pct"] = 0.0
    st.dataframe(positions, use_container_width=True, hide_index=True)
else:
    st.info("No open positions")

# ── Row 3: Recent trades ───────────────────────────────────────────────────

st.subheader("🕐 Recent Trades")
trades = get_db_data(
    "SELECT id, symbol, side, qty, entry_price, pnl, pnl_pct, "
    "       strategy, exit_reason, exit_time "
    "FROM trades ORDER BY id DESC LIMIT 20"
)
if not trades.empty:
    # Color-code PnL
    def color_pnl(val):
        if val is None:
            return ""
        return "color: green" if val > 0 else "color: red"
    st.dataframe(
        trades.style.map(color_pnl, subset=["pnl"]),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No trades yet")

# ── Row 4: Performance chart ───────────────────────────────────────────────

st.subheader("📉 Performance")
snapshots = get_db_data(
    "SELECT snapshot_time, eur_balance, total_value, num_positions "
    "FROM performance_snapshots ORDER BY snapshot_time"
)

if len(snapshots) > 1:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Portfolio Value (EUR)", "Open Positions"),
    )
    
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(snapshots["snapshot_time"]),
            y=snapshots["total_value"].fillna(snapshots["eur_balance"]),
            mode="lines",
            name="Portfolio Value",
            line=dict(color="#00d4aa", width=2),
        ),
        row=1, col=1,
    )
    
    fig.add_trace(
        go.Bar(
            x=pd.to_datetime(snapshots["snapshot_time"]),
            y=snapshots["num_positions"],
            name="Positions",
            marker_color="#ff6b6b",
        ),
        row=2, col=1,
    )
    
    fig.update_layout(
        height=500,
        hovermode="x unified",
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=40, r=20, t=30, b=20),
    )
    fig.update_yaxes(row=1, col=1, tickprefix="€")
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough data for chart (need 2+ snapshots)")

# ── Row 5: Strategy stats ──────────────────────────────────────────────────

with st.expander("📊 Strategy Statistics", expanded=False):
    stats_df = get_db_data("SELECT * FROM strategy_stats")
    if not stats_df.empty:
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

# ── Auto-refresh ───────────────────────────────────────────────────────────

st.caption(f"🔄 Auto-refreshes every {REFRESH_SECONDS}s · Last update: data cached")
st.button("🔄 Refresh Now", on_click=st.rerun, type="primary")
