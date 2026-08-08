"""
Revolut X Trader Database — SQLite persistence for trades, decisions, positions, strategy stats.
"""
import sqlite3
import os
import json
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "trader.db")


def get_db() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init():
    """Initialize database tables"""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL,
            price REAL,
            entry_price REAL,
            pnl REAL,
            pnl_pct REAL,
            fee REAL DEFAULT 0,
            strategy TEXT DEFAULT 'day_trader',
            exit_reason TEXT,
            entry_time TIMESTAMP,
            exit_time TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            cycle INTEGER,
            strategy TEXT DEFAULT 'day_trader',
            direction TEXT,
            symbol TEXT,
            ai_reasoning TEXT,
            price_entry REAL,
            price_target REAL,
            price_stop REAL,
            size REAL,
            eur_balance REAL,
            num_positions INTEGER,
            market_conditions TEXT
        );
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            strategy TEXT DEFAULT 'day_trader',
            qty REAL,
            entry_price REAL,
            entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            stop_loss REAL,
            take_profit REAL,
            is_open INTEGER DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS strategy_stats (
            strategy TEXT PRIMARY KEY,
            total_trades INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            total_pnl REAL DEFAULT 0,
            avg_win REAL DEFAULT 0,
            avg_loss REAL DEFAULT 0,
            win_rate REAL DEFAULT 0,
            consecutive_losses INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS performance_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            eur_balance REAL,
            total_value REAL,
            num_positions INTEGER
        );
    """)
    conn.commit()
    conn.close()
    return True


def log_trade(symbol, side, qty, price=None, entry_price=None, strategy="day_trader", exit_reason=None):
    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()
    if side == "sell" and entry_price and price and entry_price > 0:
        pnl = (price - entry_price) * qty
        pnl_pct = (price - entry_price) / entry_price * 100
        fee = price * qty * 0.0009
    else:
        pnl = pnl_pct = fee = None
    conn.execute("""
        INSERT INTO trades (symbol, side, qty, price, entry_price, pnl, pnl_pct, fee, strategy, exit_reason, exit_time)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (symbol, side, qty, price, entry_price, pnl, pnl_pct, fee, strategy, exit_reason, now if side == "sell" else None))
    if side == "sell" and pnl is not None:
        _update_stats(conn, strategy, pnl > 0, pnl, pnl_pct)
    conn.commit()
    tid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return tid


def _update_stats(conn, strategy, was_win, pnl, pnl_pct):
    stats = conn.execute("SELECT * FROM strategy_stats WHERE strategy=?", (strategy,)).fetchone()
    if stats:
        new_total = stats["total_trades"] + 1
        new_wins = stats["wins"] + (1 if was_win else 0)
        new_losses = stats["losses"] + (0 if was_win else 1)
        new_consec = (stats["consecutive_losses"] + 1) if not was_win else 0
        conn.execute("""
            UPDATE strategy_stats SET total_trades=?, wins=?, losses=?, total_pnl=total_pnl+?,
                win_rate=CAST(? AS REAL)/?, consecutive_losses=?, updated_at=CURRENT_TIMESTAMP
            WHERE strategy=?
        """, (new_total, new_wins, new_losses, pnl, new_wins, new_total, new_consec, strategy))
    else:
        conn.execute("""
            INSERT INTO strategy_stats(strategy,total_trades,wins,losses,total_pnl,win_rate,consecutive_losses)
            VALUES (?,1,?,?,?,?,?)
        """, (strategy, 1 if was_win else 0, 0 if was_win else 1, pnl, 1.0 if was_win else 0.0, 0 if was_win else 1))


def log_decision(direction, strategy="day_trader", cycle=None, symbol=None, ai_reasoning=None,
                 price_entry=None, price_target=None, price_stop=None, size=None,
                 eur_balance=None, num_positions=None, market_conditions=None):
    conn = get_db()
    conn.execute("""
        INSERT INTO decisions(strategy,cycle,direction,symbol,ai_reasoning,price_entry,price_target,
                              price_stop,size,eur_balance,num_positions,market_conditions)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (strategy, cycle, direction, symbol, ai_reasoning, price_entry, price_target,
          price_stop, size, eur_balance, num_positions, market_conditions))
    conn.commit()
    did = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return did


def save_snapshot(eur_balance, total_value, num_positions):
    conn = get_db()
    conn.execute("INSERT INTO performance_snapshots(eur_balance,total_value,num_positions) VALUES (?,?,?)",
                 (eur_balance, total_value, num_positions))
    conn.commit()
    conn.close()


def get_strategy_performance(strategy):
    conn = get_db()
    row = conn.execute("SELECT * FROM strategy_stats WHERE strategy=?", (strategy,)).fetchone()
    conn.close()
    return dict(row) if row else None


if __name__ == "__main__":
    init()
    print(f"✅ Database: {DB_PATH}")
    conn = get_db()
    tables = [r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    conn.close()
    print(f"   Tables: {', '.join(tables)}")