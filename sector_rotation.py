"""
Crypto Sector Rotation Tracker.
Implements multi-timeframe sector momentum analysis — money flow between sectors.

Sectors: L1, L2, DeFi, Meme, AI, Gaming, RWA, Exchange, Oracle, Storage, Privacy
Methodology: track 1h/4h/24h momentum per sector, detect rotation patterns.
Used to bias coin selection toward sectors with positive flow.
"""

import time
import logging
from collections import defaultdict
from typing import Optional

log = logging.getLogger(__name__)

# ── Sector Definitions ──
SECTORS: dict[str, list[str]] = {
    "L1": ["BTC", "ETH", "SOL", "AVAX", "BNB", "SUI", "APT", "SEI", "TON", "NEAR",
           "INJ", "DOT", "ADA", "ATOM", "FTM", "ALGO", "ICP", "XRP", "LTC", "BCH",
           "TRX", "HBAR", "KAS", "CORE", "ZETA", "ROSE", "CKB", "FLOW"],
    "L2": ["ARB", "OP", "MATIC", "POL", "STRK", "ZK", "ZKSYNC", "MANTA", "MNT",
           "METIS", "IMX", "BOBA", "CANTO", "CELO", "BLAST", "TAIKO", "SCROLL", "LINEA"],
    "DeFi": ["UNI", "AAVE", "MKR", "CRV", "SNX", "COMP", "GMX", "DYDX", "LDO",
             "PENDLE", "RUNE", "JUP", "RAY", "ORCA", "1INCH", "BAL", "SUSHI",
             "YFI", "WOO", "JOE", "CAKE", "GNS", "PERP", "KWENTA", "BANANA"],
    "Meme": ["DOGE", "SHIB", "PEPE", "BONK", "WIF", "FLOKI", "MEME", "POPCAT",
             "MOG", "BRETT", "TOSHI", "PENGU", "PNUT", "GOAT", "MEW", "TURBO",
             "MYRO", "SAMO", "WEN", "KIBA", "BOME", "SLERF", "PURR", "KHAI"],
    "AI": ["RENDER", "TAO", "FET", "AGIX", "OCEAN", "WLD", "AKT", "AKASH",
           "GRT", "CTXC", "NMR", "TRAC", "ZIG", "PAAL", "OLAS", "DEAI",
           "SPEC", "NOS", "AIOZ", "AI", "CLORE", "ATOR", "LMWR"],
    "Gaming": ["GALA", "SAND", "MANA", "AXS", "ENJ", "ILV", "MAGIC", "GODS",
               "PRIME", "PIXEL", "XAI", "BIGTIME", "MAVIA", "ACE", "NYAN",
               "SUPER", "PORTAL", "SFUND", "CROWN", "NAKA"],
    "RWA": ["ONDO", "OM", "CFG", "MPL", "TRU", "GFI", "USDY", "RIO",
            "PRCL", "CHEX", "LAND", "PROPS", "DUSK", "CPOOL", "CTC"],
    "Exchange": ["OKB", "BGB", "CRO", "KCS", "GT", "MX", "WXT", "BTTOLD"],
    "Oracle": ["LINK", "PYTH", "BAND", "API3", "UMA", "TRB", "DIA", "NEST"],
    "Storage": ["FIL", "AR", "STORJ", "HOT", "ANKR", "BTTOLD2", "SC", "BLZ"],
    "Privacy": ["XMR", "ZEC", "ROSE", "SCRT", "NYM", "ZEN", "BEAM", "ARRR"],
}

# Reverse lookup: coin → sector
COIN_SECTOR: dict[str, str] = {}
for sector, coins in SECTORS.items():
    for coin in coins:
        COIN_SECTOR[coin.upper()] = sector


class SectorTracker:
    """Tracks multi-timeframe momentum per sector."""
    
    def __init__(self):
        self.momentum: dict[str, dict[str, float]] = defaultdict(dict)  # {sector: {tf: mom%}}
        self.last_update: float = 0
        self.update_interval: float = 120  # Update every 2 minutes
    
    def update(self, mids: dict, candles_5m: dict, candles_1h: dict):
        """Update sector momentum from current prices and candles."""
        now = time.time()
        if now - self.last_update < self.update_interval:
            return
        self.last_update = now
        
        # Aggregate per sector
        sector_prices = defaultdict(list)
        for coin, price in mids.items():
            sector = COIN_SECTOR.get(coin.upper(), "")
            if sector and price > 0:
                sector_prices[sector].append(price)
        
        # Compute sector momentum from 5m candles
        sector_mom5 = {}
        for sector in SECTORS:
            coins = SECTORS[sector]
            mom_values = []
            for coin in coins:
                candles = candles_5m.get(coin.upper(), [])
                if len(candles) >= 12:  # Need at least 1 hour of data
                    closes = [float(c.get('c', c.get('close', 0))) for c in candles]
                    if closes[-1] > 0 and closes[0] > 0:
                        mom = (closes[-1] - closes[0]) / closes[0] * 100
                        mom_values.append(mom)
            if mom_values:
                sector_mom5[sector] = sum(mom_values) / len(mom_values)
        
        # Compute sector momentum from 1h candles (4h lookback)
        sector_mom1h = {}
        for sector in SECTORS:
            coins = SECTORS[sector]
            mom_values = []
            for coin in coins:
                candles = candles_1h.get(coin.upper(), [])
                if len(candles) >= 4:
                    closes = [float(c.get('c', c.get('close', 0))) for c in candles]
                    if closes[-1] > 0 and closes[-4] > 0:
                        mom = (closes[-1] - closes[-4]) / closes[-4] * 100
                        mom_values.append(mom)
            if mom_values:
                sector_mom1h[sector] = sum(mom_values) / len(mom_values)
        
        # Store
        for sector in SECTORS:
            self.momentum[sector] = {
                "5m_1h": round(sector_mom5.get(sector, 0), 2),
                "1h_4h": round(sector_mom1h.get(sector, 0), 2),
            }
        
        # Log top/bottom sectors
        if sector_mom5:
            ranked = sorted(sector_mom5.items(), key=lambda x: x[1], reverse=True)
            top3 = ranked[:3]
            bot3 = ranked[-3:]
            log.info(f"  🔄 Sectors: {' | '.join(f'{s}:{m:+.1f}%' for s,m in top3)}   vs   {' | '.join(f'{s}:{m:+.1f}%' for s,m in bot3)}")
    
    def get_sector_score(self, coin: str) -> float:
        """Return -1.0 to +1.0 score for a coin's sector momentum."""
        sector = COIN_SECTOR.get(coin.upper(), "")
        if not sector or sector not in self.momentum:
            return 0
        mom = self.momentum[sector]
        score_1h = mom.get("5m_1h", 0)
        score_4h = mom.get("1h_4h", 0)
        # Weight: 60% recent (1h), 40% medium (4h)
        combined = score_1h * 0.6 + score_4h * 0.4
        # Normalize to -1 to +1 range (cap at ±3%)
        return max(-1.0, min(1.0, combined / 3.0))
    
    def get_hot_sectors(self, min_mom: float = 0.5) -> list[str]:
        """Return sectors with positive 1h momentum above threshold."""
        hot = []
        for sector, mom in self.momentum.items():
            if mom.get("5m_1h", 0) > min_mom:
                hot.append(sector)
        return hot
    
    def get_cold_sectors(self, max_mom: float = -0.3) -> list[str]:
        """Return sectors with negative 1h momentum."""
        cold = []
        for sector, mom in self.momentum.items():
            if mom.get("5m_1h", 0) < max_mom:
                cold.append(sector)
        return cold
    
    def rotation_signal(self) -> str:
        """Return a rotation summary string."""
        if not self.momentum:
            return "no data"
        ranked = sorted(self.momentum.items(), 
                       key=lambda x: x[1].get("5m_1h", 0), reverse=True)
        hot = [s for s, m in ranked[:3] if m.get("5m_1h", 0) > 0]
        cold = [s for s, m in ranked[-3:] if m.get("5m_1h", 0) < 0]
        if hot and cold:
            return f"flow: {', '.join(hot)} → avoiding: {', '.join(cold)}"
        elif hot:
            return f"broad bid: {', '.join(hot)} leading"
        elif cold:
            return f"risk-off: {', '.join(cold)} bleeding"
        return "sideways — no sector trend"


# Singleton
sector_tracker = SectorTracker()
