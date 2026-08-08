"""
US Economic Calendar — scheduled events that move markets
CPI, Jobs, Fed, GDP, Tax — all predictable, all move crypto
"""
from datetime import datetime, timedelta
import urllib.request, ssl, json, re

# Known monthly/quarterly events (approximate schedule)
GLOBAL_EVENTS = [
    # US
    {"name": "US CPI (Inflation)", "desc": "Biggest mover. High CPI = risk-off.", "schedule": "2nd Wed monthly 13:30 UTC"},
    {"name": "US Jobs Report (NFP)", "desc": "Weak = Fed cuts, Strong = Fed hikes.", "schedule": "1st Fri monthly 13:30 UTC"},
    {"name": "Fed Interest Rate Decision", "desc": "Rate cuts = crypto up. Rate hikes = crypto down.", "schedule": "Every 6 weeks Wed"},
    {"name": "US GDP", "desc": "Growth data. Affects risk appetite.", "schedule": "Last week Jan/Apr/Jul/Oct"},
    {"name": "US Tax Day", "desc": "People sell crypto to pay taxes. April dump.", "schedule": "April 15"},
    
    # Europe
    {"name": "ECB Interest Rate Decision", "desc": "EU rate moves affect EUR pairs and risk.", "schedule": "Every 6 weeks Thu"},
    {"name": "EU CPI (Inflation)", "desc": "Eurozone inflation. Moves EUR pairs.", "schedule": "Last week monthly"},
    {"name": "German GDP / EU GDP", "desc": "Eurozone growth. Affects EUR.", "schedule": "Quarterly"},
    
    # UK
    {"name": "BOE Interest Rate Decision", "desc": "UK rates. GBP pairs affected.", "schedule": "Every 6 weeks Thu"},
    {"name": "UK CPI", "desc": "UK inflation. GBP volatility.", "schedule": "2nd week monthly"},
    
    # China
    {"name": "China GDP", "desc": "Chinese growth affects BTC (mining, demand).", "schedule": "Quarterly Jan/Apr/Jul/Oct"},
    {"name": "China PMI (Manufacturing)", "desc": "Leading indicator for Chinese economy.", "schedule": "Last day monthly"},
    
    # Japan
    {"name": "BOJ Interest Rate Decision", "desc": "Japanese rates. Yen carry trade affects crypto.", "schedule": "Every 6 weeks"},
    
    # Global
    {"name": "Oil Price (OPEC)", "desc": "Oil shocks affect inflation = rate decisions.", "schedule": "As announced"},
    {"name": "US Treasury Report", "desc": "Shows who holds US debt. Big moves possible.", "schedule": "Monthly"},
]

def get_upcoming_economic_events():
    """Return any major economic events happening this week globally"""
    now = datetime.now()
    today = now.day
    month = now.month
    weekday = now.weekday()
    
    signals = []
    
    # US Tax Season (April)
    if month in [3, 4]:
        if month == 4 and today <= 15:
            signals.append("⚠️ US TAX SEASON (Apr 1-15) — selling pressure")
        elif month == 3 and today >= 15:
            signals.append("⚠️ US TAX SEASON approaching (Apr 15)")
    
    # CPI week (2nd week)
    if 7 <= today <= 14:
        signals.append("📊 CPI week — inflation data coming, expect volatility")
    
    # Jobs week (1st week)
    if today <= 7:
        signals.append("📊 Jobs Report week — NFP Friday")
    
    # EU events
    if month in [1, 3, 5, 7, 9, 11] and today >= 20:
        signals.append("🏛️ ECB decision window — EU interest rates")
    if 25 <= today <= 31:
        signals.append("📊 EU CPI / German data — end of month")
    
    # China GDP quarter ends
    if month in [1, 4, 7, 10] and today >= 15:
        signals.append("🏭 China GDP window — affects BTC demand")
    
    # BOJ (Japan)
    if month in [3, 6, 9, 12] and today >= 15:
        signals.append("🏯 BOJ rate decision — Yen carry trade affects all markets")
    
    # Fed window
    if month in [1, 3, 5, 7, 9, 11, 12] and today >= 10:
        signals.append("🏦 Fed decision possible this month — rate sensitive")
    
    # Try to get actual earnings calendar (real data)
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(
            "https://r.jina.ai/https://www.forexfactory.com/calendar",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            content = resp.read().decode()
            # Look for high-impact events
            for event in ["CPI", "FOMC", "NFP", "Fed", "GDP", "Interest Rate", "Employment"]:
                if event.lower() in content.lower():
                    # Find surrounding text
                    idx = content.lower().find(event.lower())
                    snippet = content[max(0,idx-30):idx+60]
                    clean = re.sub(r'<[^>]+>', '', snippet).strip()
                    signals.append(f"📊 {clean[:100]}")
                    if len(signals) >= 4:
                        break
    except:
        pass
    
    if not signals:
        signals.append("No major US economic events this week")
    
    return signals[:4]

if __name__ == "__main__":
    print(f"US Economic Calendar — {datetime.now().strftime('%B %Y')}:")
    for s in get_upcoming_economic_events():
        print(f"  • {s}")