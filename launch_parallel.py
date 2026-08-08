#!/usr/bin/env python3
"""Launch both day trader and scalper in parallel, fully detached"""
import subprocess, os, sys

TRADER_DIR = "/home/ubuntu/revolut-x-trader"
VENV_PYTHON = "/tmp/trading-venv/bin/python3"

# Load API key
key = ""
with open(os.path.expanduser("~/.hermes/.env")) as f:
    for line in f:
        if "OPENROUTER_API_KEY" in line and "=" in line and not line.strip().startswith("#"):
            raw = line.split("=", 1)[1].strip()
            for q in ["'", '"']:
                raw = raw.strip(q)
            key = raw.strip()
            break

env = os.environ.copy()
env["OPENROUTER_API_KEY"] = key

# Kill any existing
for name in ["day_trader", "scalper"]:
    try:
        with open(f"/tmp/{name}.pid") as f:
            old = f.read().strip()
            if old:
                os.system(f"kill {old} 2>/dev/null")
    except:
        pass

# Start day trader (5 min cycles, AI-powered)
p1 = subprocess.Popen(
    [VENV_PYTHON, os.path.join(TRADER_DIR, "day_trader.py"), "--interval", "5"],
    stdout=open(os.path.join(TRADER_DIR, "logs", "day_trader.log"), "a"),
    stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    cwd=TRADER_DIR,
    env=env,
    start_new_session=True,
)
with open("/tmp/day_trader.pid", "w") as f:
    f.write(str(p1.pid))

# Start scalper (60s, pure support dips, zero AI cost)
p2 = subprocess.Popen(
    [VENV_PYTHON, os.path.join(TRADER_DIR, "scalper.py")],
    stdout=open(os.path.join(TRADER_DIR, "logs", "scalper.log"), "a"),
    stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    cwd=TRADER_DIR,
    env=env,
    start_new_session=True,
)
with open("/tmp/scalper.pid", "w") as f:
    f.write(str(p2.pid))

print(f"DONE")
print(f"  Day trader: PID {p1.pid} → logs/day_trader.log")
print(f"  Scalper:    PID {p2.pid} → logs/scalper.log")
print(f"  API key: {'✅' if key else '❌'}")