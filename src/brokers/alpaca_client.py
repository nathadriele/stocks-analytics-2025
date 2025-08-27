# src/brokers/alpaca_client.py
from __future__ import annotations
import os, json, time, requests, pandas as pd

ALPACA_KEY = os.getenv("ALPACA_KEY_ID")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

def _headers():
    if not ALPACA_KEY or not ALPACA_SECRET:
        raise EnvironmentError("Defina ALPACA_KEY_ID e ALPACA_SECRET_KEY.")
    return {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "Content-Type": "application/json",
    }

def submit_order(symbol: str, qty: float, side: str, type: str="market", tif: str="day"):
    url = f"{ALPACA_BASE}/v2/orders"
    payload = {"symbol": symbol, "qty": abs(qty), "side": side, "type": type, "time_in_force": tif}
    r = requests.post(url, headers=_headers(), data=json.dumps(payload), timeout=30)
    r.raise_for_status()
    return r.json()

def place_from_signals(signals_path: str, top_n: int = 3):
    df = pd.read_parquet(signals_path) if signals_path.endswith(".parquet") else pd.read_csv(signals_path)
    today = df["date"].max()
    day_df = df[df["date"] == today].copy()
    # seleciona top-n por probabilidade
    day_df = day_df.sort_values("proba_up", ascending=False).head(top_n)
    results = []
    for _, row in day_df.iterrows():
        sym = str(row["ticker"])
        side = "buy" if row["signal"] > 0 else "sell"
        res = submit_order(sym, qty=1, side=side)
        results.append(res); time.sleep(0.2)
    return results
