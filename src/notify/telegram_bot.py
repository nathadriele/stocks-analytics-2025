# src/notify/telegram_bot.py
from __future__ import annotations
import os, requests, pandas as pd

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # seu chat/channel id

def send_message(text: str):
    if not TOKEN or not CHAT_ID:
        raise EnvironmentError("Defina TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID.")
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    r = requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=30)
    r.raise_for_status()
    return r.json()

def send_trades_summary(trades_path: str):
    df = pd.read_parquet(trades_path)
    last = df[df["date"] == df["date"].max()]
    lines = ["📣 *Trades Executados Hoje*"]
    for _, r in last.iterrows():
        lines.append(f"{r['date']} | {r['ticker']} | {r['action']} @ {r['price']:.2f} | qty {r['shares']:.2f}")
    return send_message("\n".join(lines))
