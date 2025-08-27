# src/data/bulk_ingest.py
from __future__ import annotations
import argparse, os, sqlite3
import pandas as pd
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

def parse_args():
    p = argparse.ArgumentParser(description="Bulk ingest (objetivo: >1M linhas).")
    p.add_argument("--tickers-path", required=True, help="CSV com uma coluna 'ticker'.")
    p.add_argument("--provider", choices=["stooq","tiingo","yfinance"], default="stooq")
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--out-parquet", default="data/processed/prices.parquet")
    p.add_argument("--sqlite-path", default="storage/app.db")
    p.add_argument("--table", default="prices")
    p.add_argument("--max-workers", type=int, default=8)
    return p.parse_args()

def _load_tickers(path: str) -> List[str]:
    df = pd.read_csv(path)
    col = [c for c in df.columns if c.lower() in ["ticker","symbol"]][0]
    return [str(x).strip() for x in df[col].dropna().unique().tolist()]

def _fetch_one(provider: str, ticker: str, start: str, end: str) -> pd.DataFrame:
    if provider == "yfinance":
        import yfinance as yf
        d = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if d.empty: return pd.DataFrame()
        d = d.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
        d = d.reset_index().rename(columns={"Date":"date"})
        d["ticker"] = ticker.upper()
        return d[["date","ticker","open","high","low","close","volume"]]
    else:
        from src.data.alt_provider import get_prices
        return get_prices(provider, [ticker], start, end)

def main():
    a = parse_args()
    os.makedirs(os.path.dirname(a.out_parquet), exist_ok=True)
    os.makedirs(os.path.dirname(a.sqlite_path), exist_ok=True)

    tickers = _load_tickers(a.tickers_path)
    frames = []
    with ThreadPoolExecutor(max_workers=a.max_workers) as ex:
        futs = {ex.submit(_fetch_one, a.provider, t, a.start, a.end): t for t in tickers}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                df = fut.result()
                if not df.empty:
                    frames.append(df)
                    print(f"[OK] {t}: {len(df):,} linhas")
                else:
                    print(f"[WARN] {t}: vazio")
            except Exception as e:
                print(f"[ERR] {t}: {e}")

    if not frames:
        raise RuntimeError("Nenhum dado coletado.")
    allp = pd.concat(frames, ignore_index=True)
    allp["date"] = pd.to_datetime(allp["date"]).dt.tz_localize(None)
    allp = allp.sort_values(["date","ticker"]).reset_index(drop=True)
    allp.to_parquet(a.out_parquet, index=False)
    print(f"[OK] Parquet salvo: {a.out_parquet} | rows={len(allp):,}")

    # SQLite incremental
    con = sqlite3.connect(a.sqlite_path)
    allp.to_sql(a.table, con, if_exists="append", index=False)
    con.close()
    print(f"[OK] SQLite append -> {a.sqlite_path}:{a.table}")

if __name__ == "__main__":
    main()
