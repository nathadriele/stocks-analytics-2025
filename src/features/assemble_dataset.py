# src/features/assemble_dataset.py
from __future__ import annotations
import argparse, os, json
import numpy as np, pandas as pd

def parse_args():
    p = argparse.ArgumentParser(description="Assemble: cria dataset unificado e salva feature sets.")
    p.add_argument("--input-path", default="data/analytics/features.parquet", help="Se existir, usa; senão tenta criar a partir de prices.")
    p.add_argument("--prices-path", default="data/processed/prices.parquet", help="Fallback para gerar features mínimas.")
    p.add_argument("--out-path", default="data/analytics/features.parquet")
    p.add_argument("--feature-list", default="src/features/feature_list.txt")
    p.add_argument("--date-col", default="date")
    p.add_argument("--ticker-col", default="ticker")
    return p.parse_args()

def _read_df(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

def _write_df(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".parquet"): df.to_parquet(path, index=False)
    else: df.to_csv(path, index=False)

def _read_features(path: str):
    if not os.path.exists(path): return None
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

def _basic_features_from_prices(prices: pd.DataFrame, date_col: str, ticker_col: str) -> pd.DataFrame:
    df = prices[[date_col, ticker_col, "close"]].sort_values([ticker_col, date_col]).copy()
    df["ret_1d"] = df.groupby(ticker_col)["close"].pct_change()
    df["ret_5d"] = df.groupby(ticker_col)["close"].pct_change(5)
    df["ret_21d"] = df.groupby(ticker_col)["close"].pct_change(21)
    df["target_up_5d"] = (df.groupby(ticker_col)["close"].shift(-5) / df["close"] - 1 > 0).astype(int)
    df["dow"] = pd.to_datetime(df[date_col]).dt.dayofweek
    df["month"] = pd.to_datetime(df[date_col]).dt.month
    return df.dropna().reset_index(drop=True)

def main():
    a = parse_args()
    if os.path.exists(a.input_path):
        df = _read_df(a.input_path)
    else:
        prices = _read_df(a.prices_path)
        df = _basic_features_from_prices(prices, a.date_col, a.ticker_col)

    features = _read_features(a.feature_list)
    # Define conjuntos
    to_predict = ["target_up_5d"] if "target_up_5d" in df.columns else (["target_up"] if "target_up" in df.columns else [])
    numeric = [c for c in (features or df.columns.tolist())
               if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and c not in to_predict]
    dummies = []  # se já tiver one-hot no dataset, incluir aqui

    # Ordena e salva
    cols = [a.date_col, a.ticker_col] + numeric + to_predict
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values([a.date_col, a.ticker_col]).dropna().reset_index(drop=True)
    _write_df(df, a.out_path)

    # metadados
    meta = {
        "TO_PREDICT": to_predict,
        "NUMERIC": numeric,
        "DUMMIES": dummies,
        "rows": len(df),
        "cols": df.shape[1],
        "path": a.out_path,
    }
    with open("data/analytics/_feature_sets.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Dataset unificado em {a.out_path} | rows={len(df):,}")

if __name__ == "__main__":
    main()
