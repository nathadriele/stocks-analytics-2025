# src/data/alt_provider.py
from __future__ import annotations

import os
from typing import Iterable, List, Optional
import pandas as pd
import numpy as np

def _normalize(df: pd.DataFrame, ticker: str, price_cols_map: dict) -> pd.DataFrame:
    df = df.rename(columns=price_cols_map)
    required = ["date", "open", "high", "low", "close", "volume"]
    for c in required:
        if c not in df.columns:
            df[c] = np.nan
    out = df[required].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out["ticker"] = ticker.upper()
    return out.sort_values("date")

def fetch_stooq(ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """
    Baixa dados do Stooq via pandas_datareader.
    Requer: pip install pandas_datareader
    """
    from pandas_datareader import data as pdr
    df = pdr.DataReader(ticker, "stooq", start=start, end=end).reset_index()
    # Stooq retorna colunas: Date, Open, High, Low, Close, Volume
    return _normalize(df, ticker, {"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})

def fetch_tiingo(ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """
    Baixa dados do Tiingo via pandas_datareader (necessita TIINGO_API_KEY no ambiente).
    Requer: pip install pandas_datareader
    """
    from pandas_datareader import data as pdr
    api_key = os.getenv("TIINGO_API_KEY")
    if not api_key:
        raise EnvironmentError("Defina TIINGO_API_KEY no ambiente para usar Tiingo.")
    df = pdr.DataReader(ticker, "tiingo", start=start, end=end, api_key=api_key).reset_index()
    # Tiingo: date, open, high, low, close, volume
    return _normalize(df, ticker, {"date": "date", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})

def get_prices(provider: str, tickers: Iterable[str], start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """
    provider: 'stooq' (livre) ou 'tiingo' (chave).
    Retorna DataFrame long: date, ticker, open, high, low, close, volume
    """
    out: List[pd.DataFrame] = []
    for t in tickers:
        t = t.strip()
        if not t:
            continue
        if provider.lower() == "stooq":
            df = fetch_stooq(t, start, end)
        elif provider.lower() == "tiingo":
            df = fetch_tiingo(t, start, end)
        else:
            raise ValueError("Provider inválido. Use 'stooq' ou 'tiingo'.")
        out.append(df)
    if not out:
        return pd.DataFrame(columns=["date","ticker","open","high","low","close","volume"])
    return pd.concat(out, axis=0, ignore_index=True)
