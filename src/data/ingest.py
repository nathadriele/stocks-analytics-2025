"""
Ingestão de dados de mercado (OHLCV) via yfinance.

- Lê configurações de src.config (.env).
- Faz download histórico diário para os tickers definidos.
- Persiste em Parquet (data/processed/prices.parquet) e, opcionalmente, em SQLite (tabela 'prices').
- Suporta atualização incremental (anexa novas datas sem duplicar).
- Disponibiliza CLI com Typer.

Uso:
    python -m src.data.ingest
    python -m src.data.ingest --tickers AAPL MSFT SPY --start 2015-01-01 --to-db
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer
import yfinance as yf

from src import config
from src.utils.io import save_parquet, load_parquet, save_to_db

app = typer.Typer(add_completion=False, help="Ingestão de dados de mercado (yfinance)")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nomes de colunas e garante presença de 'adjclose'.
    """
    df = df.copy()
    df.columns = [str(c).replace(" ", "_").lower() for c in df.columns]
    if "adj_close" in df.columns and "adjclose" not in df.columns:
        df = df.rename(columns={"adj_close": "adjclose"})
    return df


def _to_long(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Converte o multiindex de yfinance para formato long com coluna 'ticker'.
    """
    frames = []
    for t in tickers:
        if t in df:
            sub = df[t].copy()
            sub = _normalize_columns(sub)
            sub["ticker"] = t
            frames.append(sub.reset_index().rename(columns={"Date": "date"}))
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True)
    if "date" in full.columns:
        full["date"] = pd.to_datetime(full["date"]).dt.tz_localize(None)
    full = full.sort_values(["ticker", "date"]).reset_index(drop=True)
    return full


def _download_yf(
    tickers: List[str],
    start: str,
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Baixa dados via yfinance no formato long.
    """
    if not tickers:
        return pd.DataFrame()

    dl = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto
