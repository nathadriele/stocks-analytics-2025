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
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if dl.empty:
        return pd.DataFrame()
    return _to_long(dl, tickers)


def _merge_incremental(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """
    Faz merge incremental removendo duplicatas (ticker, date).
    """
    if existing is None or existing.empty:
        return new.sort_values(["ticker", "date"]).reset_index(drop=True)
    if new is None or new.empty:
        return existing.sort_values(["ticker", "date"]).reset_index(drop=True)

    cols = list(set(existing.columns).union(set(new.columns)))
    existing = existing.reindex(columns=cols)
    new = new.reindex(columns=cols)

    merged = pd.concat([existing, new], ignore_index=True)
    merged = merged.drop_duplicates(subset=["ticker", "date"]).sort_values(
        ["ticker", "date"]
    )
    return merged.reset_index(drop=True)


def _infer_incremental_start(existing: pd.DataFrame, default_start: str) -> str:
    """
    Define a data inicial para atualização incremental:
    - Se já existe histórico, começa no dia seguinte à última data.
    - Caso contrário, usa default_start.
    """
    if existing is None or existing.empty or "date" not in existing.columns:
        return default_start
    last_dt = pd.to_datetime(existing["date"]).max()
    return (last_dt + timedelta(days=1)).strftime("%Y-%m-%d")


@app.command("run")
def run(
    tickers: List[str] = typer.Option(
        None,
        "--tickers",
        "-t",
        help="Lista de tickers (se omitido, usa TICKERS do .env)",
    ),
    start: str = typer.Option(
        None,
        "--start",
        "-s",
        help="Data inicial YYYY-MM-DD (se omitido, usa DATA_START do .env ou incremental)",
    ),
    end: Optional[str] = typer.Option(
        None, "--end", "-e", help="Data final YYYY-MM-DD (opcional)"
    ),
    interval: str = typer.Option(
        "1d",
        "--interval",
        "-i",
        help="Intervalo do yfinance (ex.: 1d, 1wk, 1mo). Padrão: 1d",
    ),
    to_db: bool = typer.Option(
        False,
        "--to-db",
        help="Se presente, persiste também no SQLite (tabela 'prices')",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Se presente, ignora incremental e reescreve todo o histórico",
    ),
):
    """
    Executa a ingestão de dados:
    - Baixa histórico via yfinance
    - Atualiza parquet incrementalmente (ou sobrescreve com --overwrite)
    - Opcionalmente escreve no SQLite
    """
    prices_path = config.PROCESSED_DIR / "prices.parquet"

    if tickers is None or len(tickers) == 0:
        tickers = config.TICKERS
    if start is None:
        start = config.DATA_START

    if not tickers:
        typer.secho("Nenhum ticker informado. Abortando.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.secho(
        f"[ingest] Tickers={tickers} | Interval={interval}", fg=typer.colors.CYAN
    )

    existing = None if overwrite else load_parquet(prices_path)

    if not overwrite:
        start = _infer_incremental_start(existing, start)
        typer.secho(f"[ingest] Modo incremental: start={start}", fg=typer.colors.YELLOW)
    else:
        typer.secho("[ingest] Modo overwrite: baixando histórico completo", fg=typer.colors.YELLOW)

    df_new = _download_yf(tickers, start=start, end=end, interval=interval)
    if df_new.empty:
        typer.secho("[ingest] Nenhum dado novo disponível.", fg=typer.colors.YELLOW)
        if existing is not None and not existing.empty:
            save_parquet(existing, prices_path, index=False)
            if to_db:
                save_to_db(existing, table_name="prices", if_exists="replace")
        raise typer.Exit()

    df_all = _merge_incremental(existing, df_new)

    df_all["date"] = pd.to_datetime(df_all["date"]).dt.tz_localize(None)
    df_all = df_all.sort_values(["ticker", "date"]).reset_index(drop=True)

    save_parquet(df_all, prices_path, index=False)

    if to_db:
        save_to_db(df_all, table_name="prices", if_exists="replace")

    nrows = len(df_all)
    first = df_all["date"].min()
    last = df_all["date"].max()
    typer.secho(
        f"[ingest] OK: {nrows} linhas | período: {first.date()} → {last.date()} | arquivo: {prices_path}",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
