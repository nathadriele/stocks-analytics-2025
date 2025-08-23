"""
Geração de features para os dados de mercado.

- Lê dados de preços ajustados em Parquet (produzidos por src.data.ingest).
- Calcula retornos em múltiplos períodos.
- Adiciona indicadores técnicos (ex.: SMA, EMA, RSI, MACD, Bollinger Bands).
- Gera variáveis de calendário (dia da semana, mês).
- Salva dataset de features em Parquet e, opcionalmente, em SQLite.

Uso:
    python -m src.features.build_features
"""

from __future__ import annotations

import typer
import pandas as pd
import numpy as np
import ta  # biblioteca técnica (baseada no pandas)

from src import config
from src.utils.io import load_parquet, save_parquet, save_to_db

app = typer.Typer(add_completion=False, help="Construção de features de mercado")


def generate_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Gera features a partir dos preços ajustados.
    Espera colunas: ['date', 'open', 'high', 'low', 'close', 'adjclose', 'volume', 'ticker']
    """
    df = prices.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    df["ret_1d"] = df.groupby("ticker")["adjclose"].pct_change(1)
    df["ret_5d"] = df.groupby("ticker")["adjclose"].pct_change(5)
    df["ret_21d"] = df.groupby("ticker")["adjclose"].pct_change(21)

    df["vol_21d"] = df.groupby("ticker")["ret_1d"].rolling(21).std().reset_index(level=0, drop=True)

    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # Indicadores técnicos (via ta)
    def add_ta_features(sub):
        sub = sub.copy()
        sub["sma_20"] = ta.trend.sma_indicator(sub["adjclose"], window=20)
        sub["ema_20"] = ta.trend.ema_indicator(sub["adjclose"], window=20)

        sub["rsi_14"] = ta.momentum.rsi(sub["adjclose"], window=14)

        macd = ta.trend.MACD(sub["adjclose"])
        sub["macd"] = macd.macd()
        sub["macd_signal"] = macd.macd_signal()

        bb = ta.volatility.BollingerBands(sub["adjclose"], window=20, window_dev=2)
        sub["bb_high"] = bb.bollinger_hband()
        sub["bb_low"] = bb.bollinger_lband()

        return sub

    df = df.groupby("ticker", group_keys=False).apply(add_ta_features)

    df["target_reg_5d"] = df.groupby("ticker")["adjclose"].pct_change(5).shift(-5)
    df["target_cls_5d"] = (df["target_reg_5d"] > 0).astype(int)

    return df


@app.command("run")
def run(to_db: bool = typer.Option(False, "--to-db", help="Persistir também em SQLite")):
    """
    Executa geração de features a partir de prices.parquet.
    """
    prices_path = config.PROCESSED_DIR / "prices.parquet"
    features_path = config.ANALYTICS_DIR / "features.parquet"

    prices = load_parquet(prices_path)
    if prices is None or prices.empty:
        typer.secho(f"[features] Arquivo de preços não encontrado em {prices_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.secho(f"[features] Gerando features para {len(prices['ticker'].unique())} tickers...", fg=typer.colors.CYAN)
    features = generate_features(prices)

    save_parquet(features, features_path, index=False)

    if to_db:
        save_to_db(features, table_name="features", if_exists="replace")

    typer.secho(f"[features] OK: {len(features)} linhas | arquivo: {features_path}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
