"""
Simulador de backtest baseado em sinais discretos (+1, 0, -1).

- Execução next day: posição do dia t = sinal do dia t-1.
- Retorno diário = posição_t * retorno_mkt_t - custos_em_mudança_de_posição.
- Custos modelados via bps (ex.: fee_bps=5 e slippage_bps=2).

Entradas:
- data/processed/prices.parquet  (colunas: date, ticker, adjclose, ...)
- data/signals/signals.parquet   (colunas: date, ticker, signal_reg/signal_cls)

Saídas:
- data/backtests/equity.parquet      (date, ret_port, equity)
- data/backtests/positions.parquet   (date, ticker, position)
- data/backtests/summary.json        (métricas)

Uso:
    python -m src.backtest.simulator --signal-col signal_cls --fee-bps 5 --slippage-bps 2
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from src import config
from src.utils.io import load_parquet, save_parquet

app = typer.Typer(add_completion=False, help="Backtest simulator (equally-weighted portfolio)")


@dataclass
class Costs:
    fee_bps: float = 5.0       # custo por lado (ex.: corretagem + taxas), em bps
    slippage_bps: float = 2.0  # deslizamento efetivo por mudança, em bps

    @property
    def total_bps(self) -> float:
        return self.fee_bps + self.slippage_bps

    @property
    def total_rate(self) -> float:
        return self.total_bps / 10_000.0


def compute_metrics(equity: pd.Series, ret_daily: pd.Series, trading_days: int = 252) -> dict:
    """
    Calcula métricas padrão de desempenho.
    """
    eq = equity.dropna()
    r = ret_daily.dropna()

    if eq.empty or r.empty:
        return {
            "CAGR": None,
            "Sharpe": None,
            "Vol_Ann": None,
            "MaxDrawdown": None,
            "N_Days": int(len(r)),
            "Start": None,
            "End": None,
        }

    n_days = len(r)
    start_val, end_val = float(eq.iloc[0]), float(eq.iloc[-1])
    years = max(n_days / trading_days, 1e-9)
    cagr = (end_val / start_val) ** (1 / years) - 1 if start_val > 0 else None

    mu = r.mean()
    sigma = r.std(ddof=0)
    vol_ann = sigma * np.sqrt(trading_days) if sigma is not None else None
    sharpe = (mu / sigma) * np.sqrt(trading_days) if sigma and sigma > 0 else None

    roll_max = eq.cummax()
    drawdown = eq / roll_max - 1.0
    maxdd = drawdown.min()

    return {
        "CAGR": None if cagr is None else float(cagr),
        "Sharpe": None if sharpe is None else float(sharpe),
        "Vol_Ann": None if vol_ann is None else float(vol_ann),
        "MaxDrawdown": None if maxdd is None else float(maxdd),
        "N_Days": int(n_days),
        "Start": str(eq.index.min().date()) if hasattr(eq.index.min(), "date") else None,
        "End": str(eq.index.max().date()) if hasattr(eq.index.max(), "date") else None,
    }


def prepare_inputs(signal_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices_path = config.PROCESSED_DIR / "prices.parquet"
    signals_path = config.SIGNALS_DIR / "signals.parquet"

    prices = load_parquet(prices_path)
    if prices is None or prices.empty:
        raise typer.Exit(f"[backtest] Arquivo de preços inexistente: {prices_path}")

    signals = load_parquet(signals_path)
    if signals is None or signals.empty:
        raise typer.Exit(f"[backtest] Arquivo de sinais inexistente: {signals_path}")

    if signal_col not in signals.columns:
        available = [c for c in signals.columns if c.startswith("signal_")]
        raise typer.Exit(f"[backtest] Coluna '{signal_col}' não encontrada. Disponíveis: {available}")

    prices = prices[["date", "ticker", "adjclose"]].copy()
    signals = signals[["date", "ticker", signal_col]].copy().rename(columns={signal_col: "signal"})

    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    signals["date"] = pd.to_datetime(signals["date"]).dt.tz_localize(None)
    prices = prices.sort_values(["ticker", "date"])
    signals = signals.sort_values(["ticker", "date"])

    return prices, signals


def build_positions(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói posições com execução next day: position_t = signal_{t-1}.
    """
    df = signals.copy()
    df["position"] = df.groupby("ticker")["signal"].shift(1)
    df["position"] = df["position"].fillna(0).astype(int)
    return df


def backtest(prices: pd.DataFrame, positions: pd.DataFrame, costs: Costs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Realiza o backtest por ticker e agrega em carteira equally-weighted.
    Retorna:
      - positions_out: DataFrame com date, ticker, position, ret_mkt, ret_after_cost
      - equity: DataFrame com date, ret_port, equity
    """
    df = pd.merge(prices, positions[["date", "ticker", "position"]], on=["date", "ticker"], how="inner")

    df["ret_mkt"] = df.groupby("ticker")["adjclose"].pct_change().fillna(0.0)

    df["position_prev"] = df.groupby("ticker")["position"].shift(1).fillna(0)
    df["turnover_units"] = (df["position"] - df["position_prev"]).abs()

    df["ret_gross"] = df["position"] * df["ret_mkt"]

    df["cost"] = df["turnover_units"] * costs.total_rate

    df["ret_net"] = df["ret_gross"] - df["cost"]

    daily = (
        df.groupby("date")
        .agg(ret_port=("ret_net", "mean"))
        .reset_index()
        .sort_values("date")
    )
    daily["equity"] = (1.0 + daily["ret_port"]).cumprod()

    positions_out = df[["date", "ticker", "position", "ret_mkt", "ret_gross", "cost", "ret_net"]].copy()
    equity = daily[["date", "ret_port", "equity"]].copy()

    equity.set_index("date", inplace=True)
    return positions_out, equity


@app.command("run")
def run(
    signal_col: str = typer.Option(
        "signal_cls",
        "--signal-col",
        "-s",
        help="Coluna de sinal a usar (ex.: signal_cls, signal_reg).",
    ),
    fee_bps: float = typer.Option(5.0, "--fee-bps", help="Custos em bps por mudança (fee)."),
    slippage_bps: float = typer.Option(2.0, "--slippage-bps", help="Slippage em bps por mudança."),
):
    """
    Executa o backtest e salva resultados e métricas.
    """
    typer.secho(
        f"[backtest] Iniciando com signal_col={signal_col}, fee_bps={fee_bps}, slippage_bps={slippage_bps}",
        fg=typer.colors.CYAN,
    )

    prices, signals = prepare_inputs(signal_col=signal_col)
    positions = build_positions(signals)

    results_dir = config.BACKTESTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    positions_out, equity = backtest(prices, positions, Costs(fee_bps=fee_bps, slippage_bps=slippage_bps))

    save_parquet(positions_out, results_dir / "positions.parquet", index=False)
    equity_to_save = equity.reset_index()
    save_parquet(equity_to_save, results_dir / "equity.parquet", index=False)

    summary = compute_metrics(
        equity=equity["equity"],
        ret_daily=equity["ret_port"],
        trading_days=252,
    )
    summary_path = results_dir / "summary.json"
    summary_json = {
        "signal_col": signal_col,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "metrics": summary,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)
    typer.secho(f"[backtest] Métricas salvas em {summary_path}", fg=typer.colors.GREEN)

    typer.secho(
        f"[backtest] CAGR={summary.get('CAGR'):.4f} | Sharpe={summary.get('Sharpe'):.2f} | "
        f"VolAnn={summary.get('Vol_Ann'):.2f} | MaxDD={summary.get('MaxDrawdown'):.2f}",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
