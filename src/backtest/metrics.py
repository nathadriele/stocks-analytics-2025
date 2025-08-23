"""
Métricas de desempenho para backtests e estratégias de trading.

Pode ser usado isoladamente em notebooks ou integrado a src/backtest/simulator.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cagr(equity: pd.Series, trading_days: int = 252) -> float | None:
    """
    Calcula CAGR (Compound Annual Growth Rate).
    """
    eq = equity.dropna()
    if eq.empty:
        return None
    n_days = len(eq)
    years = n_days / trading_days
    start, end = float(eq.iloc[0]), float(eq.iloc[-1])
    if start <= 0:
        return None
    return (end / start) ** (1 / years) - 1


def sharpe_ratio(returns: pd.Series, trading_days: int = 252, rf: float = 0.0) -> float | None:
    """
    Calcula Sharpe Ratio anualizado.
    rf = taxa livre de risco (diária).
    """
    r = returns.dropna()
    if r.empty:
        return None
    excess = r - rf
    mu = excess.mean()
    sigma = excess.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        return None
    return (mu / sigma) * np.sqrt(trading_days)


def sortino_ratio(returns: pd.Series, trading_days: int = 252, rf: float = 0.0) -> float | None:
    """
    Calcula Sortino Ratio anualizado (penaliza apenas retornos negativos).
    """
    r = returns.dropna()
    if r.empty:
        return None
    downside = r[r < 0]
    if downside.std(ddof=0) == 0 or np.isnan(downside.std(ddof=0)):
        return None
    mu = (r - rf).mean()
    sigma_down = downside.std(ddof=0)
    return (mu / sigma_down) * np.sqrt(trading_days)


def volatility(returns: pd.Series, trading_days: int = 252) -> float | None:
    """
    Volatilidade anualizada.
    """
    r = returns.dropna()
    if r.empty:
        return None
    return r.std(ddof=0) * np.sqrt(trading_days)


def max_drawdown(equity: pd.Series) -> float | None:
    """
    Máximo drawdown (equity / rolling max - 1).
    """
    eq = equity.dropna()
    if eq.empty:
        return None
    roll_max = eq.cummax()
    drawdown = eq / roll_max - 1.0
    return float(drawdown.min())


def hit_ratio(returns: pd.Series) -> float | None:
    """
    Percentual de trades/retornos positivos (Win Rate).
    """
    r = returns.dropna()
    if r.empty:
        return None
    return (r > 0).sum() / len(r)


def trade_statistics(returns: pd.Series) -> dict:
    """
    Estatísticas adicionais para análise de trades.
    """
    r = returns.dropna()
    if r.empty:
        return {}
    return {
        "mean": float(r.mean()),
        "median": float(r.median()),
        "std": float(r.std(ddof=0)),
        "skew": float(r.skew()),
        "kurtosis": float(r.kurtosis()),
        "min": float(r.min()),
        "max": float(r.max()),
    }
