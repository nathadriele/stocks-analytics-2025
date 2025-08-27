# src/backtest/vector_backtester.py
from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd


# =======================
# ======= Utils =========
# =======================

def _load_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("Formato não suportado. Use .parquet ou .csv")


def _pivot_prices(df_prices: pd.DataFrame, price_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (preços, retornos) pivotados por [date x ticker]."""
    req = {"date", "ticker", price_col}
    missing = req - set(df_prices.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em preços: {missing}")
    px = df_prices[["date", "ticker", price_col]].copy()
    px["date"] = pd.to_datetime(px["date"])
    wide = px.pivot_table(index="date", columns="ticker", values=price_col, aggfunc="last").sort_index()
    rets = wide.pct_change().fillna(0.0)
    return wide, rets


def _align_signals(signals: pd.DataFrame) -> pd.DataFrame:
    req = {"date", "ticker", "signal", "weight"}
    missing = req - set(signals.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em sinais: {missing}")
    out = signals.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["signal"] = out["signal"].astype(float)
    out["weight"] = out["weight"].astype(float)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def _rebalance_cost_delta_w(w_prev: np.ndarray, w_tgt: np.ndarray) -> Tuple[float, np.ndarray]:
    """Cálculo de turnover simples: notional transacionado ~ sum(|delta_w|).
    Retorna (turnover, delta_w)."""
    delta = w_tgt - (w_prev if w_prev is not None else np.zeros_like(w_tgt))
    turnover = float(np.nansum(np.abs(delta)))
    return turnover, delta


def _max_drawdown(equity: pd.Series) -> Tuple[float, float, float]:
    """Max Drawdown absoluto: retorna (max_dd, peak_value, trough_value)."""
    cummax = equity.cummax()
    dd = (equity / cummax) - 1.0
    mdd = dd.min() if len(dd) else 0.0
    # valores de pico e vale (para referência)
    idx_trough = dd.idxmin() if len(dd) else None
    if idx_trough is None:
        return 0.0, float(equity.iloc[0]), float(equity.iloc[-1])
    idx_peak = equity.loc[:idx_trough].idxmax()
    return float(mdd), float(equity.loc[idx_peak]), float(equity.loc[idx_trough])


def _annualize_ratio(daily_ret: pd.Series, rf: float = 0.0) -> Dict[str, float]:
    """Métricas anuais padrão: CAGR, vol, Sharpe, Sortino, winrate."""
    if len(daily_ret) < 2:
        return {"cagr": 0.0, "vol_ann": 0.0, "sharpe": 0.0, "sortino": 0.0, "winrate": 0.0}
    n_years = len(daily_ret) / 252.0
    cum = float((1.0 + daily_ret).prod())
    cagr = cum ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0
    vol_ann = float(daily_ret.std(ddof=0)) * np.sqrt(252.0)
    mean_daily = float(daily_ret.mean())
    sharpe = ((mean_daily - rf / 252.0) / (daily_ret.std(ddof=0) + 1e-12)) * np.sqrt(252.0)
    downside = daily_ret.copy()
    downside[downside > 0] = 0
    down_std = float(np.sqrt((downside ** 2).mean())) * np.sqrt(252.0)
    sortino = ((mean_daily - rf / 252.0) / (down_std + 1e-12)) * np.sqrt(252.0)
    winrate = float((daily_ret > 0).mean())
    return {
        "cagr": float(cagr),
        "vol_ann": float(vol_ann),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "winrate": float(winrate),
    }


def _buyhold_benchmark(px: pd.DataFrame, ticker: Optional[str], start_equity: float) -> pd.Series:
    """Benchmark simples:
    - se ticker fornecido: buy&hold desse ticker (coluna deve existir no pivot).
    - senão: equal-weight estático do universo disponível no primeiro dia."""
    if ticker is not None:
        if ticker not in px.columns:
            raise ValueError(f"Ticker de benchmark '{ticker}' não encontrado em preços.")
        bench = px[ticker].ffill().bfill()
        ret = bench.pct_change().fillna(0.0)
        eq = (1.0 + ret).cumprod() * start_equity
        return eq
    # equal-weight estático:
    first = px.iloc[0].dropna()
    cols = first.index.tolist()
    w0 = np.array([1.0 / len(cols)] * len(cols))
    rets = px[cols].pct_change().fillna(0.0)
    port = (rets * w0).sum(axis=1)
    eq = (1.0 + port).cumprod() * start_equity
    return eq


# ==================================
# ========== Backtester =============
# ==================================

def backtest_vector(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    price_col_exec: str = "close",
    initial_capital: float = 100_000.0,
    cost_bps: float = 5.0,
    slippage_pct: float = 0.0,
    normalize_weights: bool = True,
    max_gross_leverage: float = 1.0,
    benchmark_ticker: Optional[str] = None,
) -> Dict[str, object]:
    """
    Backtest vetorial com rebalance diário para pesos-alvo vindos de 'signals'.
    - prices: DataFrame long com date,ticker,close (e opcional open) — já lido.
    - signals: DataFrame com date,ticker,signal,weight (lag aplicado previamente).
    - price_col_exec: coluna usada para preço de execução e cálculo dos retornos (close por padrão).
    - cost_bps: custo por lado em basis points sobre o notional transacionado (ex.: 5 = 0.05%).
    - slippage_pct: slippage percentual aplicado ao notional transacionado (ex.: 0.10 = 0.10%).
    - normalize_weights: normaliza por data para respeitar max_gross_leverage.
    - max_gross_leverage: limite para soma(|weights|) por data (1.0 = 100% do capital).
    """
    # Pivot de preços e retornos
    px_wide, rets = _pivot_prices(prices, price_col_exec)

    # Tabela de pesos alvo
    sig = _align_signals(signals)
    # pivotando pesos (aceita negativos)
    w_tgt = sig.pivot_table(index="date", columns="ticker", values="weight", aggfunc="last").reindex(px_wide.index).fillna(0.0)

    # normalização por gross leverage
    if normalize_weights:
        gross = np.abs(w_tgt).sum(axis=1)
        scale = (max_gross_leverage / gross).clip(upper=1.0)  # se excede, escala p/ baixo
        w_tgt = (w_tgt.T * scale.replace([np.inf, -np.inf], 1.0).fillna(1.0)).T

    # inicializações
    dates = px_wide.index
    tickers = px_wide.columns
    nT = len(dates)
    nN = len(tickers)

    w_prev = np.zeros(nN)  # pesos do dia anterior
    equity = np.zeros(nT)
    daily_ret = np.zeros(nT)
    daily_cost = np.zeros(nT)
    daily_turnover = np.zeros(nT)
    trades_count = np.zeros(nT)

    equity[0] = initial_capital

    # loop por data (vetorial nas operações de coluna, mas iteramos por tempo para custo e pesos)
    for t in range(1, nT):
        # pesos alvo no dia t-1 (efeito do rebalance no fim do dia t-1 para capturar retorno de t)
        w_target = w_tgt.iloc[t - 1].values.copy()

        # turnover e trades (delta em relação a w_prev)
        turnover_t, delta_w = _rebalance_cost_delta_w(w_prev, w_target)
        daily_turnover[t] = turnover_t
        trades_count[t] = np.count_nonzero(np.abs(delta_w) > 1e-12)

        # custo por notional transacionado
        cost_rate = (cost_bps / 10_000.0) + float(slippage_pct)
        cost_cash = equity[t - 1] * turnover_t * cost_rate  # simples: |delta_w| somado * equity
        daily_cost[t] = cost_cash / max(equity[t - 1], 1e-12)

        # retorno do portfólio em t: pesos (aplicados no final de t-1) * retornos de t
        r_t = rets.iloc[t].values  # vetor de retornos por ativo no dia t
        port_ret_gross = float(np.nansum(w_target * r_t))

        # retorno líquido após custo
        port_ret_net = port_ret_gross - daily_cost[t]
        daily_ret[t] = port_ret_net
        equity[t] = equity[t - 1] * (1.0 + port_ret_net)

        # atualiza pesos "anteriores" para próxima data = w_target ajustado pela variação do dia t
        # (aproximação: mantemos plano; rebalan. só no fim do dia)
        w_prev = w_target

    # métricas de risco/retorno
    equity_series = pd.Series(equity, index=dates, name="equity")
    ret_series = pd.Series(daily_ret, index=dates, name="ret")
    cost_series = pd.Series(daily_cost, index=dates, name="cost")
    turnover_series = pd.Series(daily_turnover, index=dates, name="turnover")
    trades_series = pd.Series(trades_count, index=dates, name="trades")

    ann = _annualize_ratio(ret_series[1:])  # ignora o primeiro (zero)
    mdd, peak_v, trough_v = _max_drawdown(equity_series)

    # benchmark
    bench_eq = _buyhold_benchmark(px_wide, benchmark_ticker, initial_capital)
    bench_ret = bench_eq.pct_change().fillna(0.0)
    bench_ann = _annualize_ratio(bench_ret[1:])
    bench_mdd, _, _ = _max_drawdown(bench_eq)

    summary = {
        "initial_capital": float(initial_capital),
        "final_equity": float(equity_series.iloc[-1]),
        "tot_return": float(equity_series.iloc[-1] / equity_series.iloc[0] - 1.0),
        "metrics": {
            "cagr": ann["cagr"],
            "vol_ann": ann["vol_ann"],
            "sharpe": ann["sharpe"],
            "sortino": ann["sortino"],
            "max_drawdown": float(mdd),
            "winrate": ann["winrate"],
            "avg_turnover": float(turnover_series.mean()),
            "avg_trades_per_day": float(trades_series.mean()),
            "avg_cost_daily": float(cost_series.mean()),
        },
        "benchmark": {
            "final_equity": float(bench_eq.iloc[-1]),
            "cagr": bench_ann["cagr"],
            "vol_ann": bench_ann["vol_ann"],
            "sharpe": bench_ann["sharpe"],
            "sortino": bench_ann["sortino"],
            "max_drawdown": float(bench_mdd),
        },
    }

    out = pd.DataFrame({
        "equity": equity_series,
        "ret": ret_series,
        "cost": cost_series,
        "turnover": turnover_series,
        "trades": trades_series,
        "bench_equity": bench_eq,
    })

    return {"summary": summary, "timeseries": out}


# ==================================
# ============ CLI =================
# ==================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest vetorial de estratégias baseadas em sinais.")
    p.add_argument("--prices-path", required=True, help="Arquivo de preços (parquet/csv) com colunas: date,ticker,close[,open].")
    p.add_argument("--signals-path", required=True, help="Arquivo de sinais (parquet/csv) gerado em data/signals/ (Arquivo 3).")
    p.add_argument("--price-col-exec", default="close", help="Coluna de preço para execução e retornos (default: close).")
    p.add_argument("--initial-capital", type=float, default=100000.0, help="Capital inicial (default: 100k).")
    p.add_argument("--cost-bps", type=float, default=5.0, help="Custo por lado em bps (default: 5 = 0.05%).")
    p.add_argument("--slippage-pct", type=float, default=0.0, help="Slippage percentual adicional (ex.: 0.10 = 0.10%).")
    p.add_argument("--no-normalize", action="store_true", help="Não normalizar pesos por gross leverage.")
    p.add_argument("--max-gross-leverage", type=float, default=1.0, help="Limite para soma(|weights|) por data (default: 1.0).")
    p.add_argument("--benchmark-ticker", default=None, help="Ticker de benchmark (ex.: SPY). Se ausente, usa equal-weight estático.")
    p.add_argument("--out-dir", default="data/backtests", help="Diretório de saída (equity/summary).")
    p.add_argument("--strategy-name", default=None, help="Nome para os arquivos de saída; se ausente, tenta ler de 'signals'.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    prices = _load_table(args.prices_path)
    signals = _load_table(args.signals_path)

    strat = args.strategy_name
    if strat is None:
        if "strategy" in signals.columns:
            strat = str(signals["strategy"].iloc[0])
        else:
            strat = "strategy"

    res = backtest_vector(
        prices=prices,
        signals=signals,
        price_col_exec=args.price_col_exec,
        initial_capital=args.initial_capital,
        cost_bps=args.cost_bps,
        slippage_pct=args.slippage_pct,
        normalize_weights=not args.no_normalize,
        max_gross_leverage=args.max_gross_leverage,
        benchmark_ticker=(None if args.benchmark_ticker in [None, "None", ""] else args.benchmark_ticker),
    )

    eq_path_parquet = os.path.join(args.out_dir, f"equity_{strat}.parquet")
    eq_path_csv = os.path.join(args.out_dir, f"equity_{strat}.csv")
    res["timeseries"].to_parquet(eq_path_parquet, index=True)
    res["timeseries"].to_csv(eq_path_csv, index=True)

    summ = res["summary"]
    summ["params"] = {
        "prices_path": args.prices_path,
        "signals_path": args.signals_path,
        "price_col_exec": args.price_col_exec,
        "initial_capital": args.initial_capital,
        "cost_bps": args.cost_bps,
        "slippage_pct": args.slippage_pct,
        "normalize_weights": not args.no_normalize,
        "max_gross_leverage": args.max_gross_leverage,
        "benchmark_ticker": args.benchmark_ticker,
    }
    with open(os.path.join(args.out_dir, f"summary_{strat}.json"), "w") as f:
        json.dump(summ, f, indent=2)

    print(f"[OK] Backtest concluído: {eq_path_parquet}")
    print(f"     Final equity: {summ['final_equity']:.2f} | CAGR: {summ['metrics']['cagr']:.2%} | Sharpe: {summ['metrics']['sharpe']:.2f}")
    print(f"     Benchmark CAGR: {summ['benchmark']['cagr']:.2%}")


if __name__ == "__main__":
    main()
