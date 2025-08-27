# src/backtest/exact_simulator.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


def _pivot_prices(df_prices: pd.DataFrame, price_col: str) -> pd.DataFrame:
    req = {"date", "ticker", price_col}
    missing = req - set(df_prices.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em preços: {missing}")
    px = df_prices[["date", "ticker", price_col]].copy()
    px["date"] = pd.to_datetime(px["date"])
    wide = px.pivot_table(index="date", columns="ticker", values=price_col, aggfunc="last").sort_index()
    return wide


def _align_signals(signals: pd.DataFrame) -> pd.DataFrame:
    req = {"date", "ticker", "signal", "weight", "proba_up"}
    missing = req - set(signals.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em sinais: {missing}")
    out = signals.copy()
    out["date"] = pd.to_datetime(out["date"])
    # normaliza campos essenciais
    out["signal"] = out["signal"].astype(float)
    out["weight"] = out["weight"].astype(float)
    out["proba_up"] = out["proba_up"].astype(float)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def _annual_metrics(daily_ret: pd.Series, rf: float = 0.0) -> Dict[str, float]:
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
    sortino = ((mean_daily - rf / 252.0) / (np.sqrt((downside**2).mean()) + 1e-12)) * np.sqrt(252.0)
    winrate = float((daily_ret > 0).mean())
    return {"cagr": cagr, "vol_ann": vol_ann, "sharpe": sharpe, "sortino": sortino, "winrate": winrate}


@dataclass
class Params:
    price_col: str = "close"
    initial_capital: float = 100_000.0
    max_positions: int = 5          # número máximo de posições simultâneas
    allow_short: bool = False       # habilita short selling
    cost_bps: float = 5.0           # custo por lado em bps (0.05% => 5)
    slippage_pct: float = 0.0       # slippage percentual (ex.: 0.0005 = 0.05%)
    gross_leverage: float = 1.0     # soma(|weights|) máxima
    stop_loss_pct: float = 0.10     # -10% para long; para short é +10% contra
    take_profit_pct: float = 0.20   # +20% para long; -20% a favor do short
    allocation_mode: str = "equal"  # "equal" ou "prob"
    benchmark_ticker: Optional[str] = None


# ===========================
# == Exact Backtest Engine ==
# ===========================

def exact_backtest(
    prices_long: pd.DataFrame,
    signals_long: pd.DataFrame,
    params: Params,
) -> Dict[str, object]:
    # Preparação
    px_wide = _pivot_prices(prices_long, params.price_col)
    sig = _align_signals(signals_long)

    # Índices principais
    dates = px_wide.index
    tickers = px_wide.columns.tolist()
    nT = len(dates)

    # Tabelas auxiliares de sinais por data
    sig_by_date: Dict[pd.Timestamp, pd.DataFrame] = {
        d: g.drop(columns=["date"]).reset_index(drop=True)
        for d, g in sig.groupby("date", sort=True)
    }

    # Estado do portfólio
    cash = float(params.initial_capital)
    positions: Dict[str, Dict[str, float]] = {}  # ticker -> {"shares": s, "entry": px, "dir": +1/-1}
    equity_series: List[float] = []
    cash_series: List[float] = []
    gross_expo_series: List[float] = []
    net_expo_series: List[float] = []
    daily_ret: List[float] = []
    turnover_series: List[float] = []
    trades_log: List[Dict[str, object]] = []
    positions_snap: List[Dict[str, object]] = []

    cost_rate = (params.cost_bps / 10_000.0) + float(params.slippage_pct)

    # Helpers
    def _pos_value(dt_idx: int) -> float:
        px = px_wide.iloc[dt_idx]
        val = 0.0
        for tkr, p in positions.items():
            if tkr in px.index and not np.isnan(px[tkr]):
                val += p["shares"] * px[tkr]
        return float(val)

    def _record_positions(dt_idx: int):
        dt = dates[dt_idx]
        px = px_wide.iloc[dt_idx]
        for tkr, p in positions.items():
            pr = float(px[tkr]) if tkr in px.index and not np.isnan(px[tkr]) else np.nan
            positions_snap.append({
                "date": dt,
                "ticker": tkr,
                "shares": float(p["shares"]),
                "direction": int(p["dir"]),
                "entry_price": float(p["entry"]),
                "mark_price": pr,
                "market_value": float(p["shares"] * pr) if not np.isnan(pr) else np.nan,
            })

    def _close_position(tkr: str, price: float, dt, reason: str):
        nonlocal cash
        if tkr not in positions:
            return 0.0, 0.0
        pos = positions[tkr]
        shares = pos["shares"]
        notional = abs(shares) * price
        # custo de trading
        trade_cost = notional * cost_rate
        cash += shares * price  # vende (ou compra de volta no short)
        cash -= trade_cost
        trades_log.append({
            "date": dt, "ticker": tkr, "action": "CLOSE",
            "shares": float(-shares), "price": float(price),
            "notional": float(notional), "cost": float(trade_cost),
            "reason": reason
        })
        del positions[tkr]
        return notional, trade_cost

    def _open_position(tkr: str, direction: int, target_cash: float, price: float, dt, reason: str):
        """Abre posição com valor abs(target_cash). direction: +1 long, -1 short."""
        nonlocal cash
        if price <= 0 or target_cash <= 0:
            return 0.0, 0.0
        shares = (target_cash / price) * direction
        notional = abs(target_cash)
        trade_cost = notional * cost_rate
        cash -= shares * price  # compra (ou recebe cash na venda a descoberto se direction=-1)
        cash -= trade_cost
        positions[tkr] = {"shares": float(shares), "entry": float(price), "dir": int(direction)}
        trades_log.append({
            "date": dt, "ticker": tkr, "action": "OPEN_LONG" if direction > 0 else "OPEN_SHORT",
            "shares": float(shares), "price": float(price),
            "notional": float(notional), "cost": float(trade_cost),
            "reason": reason
        })
        return notional, trade_cost

    # Loop temporal (EOD → EOD)
    equity_prev = params.initial_capital  # para retorno diário
    for t in range(nT):
        dt = dates[t]
        px_today = px_wide.iloc[t]

        # 1) Stop-loss / Take-profit no preço de hoje
        to_close = []
        for tkr, pos in positions.items():
            if tkr not in px_today.index or np.isnan(px_today[tkr]):
                continue
            entry = pos["entry"]
            price = float(px_today[tkr])
            move = (price / entry) - 1.0
            if pos["dir"] > 0:
                # LONG: SL se move <= -stop_loss; TP se move >= take_profit
                if move <= -params.stop_loss_pct:
                    to_close.append((tkr, price, "stop_loss"))
                elif move >= params.take_profit_pct:
                    to_close.append((tkr, price, "take_profit"))
            else:
                # SHORT: contra é move >= +stop_loss; a favor é move <= -take_profit
                if move >= params.stop_loss_pct:
                    to_close.append((tkr, price, "stop_loss"))
                elif move <= -params.take_profit_pct:
                    to_close.append((tkr, price, "take_profit"))
        for tkr, price, reason in to_close:
            _close_position(tkr, price, dt, reason)

        # 2) Processa sinais do dia para decidir carteira alvo
        todays = sig_by_date.get(dt, None)
        if todays is None:
            todays = pd.DataFrame(columns=["ticker", "signal", "weight", "proba_up"])

        # candidatos long e short
        longs = todays[todays["signal"] > 0].copy()
        shorts = todays[todays["signal"] < 0].copy() if params.allow_short else pd.DataFrame(columns=longs.columns)

        # ranking por probabilidade (long desc, short asc)
        longs = longs.sort_values("proba_up", ascending=False)
        shorts = shorts.sort_values("proba_up", ascending=True)

        # consolidar lista alvo considerando max_positions (total)
        target_list = []
        for _, row in longs.iterrows():
            target_list.append((row["ticker"], +1, float(row["proba_up"])))
        for _, row in shorts.iterrows():
            target_list.append((row["ticker"], -1, float(1.0 - row["proba_up"])))  # "edge" p/ short

        # ordena por "edge" desc
        target_list.sort(key=lambda x: x[2], reverse=True)
        target_list = target_list[: max(0, params.max_positions)]

        # 3) Fecha posições que não estão mais no target_list
        target_names = {tk for (tk, _, _) in target_list}
        to_close_rebal = []
        for tkr in list(positions.keys()):
            if tkr not in target_names:
                price = float(px_today[tkr]) if tkr in px_today.index and not np.isnan(px_today[tkr]) else np.nan
                if not np.isnan(price):
                    to_close_rebal.append((tkr, price))
        for tkr, price in to_close_rebal:
            _close_position(tkr, price, dt, "rebalance_exit")

        # 4) Define pesos desejados (equal ou proporcional ao "edge")
        n_sel = len(target_list)
        weights: Dict[str, float] = {}
        if n_sel > 0:
            if params.allocation_mode == "prob":
                edges = np.array([edge for (_, _, edge) in target_list], dtype=float)
                # normaliza pela soma de edges
                if edges.sum() <= 0:
                    edges = np.ones_like(edges)
                alloc = edges / edges.sum()
            else:
                alloc = np.array([1.0 / n_sel] * n_sel)

            # aplica gross_leverage
            alloc = alloc * params.gross_leverage

            for (i, (tkr, direction, _edge)) in enumerate(target_list):
                weights[tkr] = alloc[i] * (1 if direction > 0 else -1)

        # 5) Executa rebalance no preço de hoje
        #     - fecha diferenças (parcial se mudasse direção, mas aqui fechamos fora do target anteriormente)
        #     - abre novas posições conforme 'weights'
        eq_before = cash + _pos_value(t)
        turnover = 0.0

        # atualiza posições existentes (se permaneceram no target, ajusta tamanho)
        for tkr, w in weights.items():
            price = float(px_today[tkr]) if tkr in px_today.index and not np.isnan(px_today[tkr]) else np.nan
            if np.isnan(price) or price <= 0:
                continue

            target_notional = abs(w) * eq_before
            direction = 1 if w > 0 else -1

            if tkr in positions:
                # ajuste (compra/venda parcial)
                current_shares = positions[tkr]["shares"]
                current_dir = positions[tkr]["dir"]
                current_notional = abs(current_shares) * price
                delta_notional = target_notional - current_notional

                if direction != current_dir:
                    # se mudou direção, fecha e abre
                    notional_closed, _ = _close_position(tkr, price, dt, "flip_direction")
                    turnover += notional_closed
                    notional_opened, _ = _open_position(tkr, direction, target_notional, price, dt, "flip_open")
                    turnover += notional_opened
                else:
                    if abs(delta_notional) / max(eq_before, 1e-12) > 1e-6:
                        # trade incremental
                        delta_shares = (delta_notional / price) * direction
                        notional = abs(delta_notional)
                        trade_cost = notional * cost_rate
                        # aplica
                        cash -= delta_shares * price
                        cash -= trade_cost
                        positions[tkr]["shares"] += delta_shares
                        turnover += notional
                        trades_log.append({
                            "date": dt, "ticker": tkr,
                            "action": "ADJUST",
                            "shares": float(delta_shares),
                            "price": float(price),
                            "notional": float(notional),
                            "cost": float(trade_cost),
                            "reason": "rebalance_adjust"
                        })
            else:
                # abrir nova
                notional_opened, _ = _open_position(tkr, direction, target_notional, price, dt, "rebalance_open")
                turnover += notional_opened

        # equity no fim do dia t (após rebalance/custos)
        eq_after = cash + _pos_value(t)
        equity_series.append(eq_after)
        cash_series.append(cash)
        gross_expo_series.append(sum(abs(p["shares"]) * float(px_today[tkr]) for tkr, p in positions.items() if tkr in px_today.index and not np.isnan(px_today[tkr])))
        net_expo_series.append(sum(p["shares"] * float(px_today[tkr]) for tkr, p in positions.items() if tkr in px_today.index and not np.isnan(px_today[tkr])))
        turnover_series.append(turnover / max(eq_before, 1e-12))

        # retorno diário (EOD→EOD)
        if len(equity_series) == 1:
            daily_ret.append(0.0)
        else:
            r = (equity_series[-1] / equity_prev) - 1.0
            daily_ret.append(float(r))
        equity_prev = equity_series[-1]

        # snapshot de posições
        _record_positions(t)

    # séries finais
    equity_ts = pd.Series(equity_series, index=dates, name="equity")
    cash_ts = pd.Series(cash_series, index=dates, name="cash")
    gross_ts = pd.Series(gross_expo_series, index=dates, name="gross_exposure")
    net_ts = pd.Series(net_expo_series, index=dates, name="net_exposure")
    ret_ts = pd.Series(daily_ret, index=dates, name="ret")
    turnover_ts = pd.Series(turnover_series, index=dates, name="turnover")

    # métricas
    ann = _annual_metrics(ret_ts[1:])
    mdd = _max_drawdown(equity_ts)

    # saída
    timeseries = pd.concat([equity_ts, cash_ts, gross_ts, net_ts, ret_ts, turnover_ts], axis=1)

    return {
        "timeseries": timeseries,
        "positions": pd.DataFrame(positions_snap),
        "trades": pd.DataFrame(trades_log),
        "summary": {
            "initial_capital": float(params.initial_capital),
            "final_equity": float(equity_ts.iloc[-1]) if len(equity_ts) else float(params.initial_capital),
            "tot_return": float(equity_ts.iloc[-1] / equity_ts.iloc[0] - 1.0) if len(equity_ts) else 0.0,
            "metrics": {
                "cagr": float(ann["cagr"]),
                "vol_ann": float(ann["vol_ann"]),
                "sharpe": float(ann["sharpe"]),
                "sortino": float(ann["sortino"]),
                "max_drawdown": float(mdd),
                "winrate": float(ann["winrate"]),
                "avg_turnover": float(turnover_ts.mean()),
                "avg_trades_per_day": float((pd.Series([abs(x) for x in turnover_series]) > 0).mean()),
            },
            "params": vars(params),
        },
    }


# =======================
# ======== CLI ==========
# =======================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulação EXATA (linha-a-linha) com reinvestimento, SL/TP e limite de posições.")
    p.add_argument("--prices-path", required=True, help="Arquivo de preços (parquet/csv) com colunas: date,ticker,close.")
    p.add_argument("--signals-path", required=True, help="Arquivo de sinais (parquet/csv) com: date,ticker,signal,weight,proba_up,model_name,strategy.")
    p.add_argument("--price-col", default="close", help="Coluna de preço para execução (default: close).")
    p.add_argument("--initial-capital", type=float, default=100000.0)
    p.add_argument("--max-positions", type=int, default=5)
    p.add_argument("--allow-short", action="store_true")
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--slippage-pct", type=float, default=0.0)
    p.add_argument("--gross-leverage", type=float, default=1.0)
    p.add_argument("--stop-loss-pct", type=float, default=0.10)
    p.add_argument("--take-profit-pct", type=float, default=0.20)
    p.add_argument("--allocation-mode", choices=["equal", "prob"], default="equal")
    p.add_argument("--benchmark-ticker", default=None)
    p.add_argument("--out-dir", default="data/backtests/exact")
    p.add_argument("--strategy-name", default=None, help="Nome de estratégia para os arquivos; se None, tenta inferir de signals.strategy.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    prices = _load_table(args.prices_path)
    signals = _load_table(args.signals_path)

    # inferir strategy name
    strat = args.strategy_name
    if strat is None:
        strat = str(signals["strategy"].iloc[0]) if "strategy" in signals.columns and len(signals) else "exact_strategy"

    params = Params(
        price_col=args.price_col,
        initial_capital=args.initial_capital,
        max_positions=args.max_positions,
        allow_short=bool(args.allow_short),
        cost_bps=args.cost_bps,
        slippage_pct=args.slippage_pct,
        gross_leverage=args.gross_leverage,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        allocation_mode=args.allocation_mode,
        benchmark_ticker=(None if args.benchmark_ticker in [None, "None", ""] else args.benchmark_ticker),
    )

    res = exact_backtest(prices, signals, params)

    # salvar artefatos
    ts = res["timeseries"]
    pos = res["positions"]
    trd = res["trades"]
    summ = res["summary"]

    ts_path_parquet = os.path.join(args.out_dir, f"equity_exact_{strat}.parquet")
    ts_path_csv = os.path.join(args.out_dir, f"equity_exact_{strat}.csv")
    pos_path = os.path.join(args.out_dir, "positions.parquet")
    trd_path = os.path.join(args.out_dir, "trades.parquet")
    summ_path = os.path.join(args.out_dir, f"summary_exact_{strat}.json")

    ts.to_parquet(ts_path_parquet, index=True)
    ts.to_csv(ts_path_csv, index=True)
    pos.to_parquet(pos_path, index=False)
    trd.to_parquet(trd_path, index=False)

    with open(summ_path, "w") as f:
        json.dump(summ, f, indent=2)

    print(f"[OK] Exact backtest salvo em:\n  {ts_path_parquet}\n  {pos_path}\n  {trd_path}\n  {summ_path}")
    print(f"     Final equity: {summ['final_equity']:.2f} | CAGR: {summ['metrics']['cagr']:.2%} | Sharpe: {summ['metrics']['sharpe']:.2f}")
    print(f"     MaxDD: {summ['metrics']['max_drawdown']:.2%} | WinRate: {summ['metrics']['winrate']:.2%}")


if __name__ == "__main__":
    main()
