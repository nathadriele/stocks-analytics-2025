import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.backtest.simulator import backtest, Costs, compute_metrics


def _mk_dates(n=5, start="2025-01-01"):
    start_dt = pd.to_datetime(start)
    return [start_dt + timedelta(days=i) for i in range(n)]


def test_backtest_basic_costs_and_shapes():
    # ----- preços (close-to-close) para 1 ticker -----
    dates = _mk_dates(5, "2025-01-01")
    prices = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAA"] * 5,
            "adjclose": [100.0, 101.0, 102.0, 103.0, 104.0],
        }
    )

    # ----- posições (execução next day no simulador) -----
    # posição muda em t=1 (0->+1) e t=3 (+1->-1) => deve incorrer custo nesses dias
    positions = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAA"] * 5,
            "position": [0, 1, 1, -1, -1],
        }
    )

    costs = Costs(fee_bps=5.0, slippage_bps=2.0)

    positions_out, equity = backtest(prices, positions, costs)

    assert set(["date", "ticker", "position", "ret_mkt", "ret_gross", "cost", "ret_net"]).issubset(
        positions_out.columns
    )
    assert set(["ret_port", "equity"]).issubset(equity.columns)
    assert len(equity) == len(dates)

    # ----- custos aplicados quando há mudança de posição -----
    # como o simulador calcula turnover com base em position_t vs position_{t-1},
    # mudanças ocorrem nos índices 1 e 3 (datas[1], datas[3])
    po = positions_out.sort_values("date").reset_index(drop=True)
    costs_on_days = po.groupby("date")["cost"].sum()
    assert costs_on_days.iloc[1] > 0  # dia da mudança 0->+1
    assert costs_on_days.iloc[3] > 0  # dia da mudança +1->-1

    summary = compute_metrics(
        equity=equity.set_index("date")["equity"],
        ret_daily=equity.set_index("date")["ret_port"],
        trading_days=252,
    )
    for k in ["CAGR", "Sharpe", "Vol_Ann", "MaxDrawdown", "N_Days", "Start", "End"]:
        assert k in summary
    assert summary["N_Days"] == len(dates)
