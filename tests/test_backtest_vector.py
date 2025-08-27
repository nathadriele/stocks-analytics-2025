# tests/test_backtest_vector.py
import os
import pandas as pd

def test_vector_equity_has_cols():
    bdir = "data/backtests"
    files = [f for f in os.listdir(bdir) if f.startswith("equity_") and f.endswith(".parquet")]
    assert files, "Rode make backtest_vector antes dos testes."
    df = pd.read_parquet(os.path.join(bdir, files[0]))
    for col in ["equity","ret"]:
        assert col in df.columns
