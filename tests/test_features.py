import pandas as pd
from src.features.build_features import generate_features


def test_generate_features_basic():
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "date": dates.tolist() * 2,
        "ticker": ["AAA"] * 30 + ["BBB"] * 30,
        "open": 100,
        "high": 105,
        "low": 95,
        "close": 100,
        "adjclose": list(range(100, 130)) + list(range(200, 230)),
        "volume": 1000,
    })

    features = generate_features(df)

    expected_cols = [
        "ret_1d", "ret_5d", "ret_21d", "vol_21d",
        "dow", "month", "sma_20", "ema_20", "rsi_14",
        "macd", "macd_signal", "bb_high", "bb_low",
        "target_reg_5d", "target_cls_5d",
    ]
    for col in expected_cols:
        assert col in features.columns

    assert set(features["ticker"].unique()) == {"AAA", "BBB"}

    assert features["target_reg_5d"].notna().sum() > 0
    assert features["target_cls_5d"].isin([0, 1]).any()
