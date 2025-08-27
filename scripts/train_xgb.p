# scripts/train_xgb.py
from __future__ import annotations
import argparse, os, pandas as pd
from src.models.xgb import TrainConfigXGB, train_xgb

def parse_args():
    p = argparse.ArgumentParser(description="Treino XGBoost com TimeSeriesSplit + RandomizedSearchCV")
    p.add_argument("--data-path", required=True)
    p.add_argument("--features-file", default="src/features/feature_list.txt")
    p.add_argument("--date-col", default="date")
    p.add_argument("--ticker-col", default="ticker")
    p.add_argument("--target-candidates", default="target_up_5d,target_up")
    p.add_argument("--ret5-col", default="ret_5d")
    p.add_argument("--close-col", default="close")
    p.add_argument("--test-size-ratio", type=float, default=0.2)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--analytics-dir", default="data/analytics")
    return p.parse_args()

def _read_features(path: str):
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

def main():
    a = parse_args()
    df = pd.read_parquet(a.data_path) if a.data_path.endswith(".parquet") else pd.read_csv(a.data_path)
    feats = _read_features(a.features_file)
    cfg = TrainConfigXGB(
        features=feats, date_col=a.date_col, ticker_col=a.ticker_col,
        target_col_candidates=tuple([c.strip() for c in a.target_candidates.split(",")]),
        close_col=a.close_col, ret5_col=a.ret5_col,
        models_dir=a.models_dir, analytics_dir=a.analytics_dir,
        test_size_ratio=a.test_size_ratio, n_splits_cv=a.n_splits,
        calibrate=not a.no_calibrate
    )
    rep = train_xgb(df, cfg, model_name="xgb_baseline")
    print(rep)

if __name__ == "__main__":
    main()
