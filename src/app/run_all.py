# src/app/run_all.py
from __future__ import annotations
import subprocess, shlex, os, sys

def run(cmd: str):
    print(f"\n$ {cmd}")
    rc = subprocess.call(shlex.split(cmd))
    if rc != 0:
        sys.exit(rc)

def main():
    DATA = "data/analytics/features.parquet"
    FEATURES = "src/features/feature_list.txt"
    PRICES = "data/processed/prices.parquet"

    # 1) Assemble (garante dataset unificado)
    run(f"python -m src.features.assemble_dataset --input-path {DATA} --prices-path {PRICES} --out-path {DATA} --feature-list {FEATURES}")

    # 2) Treinos
    run(f"python -m scripts.train_trees --data-path {DATA} --features-file {FEATURES} --model both")
    try:
        run(f"python -m scripts.train_xgb --data-path {DATA} --features-file {FEATURES}")
    except SystemExit:
        print("[WARN] XGBoost não treinado (instale xgboost). Seguindo...")

    # 3) Sinais (duas estratégias)
    run("python -m src.strategy.generate_signals --strategy long_only_threshold --thr 0.60")
    run("python -m src.strategy.generate_signals --strategy long_short_threshold --thr-long 0.65 --thr-short 0.40")

    # 4) Backtests
    run("python -m src.backtest.vector_backtester --prices-path data/processed/prices.parquet --signals-path data/signals/signals_long_only_threshold.parquet --benchmark-ticker SPY")
    run("python -m src.backtest.vector_backtester --prices-path data/processed/prices.parquet --signals-path data/signals/signals_long_short_threshold.parquet --benchmark-ticker SPY")

    # 5) Exato
    run("python -m src.backtest.exact_simulator --prices-path data/processed/prices.parquet --signals-path data/signals/signals_long_only_threshold.parquet --max-positions 5 --stop-loss-pct 0.10 --take-profit-pct 0.20")

    # 6) Relatório
    run("python -m src.reports.make_report")

    print("\n[OK] Pipeline fim-a-fim concluído.")

if __name__ == "__main__":
    main()
