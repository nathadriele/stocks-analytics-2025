# scripts/train_trees.py
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from src.models.baselines import (
    TrainConfig,
    train_decision_tree,
    train_random_forest,
)


EXCLUDE_COLS_DEFAULT = {
    "date", "datetime", "timestamp",
    "ticker", "symbol",
    "close", "open", "high", "low", "volume",
    "target_up", "target_up_5d", "ret_5d",
}


def _read_features_from_file(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de features não encontrado: {path}")

    if path.endswith((".yml", ".yaml")):
        import yaml  # lazy import
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        # aceita formatos:
        if isinstance(data, dict) and "features" in data and isinstance(data["features"], list):
            return [str(x) for x in data["features"]]
        # ou um dict onde as chaves são features
        return [str(k) for k in data.keys()]
    else:
        # TXT/CSV (uma feature por linha)
        with open(path, "r") as f:
            lines = [ln.strip() for ln in f.readlines()]
        return [ln for ln in lines if ln and not ln.startswith("#")]


def _auto_features(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> List[str]:
    exclude = set(EXCLUDE_COLS_DEFAULT)
    if exclude_cols:
        exclude |= set(exclude_cols)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in num_cols if c not in exclude]
    if not feats:
        raise ValueError("Detecção automática não encontrou features numéricas válidas.")
    return feats


def _load_dataframe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset não encontrado: {path}")
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("Formato não suportado. Use .parquet ou .csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Treino de DecisionTree/RandomForest com GridSearch + TimeSeriesSplit"
    )
    p.add_argument(
        "--data-path",
        required=True,
        help="Caminho do dataset consolidado (Parquet ou CSV) com features e alvo."
    )
    p.add_argument(
        "--features",
        default=None,
        help="Lista de features separadas por vírgula. Ex: ret_1d,sma_10,ema_10"
    )
    p.add_argument(
        "--features-file",
        default=None,
        help="Arquivo (TXT/YAML) com uma feature por linha (ou chave 'features:' em YAML)."
    )
    p.add_argument(
        "--exclude-cols",
        default=None,
        help="Colunas para excluir da detecção automática (vírgulas)."
    )
    p.add_argument(
        "--date-col",
        default="date",
        help="Nome da coluna de data (default: date)."
    )
    p.add_argument(
        "--ticker-col",
        default="ticker",
        help="Nome da coluna de ticker (default: ticker)."
    )
    p.add_argument(
        "--target-candidates",
        default="target_up_5d,target_up",
        help="Candidatos de alvo binário (ordem de prioridade), separados por vírgula."
    )
    p.add_argument(
        "--ret5-col",
        default="ret_5d",
        help="Coluna de retorno 5d (fallback para inferir alvo)."
    )
    p.add_argument(
        "--close-col",
        default="close",
        help="Coluna de preço (fallback para inferir alvo por variação futura)."
    )
    p.add_argument(
        "--test-size-ratio",
        type=float,
        default=0.2,
        help="Proporção temporal para teste (default: 0.2)."
    )
    p.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="N de splits no TimeSeriesSplit (default: 5)."
    )
    p.add_argument(
        "--model",
        choices=["dt", "rf", "both"],
        default="both",
        help="Qual modelo treinar (Decision Tree, Random Forest ou ambos)."
    )
    p.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Desabilita calibração de probabilidade (Platt)."
    )
    p.add_argument(
        "--models-dir",
        default="models",
        help="Diretório para salvar modelos/relatórios (default: models)."
    )
    p.add_argument(
        "--analytics-dir",
        default="data/analytics",
        help="Diretório para salvar predições (default: data/analytics)."
    )
    return p.parse_args()


def main():
    args = parse_args()

    df = _load_dataframe(args.data_path)

    # features
    features: Optional[List[str]] = None
    if args.features:
        features = [c.strip() for c in args.features.split(",") if c.strip()]
    elif args.features_file:
        features = _read_features_from_file(args.features_file)
    else:
        excl = [c.strip() for c in args.exclude_cols.split(",")] if args.exclude_cols else None
        features = _auto_features(df, exclude_cols=excl)

    # config
    cfg = TrainConfig(
        features=features,
        date_col=args.date_col,
        ticker_col=args.ticker_col,
        target_col_candidates=tuple([c.strip() for c in args.target_candidates.split(",")]),
        close_col=args.close_col,
        ret5_col=args.ret5_col,
        models_dir=args.models_dir,
        analytics_dir=args.analytics_dir,
        test_size_ratio=args.test_size_ratio,
        n_splits_cv=args.n_splits,
        calibrate=not args.no_calibrate,
    )

    os.makedirs(cfg.models_dir, exist_ok=True)

    reports: Dict[str, Dict] = {}

    if args.model in ("dt", "both"):
        rep_dt = train_decision_tree(df.copy(), cfg, model_name="dt_baseline")
        reports["dt_baseline"] = rep_dt

    if args.model in ("rf", "both"):
        rep_rf = train_random_forest(df.copy(), cfg, model_name="rf_baseline")
        reports["rf_baseline"] = rep_rf

    summary_path = os.path.join(cfg.models_dir, "_last_training.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "data_path": args.data_path,
                "features_used": features,
                "models_trained": list(reports.keys()),
                "reports": reports,
            },
            f,
            indent=2,
        )

    print(f"[OK] Treino finalizado. Sumário: {summary_path}")
    for k, r in reports.items():
        print(f"  - {k}: AUC={r['metrics_test'].get('auc'):.4f} "
              f"ACC={r['metrics_test'].get('accuracy'):.4f} "
              f"F1={r['metrics_test'].get('f1'):.4f}")
        print(f"    Model: {r['artifacts']['model_path']}")
        print(f"    Preds: {r['artifacts']['preds_path']}")


if __name__ == "__main__":
    main()
