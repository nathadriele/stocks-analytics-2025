# src/models/xgb.py
from __future__ import annotations

import os, json
from typing import Dict, List, Optional, Tuple
import numpy as np, pandas as pd

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump
from dataclasses import dataclass

try:
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError("Instale xgboost: pip install xgboost") from e

@dataclass
class TrainConfigXGB:
    features: List[str]
    date_col: str = "date"
    ticker_col: str = "ticker"
    target_col_candidates: Tuple[str, ...] = ("target_up_5d","target_up")
    close_col: str = "close"
    ret5_col: str = "ret_5d"
    models_dir: str = "models"
    analytics_dir: str = "data/analytics"
    test_size_ratio: float = 0.2
    n_splits_cv: int = 5
    random_state: int = 42
    calibrate: bool = True

def _ensure_dirs(cfg: TrainConfigXGB) -> None:
    os.makedirs(cfg.models_dir, exist_ok=True)
    os.makedirs(cfg.analytics_dir, exist_ok=True)

def _infer_target(df: pd.DataFrame, cfg: TrainConfigXGB) -> pd.Series:
    for col in cfg.target_col_candidates:
        if col in df.columns:
            return df[col].astype(int)
    if cfg.ret5_col in df.columns:
        return (df[cfg.ret5_col].astype(float) > 0).astype(int)
    if cfg.close_col in df.columns:
        fwd = df[cfg.close_col].shift(-5)
        ret5 = (fwd/df[cfg.close_col]) - 1.0
        return (ret5 > 0).astype(int)
    raise ValueError("Não foi possível inferir o alvo.")

def _split_time(df: pd.DataFrame, ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df); ts = max(1, int(n*ratio))
    return df.iloc[:-ts].copy(), df.iloc[-ts:].copy()

def _xy(df: pd.DataFrame, cfg: TrainConfigXGB) -> tuple[np.ndarray, np.ndarray]:
    X = df[cfg.features].astype(float).values
    y = _infer_target(df, cfg).values
    return X, y

def _metrics(y_true, proba, thr=0.5) -> Dict[str, float]:
    yhat = (proba >= thr).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, proba)) if len(np.unique(y_true))>1 else np.nan,
        "accuracy": float(accuracy_score(y_true, yhat)),
        "f1": float(f1_score(y_true, yhat, zero_division=0)),
    }

def _save_preds(df_idx: pd.DataFrame, proba: np.ndarray, cfg: TrainConfigXGB, name: str) -> str:
    out = df_idx[[cfg.date_col, cfg.ticker_col]].copy() if cfg.ticker_col in df_idx.columns else df_idx[[cfg.date_col]].copy()
    out["proba_up"] = proba
    path = os.path.join(cfg.analytics_dir, f"pred_{name}.parquet")
    out.to_parquet(path, index=False)
    return path

def train_xgb(df: pd.DataFrame, cfg: TrainConfigXGB, model_name="xgb") -> Dict[str, object]:
    _ensure_dirs(cfg)
    df = df.sort_values(cfg.date_col).reset_index(drop=True)
    df_tr, df_te = _split_time(df, cfg.test_size_ratio)
    Xtr, Ytr = _xy(df_tr, cfg)
    Xte, Yte = _xy(df_te, cfg)

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=400,
            random_state=cfg.random_state,
            tree_method="hist",
            n_jobs=-1,
        ))
    ])
    param_dist = {
        "clf__n_estimators": [300, 400, 600, 800],
        "clf__max_depth": [3, 4, 5, 6, 8],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "clf__subsample": [0.7, 0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__min_child_weight": [1, 3, 5, 7],
        "clf__gamma": [0, 0.1, 0.3],
    }
    cv = TimeSeriesSplit(n_splits=cfg.n_splits_cv)
    rs = RandomizedSearchCV(
        pipe, param_distributions=param_dist, n_iter=25, cv=cv,
        scoring="roc_auc", n_jobs=-1, random_state=cfg.random_state, refit=True, verbose=0
    )
    rs.fit(Xtr, Ytr)

    best = rs.best_estimator_
    if cfg.calibrate:
        best = CalibratedClassifierCV(best, method="sigmoid", cv=cv)
        best.fit(Xtr, Ytr)

    proba_te = best.predict_proba(Xte)[:,1]
    metrics = _metrics(Yte, proba_te)
    model_path = os.path.join(cfg.models_dir, f"{model_name}.joblib")
    from joblib import dump
    dump(best, model_path)
    preds_path = _save_preds(df_te[[cfg.date_col, cfg.ticker_col]] if cfg.ticker_col in df_te.columns else df_te[[cfg.date_col]], proba_te, cfg, model_name)

    report = {
        "model": "XGBoost",
        "model_name": model_name,
        "best_params": getattr(rs, "best_params_", {}),
        "metrics_test": metrics,
        "artifacts": {"model_path": model_path, "preds_path": preds_path},
    }
    with open(os.path.join(cfg.models_dir, f"{model_name}_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report
