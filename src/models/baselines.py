# src/models/baselines.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump

@dataclass
class TrainConfig:
    features: List[str]
    date_col: str = "date"
    ticker_col: str = "ticker"
    target_col_candidates: Tuple[str, ...] = ("target_up_5d", "target_up")
    close_col: str = "close"
    ret5_col: str = "ret_5d"
    models_dir: str = "models"
    analytics_dir: str = "data/analytics"
    test_size_ratio: float = 0.2
    n_splits_cv: int = 5
    random_state: int = 42
    calibrate: bool = True

def _ensure_dirs(cfg: TrainConfig) -> None:
    os.makedirs(cfg.models_dir, exist_ok=True)
    os.makedirs(cfg.analytics_dir, exist_ok=True)


def _infer_target(df: pd.DataFrame, cfg: TrainConfig) -> pd.Series:
    for col in cfg.target_col_candidates:
        if col in df.columns:
            return df[col].astype(int)

    if cfg.ret5_col in df.columns:
        return (df[cfg.ret5_col].astype(float) > 0.0).astype(int)

    if cfg.close_col in df.columns:
        fwd = df[cfg.close_col].shift(-5)
        ret5 = (fwd / df[cfg.close_col]) - 1.0
        return (ret5 > 0.0).astype(int)

    raise ValueError(
        f"{cfg.target_col_candidates} ou {cfg.ret5_col} ou {cfg.close_col}."
    )


def _train_test_split_time(df: pd.DataFrame, cfg: TrainConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    test_size = max(1, int(n * cfg.test_size_ratio))
    return df.iloc[:-test_size].copy(), df.iloc[-test_size:].copy()


def _prepare_xy(df: pd.DataFrame, cfg: TrainConfig) -> Tuple[np.ndarray, np.ndarray]:
    miss = [f for f in cfg.features if f not in df.columns]
    if miss:
        raise ValueError(f"Features ausentes no DataFrame: {miss}")

    X = df[cfg.features].astype(float).values
    y = _infer_target(df, cfg).values
    return X, y


def _evaluate(y_true: np.ndarray, proba: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    """Calcula métricas padrão de classificação (AUC/ACC/F1)."""
    y_hat = (proba >= thr).astype(int)
    out = {
        "auc": float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else np.nan,
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
    }
    return out


def _save_predictions(df_idx: pd.DataFrame, proba: np.ndarray, cfg: TrainConfig, name: str) -> str:
    """Salva arquivo com probabilidades para uso no gerador de sinais."""
    out = df_idx[[cfg.date_col, cfg.ticker_col]].copy() if cfg.ticker_col in df_idx.columns else df_idx[[cfg.date_col]].copy()
    out["proba_up"] = proba
    path = os.path.join(cfg.analytics_dir, f"pred_{name}.parquet")
    out.to_parquet(path, index=False)
    return path


# =========================
# ==== Model Training =====
# =========================

def train_decision_tree(
    df: pd.DataFrame,
    cfg: TrainConfig,
    param_grid: Optional[Dict[str, List]] = None,
    model_name: str = "dt"
) -> Dict[str, object]:
    """Treina DecisionTree com GridSearch + TimeSeriesSplit. Retorna métricas e caminhos."""
    _ensure_dirs(cfg)

    if param_grid is None:
        param_grid = {
            "clf__max_depth": [3, 5, 7, 9, None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 3, 5],
        }

    df = df.sort_values(cfg.date_col).reset_index(drop=True)
    df_train, df_test = _train_test_split_time(df, cfg)

    X_tr, y_tr = _prepare_xy(df_train, cfg)
    X_te, y_te = _prepare_xy(df_test, cfg)

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(random_state=cfg.random_state)),
    ])

    cv = TimeSeriesSplit(n_splits=cfg.n_splits_cv)
    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs.fit(X_tr, y_tr)

    best = gs.best_estimator_

    if cfg.calibrate:
        # calibração de probabilidade (Platt - sigmoid)
        best = CalibratedClassifierCV(best, method="sigmoid", cv=cv)
        best.fit(X_tr, y_tr)

    proba_te = best.predict_proba(X_te)[:, 1]
    metrics = _evaluate(y_te, proba_te, thr=0.5)

    # persistência
    model_path = os.path.join(cfg.models_dir, f"{model_name}.joblib")
    dump(best, model_path)
    preds_path = _save_predictions(df_test[[cfg.date_col, cfg.ticker_col]] if cfg.ticker_col in df_test.columns else df_test[[cfg.date_col]], proba_te, cfg, model_name)

    report = {
        "model": "DecisionTree",
        "model_name": model_name,
        "best_params": getattr(gs, "best_params_", {}),
        "metrics_test": metrics,
        "artifacts": {"model_path": model_path, "preds_path": preds_path},
    }

    # salva JSON de métrica rápida para auditoria
    with open(os.path.join(cfg.models_dir, f"{model_name}_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    return report


def train_random_forest(
    df: pd.DataFrame,
    cfg: TrainConfig,
    param_grid: Optional[Dict[str, List]] = None,
    model_name: str = "rf"
) -> Dict[str, object]:
    """Treina RandomForest com GridSearch + TimeSeriesSplit. Retorna métricas e caminhos."""
    _ensure_dirs(cfg)

    if param_grid is None:
        param_grid = {
            "clf__n_estimators": [200, 400, 800],
            "clf__max_depth": [5, 10, 15, None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2", None],
        }

    df = df.sort_values(cfg.date_col).reset_index(drop=True)
    df_train, df_test = _train_test_split_time(df, cfg)

    X_tr, y_tr = _prepare_xy(df_train, cfg)
    X_te, y_te = _prepare_xy(df_test, cfg)

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", RandomForestClassifier(random_state=cfg.random_state, n_jobs=-1)),
    ])

    cv = TimeSeriesSplit(n_splits=cfg.n_splits_cv)
    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs.fit(X_tr, y_tr)

    best = gs.best_estimator_

    if cfg.calibrate:
        best = CalibratedClassifierCV(best, method="sigmoid", cv=cv)
        best.fit(X_tr, y_tr)

    proba_te = best.predict_proba(X_te)[:, 1]
    metrics = _evaluate(y_te, proba_te, thr=0.5)

    # persistência
    model_path = os.path.join(cfg.models_dir, f"{model_name}.joblib")
    dump(best, model_path)
    preds_path = _save_predictions(df_test[[cfg.date_col, cfg.ticker_col]] if cfg.ticker_col in df_test.columns else df_test[[cfg.date_col]], proba_te, cfg, model_name)

    report = {
        "model": "RandomForest",
        "model_name": model_name,
        "best_params": getattr(gs, "best_params_", {}),
        "metrics_test": metrics,
        "artifacts": {"model_path": model_path, "preds_path": preds_path},
    }

    with open(os.path.join(cfg.models_dir, f"{model_name}_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    return report
