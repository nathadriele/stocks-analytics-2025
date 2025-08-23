"""
Treinamento de modelos de previsão de retornos de mercado.

- Lê features de data/analytics/features.parquet
- Treina modelos de:
    - Regressão (retorno futuro em 5 dias)
    - Classificação (direção positiva/negativa em 5 dias)
- Avalia com métricas adequadas
- Persiste artefatos em models/

Uso:
    python -m src.models.train --mode both
"""

from __future__ import annotations

import typer
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src import config
from src.utils.io import load_parquet, save_parquet

app = typer.Typer(add_completion=False, help="Treinamento de modelos")


def _prepare_data(features: pd.DataFrame):
    """
    Prepara X, y para regressão e classificação.
    """
    # Features numéricas (exclui targets e strings)
    exclude_cols = ["date", "ticker", "target_reg_5d", "target_cls_5d"]
    num_cols = [c for c in features.columns if c not in exclude_cols]

    X = features[num_cols].fillna(0)

    y_reg = features["target_reg_5d"].fillna(0)
    y_cls = features["target_cls_5d"].fillna(0)

    return X, y_reg, y_cls


@app.command("run")
def run(mode: str = typer.Option("both", help="Opções: reg, cls, both")):
    """
    Executa o treinamento de modelos.
    """
    features_path = config.ANALYTICS_DIR / "features.parquet"
    models_dir = config.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    features = load_parquet(features_path)
    if features is None or features.empty:
        typer.secho(f"[train] Arquivo não encontrado: {features_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    X, y_reg, y_cls = _prepare_data(features)

    # Split temporal (3 folds)
    tscv = TimeSeriesSplit(n_splits=3)

    if mode in ("reg", "both"):
        reg = LinearRegression()
        maes, mapes = [], []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
            reg.fit(X_train, y_train)
            preds = reg.predict(X_test)
            maes.append(mean_absolute_error(y_test, preds))
            mapes.append(mean_absolute_percentage_error(y_test, preds))

        joblib.dump(reg, models_dir / "model_reg.pkl")
        typer.secho(
            f"[train][reg] MAE={sum(maes)/len(maes):.4f} | MAPE={sum(mapes)/len(mapes):.4f}",
            fg=typer.colors.GREEN,
        )

    if mode in ("cls", "both"):
        cls = LogisticRegression(max_iter=500)
        accs, f1s, aucs = [], [], []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_cls.iloc[train_idx], y_cls.iloc[test_idx]
            cls.fit(X_train, y_train)
