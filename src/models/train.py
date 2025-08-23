"""
Treinamento de modelos (linear, rf, xgb, lstm) para:
- Regressão: target_reg_5d
- Classificação: target_cls_5d

Uso:
  python -m src.models.train --task both --model rf
  python -m src.models.train --task reg  --model xgb
  python -m src.models.train --task cls  --model lstm --seq-len 20 --epochs 15
"""

from __future__ import annotations

import typer
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from src import config
from src.utils.io import load_parquet
from .advanced import (
    feature_columns,
    get_regressor,
    get_classifier,
    prepare_sequences,
    build_lstm_reg,
    build_lstm_cls,
    fit_lstm,
)

app = typer.Typer(add_completion=False, help="Treinamento de modelos avançados")


def _prepare_tabular(features: pd.DataFrame):
    exclude = {"date", "ticker", "target_reg_5d", "target_cls_5d"}
    cols = [c for c in features.columns if c not in exclude and pd.api.types.is_numeric_dtype(features[c])]
    X = features[cols].fillna(0)
    y_reg = features["target_reg_5d"].fillna(0)
    y_cls = features["target_cls_5d"].fillna(0)
    return X, y_reg, y_cls, cols


def _save_model(model, path: Path, algo: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    algo = algo.lower()
    if algo in {"linear", "rf"}:
        joblib.dump(model, path)
    elif algo == "xgb":
        # xgboost possui método save_model
        model.save_model(str(path))
    elif algo == "lstm":
        model.save(str(path))
    else:
        raise ValueError(f"Algo desconhecido: {algo}")


@app.command("run")
def run(
    task: str = typer.Option("both", help="reg | cls | both"),
    model: str = typer.Option("rf", help="linear | rf | xgb | lstm"),
    seq_len: int = typer.Option(20, help="Comprimento da janela para LSTM"),
    epochs: int = typer.Option(20, help="Épocas para LSTM"),
    batch_size: int = typer.Option(64, help="Batch size para LSTM"),
):
    """
    Treina modelos conforme 'task' e 'model', e salva em models/.
    """
    features_path = config.ANALYTICS_DIR / "features.parquet"
    out_dir = config.MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = load_parquet(features_path)
    if feats is None or feats.empty:
        typer.secho(f"[train] Arquivo não encontrado: {features_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    feats = feats.sort_values(["ticker", "date"]).reset_index(drop=True)

    # TABULAR (linear / rf / xgb)
    if model in {"linear", "rf", "xgb"}:
        X, y_reg, y_cls, cols = _prepare_tabular(feats)
        tscv = TimeSeriesSplit(n_splits=3)

        if task in {"reg", "both"}:
            reg = get_regressor(model)
            maes, mapes = [], []
            for tr, te in tscv.split(X):
                reg.fit(X.iloc[tr], y_reg.iloc[tr])
                pred = reg.predict(X.iloc[te])
                maes.append(mean_absolute_error(y_reg.iloc[te], pred))
                mapes.append(mean_absolute_percentage_error(y_reg.iloc[te], pred))
            # fit final em todo dataset e salvar
            reg.fit(X, y_reg)
            path = out_dir / f"model_reg_{model}.{'pkl' if model!='xgb' else 'json'}"
            _save_model(reg, path, model)
            typer.secho(f"[train][{model}][reg] MAE={np.mean(maes):.4f} | MAPE={np.mean(mapes):.4f} | saved={path}", fg=typer.colors.GREEN)

        if task in {"cls", "both"}:
            cls = get_classifier(model)
            accs, f1s, aucs = [], [], []
            for tr, te in tscv.split(X):
                cls.fit(X.iloc[tr], y_cls.iloc[tr])
                preds = cls.predict(X.iloc[te])
                probs = getattr(cls, "predict_proba", None)
                if probs is not None:
                    pr = probs(X.iloc[te])[:, 1]
                else:
                    pr = cls.decision_function(X.iloc[te])
                accs.append(accuracy_score(y_cls.iloc[te], preds))
                f1s.append(f1_score(y_cls.iloc[te], preds))
                # AUC pode falhar se só houver uma classe; trate:
                try:
                    aucs.append(roc_auc_score(y_cls.iloc[te], pr))
                except Exception:
                    pass
            # fit final
            cls.fit(X, y_cls)
            path = out_dir / f"model_cls_{model}.{'pkl' if model!='xgb' else 'json'}"
            _save_model(cls, path, model)
            typer.secho(f"[train][{model}][cls] Acc={np.mean(accs):.3f} | F1={np.mean(f1s):.3f} | AUC={np.mean(aucs) if aucs else float('nan'):.3f} | saved={path}", fg=typer.colors.GREEN)

    # LSTM
    elif model == "lstm":
        cols = feature_columns(feats)
        # REG
        if task in {"reg", "both"}:
            data = prepare_sequences(feats, cols, target_col="target_reg_5d", seq_len=seq_len)
            if len(data.X) == 0:
                typer.secho("[train][lstm][reg] Dados insuficientes para sequências.", fg=typer.colors.RED)
            else:
                mdl = build_lstm_reg((seq_len, len(cols)))
                fit_lstm(mdl, data.X, data.y, epochs=epochs, batch_size=batch_size)
                path = out_dir / f"model_reg_lstm.keras"
                _save_model(mdl, path, "lstm")
                typer.secho(f"[train][lstm][reg] saved={path}", fg=typer.colors.GREEN)

        # CLS
        if task in {"cls", "both"}:
            data = prepare_sequences(feats, cols, target_col="target_cls_5d", seq_len=seq_len)
            if len(data.X) == 0:
                typer.secho("[train][lstm][cls] Dados insuficientes para sequências.", fg=typer.colors.RED)
            else:
                mdl = build_lstm_cls((seq_len, len(cols)))
                fit_lstm(mdl, data.X, data.y, epochs=epochs, batch_size=batch_size)
                path = out_dir / f"model_cls_lstm.keras"
                _save_model(mdl, path, "lstm")
                typer.secho(f"[train][lstm][cls] saved={path}", fg=typer.colors.GREEN)
    else:
        raise typer.Exit(f"Modelo inválido: {model}")


if __name__ == "__main__":
    app()
