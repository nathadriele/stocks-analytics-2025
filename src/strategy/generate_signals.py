"""
Geração de sinais a partir de modelos (linear, rf, xgb, lstm).

Uso:
  python -m src.strategy.generate_signals --mode both --algo rf
  python -m src.strategy.generate_signals --mode reg  --algo xgb
  python -m src.strategy.generate_signals --mode cls  --algo lstm --seq-len 20
"""

from __future__ import annotations

import typer
import joblib
import pandas as pd
from pathlib import Path

from src import config
from src.utils.io import load_parquet, save_parquet
from src.models.advanced import feature_columns, prepare_sequences_infer

app = typer.Typer(add_completion=False, help="Geração de sinais (RF/XGB/LSTM)")


def _to_signal_from_regression(pred: float, threshold: float = 0.0) -> int:
    if pred > threshold:
        return 1
    elif pred < threshold:
        return -1
    return 0


def _to_signal_from_classification(prob: float, threshold: float = 0.5) -> int:
    return 1 if prob > threshold else -1


def _load_model(algo: str, task: str):
    algo = algo.lower()
    task = task.lower()
    if algo in {"linear", "rf"}:
        path = config.MODELS_DIR / f"model_{task}_{algo}.pkl"
        return joblib.load(path), path
    if algo == "xgb":
        from xgboost import XGBRegressor, XGBClassifier
        path = config.MODELS_DIR / f"model_{task}_xgb.json"
        model = XGBRegressor() if task == "reg" else XGBClassifier()
        model.load_model(str(path))
        return model, path
    if algo == "lstm":
        from tensorflow.keras.models import load_model
        path = config.MODELS_DIR / f"model_{task}_lstm.keras"
        return load_model(str(path)), path
    raise ValueError(f"Algo desconhecido: {algo}")


@app.command("run")
def run(
    mode: str = typer.Option("both", help="reg | cls | both"),
    algo: str = typer.Option("rf", help="linear | rf | xgb | lstm"),
    seq_len: int = typer.Option(20, help="Comprimento da janela (LSTM)"),
):
    """
    Gera sinais e salva em data/signals/signals.parquet.
    """
    features_path = config.ANALYTICS_DIR / "features.parquet"
    signals_path = config.SIGNALS_DIR / "signals.parquet"

    feats = load_parquet(features_path)
    if feats is None or feats.empty:
        typer.secho(f"[signals] Arquivo não encontrado: {features_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    feats = feats.sort_values(["ticker", "date"]).reset_index(drop=True)
    exclude = {"date", "ticker", "target_reg_5d", "target_cls_5d"}
    X = feats[[c for c in feats.columns if c not in exclude and pd.api.types.is_numeric_dtype(feats[c])]].fillna(0)
    base = feats[["date", "ticker"]].copy()

    out = base.copy()

    # REG
    if mode in {"reg", "both"}:
        model, path = _load_model(algo, "reg")
        if algo == "lstm":
            cols = feature_columns(feats)
            data = prepare_sequences_infer(feats, cols, seq_len=seq_len)
            if len(data.X) == 0:
                typer.secho("[signals][lstm][reg] Dados insuficientes para inferência.", fg=typer.colors.RED)
            else:
                yhat = model.predict(data.X).reshape(-1)
                dfp = data.index.copy()
                dfp["signal_reg"] = [ _to_signal_from_regression(v, threshold=0.0) for v in yhat ]
                out = out.merge(dfp[["ticker","date","signal_reg"]], on=["ticker","date"], how="left")
        else:
            yhat = model.predict(X)
            out["signal_reg"] = [ _to_signal_from_regression(v, threshold=0.0) for v in yhat ]

    # CLS
    if mode in {"cls", "both"}:
        model, path = _load_model(algo, "cls")
        if algo == "lstm":
            cols = feature_columns(feats)
            data = prepare_sequences_infer(feats, cols, seq_len=seq_len)
            if len(data.X) == 0:
                typer.secho("[signals][lstm][cls] Dados insuficientes para inferência.", fg=typer.colors.RED)
            else:
                prob = model.predict(data.X).reshape(-1)
                dfp = data.index.copy()
                dfp["signal_cls"] = [ _to_signal_from_classification(p, threshold=0.5) for p in prob ]
                out = out.merge(dfp[["ticker","date","signal_cls"]], on=["ticker","date"], how="left")
        else:
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X)[:, 1]
            else:
                prob = model.decision_function(X)
            out["signal_cls"] = [ _to_signal_from_classification(p, threshold=0.5) for p in prob ]

    for col in ["signal_reg", "signal_cls"]:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int)

    save_parquet(out, signals_path, index=False)
    typer.secho(f"[signals] OK: {len(out)} linhas salvas em {signals_path} (algo={algo}, mode={mode})", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
