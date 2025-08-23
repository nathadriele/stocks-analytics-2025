"""
Geração de sinais de trading a partir de previsões dos modelos.

- Lê features em data/analytics/features.parquet
- Carrega modelos treinados (regressão e/ou classificação)
- Converte previsões em sinais de trading (long, short, flat)
- Salva sinais em data/signals/signals.parquet

Uso:
    python -m src.strategy.generate_signals --mode both
"""

from __future__ import annotations

import typer
import joblib
import pandas as pd
from pathlib import Path

from src import config
from src.utils.io import load_parquet, save_parquet

app = typer.Typer(add_completion=False, help="Geração de sinais de trading")


def _to_signal_from_regression(pred: float, threshold: float = 0.0) -> int:
    """
    Converte previsão de retorno (regressão) em sinal.
    - Se retorno previsto > threshold → long (+1)
    - Se retorno previsto < threshold → short (-1)
    - Caso contrário → flat (0)
    """
    if pred > threshold:
        return 1
    elif pred < threshold:
        return -1
    return 0


def _to_signal_from_classification(prob: float, threshold: float = 0.5) -> int:
    """
    Converte probabilidade prevista em sinal.
    - prob > threshold → long (+1)
    - prob <= threshold → short (-1)
    """
    return 1 if prob > threshold else -1


@app.command("run")
def run(mode: str = typer.Option("both", help="Opções: reg, cls, both")):
    """
    Executa geração de sinais.
    """
    features_path = config.ANALYTICS_DIR / "features.parquet"
    signals_path = config.SIGNALS_DIR / "signals.parquet"

    features = load_parquet(features_path)
    if features is None or features.empty:
        typer.secho(f"[signals] Arquivo não encontrado: {features_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    exclude_cols = ["date", "ticker", "target_reg_5d", "target_cls_5d"]
    X = features[[c for c in features.columns if c not in exclude_cols]].fillna(0)

    df_signals = features[["date", "ticker"]].copy()

    if mode in ("reg", "both"):
        model_reg = joblib.load(config.MODELS_DIR / "model_reg.pkl")
        preds_reg = model_reg.predict(X)
        df_signals["signal_reg"] = [ _to_signal_from_regression(p, threshold=0.0) for p in preds_reg ]

    if mode in ("cls", "both"):
        model_cls = joblib.load(config.MODELS_DIR / "model_cls.pkl")
        probs_cls = model_cls.predict_proba(X)[:, 1]
        df_signals["signal_cls"] = [ _to_signal_from_classification(p, threshold=0.5) for p in probs_cls ]

    save_parquet(df_signals, signals_path, index=False)
    typer.secho(f"[signals] OK: {len(df_signals)} sinais salvos em {signals_path}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
