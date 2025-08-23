"""
Geração de previsões diárias a partir dos modelos treinados.

Fluxo:
- Lê features em data/analytics/features.parquet
- Seleciona o "último dia disponível" (ou uma data alvo via --date)
- Carrega os modelos em models/ (reg e cls, se existirem)
- Gera previsões (y_reg_pred, y_cls_prob) por ticker
- Salva em data/analytics/predictions_daily.parquet (append seguro por timestamp de geração)

Uso:
    python -m src.app.predict_daily
    python -m src.app.predict_daily --date 2025-05-19
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import typer

from src import config
from src.utils.io import load_parquet, save_parquet

app = typer.Typer(add_completion=False, help="Previsões diárias (modelos reg e cls)")


def _pick_feature_rows(features: pd.DataFrame, target_date: Optional[str]) -> pd.DataFrame:
    """
    Seleciona as linhas de features para prever.
    - Se target_date for None: pega o último dia disponível por ticker.
    - Se target_date existir: filtra exatamente aquela data.
    """
    df = features.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    if target_date:
        dt = pd.to_datetime(target_date).tz_localize(None)
        sel = df[df["date"] == dt].copy()
        if sel.empty:
            raise typer.Exit(f"[predict_daily] Não há features para a data {target_date}")
        return sel

    idx = df.groupby("ticker")["date"].transform("max") == df["date"]
    sel = df[idx].copy()
    if sel.empty:
        raise typer.Exit("[predict_daily] Não foi possível encontrar último dia por ticker.")
    return sel


def _prepare_X(features: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara matriz de features X limpando colunas não-numéricas/targets.
    """
    exclude = {"date", "ticker", "target_reg_5d", "target_cls_5d"}
    cols = [c for c in features.columns if c not in exclude]
    X = features[cols].fillna(0)
    return X


def _load_model(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


@app.command("run")
def run(
    date: Optional[str] = typer.Option(
        None,
        "--date",
        "-d",
        help="Data exata YYYY-MM-DD para gerar previsões; se omitido, usa o último dia disponível.",
    )
):
    """
    Executa a geração de previsões diárias e persiste em predictions_daily.parquet.
    """
    features_path = config.ANALYTICS_DIR / "features.parquet"
    out_path = config.ANALYTICS_DIR / "predictions_daily.parquet"

    features = load_parquet(features_path)
    if features is None or features.empty:
        typer.secho(f"[predict_daily] Arquivo de features inexistente: {features_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    feat_sel = _pick_feature_rows(features, date)
    X = _prepare_X(feat_sel)

    model_reg = _load_model(config.MODELS_DIR / "model_reg.pkl")
    model_cls = _load_model(config.MODELS_DIR / "model_cls.pkl")

    if model_reg is None and model_cls is None:
        typer.secho("[predict_daily] Nenhum modelo encontrado em models/. Treine antes de prever.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    preds = feat_sel[["date", "ticker"]].copy()

    if model_reg is not None:
        y_reg = model_reg.predict(X)
        preds["y_reg_pred_5d"] = y_reg

    if model_cls is not None:
        try:
            y_prob = model_cls.predict_proba(X)[:, 1]
        except Exception:
            y_prob = model_cls.decision_function(X)
        preds["y_cls_prob_up_5d"] = y_prob

    preds["generated_at"] = pd.Timestamp.utcnow().tz_localize(None)

    if out_path.exists():
        old = pd.read_parquet(out_path)
        merged = pd.concat([old, preds], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date", "ticker", "generated_at"])
        save_parquet(merged, out_path, index=False)
    else:
        save_parquet(preds, out_path, index=False)

    n_rows = len(preds)
    day = preds["date"].iloc[0].date()
    typer.secho(f"[predict_daily] OK: {n_rows} previsões para {day} salvas em {out_path}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
