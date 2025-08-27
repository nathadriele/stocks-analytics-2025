# src/strategy/generate_signals.py
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _read_pred_files(pred_dir: str, explicit_paths: Optional[List[str]] = None) -> List[pd.DataFrame]:
    """Lê 1+ arquivos de predição (parquet ou csv) contendo ao menos: date, [ticker], proba_up."""
    paths = []
    if explicit_paths:
        paths = explicit_paths
    else:
        paths = sorted(glob.glob(os.path.join(pred_dir, "pred_*.parquet"))) + \
                sorted(glob.glob(os.path.join(pred_dir, "pred_*.csv")))

    if not paths:
        raise FileNotFoundError(f"Nenhuma predição encontrada em {pred_dir} (esperado pred_*.parquet/csv)")

    dfs = []
    for p in paths:
        if p.endswith(".parquet"):
            df = pd.read_parquet(p)
        elif p.endswith(".csv"):
            df = pd.read_csv(p)
        else:
            continue
        # normaliza colunas
        cols = {c.lower(): c for c in df.columns}
        # exige 'date' e 'proba_up'
        # ticker é fortemente recomendado
        assert "date" in [c.lower() for c in df.columns], f"'date' ausente em {p}"
        assert "proba_up" in [c.lower() for c in df.columns], f"'proba_up' ausente em {p}"

        df = df.rename(columns={cols.get("date", "date"): "date",
                                cols.get("ticker", "ticker"): "ticker",
                                cols.get("proba_up", "proba_up"): "proba_up"})
        if "ticker" not in df.columns:
            df["ticker"] = "ALL"  # fallback
        df["model_name"] = os.path.splitext(os.path.basename(p))[0].replace("pred_", "")
        dfs.append(df[["date", "ticker", "proba_up", "model_name"]].copy())

    return dfs


def _ensemble_mean(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Junta por média de probabilidade quando existem múltiplos arquivos."""
    df = dfs[0].copy()
    if len(dfs) == 1:
        df["proba_up_ens"] = df["proba_up"]
        df["model_name"] = df["model_name"].astype(str)
        return df.rename(columns={"proba_up_ens": "proba_up"})

    # merge incremental por ['date','ticker']
    base = dfs[0][["date", "ticker"]].drop_duplicates().copy()
    stack = []
    for x in dfs:
        stack.append(x[["date", "ticker", "proba_up"]].rename(columns={"proba_up": f"p_{len(stack)}"}))
    out = base
    for s in stack:
        out = out.merge(s, on=["date", "ticker"], how="outer")

    prob_cols = [c for c in out.columns if c.startswith("p_")]
    out["proba_up"] = out[prob_cols].mean(axis=1)
    out["model_name"] = "ensemble_mean"
    out = out[["date", "ticker", "proba_up", "model_name"]].dropna(subset=["proba_up"]).copy()
    return out


def _apply_lag(df: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    """Desloca a 'data' do sinal em +lag_days (sem look-ahead na execução)."""
    if lag_days <= 0:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]) + pd.to_timedelta(lag_days, unit="D")
    return df


def strategy_long_only_threshold(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    out = df.copy()
    out["signal"] = (out["proba_up"] >= thr).astype(int)  # 1 ou 0
    out["weight"] = out["signal"].astype(float)           # peso 1.0 quando 1
    out["strategy"] = f"long_only_threshold_{thr:.2f}"
    return out


def strategy_long_short_threshold(df: pd.DataFrame, thr_long: float, thr_short: float) -> pd.DataFrame:
    out = df.copy()
    long_ = (out["proba_up"] >= thr_long).astype(int)
    short_ = (out["proba_up"] <= thr_short).astype(int) * -1
    out["signal"] = (long_ + short_)  # +1, 0 ou -1
    out["weight"] = out["signal"].astype(float)
    out["strategy"] = f"long_short_threshold_L{thr_long:.2f}_S{thr_short:.2f}"
    return out


def strategy_topk_long_only(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Para cada data, seleciona top-k por probabilidade, equal-weight (1 para selecionados, 0 caso contrário)."""
    out = df.copy()
    out["rank"] = out.groupby("date")["proba_up"].rank(method="first", ascending=False)
    out["signal"] = (out["rank"] <= k).astype(int)
    # equal-weight entre os selecionados
    # peso somado por data = min(k, n_dia)
    denom = out.groupby("date")["signal"].transform(lambda s: max(1, int(s.sum())))
    out["weight"] = out["signal"] / denom
    out["strategy"] = f"topk_long_only_k{k}"
    return out.drop(columns=["rank"])


def strategy_prob_weighted(df: pd.DataFrame, center: float = 0.5, scale: float = 2.0, clip: Tuple[float, float] = (0.0, 1.0)) -> pd.DataFrame:
    """Peso contínuo proporcional à probabilidade.
    weight = clip( (proba - center) * scale, [0..1] ); sinal 1 se weight>0, senão 0.
    """
    out = df.copy()
    w = (out["proba_up"] - center) * scale
    w = w.clip(lower=clip[0], upper=clip[1])
    out["weight"] = w
    out["signal"] = (out["weight"] > 0).astype(int)
    out["strategy"] = f"prob_weighted_c{center:.2f}_s{scale:.1f}"
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Geração de sinais de trading a partir de probabilidades.")
    p.add_argument("--pred-dir", default="data/analytics", help="Diretório com arquivos pred_*.parquet/csv")
    p.add_argument("--pred-files", default=None, help="Lista de caminhos separados por vírgulas (opcional).")
    p.add_argument("--ensemble", choices=["mean", "none"], default="mean", help="Como combinar múltiplas predições.")
    p.add_argument("--strategy", choices=["long_only_threshold", "long_short_threshold", "topk_long_only", "prob_weighted"], default="long_only_threshold")
    p.add_argument("--thr", type=float, default=0.60, help="Threshold para long_only_threshold (padrão: 0.60).")
    p.add_argument("--thr-long", type=float, default=0.65, help="Threshold long (long_short_threshold).")
    p.add_argument("--thr-short", type=float, default=0.40, help="Threshold short (long_short_threshold).")
    p.add_argument("--topk", type=int, default=5, help="k para topk_long_only.")
    p.add_argument("--center", type=float, default=0.50, help="Centro do peso para prob_weighted.")
    p.add_argument("--scale", type=float, default=2.0, help="Escala do peso para prob_weighted.")
    p.add_argument("--clip-low", type=float, default=0.0, help="Clipping inferior do peso contínuo.")
    p.add_argument("--clip-high", type=float, default=1.0, help="Clipping superior do peso contínuo.")
    p.add_argument("--lag-days", type=int, default=1, help="Dias para deslocar a data do sinal (evita look-ahead).")
    p.add_argument("--out-dir", default="data/signals", help="Diretório de saída para sinais.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # carregar previsões
    explicit = [s.strip() for s in args.pred_files.split(",")] if args.pred_files else None
    dfs = _read_pred_files(args.pred_dir, explicit_paths=explicit)

    # ensemble
    if args.ensemble == "mean":
        df = _ensemble_mean(dfs)
    else:
        # usa a primeira (ou única) previsão
        df = dfs[0].copy()

    # ordena e aplica lag de execução
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    df = _apply_lag(df, args.lag_days)

    # aplica estratégia
    if args.strategy == "long_only_threshold":
        out = strategy_long_only_threshold(df, thr=args.thr)
    elif args.strategy == "long_short_threshold":
        out = strategy_long_short_threshold(df, thr_long=args.thr_long, thr_short=args.thr_short)
    elif args.strategy == "topk_long_only":
        out = strategy_topk_long_only(df, k=args.topk)
    elif args.strategy == "prob_weighted":
        out = strategy_prob_weighted(df, center=args.center, scale=args.scale, clip=(args.clip_low, args.clip_high))
    else:
        raise ValueError("Estratégia inválida.")

    cols = ["date", "ticker", "signal", "weight", "proba_up", "model_name", "strategy"]
    out = out[cols].copy()
    out["date"] = pd.to_datetime(out["date"])

    base = args.strategy
    pq_path = os.path.join(args.out_dir, f"signals_{base}.parquet")
    csv_path = os.path.join(args.out_dir, f"signals_{base}.csv")
    out.to_parquet(pq_path, index=False)
    out.to_csv(csv_path, index=False)

    meta = {
        "strategy": args.strategy,
        "records": int(len(out)),
        "start_date": str(out["date"].min().date()) if len(out) else None,
        "end_date": str(out["date"].max().date()) if len(out) else None,
        "params": {
            "ensemble": args.ensemble,
            "thr": args.thr,
            "thr_long": args.thr_long,
            "thr_short": args.thr_short,
            "topk": args.topk,
            "center": args.center,
            "scale": args.scale,
            "clip_low": args.clip_low,
            "clip_high": args.clip_high,
            "lag_days": args.lag_days,
            "pred_dir": args.pred_dir,
            "pred_files": explicit,
        },
        "outputs": {"parquet": pq_path, "csv": csv_path},
    }
    with open(os.path.join(args.out_dir, "_last_signals_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Sinais gerados: {pq_path}")
    print(f"      Linhas: {meta['records']} | Período: {meta['start_date']} → {meta['end_date']}")


if __name__ == "__main__":
    main()
