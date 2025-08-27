# src/reports/make_report.py
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class Paths:
    backtests_dir: str = "data/backtests"
    exact_dir: str = "data/backtests/exact"
    reports_dir: str = "reports"
    img_dir: str = "reports/img"


def _ensure_dirs(p: Paths) -> None:
    os.makedirs(p.reports_dir, exist_ok=True)
    os.makedirs(p.img_dir, exist_ok=True)


def _load_table(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Formato não suportado (use parquet/csv): {path}")


def _collect_vector(backtests_dir: str) -> List[Dict]:
    """Lê summaries e equity dos backtests vetoriais."""
    out = []
    for summ_path in sorted(glob.glob(os.path.join(backtests_dir, "summary_*.json"))):
        name = os.path.splitext(os.path.basename(summ_path))[0].replace("summary_", "")
        equity_parq = os.path.join(backtests_dir, f"equity_{name}.parquet")
        equity_csv = os.path.join(backtests_dir, f"equity_{name}.csv")
        eq_path = equity_parq if os.path.exists(equity_parq) else equity_csv if os.path.exists(equity_csv) else None
        if eq_path is None:
            # pula se não houver timeseries
            continue
        with open(summ_path, "r") as f:
            summ = json.load(f)
        eq = _load_table(eq_path)
        # normaliza índice de data
        if "date" in eq.columns:
            eq["date"] = pd.to_datetime(eq["date"])
            eq = eq.set_index("date").sort_index()
        elif eq.index.name is None:
            # pode vir com index numérico; tenta inferir coluna compatível
            pass
        out.append({
            "name": name,
            "kind": "vector",
            "summary": summ,
            "equity": eq,
            "bench_available": "bench_equity" in eq.columns,
        })
    return out


def _collect_exact(exact_dir: str) -> List[Dict]:
    """Lê summaries e equity dos backtests exatos."""
    out = []
    for summ_path in sorted(glob.glob(os.path.join(exact_dir, "summary_exact_*.json"))):
        name = os.path.splitext(os.path.basename(summ_path))[0].replace("summary_exact_", "")
        equity_parq = os.path.join(exact_dir, f"equity_exact_{name}.parquet")
        equity_csv = os.path.join(exact_dir, f"equity_exact_{name}.csv")
        eq_path = equity_parq if os.path.exists(equity_parq) else equity_csv if os.path.exists(equity_csv) else None
        if eq_path is None:
            continue
        with open(summ_path, "r") as f:
            summ = json.load(f)
        eq = _load_table(eq_path)
        if "date" in eq.columns:
            eq["date"] = pd.to_datetime(eq["date"])
            eq = eq.set_index("date").sort_index()
        out.append({
            "name": name,
            "kind": "exact",
            "summary": summ,
            "equity": eq,
            "bench_available": False,  # exato não carrega bench (opcional no futuro)
        })
    return out


def _format_pct(x: float) -> str:
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "-"


def _format_num(x: float) -> str:
    try:
        return f"{x:,.2f}"
    except Exception:
        return "-"


def _rolling_sharpe(ret: pd.Series, window: int = 126) -> pd.Series:
    if len(ret) < 2:
        return pd.Series([], dtype=float)
    r = ret.rolling(window=window)
    mu = r.mean()
    sd = r.std(ddof=0)
    out = (mu / (sd + 1e-12)) * np.sqrt(252.0)
    return out


def _drawdown_series(equity: pd.Series) -> pd.Series:
    if equity.isna().all() or len(equity) == 0:
        return equity * np.nan
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return dd


def _align_indices(series_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """Alinha diversas séries pelo índice (date) com outer join."""
    df = pd.DataFrame(index=pd.Index([], name="date"))
    for k, s in series_dict.items():
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        df = df.join(s.rename(k), how="outer")
    return df.sort_index()


def build_report(paths: Paths, rolling_win: int = 126, top_n_plot: int = 6) -> Tuple[pd.DataFrame, Dict[str, str]]:
    _ensure_dirs(paths)

    # coleta
    vec = _collect_vector(paths.backtests_dir)
    ext = _collect_exact(paths.exact_dir)
    runs = vec + ext

    if not runs:
        raise RuntimeError("Nenhum backtest encontrado. Rode primeiro os alvos de backtest.")

    # tabela comparativa
    rows = []
    for r in runs:
        summ = r["summary"]
        name = r["name"]
        kind = r["kind"]
        m = summ.get("metrics", {})
        rows.append({
            "strategy": name,
            "type": kind,
            "final_equity": summ.get("final_equity", np.nan),
            "tot_return": summ.get("tot_return", np.nan),
            "cagr": m.get("cagr", np.nan),
            "sharpe": m.get("sharpe", np.nan),
            "sortino": m.get("sortino", np.nan),
            "vol_ann": m.get("vol_ann", np.nan),
            "max_drawdown": m.get("max_drawdown", np.nan),
            "winrate": m.get("winrate", np.nan),
            "avg_turnover": m.get("avg_turnover", np.nan),
        })

    cmp_df = pd.DataFrame(rows).sort_values(["type", "cagr", "sharpe"], ascending=[True, False, False])

    # === Curvas para gráfico ===
    # Equity por estratégia
    eq_map: Dict[str, pd.Series] = {}
    # Benchmark preferencial: se houver bench_equity em algum vetorial, usa o de maior interseção de datas
    bench_series: Optional[pd.Series] = None
    max_len = -1

    for r in runs:
        eq = r["equity"]
        if "equity" in eq.columns:
            eq_map[f"{r['type']}:{r['name']}"] = eq["equity"].astype(float)
        # preenche benchmark se existir e for o mais longo
        if r["bench_available"] and "bench_equity" in eq.columns:
            s = eq["bench_equity"].astype(float)
            if len(s.dropna()) > max_len:
                bench_series = s
                max_len = len(s.dropna())

    # escolhe top_n_plot por Sharpe
    top_names = (
        cmp_df.sort_values(["sharpe", "cagr"], ascending=False)
        .head(top_n_plot)["strategy"]
        .tolist()
    )
    # filtra map para top
    eq_map_plot = {k: v for k, v in eq_map.items() if k.split(":", 1)[1] in top_names}

    # Alinha para plot
    curves = _align_indices(eq_map_plot)
    if bench_series is not None:
        bench_df = _align_indices({"Benchmark": bench_series})
        curves = curves.join(bench_df, how="outer")

    img_paths: Dict[str, str] = {}

    # === Equity curves ===
    plt.figure(figsize=(12, 6))
    for col in curves.columns:
        curves[col].plot()
    plt.title("Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    eq_img = os.path.join(paths.img_dir, "equity_curves.png")
    plt.savefig(eq_img, dpi=140)
    plt.close()
    img_paths["equity"] = os.path.relpath(eq_img, start=paths.reports_dir)

    # === Rolling Sharpe (126d) ===
    roll_map = {}
    for col in curves.columns:
        # converte equity -> ret diário
        eq = curves[col].dropna()
        ret = eq.pct_change().fillna(0.0)
        roll_map[col] = _rolling_sharpe(ret, window=rolling_win)

    roll_df = _align_indices(roll_map)
    plt.figure(figsize=(12, 6))
    for col in roll_df.columns:
        roll_df[col].plot()
    plt.title(f"Rolling Sharpe ({rolling_win}d)")
    plt.xlabel("Date")
    plt.ylabel("Sharpe")
    plt.tight_layout()
    rs_img = os.path.join(paths.img_dir, f"rolling_sharpe_{rolling_win}d.png")
    plt.savefig(rs_img, dpi=140)
    plt.close()
    img_paths["rolling_sharpe"] = os.path.relpath(rs_img, start=paths.reports_dir)

    # === Drawdown ===
    dd_map = {}
    for col in curves.columns:
        eq = curves[col].dropna()
        dd_map[col] = _drawdown_series(eq)
    dd_df = _align_indices(dd_map)
    plt.figure(figsize=(12, 6))
    for col in dd_df.columns:
        dd_df[col].plot()
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    dd_img = os.path.join(paths.img_dir, "drawdown.png")
    plt.savefig(dd_img, dpi=140)
    plt.close()
    img_paths["drawdown"] = os.path.relpath(dd_img, start=paths.reports_dir)

    # === Markdown report ===
    md_path = os.path.join(paths.reports_dir, "backtest_results.md")
    with open(md_path, "w") as f:
        f.write("# Backtest Results\n\n")
        f.write("## Summary Table\n\n")

        # tabela em markdown
        show = cmp_df.copy()
        show["final_equity"] = show["final_equity"].map(_format_num)
        show["tot_return"] = show["tot_return"].map(_format_pct)
        show["cagr"] = show["cagr"].map(_format_pct)
        show["vol_ann"] = show["vol_ann"].map(_format_pct)
        show["sharpe"] = show["sharpe"].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")
        show["sortino"] = show["sortino"].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")
        show["max_drawdown"] = show["max_drawdown"].map(_format_pct)
        show["winrate"] = show["winrate"].map(_format_pct)
        show["avg_turnover"] = show["avg_turnover"].map(_format_pct)

        f.write(show.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Equity Curves\n\n")
        f.write(f"![Equity Curves]({img_paths['equity']})\n\n")

        f.write(f"## Rolling Sharpe ({rolling_win}d)\n\n")
        f.write(f"![Rolling Sharpe]({img_paths['rolling_sharpe']})\n\n")

        f.write("## Drawdown\n\n")
        f.write(f"![Drawdown]({img_paths['drawdown']})\n\n")

        f.write("---\n")
        f.write("_Report generated by `src/reports/make_report.py`_\n")

    return show, img_paths, md_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gera relatório consolidado (tabela + gráficos) dos backtests.")
    p.add_argument("--backtests-dir", default="data/backtests")
    p.add_argument("--exact-dir", default="data/backtests/exact")
    p.add_argument("--out-reports-dir", default="reports")
    p.add_argument("--rolling-window", type=int, default=126)
    p.add_argument("--top-n-plot", type=int, default=6, help="Máximo de estratégias a plotar (por Sharpe).")
    return p.parse_args()


def main():
    args = parse_args()
    paths = Paths(
        backtests_dir=args.backtests_dir,
        exact_dir=args.exact_dir,
        reports_dir=args.out_reports_dir,
        img_dir=os.path.join(args.out_reports_dir, "img"),
    )
    show, imgs, md = build_report(paths, rolling_win=args.rolling_window, top_n_plot=args.top_n_plot)
    print("[OK] Relatório gerado:")
    print(" - Tabela:")
    print(show.head(20).to_string(index=False))
    print(f" - Markdown: {md}")
    for k, v in imgs.items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    main()
