# src/eda/eda_report.py
from __future__ import annotations
import argparse, os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="Gera EDA (alvo, correlações, cortes por ticker/mês).")
    p.add_argument("--data-path", default="data/analytics/features.parquet")
    p.add_argument("--features-file", default="src/features/feature_list.txt")
    p.add_argument("--date-col", default="date")
    p.add_argument("--ticker-col", default="ticker")
    p.add_argument("--target", default=None, help="Ex.: target_up_5d ou target_up; se None, tenta inferir.")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--img-dir", default="reports/img")
    return p.parse_args()

def _read_df(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

def _read_features(path: str):
    if not os.path.exists(path): return None
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

def main():
    a = parse_args()
    os.makedirs(a.reports_dir, exist_ok=True)
    os.makedirs(a.img_dir, exist_ok=True)
    df = _read_df(a.data_path)
    df[a.date_col] = pd.to_datetime(df[a.date_col])

    # infer target
    target = a.target
    if target is None:
        for cand in ["target_up_5d","target_up"]:
            if cand in df.columns:
                target = cand; break
    if target is None and "ret_5d" in df.columns:
        df["target_up_5d"] = (df["ret_5d"] > 0).astype(int)
        target = "target_up_5d"
    if target is None:
        raise ValueError("Não foi possível inferir o target.")

    # distribuição do alvo
    tgt_rate = df[target].mean()
    plt.figure(figsize=(5,3))
    df[target].value_counts().sort_index().plot(kind="bar")
    plt.title(f"Target distribution ({target})  |  mean={tgt_rate:.2f}")
    plt.tight_layout()
    tgt_img = os.path.join(a.img_dir, "target_hist.png")
    plt.savefig(tgt_img, dpi=140); plt.close()

    # correlações com o target (numéricas)
    feats = _read_features(a.features_file) or df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in feats if c in df.columns]
    num = df[feats].select_dtypes(include=[np.number])
    # encode target numérico
    df["_tgt_"] = df[target].astype(float)
    corr = num.corrwith(df["_tgt_"]).sort_values(ascending=False)

    # heatmap (top 25 por abs)
    top = corr.abs().sort_values(ascending=False).head(25).index.tolist()
    C = df[top + ["_tgt_"]].corr()
    plt.figure(figsize=(10,8))
    import matplotlib
    im = plt.imshow(C, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(C.columns)), C.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(C.index)), C.index, fontsize=8)
    plt.title("Correlation Heatmap (top 25 by |corr|)")
    plt.tight_layout()
    corr_img = os.path.join(a.img_dir, "corr_heatmap.png")
    plt.savefig(corr_img, dpi=140); plt.close()

    # cortes por mês e por ticker (média do target)
    df["month"] = df[a.date_col].dt.month
    by_month = df.groupby("month")[target].mean().rename("target_mean_by_month")
    by_ticker = df.groupby(a.ticker_col)[target].mean().rename("target_mean_by_ticker").sort_values(ascending=False)

    # gráficos
    plt.figure(figsize=(8,3))
    by_month.plot(kind="bar"); plt.title("Target mean by month"); plt.tight_layout()
    month_img = os.path.join(a.img_dir, "target_by_month.png")
    plt.savefig(month_img, dpi=140); plt.close()

    # top 20 tickers
    plt.figure(figsize=(10,4))
    by_ticker.head(20).plot(kind="bar"); plt.title("Target mean by ticker (Top 20)"); plt.tight_layout()
    tick_img = os.path.join(a.img_dir, "target_by_ticker.png")
    plt.savefig(tick_img, dpi=140); plt.close()

    # markdown
    md = os.path.join(a.reports_dir, "eda_summary.md")
    with open(md, "w") as f:
        f.write("# EDA Summary\n\n")
        f.write(f"- Dataset: `{a.data_path}`\n")
        f.write(f"- Rows: **{len(df):,}** | Columns: **{df.shape[1]}**\n")
        f.write(f"- Target: **{target}** | Positive rate: **{tgt_rate:.2f}**\n\n")
        f.write("## Correlations with target (top 20)\n\n")
        f.write(corr.sort_values(key=np.abs, ascending=False).head(20).to_frame("corr").to_markdown())
        f.write("\n\n")
        f.write("## Plots\n\n")
        f.write(f"![Target]({os.path.relpath(tgt_img, a.reports_dir)})\n\n")
        f.write(f"![Heatmap]({os.path.relpath(corr_img, a.reports_dir)})\n\n")
        f.write(f"![By Month]({os.path.relpath(month_img, a.reports_dir)})\n\n")
        f.write(f"![By Ticker]({os.path.relpath(tick_img, a.reports_dir)})\n\n")

    print(f"[OK] EDA gerada em: {md}")

if __name__ == "__main__":
    main()
