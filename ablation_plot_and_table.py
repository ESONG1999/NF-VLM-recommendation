import re
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ===================== 解析 metrics 列 =====================

def parse_metrics_str(m):
    """
    m 形如: "[Test-FullRanking] {'AUC': 0.80, 'NDCG@10': 0.31, 'Recall@10': 0.36}"
    返回 dict: {"AUC": 0.80, "NDCG@10": 0.31, "Recall@10": 0.36}
    """
    if not isinstance(m, str):
        return {}

    # 提取大括号里的部分
    match = re.search(r"\{.*\}", m)
    if not match:
        return {}

    d_str = match.group(0)
    try:
        d = ast.literal_eval(d_str)
        return d
    except Exception:
        return {}


def load_ablation_csv(path):
    df = pd.read_csv(path)

    # 解析 metrics 列
    metrics_parsed = df["metrics"].apply(parse_metrics_str)
    df["AUC"] = metrics_parsed.apply(lambda d: d.get("AUC", np.nan))
    df["NDCG10"] = metrics_parsed.apply(lambda d: d.get("NDCG@10", np.nan))
    df["Recall10"] = metrics_parsed.apply(lambda d: d.get("Recall@10", np.nan))

    # 只保留 nfvlm（你也可以留 baseline 做对比）
    df_nf = df[df["model"] == "nfvlm"].copy()

    # 类型转换
    df_nf["nf_rank"] = pd.to_numeric(df_nf["nf_rank"], errors="coerce")
    df_nf["nf_reg"] = pd.to_numeric(df_nf["nf_reg"], errors="coerce")

    return df, df_nf


# ===================== 图 1：rank → NDCG@10 曲线 =====================

def plot_rank_vs_ndcg(df_nf, out_png="rank_vs_ndcg10.png", mode="best_over_reg"):
    """
    mode:
      - "best_over_reg": 对每个 rank 取在不同 reg 中 NDCG10 最高的那一个
      - "mean_over_reg": 对每个 rank 在不同 reg 上取平均 NDCG10
    """
    group = df_nf.groupby("nf_rank")

    rows = []
    for r, g in group:
        if mode == "best_over_reg":
            val = g["NDCG10"].max()
        else:
            val = g["NDCG10"].mean()
        rows.append((r, val))

    res = pd.DataFrame(rows, columns=["nf_rank", "NDCG10"]).sort_values("nf_rank")

    plt.figure()
    plt.plot(res["nf_rank"], res["NDCG10"], marker="o")
    plt.xlabel("NF Rank")
    plt.ylabel("NDCG@10")
    plt.title(f"NF Rank vs NDCG@10 ({mode})")
    plt.grid(True)
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"[Saved] {out_png}")


# ===================== 图 2：noise_reg → Recall@10 曲线 =====================

def plot_reg_vs_recall(df_nf, out_png="reg_vs_recall10.png", mode="best_over_rank"):
    """
    mode:
      - "best_over_rank": 对每个 reg 取在不同 rank 中 Recall10 最高的那一个
      - "mean_over_rank": 对每个 reg 在不同 rank 上取平均 Recall10
    """
    group = df_nf.groupby("nf_reg")

    rows = []
    for lam, g in group:
        if mode == "best_over_rank":
            val = g["Recall10"].max()
        else:
            val = g["Recall10"].mean()
        rows.append((lam, val))

    res = pd.DataFrame(rows, columns=["nf_reg", "Recall10"]).sort_values("nf_reg")

    plt.figure()
    plt.plot(res["nf_reg"], res["Recall10"], marker="o")
    plt.xlabel("Noise regularization λ")
    plt.ylabel("Recall@10")
    plt.title(f"Noise Reg vs Recall@10 ({mode})")
    plt.xscale("log")  # λ 通常是 log-scale
    plt.grid(True)
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"[Saved] {out_png}")


# ===================== 图 3：heatmap（rank × reg → NDCG@10） =====================

def plot_heatmap_rank_reg(
    df_nf,
    metric="NDCG10",
    out_png="heatmap_rank_reg_ndcg10.png",
    agg="mean",
):
    """
    metric: "NDCG10" 或 "Recall10" 等
    agg: "mean" 或 "max"
    """
    if agg == "mean":
        table = df_nf.pivot_table(
            index="nf_rank",
            columns="nf_reg",
            values=metric,
            aggfunc="mean",
        )
    else:
        table = df_nf.pivot_table(
            index="nf_rank",
            columns="nf_reg",
            values=metric,
            aggfunc="max",
        )

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        table,
        annot=True,
        fmt=".3f",
        cmap="viridis",
    )
    plt.title(f"{metric} over (rank, λ) [{agg}]")
    plt.xlabel("Noise regularization λ")
    plt.ylabel("NF Rank")
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"[Saved] {out_png}")


# ===================== LaTeX 表格生成 =====================

def generate_latex_table_rank_reg(df_nf, metric="NDCG10"):
    """
    生成一个 NeurIPS 风格 LaTeX 表格：
      Rows: nf_rank
      Cols: nf_reg
      Cell: metric value (比如 NDCG@10)，保留3位小数
    """

    table = df_nf.pivot_table(
        index="nf_rank",
        columns="nf_reg",
        values=metric,
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)

    ranks = list(table.index)
    regs = list(table.columns)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l" + "c" * len(regs) + "}")
    lines.append(r"\toprule")

    # header
    reg_header = " & " + " & ".join([rf"$\lambda={lam}$" for lam in regs]) + r" \\"
    lines.append(r"Rank $\backslash$ $\lambda$" + reg_header)
    lines.append(r"\midrule")

    # rows
    for r in ranks:
        row_vals = []
        for lam in regs:
            v = table.loc[r, lam]
            if pd.isna(v):
                row_vals.append("--")
            else:
                row_vals.append(f"{v:.3f}")
        line = f"{r} & " + " & ".join(row_vals) + r" \\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Ablation on NF rank and noise regularization $\lambda$ using " + metric + r".}")
    lines.append(r"\label{tab:nf_ablation_" + metric.lower() + r"}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)
    print("\n================ LaTeX Table ================\n")
    print(latex_str)
    print("\n=============================================\n")
    return latex_str

def generate_baseline_vs_nfvlm_table(df_all):
    """
    输入完整的 ablation CSV（包含 baseline + nfvlm）
    输出：
        一个 latex 表格
        自动选择 NF-VLM 的最佳 NDCG@10（或可修改成 Recall）
    """

    # 解析 metrics
    metrics_parsed = df_all["metrics"].apply(parse_metrics_str)
    df_all["AUC"] = metrics_parsed.apply(lambda d: d.get("AUC", np.nan))
    df_all["NDCG10"] = metrics_parsed.apply(lambda d: d.get("NDCG@10", np.nan))
    df_all["Recall10"] = metrics_parsed.apply(lambda d: d.get("Recall@10", np.nan))

    # ========= Baselines: MLP / MF =========
    df_mlp = df_all[df_all["model"] == "mlp"]
    df_mf  = df_all[df_all["model"] == "mf"]

    if len(df_mlp) == 0 or len(df_mf) == 0:
        print("Baseline rows not found (mlp/mf).")
        return ""

    mlp_best = df_mlp.loc[df_mlp["NDCG10"].idxmax()]
    mf_best  = df_mf.loc[df_mf["NDCG10"].idxmax()]

    # ========= NF-VLM: 从所有 rank/λ 组合中取最佳 =========
    df_nf = df_all[df_all["model"] == "nfvlm"]
    nf_best = df_nf.loc[df_nf["NDCG10"].idxmax()]

    # ========= 构建 latex 表格 =========
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & AUC & NDCG@10 & Recall@10 \\")
    lines.append(r"\midrule")

    def fmt(x): return f"{x:.3f}" if pd.notna(x) else "--"

    lines.append(
        "MLP & {} & {} & {} \\\\".format(
            fmt(mlp_best["AUC"]), fmt(mlp_best["NDCG10"]), fmt(mlp_best["Recall10"])
        )
    )

    lines.append(
        "MF & {} & {} & {} \\\\".format(
            fmt(mf_best["AUC"]), fmt(mf_best["NDCG10"]), fmt(mf_best["Recall10"])
        )
    )

    lines.append(
        r"NF-VLM (best) & {} & {} & {} \\".format(
            fmt(nf_best["AUC"]), fmt(nf_best["NDCG10"]), fmt(nf_best["Recall10"])
        )
        + rf"  % rank={int(nf_best['nf_rank'])}, lambda={nf_best['nf_reg']}"
    )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Comparison of MLP, MF, and NF-VLM on full-ranking metrics.}")
    lines.append(r"\label{tab:baseline_nfvlm}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)
    print("\n================ Baseline vs NF-VLM LaTeX Table ================\n")
    print(latex_str)
    print("\n================================================================\n")

    return latex_str


# ===================== main =====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="ablation_CLIP_*.csv")
    args = parser.parse_args()

    df_all, df_nf = load_ablation_csv(args.csv)

    # 图1: rank vs NDCG@10
    plot_rank_vs_ndcg(df_nf, out_png="rank_vs_ndcg10_best.png", mode="best_over_reg")

    # 图2: reg vs Recall@10
    plot_reg_vs_recall(df_nf, out_png="reg_vs_recall10_best.png", mode="best_over_rank")

    # 图3: heatmap
    plot_heatmap_rank_reg(df_nf, metric="NDCG10", out_png="heatmap_rank_reg_ndcg10.png", agg="mean")

    # 生成 LaTeX 表格 (NDCG@10)
    generate_latex_table_rank_reg(df_nf, metric="NDCG10")

    # Baseline vs NF-VLM 三行表格
    generate_baseline_vs_nfvlm_table(df_all)
