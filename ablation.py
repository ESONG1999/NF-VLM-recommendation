# ablation.py
import os
import csv
from datetime import datetime
import subprocess

# ---------------------------------------------------------
# 固定数据根目录
# ---------------------------------------------------------
DATASET_ROOT = "data/amazon2023"

# 只使用 CLIP
ITEM2IDX = "item2idx_clip.pkl"
ITEM_EMB = "item_clip.pt"

# ---------------------------------------------------------
# baseline 模型
# ---------------------------------------------------------
MODELS = ["mlp", "mf", "nfvlm"]

# NF-VLM 的 ablation
NF_RANKS = [8, 16, 32, 64]
NF_REGS = [0, 1e-4, 1e-3, 3e-3]

# 超参数
BATCH_SIZE = 1024
EPOCHS = 20
LR = 1e-3

# 结果文件
LOG_FILE = f"ablation_CLIP_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"


def run_experiment(model, nf_rank=None, nf_reg=None):
    """
    调用 train_clean.py 进行实验
    """

    cmd = [
        "python",
        "train_clean.py",
        "--dataset_root", DATASET_ROOT,
        "--item2idx", ITEM2IDX,
        "--item_emb", ITEM_EMB,
        "--model_type", model,
        "--batch_size", str(BATCH_SIZE),
        "--epochs", str(EPOCHS),
        "--lr", str(LR),
        "--device", "cuda",
        "--nf_noise_reg", str(nf_reg if nf_reg is not None else 0),
    ]

    # only nfvlm uses nf_rank
    if nf_rank is not None:
        cmd += ["--nf_rank", str(nf_rank)]

    print("=============================================================")
    print("Running:", " ".join(cmd))
    print("=============================================================")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # 打印输出到屏幕
    print(result.stdout)
    print(result.stderr)

    # 提取最终 test metrics
    last_line = ""
    for line in result.stdout.split("\n"):
        if line.startswith("[Test]"):
            last_line = line
            break

    return last_line.strip()


def main():

    # 创建 CSV 文件
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "nf_rank", "nf_reg", "metrics"])

    # -------------------------------
    # Ablation A: baseline models
    # -------------------------------
    for m in MODELS:
        if m != "nfvlm":   # baseline 无需 rank/reg
            metrics = run_experiment(m)
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([m, "-", "-", metrics])

    # -------------------------------
    # Ablation B: NF-VLM rank/reg
    # -------------------------------
    for r in NF_RANKS:
        for lam in NF_REGS:
            metrics = run_experiment("nfvlm", nf_rank=r, nf_reg=lam)
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["nfvlm", r, lam, metrics])


if __name__ == "__main__":
    main()
