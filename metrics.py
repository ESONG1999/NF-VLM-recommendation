# metrics.py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from collections import defaultdict


# ---------------------------------------------------------
# A. AUC（全局）
# ---------------------------------------------------------
def compute_auc(labels: torch.Tensor, logits: torch.Tensor):
    """
    全局 AUC，不按 user 划分
    """
    y_true = labels.numpy()
    y_score = logits.sigmoid().numpy()
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")
    return float(auc)


# ---------------------------------------------------------
# B. user-level ranking metrics
# ---------------------------------------------------------

def _dcg(rel):
    """rel: relevance array sorted by predicted score (desc)."""
    if len(rel) == 0:
        return 0.0
    return np.sum((2**rel - 1) / np.log2(np.arange(2, len(rel) + 2)))


def compute_ranking_by_user(labels, logits, user_ids, k=10):
    """
    user_ids, labels, logits 均为一维张量（整个验证集或测试集）
    对每个 user 分组计算 NDCG@K 与 Recall@K。

    返回：
    - mean_ndcg
    - mean_recall
    """
    user_groups = defaultdict(list)

    # 按 user 分组
    for u, y, s in zip(user_ids.numpy(), labels.numpy(), logits.numpy()):
        user_groups[int(u)].append((s, y))

    ndcgs = []
    recalls = []

    for u, samples in user_groups.items():
        samples = sorted(samples, key=lambda x: -x[0])  # 按 score 排序
        scores_sorted = [x[0] for x in samples]
        labels_sorted = [x[1] for x in samples]
        labels_sorted = np.array(labels_sorted)

        # top-K
        topk = labels_sorted[:k]

        # Recall@K
        num_pos = labels_sorted.sum()
        if num_pos > 0:
            recall = topk.sum() / num_pos
        else:
            recall = 0.0

        # NDCG@K
        ideal_sorted = np.sort(labels_sorted)[::-1]
        dcg = _dcg(topk)
        idcg = _dcg(ideal_sorted[:k])
        ndcg = dcg / idcg if idcg > 0 else 0.0

        recalls.append(recall)
        ndcgs.append(ndcg)

    return np.mean(ndcgs), np.mean(recalls)


# ---------------------------------------------------------
# C. eval 函数（train_clean 使用）
# ---------------------------------------------------------

@torch.no_grad()
def evaluate_model_user_level(model, loader, device):
    """
    适配 train_clean：
    返回:
        AUC（全局）
        user-level NDCG@10
        user-level Recall@10
    """
    model.eval()
    all_logits = []
    all_labels = []
    all_user_ids = []

    for u_idx, _, item_emb, y in loader:
        u_idx = u_idx.to(device)
        item_emb = item_emb.to(device)
        y = y.to(device)

        if hasattr(model, "nf"):
            logits, _, _ = model(u_idx, item_emb)
        else:
            logits = model(u_idx, item_emb)

        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
        all_user_ids.append(u_idx.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    user_ids = torch.cat(all_user_ids)

    global_auc = compute_auc(labels, logits)
    ndcg10, recall10 = compute_ranking_by_user(labels, logits, user_ids, k=10)

    return {
        "AUC": global_auc,
        "NDCG@10": ndcg10,
        "Recall@10": recall10,
    }
