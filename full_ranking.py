# full_ranking.py
import numpy as np
import torch
from tqdm import tqdm


# ----------- 修复 NumPy 2.0 版本报错的 DCG/NDCG 实现 -----------
def dcg_at_k(r, k):
    # NumPy 2.0 取消了 asfarray，使用 asarray + dtype=float
    r = np.asarray(r, dtype=float)[:k]
    if r.size == 0:
        return 0.0
    return np.sum((2 ** r - 1) / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    r = np.asarray(r, dtype=float)
    ideal = np.sort(r)[::-1]               # 正样本排在前面
    dcg_max = dcg_at_k(ideal, k)
    return dcg_at_k(r, k) / dcg_max if dcg_max > 0 else 0.0


# -------------------- per-user full ranking 评估 --------------------
@torch.no_grad()
def evaluate_full_ranking_test_only(
    model,
    user2pos: dict,
    user2cands: dict,
    item_emb_matrix: torch.Tensor,
    device="cuda",
    batch_size=1024,
    k=10,
):
    model.eval()

    ndcg_list = []
    recall_list = []

    for u_idx in tqdm(user2cands.keys(), desc="Full-ranking Test"):
        cands = user2cands[u_idx]
        pos_items = set(user2pos[u_idx])

        if len(cands) == 0 or len(pos_items) == 0:
            continue

        cand_emb = item_emb_matrix[cands].to(device)

        user_idx_tensor = torch.full(
            (len(cands),),
            fill_value=u_idx,
            dtype=torch.long,
            device=device,
        )

        # 批量 evaluate
        scores = []
        for start in range(0, len(cands), batch_size):
            end = start + batch_size
            batch_u = user_idx_tensor[start:end]
            batch_emb = cand_emb[start:end]

            if hasattr(model, "nf"):  # NF-VLM
                logit, _, _ = model(batch_u, batch_emb)
            else:                     # MLP/MF baseline
                logit = model(batch_u, batch_emb)

            scores.append(logit.cpu())

        scores = torch.cat(scores).numpy()

        # 排序
        rank_idx = np.argsort(-scores)
        sorted_items = np.asarray(cands)[rank_idx]

        # relevance vector
        rel = np.array([1 if it in pos_items else 0 for it in sorted_items], dtype=int)

        # NDCG@K
        ndcg = ndcg_at_k(rel, k)
        ndcg_list.append(ndcg)

        # Recall@K
        num_pos = len(pos_items)
        recall = rel[:k].sum() / num_pos if num_pos > 0 else 0.0
        recall_list.append(recall)

    mean_ndcg = float(np.mean(ndcg_list)) if len(ndcg_list) > 0 else 0.0
    mean_recall = float(np.mean(recall_list)) if len(recall_list) > 0 else 0.0

    return mean_ndcg, mean_recall
