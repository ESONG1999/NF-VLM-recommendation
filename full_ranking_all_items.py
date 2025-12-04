# full_ranking_all_items.py
import numpy as np
import torch
from tqdm import tqdm

@torch.no_grad()
def debug_user_ranking_all_items(
    model,
    user_idx: int,
    pos_items: list,
    item_emb_matrix: torch.Tensor,
    device: str = "cuda",
    topk: int = 10,
    batch_size: int = 2048,
):
    """
    打印某个 user 在「所有 item」上的 Top-K 排名
    """
    model.eval()

    num_items = item_emb_matrix.size(0)

    # 把全体 items 的 embedding 移上 GPU
    cand_emb = item_emb_matrix.to(device)

    print(f"\n===== DEBUG USER {user_idx} =====")
    print(f"Positive items: {pos_items}")

    scores = []

    # 正确的 batch slice 必须用 batch_emb.size(0)！
    for start in range(0, num_items, batch_size):
        end = min(start + batch_size, num_items)

        # item batch
        batch_emb = cand_emb[start:end]        # shape [cur_B, D]

        # user batch (must match emb batch size!)
        batch_user = torch.full(
            (batch_emb.size(0),),
            fill_value=user_idx,
            dtype=torch.long,
            device=device,
        )

        # run model
        if hasattr(model, "nf"):
            logits, _, _ = model(batch_user, batch_emb)
        else:
            logits = model(batch_user, batch_emb)

        scores.append(logits.cpu())

    scores = torch.cat(scores).numpy()         # shape [num_items]

    # 排序
    rank_idx = np.argsort(-scores)
    top_items = rank_idx[:topk]

    print("\nTop-10 Ranking:")
    for rank, item_id in enumerate(top_items, 1):
        score = scores[item_id]
        label = 1 if item_id in pos_items else 0
        print(f"Rank {rank:02d} | item={item_id} | score={score:.4f} | label={label}")



def dcg_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    if r.size == 0:
        return 0.0
    return np.sum((2 ** r - 1) / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    r = np.asarray(r, dtype=float)
    ideal = np.sort(r)[::-1]
    dcg_max = dcg_at_k(ideal, k)
    return dcg_at_k(r, k) / dcg_max if dcg_max > 0 else 0.0


@torch.no_grad()
def evaluate_full_ranking_all_items(
    model,
    user2pos: dict,
    item_emb_matrix: torch.Tensor,
    device="cuda",
    batch_size=2048,
    k=10,
):
    model.eval()

    num_items = item_emb_matrix.size(0)
    all_items = list(range(num_items))

    # move all item embeddings to GPU once 👇
    cand_emb = item_emb_matrix.to(device)   # shape [I, D]

    ndcg_list = []
    recall_list = []

    for u_idx in tqdm(sorted(user2pos.keys()), desc="Full-ranking(all items)"):
        pos_items = set(user2pos[u_idx])
        if len(pos_items) == 0:
            continue

        scores = []
        # batch over all items
        for start in range(0, num_items, batch_size):
            end = min(start + batch_size, num_items)
            batch_emb = cand_emb[start:end]             # [cur_B, D]

            batch_user = torch.full(
                (batch_emb.size(0),),
                fill_value=u_idx,
                dtype=torch.long,
                device=device,
            )

            if hasattr(model, "nf"):
                logits, _, _ = model(batch_user, batch_emb)
            else:
                logits = model(batch_user, batch_emb)

            scores.append(logits.cpu())

        scores = torch.cat(scores).numpy()              # [I]

        # sort items
        rank_idx = np.argsort(-scores)
        ranked_items = np.asarray(all_items)[rank_idx]

        # relevance vector
        rel = np.array([1 if it in pos_items else 0 for it in ranked_items], dtype=int)

        ndcg = ndcg_at_k(rel, k)
        ndcg_list.append(ndcg)

        recall = rel[:k].sum() / len(pos_items)
        recall_list.append(recall)

    return float(np.mean(ndcg_list)), float(np.mean(recall_list))

