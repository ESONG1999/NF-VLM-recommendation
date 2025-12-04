# debug_ranking.py
import numpy as np
import torch

@torch.no_grad()
def debug_user_ranking(
    model,
    user_idx,
    user2cands,
    user2pos,
    item_emb_matrix,
    device="cuda",
    topk=10,
):
    """
    打印某个 user 在 test 中的排序结果：
    - item_id
    - score
    - label(0/1)
    """

    cands = user2cands[user_idx]
    pos_items = set(user2pos[user_idx])

    print(f"\n========== User {user_idx} Debug ==========")
    print(f"#candidates={len(cands)} positive={len(pos_items)}")

    # embedding
    cand_emb = item_emb_matrix[cands].to(device)

    user_idx_tensor = torch.full(
        (len(cands),), fill_value=user_idx, dtype=torch.long, device=device
    )

    # compute scores
    if hasattr(model, "nf"):
        logits, _, _ = model(user_idx_tensor, cand_emb)
    else:
        logits = model(user_idx_tensor, cand_emb)

    scores = logits.cpu().numpy()
    sorted_idx = np.argsort(-scores)

    print("Top 10 Items:")
    for rank in range(min(topk, len(cands))):
        i = sorted_idx[rank]
        item_id = cands[i]
        score = scores[i]
        label = 1 if item_id in pos_items else 0
        print(f"Rank {rank+1:02d} | item={item_id} | score={score:.4f} | label={label}")
