# train_clean.py
import os
import csv
import pickle
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from models import MLPRecModel, MFRecModel, NFVLMRecModel
from metrics import evaluate_model_user_level
from collections import defaultdict
from debug_ranking import debug_user_ranking
from full_ranking_all_items import evaluate_full_ranking_all_items, debug_user_ranking_all_items



# python train_clean.py \
#   --dataset_root data/amazon2023 \
#   --item2idx item2idx_clip.pkl \
#   --item_emb item_clip.pt \
#   --model_type nfvlm \
#   --batch_size 1024 \
#   --epochs 10

def debug_test_stats(test_set):
    import numpy as np
    ys = [y for (_,_,y) in test_set.samples]
    print("Test stats:")
    print("  #samples:", len(ys))
    print("  #positive:", np.sum(np.array(ys) > 0))
    print("  #negative:", np.sum(np.array(ys) == 0))


# =========================================================
# Dataset（支持 train_clean / val_clean / test_clean）
# =========================================================
class RecDatasetClean(Dataset):
    def __init__(self, csv_path, item2idx_path, item_emb_path, user2idx=None):
        self.csv_path = csv_path

        # load item2idx & embeddings
        with open(item2idx_path, "rb") as f:
            self.item2idx: Dict[str, int] = pickle.load(f)
        self.item_emb_matrix: torch.Tensor = torch.load(item_emb_path)
        self.emb_dim = self.item_emb_matrix.size(1)

        # load csv
        self.samples: List[Tuple[int, int, float]] = []
        if user2idx is None:
            self.user2idx = {}
        else:
            self.user2idx = user2idx
        self.idx2user: List[str] = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                u_raw = row["user_id"]
                i_raw = row["item_id"]
                y = float(row["label"])

                if i_raw not in self.item2idx:
                    continue  # 只有在 item2idx 中的 item 才能训练/推理

                # map user
                if user2idx is None:
                    # train_set 建立 mapping
                    if u_raw not in self.user2idx:
                        self.user2idx[u_raw] = len(self.user2idx)
                else:
                    # test_set 或 val_set 使用 train 的 user2idx
                    if u_raw not in self.user2idx:
                        continue   # ❗ 跳过 test 中不存在于 train 的 user


                u_idx = self.user2idx[u_raw]
                i_idx = self.item2idx[i_raw]

                self.samples.append((u_idx, i_idx, y))

        self.num_users = len(self.user2idx)
        self.num_items = len(self.item2idx)

        print(
            f"[RecDatasetClean] Loaded {csv_path}: "
            f"#users={self.num_users}, #items={self.num_items}, #samples={len(self.samples)}, emb_dim={self.emb_dim}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u_idx, i_idx, y = self.samples[idx]
        item_emb = self.item_emb_matrix[i_idx]  # [D]
        return (
            torch.tensor(u_idx, dtype=torch.long),
            torch.tensor(i_idx, dtype=torch.long),
            item_emb,  # [D]
            torch.tensor(y, dtype=torch.float32),
        )

def build_user2pos_from_test(test_set):
    """
    只用 test_set 中的样本，构建每个 user 的正样本 item 列表
    （item_idx 是 embedding 索引）
    """
    user2pos = defaultdict(list)
    for (u_idx, i_idx, y) in test_set.samples:
        if y > 0:
            user2pos[u_idx].append(i_idx)
    return user2pos

def build_test_user_item_maps(test_set):
    """
    从 test_set 构建每个 user 的：
    - user2pos: 该 user 在 test 中的正样本 item 列表
    - user2cands: 该 user 在 test 中出现在过的所有 item（正 + 负）

    注意：这里只用 test_clean.csv，不做“全 item 排序”。
    """
    user2pos = defaultdict(list)
    user2cands = defaultdict(list)

    for (u_idx, i_idx, y) in test_set.samples:
        user2cands[u_idx].append(i_idx)
        if y > 0:
            user2pos[u_idx].append(i_idx)

    return user2pos, user2cands

        
def collate_fn(batch):
    u, i, e, y = zip(*batch)
    return (
        torch.stack(u, dim=0),
        torch.stack(i, dim=0),
        torch.stack(e, dim=0),
        torch.stack(y, dim=0),
    )


# =========================================================
# Training
# =========================================================
def train_epoch(model, loader, optimizer, device, nf_noise_reg=0.0):
    model.train()
    bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0

    for u_idx, i_idx, item_emb, y in tqdm(loader, desc="Train", leave=False):
        u_idx = u_idx.to(device)
        item_emb = item_emb.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        if isinstance(model, NFVLMRecModel):
            logits, _, n = model(u_idx, item_emb)
            loss = bce(logits, y) + nf_noise_reg * (n.pow(2).mean())
        else:
            logits = model(u_idx, item_emb)
            loss = bce(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []

    for u_idx, i_idx, item_emb, y in tqdm(loader, desc="Eval", leave=False):
        u_idx = u_idx.to(device)
        item_emb = item_emb.to(device)
        y = y.to(device)

        if isinstance(model, NFVLMRecModel):
            logits, _, _ = model(u_idx, item_emb)
        else:
            logits = model(u_idx, item_emb)

        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    auc = compute_auc(all_labels, all_logits)
    ndcg10, recall10 = compute_ranking_metrics(all_labels, all_logits, k=10)
    return {"AUC": auc, "NDCG@10": ndcg10, "Recall@10": recall10}


# =========================================================
# Main
# =========================================================
def main_train_clean(
    dataset_root: str,
    item2idx: str,
    item_emb: str,
    model_type: str = "nfvlm",
    batch_size: int = 512,
    epochs: int = 10,
    lr: float = 1e-3,
    nf_noise_reg: float = 1e-3,
    device: str = "cuda",
):
    item2idx_path = os.path.join(dataset_root, item2idx)
    item_emb_path = os.path.join(dataset_root, item_emb)

    # Load clean datasets
    train_csv = os.path.join(dataset_root, "train_clean.csv")
    val_csv = os.path.join(dataset_root, "val_clean.csv")
    test_csv = os.path.join(dataset_root, "test_clean.csv")

    # 1. Load train
    train_set = RecDatasetClean(train_csv, item2idx_path, item_emb_path)
    
    # 2. Use train's user2idx to load val & test
    val_set = RecDatasetClean(val_csv, item2idx_path, item_emb_path, user2idx=train_set.user2idx)
    test_set = RecDatasetClean(test_csv, item2idx_path, item_emb_path, user2idx=train_set.user2idx)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    num_users = train_set.num_users
    emb_dim = train_set.emb_dim

    # Initialize Model
    if model_type == "mlp":
        model = MLPRecModel(num_users, emb_dim)
    elif model_type == "mf":
        model = MFRecModel(num_users, train_set.num_items, emb_dim)
    elif model_type == "nfvlm":
        model = NFVLMRecModel(num_users, emb_dim, nf_rank=args.nf_rank)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    best_auc = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        print(f"========== Epoch {epoch}/{epochs} ==========")
        train_loss = train_epoch(model, train_loader, optimizer, device, nf_noise_reg)
        print(f"[Train] loss={train_loss:.4f}")

        val_metrics = evaluate_model_user_level(model, val_loader, device)
        print("[Val]", val_metrics)

        if val_metrics["AUC"] > best_auc:
            best_auc = val_metrics["AUC"]
            best_state = model.state_dict()

    # load best checkpoint
    if best_state:
        model.load_state_dict(best_state)

    test_metrics = evaluate_model_user_level(model, test_loader, device)
    # print("[Test]", test_metrics)
    debug_test_stats(test_set)
    
    # ===== 新增：真正 full-ranking（所有 item 为候选） =====
    user2pos = build_user2pos_from_test(test_set)

    ndcg10, recall10 = evaluate_full_ranking_all_items(
        model,
        user2pos=user2pos,
        item_emb_matrix=test_set.item_emb_matrix,
        device=device,
        batch_size=2048,
        k=10,
    )

    # ===== Debug：随机挑一个 user 看 Top-10 =====
    some_user = sorted(user2pos.keys())[0]
    debug_user_ranking_all_items(
        model,
        user_idx=some_user,
        pos_items=user2pos[some_user],
        item_emb_matrix=test_set.item_emb_matrix,
        device=device,
        topk=10,
    )

    
    print("[Test]", {"AUC": test_metrics["AUC"], "NDCG@10": ndcg10, "Recall@10": recall10})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--item2idx", type=str, required=True)
    parser.add_argument("--item_emb", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="nfvlm",
                        choices=["nfvlm", "mlp", "mf"])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--nf_noise_reg", type=float, default=1e-3)
    parser.add_argument("--nf_rank", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    main_train_clean(
        dataset_root=args.dataset_root,
        item2idx=args.item2idx,
        item_emb=args.item_emb,
        model_type=args.model_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        nf_noise_reg=args.nf_noise_reg,
        device=args.device,
    )
