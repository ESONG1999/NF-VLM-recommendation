# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPRecModel(nn.Module):
    """
    简单 baseline：user embedding + item visual embedding → MLP → 预测点击概率
    """
    def __init__(self, num_users: int, emb_dim: int, user_emb_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, user_emb_dim)
        self.fc1 = nn.Linear(user_emb_dim + emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, 1)

    def forward(self, user_idx, item_emb):
        u = self.user_emb(user_idx)           # [B, U]
        x = torch.cat([u, item_emb], dim=-1)  # [B, U+D]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logit = self.out(x).squeeze(-1)       # [B]
        return logit


class MFRecModel(nn.Module):
    """
    经典 MF baseline：user_emb · item_emb
    item_emb 初始化为视觉 embedding + 可微调的一层线性
    """
    def __init__(self, num_users: int, num_items: int, emb_dim: int, latent_dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, latent_dim)
        self.item_proj = nn.Linear(emb_dim, latent_dim)

    def forward(self, user_idx, item_emb):
        u = self.user_emb(user_idx)          # [B, K]
        v = self.item_proj(item_emb)         # [B, K]
        logit = (u * v).sum(dim=-1)          # [B]
        return logit


class NoiseFactorization(nn.Module):
    """
    NF-VLM：把视觉 embedding 分解为 signal + noise 子空间
    简单实现：低秩投影 + 残差 = noise
    """
    def __init__(self, emb_dim: int, rank: int = 32):
        super().__init__()
        self.proj = nn.Linear(emb_dim, rank, bias=False)
        self.back = nn.Linear(rank, emb_dim, bias=False)

    def forward(self, x):
        # x: [B, D]
        z = self.proj(x)        # [B, r]
        s = self.back(z)        # [B, D] signal
        n = x - s               # [B, D] noise
        return s, n


class NFVLMRecModel(nn.Module):
    """
    NF-VLM 推荐模型：
    - item_emb → NF 分解 → signal embedding
    - user_emb + signal → MLP → logit
    同时可以在 loss 中对 noise 加 L2 penalty，鼓励噪声缩小
    """
    def __init__(self, num_users: int, emb_dim: int,
                 user_emb_dim: int = 64, nf_rank: int = 32,
                 hidden_dim: int = 256):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, user_emb_dim)
        self.nf = NoiseFactorization(emb_dim, nf_rank)

        self.fc1 = nn.Linear(user_emb_dim + emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, 1)

    def forward(self, user_idx, item_emb):
        u = self.user_emb(user_idx)          # [B, U]
        s, n = self.nf(item_emb)             # [B, D], [B, D]
        x = torch.cat([u, s], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logit = self.out(x).squeeze(-1)
        return logit, s, n
