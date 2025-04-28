# full_model_tgcn_dgnnsr.py

import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import TGCN
from static_gnn import StaticGraphEncoder
from fusion import FusionLayer
from classifier import CreditRiskClassifier
from time_encoder import MultiViewTimeEncoder  # ← ✅ MTEを使う！

class FullModel(nn.Module):
    def __init__(self, in_dim=128, out_dim=128, time_config=(16, 16, 8, 32)):
        super().__init__()
        d_re, d_ab, d_se, d_time = time_config
        self.mte = MultiViewTimeEncoder(d_re, d_ab, d_se, d_time)
        self.tgcn = TGCN(in_channels=in_dim + d_time, out_channels=out_dim)
        self.static_encoder = StaticGraphEncoder(in_channels=in_dim, hidden_channels=128, out_channels=out_dim)
        self.fusion = FusionLayer(out_dim)
        self.classifier = CreditRiskClassifier(out_dim)

    def forward(self, data_tgcn, h_static, t, t_prime, semantic_feat):
        """
        data_tgcn: PyG Temporal snapshot (x, edge_index, edge_attr)
        h_static: [B, D]
        t: [B]           - 中心ノードの時刻
        t_prime: [B, N]  - 近傍ノードの時刻
        semantic_feat: [B, N, d_se] - 時間的意味特徴（祝日/週末など）
        """
        # MTEで時間埋め込みを作成
        B = data_tgcn.x.shape[0]
        N = t_prime.shape[1]
        t_expanded = t.unsqueeze(1).expand(-1, N).reshape(-1)            # [B*N]
        t_prime_flat = t_prime.reshape(-1)                               # [B*N]
        sem_feat_flat = semantic_feat.reshape(-1, semantic_feat.shape[-1])  # [B*N, d_se]
        time_embed = self.mte(t_expanded, t_prime_flat, sem_feat_flat)  # [B*N, d_time]
        time_embed = time_embed.reshape(B, N, -1).mean(dim=1)            # [B, d_time] ← ノード単位に集約

        # 特徴ベクトルと連結
        x = data_tgcn.x  # [B, D]
        x_aug = torch.cat([x, time_embed], dim=-1)  # [B, D + d_time]

        # 時系列構造学習（TGCN）
        h_dynamic = self.tgcn(x_aug, data_tgcn.edge_index, data_tgcn.edge_attr)  # [B, out_dim]

        # 静的特徴統合
        h_fused = self.fusion(h_static, h_dynamic)

        # 分類
        y_pred = self.classifier(h_fused)
        return y_pred

