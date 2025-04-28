# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 14:36:18 2025

@author: ryuse
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfGating(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gate = nn.Linear(in_dim, in_dim)

    def forward(self, h):
        """
        h: [batch_size, in_dim]
        returns: gated h (same shape)
        """
        gate_score = torch.sigmoid(self.gate(h))  # [B, in_dim] in (0,1)
        return h * gate_score


class FusionLayer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gate_static = SelfGating(in_dim)
        self.gate_dynamic = SelfGating(in_dim)

    def forward(self, h_static, h_dynamic):
        """
        h_static: [B, D] - 静的グラフからの埋め込み
        h_dynamic: [B, D] - 動的グラフからの埋め込み

        return: [B, D] - 融合された特徴ベクトル
        """
        h_s = self.gate_static(h_static)
        h_d = self.gate_dynamic(h_dynamic)
        return h_s + h_d
