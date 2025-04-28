# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 14:33:38 2025

@author: ryuse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, in_dim, time_dim, heads=1, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.time_dim = time_dim
        self.heads = heads
        self.scale = (in_dim + time_dim) ** 0.5

        self.q_proj = nn.Linear(in_dim + time_dim, in_dim)
        self.k_proj = nn.Linear(in_dim + time_dim, in_dim)
        self.v_proj = nn.Linear(in_dim, in_dim)
        self.out_proj = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, time_embed):
        """
        h: [batch_size, num_neighbors, in_dim]
        time_embed: [batch_size, num_neighbors, time_dim]
        """
        x = torch.cat([h, time_embed], dim=-1)  # [B, N, in_dim + time_dim]

        Q = self.q_proj(x)  # [B, N, in_dim]
        K = self.k_proj(x)  # [B, N, in_dim]
        V = self.v_proj(h)  # [B, N, in_dim]

        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale  # [B, N, N]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)  # [B, N, in_dim]
        out = self.out_proj(out)  # [B, N, in_dim]

        return out, attn_weights
