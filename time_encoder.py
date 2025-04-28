# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 14:31:03 2025

@author: ryuse
"""

import torch
import torch.nn as nn
import numpy as np

class MultiViewTimeEncoder(nn.Module):
    def __init__(self, d_re, d_ab, d_se, d_out):
        super().__init__()
        self.d_re = d_re
        self.d_ab = d_ab
        self.d_se = d_se
        self.d_out = d_out

        self.omega_re = nn.Parameter(torch.randn(d_re))
        self.omega_ab = nn.Parameter(torch.randn(d_ab))

        # Time semantic encoder (e.g. weekend, holiday)
        self.semantic_encoder = nn.Linear(d_se, d_se)

        # ここが超重要な修正ポイント！
        input_dim = d_re * 2 + d_ab * 2 + d_se
        #print(f"[MTE] input_dim to self.proj: {input_dim}")
        self.proj = nn.Linear(input_dim, d_out)

    def forward(self, t, t_prime, semantic_feat):
        """
        t: [batch_size] - current time
        t_prime: [batch_size] - past event time
        semantic_feat: [batch_size, d_se] - binary features (e.g., is_weekend, is_holiday)

        Returns:
            [batch_size, d_out] - time embedding
        """
        # Relative time (t - t')
        delta_t = t - t_prime  # [batch_size]

        # [batch_size, d_re]
        z_re = torch.cat([
            torch.cos(self.omega_re * delta_t.unsqueeze(1)),
            torch.sin(self.omega_re * delta_t.unsqueeze(1))
        ], dim=-1) / np.sqrt(self.d_re)

        # Absolute time
        z_ab = torch.cat([
            torch.cos(self.omega_ab * t_prime.unsqueeze(1)),
            torch.sin(self.omega_ab * t_prime.unsqueeze(1))
        ], dim=-1) / np.sqrt(self.d_ab)

        # Semantic time encoding (e.g. holiday/weekend binary info)
        z_se = self.semantic_encoder(semantic_feat)  # [batch_size, d_se]

        # Final embedding
        z_cat = torch.cat([z_re, z_ab, z_se], dim=-1)  # [batch_size, d_re + d_ab + d_se]
        z_final = self.proj(z_cat)  # [batch_size, d_out]

        return z_final
