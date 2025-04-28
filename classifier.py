# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 14:38:18 2025

@author: ryuse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CreditRiskClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dims=[64, 32], dropout=0.2):
        super().__init__()
        layers = []
        dims = [in_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(dims[-1], 1)  # 出力は1次元（確率）
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        """
        h: [batch_size, in_dim] - 融合された特徴ベクトル

        Returns:
            [batch_size, 1] - デフォルト確率 (0〜1)
        """
        x = self.mlp(h)
        x = self.out(x)
        return self.sigmoid(x)