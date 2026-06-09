#!/usr/bin/env python3
"""
 Enter module docstring here
"""

# ----------------- Futures -----------------
from __future__ import annotations

# ----------------- Standard Library -----------------


# ----------------- Third Party Library -----------------
import torch

# ----------------- Application Imports -----------------


# ----------------- Module-level Configuration -----------------

inputs = torch.tensor([
    [0.43, 0.15, 0.89],
    [0.55, 0.87, 0.66],
    [0.57, 0.85, 0.64],
    [0.22, 0.58, 0.33],
    [0.77, 0.25, 0.10],
    [0.05, 0.80, 0.55],
])

query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(query, x_i)

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i

print(attn_scores_2)
print(attn_weights_2)
print(context_vec_2)
