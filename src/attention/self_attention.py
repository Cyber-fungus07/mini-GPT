import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        attn_scores = queries @ keys.transpose(-2, -1)

        attn_weight = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5,
            dim=-1
        )
        context_vector = attn_weight @ values
        return context_vector
