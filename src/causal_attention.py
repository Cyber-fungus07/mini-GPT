import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self,d_in,d_out,context_len,dropout,qkv_bias=False):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(p=dropout)

        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_len, context_len),diagonal=1)
        )

    def forward(self,x):
        b, num_tokens, d_in = x.shape

        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        attn_scores = queries @ keys.transpose(1,2)

        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf
        )
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec