import torch
import torch.nn as nn

class GPTEmbedding(nn.Module):
    def __init__(self,vocab_size,context_length,embedding_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )
        self.position_embedding = nn.Embedding(
            context_length,
            embedding_dim
        )

    def forward(self, token_ids):
        positions = torch.arange(
            token_ids.shape[1],
            device=token_ids.device
        )
        token_emb = self.token_embedding(token_ids)
        position_emb = self.position_embedding(positions)

        return token_emb + position_emb