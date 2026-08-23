import torch
import torch.nn.functional as F

class GPTLoss:
    def __call__(self, logits, targets):
        logits = logits.flatten(0, 1)
        targets = targets.flatten()

        return F.cross_entropy(logits, targets)


class Perplexity:
    def __call__(self, loss):
        return torch.exp(loss)