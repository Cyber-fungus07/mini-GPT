# mini-GPT

> A GPT-2 style language model built completely from scratch in PyTorch — no Hugging Face, no black boxes.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## What is this?

This project implements every building block of a GPT model from the ground up — attention, layer norm, feed-forward layers, and the full training loop — written to be readable and easy to follow. The goal is understanding, not speed.

The architecture matches **GPT-2 small** exactly, so you can later load official OpenAI weights into it.

---

## Architecture

| Hyperparameter | Value |
|---|---|
| Parameters | 163M |
| Layers | 12 |
| Embedding dim | 768 |
| Attention heads | 12 |
| Context length | 1024 |
| Vocabulary | 50,257 tokens (BPE) |

**Data flow:**

```
Text → Tokenizer → Embeddings → 12× TransformerBlock → LayerNorm → Logits → Loss
                                      ↑
                        MultiHeadAttention + FeedForward
                        (with residual connections & pre-norm)
```

---

## Project Structure

```
mini-GPT/
├── main.py                   # Entry point — run training from here
├── data/                     # Training text files
└── src/
    ├── config.py             # GPT-2 hyperparameters
    ├── attention/            # Self → Causal → Multi-head attention
    ├── model/                # Embedding, Transformer block, GPT model
    ├── data/                 # Tokenizer (tiktoken BPE) & Dataloader
    ├── training/             # Loss (cross-entropy) & Trainer class
    └── inference/            # Text generation (top-k + temperature)
```

---

## Quickstart

```bash
# Clone & set up
git clone https://github.com/Cyber-fungus07/mini-GPT.git
cd mini-GPT
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train
python main.py
```

Training logs loss every 5 steps and generates a sample after each epoch:

---

## Status

| Component | Status |
|---|---|
| Model architecture | ✅ Complete |
| Training pipeline | ✅ Complete |
| Text generation | ✅ Complete |
| Checkpointing | ✅ Complete |
| Pre-trained GPT-2 weight loading | 🔲 Planned |
