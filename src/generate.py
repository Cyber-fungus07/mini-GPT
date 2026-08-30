import torch

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def generate(model,idx,max_new_tokens,context_size,temperature=1.0,top_k=None,eos_id=None):
    for _ in range(max_new_tokens):
        # Keep only the latest context_size tokens
        idx_cond = idx[:, -context_size:]

        # Get model predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Get logits for the last token
        logits = logits[:, -1, :]

        # Top-k filtering
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]

            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits
            )

        # Temperature scaling + sampling
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)

        # Greedy decoding
        else:
            idx_next = torch.argmax(logits,dim=-1,keepdim=True)

        # Stop if EOS token is generated
        if eos_id is not None and idx_next.item() == eos_id:
            break

        # Add predicted token to sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
