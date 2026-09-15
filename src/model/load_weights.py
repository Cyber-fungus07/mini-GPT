import torch
from transformers import GPT2LMHeadModel

# Expected dims per official size; only sizes matching the model cfg can load.
GPT2_SIZES = {
    "gpt2": {"emb_dim": 768, "n_heads": 12, "n_layers": 12},
    "gpt2-medium": {"emb_dim": 1024, "n_heads": 16, "n_layers": 24},
    "gpt2-large": {"emb_dim": 1280, "n_heads": 20, "n_layers": 36},
    "gpt2-xl": {"emb_dim": 1600, "n_heads": 25, "n_layers": 48},
}


def gpt2_config(model_size="gpt2"):
    from src.config import GPT_CONFIG

    if model_size not in GPT2_SIZES:
        raise ValueError(f"Unknown model_size {model_size!r}. Choose from {sorted(GPT2_SIZES)}")

    cfg = dict(GPT_CONFIG)
    cfg.update(GPT2_SIZES[model_size])
    cfg["qkv_bias"] = True
    return cfg


def _assign(target, source, label):
    if target.shape != source.shape:
        raise ValueError(f"Shape mismatch for {label}: {tuple(target.shape)} vs {tuple(source.shape)}")
    return source.clone() if isinstance(source, torch.Tensor) else torch.tensor(source)


def load_gpt2_weights(model, model_size="gpt2", verbose=False):
    if model_size not in GPT2_SIZES:
        raise ValueError(f"Unknown model_size {model_size!r}. Choose from {sorted(GPT2_SIZES)}")

    expected = GPT2_SIZES[model_size]
    n_layers = len(model.trf_blocks)
    emb_dim = model.tok_emb.weight.shape[1]
    if n_layers != expected["n_layers"] or emb_dim != expected["emb_dim"]:
        raise ValueError(
            f"Model cfg (layers={n_layers}, emb_dim={emb_dim}) does not match "
            f"{model_size} (layers={expected['n_layers']}, emb_dim={expected['emb_dim']})"
        )

    # GPT-2 uses bias on QKV; our default from-scratch cfg disables it.
    sample_q = model.trf_blocks[0].att.W_query
    if sample_q.bias is None:
        raise ValueError(
            "Model was built with qkv_bias=False but GPT-2 weights include QKV bias. "
            "Build the model with gpt2_config() (qkv_bias=True) before loading."
        )

    hf = GPT2LMHeadModel.from_pretrained(model_size)
    sd = hf.state_dict()

    with torch.no_grad():
        model.tok_emb.weight.copy_(_assign(model.tok_emb.weight, sd["transformer.wte.weight"], "tok_emb"))
        model.pos_emb.weight.copy_(_assign(model.pos_emb.weight, sd["transformer.wpe.weight"], "pos_emb"))
        model.out_head.weight.copy_(_assign(model.out_head.weight, sd["lm_head.weight"], "out_head"))
        model.final_norm.scale.copy_(_assign(model.final_norm.scale, sd["transformer.ln_f.weight"], "ln_f.scale"))
        model.final_norm.shift.copy_(_assign(model.final_norm.shift, sd["transformer.ln_f.bias"], "ln_f.shift"))

        for i, block in enumerate(model.trf_blocks):
            p = f"transformer.h.{i}"
            if verbose:
                print(f"Loading block {i + 1}/{n_layers}...")

            block.norm1.scale.copy_(_assign(block.norm1.scale, sd[f"{p}.ln_1.weight"], f"b{i}.ln1.scale"))
            block.norm1.shift.copy_(_assign(block.norm1.shift, sd[f"{p}.ln_1.bias"], f"b{i}.ln1.shift"))
            block.norm2.scale.copy_(_assign(block.norm2.scale, sd[f"{p}.ln_2.weight"], f"b{i}.ln2.scale"))
            block.norm2.shift.copy_(_assign(block.norm2.shift, sd[f"{p}.ln_2.bias"], f"b{i}.ln2.shift"))

            # Fused QKV: (emb, 3*emb) -> transpose -> split into Q/K/V.
            c_attn_w = sd[f"{p}.attn.c_attn.weight"].T  # (3*emb, emb)
            c_attn_b = sd[f"{p}.attn.c_attn.bias"]
            q_w, k_w, v_w = c_attn_w.chunk(3, dim=0)
            q_b, k_b, v_b = c_attn_b.chunk(3, dim=0)
            block.att.W_query.weight.copy_(_assign(block.att.W_query.weight, q_w, f"b{i}.q.w"))
            block.att.W_key.weight.copy_(_assign(block.att.W_key.weight, k_w, f"b{i}.k.w"))
            block.att.W_value.weight.copy_(_assign(block.att.W_value.weight, v_w, f"b{i}.v.w"))
            block.att.W_query.bias.copy_(_assign(block.att.W_query.bias, q_b, f"b{i}.q.b"))
            block.att.W_key.bias.copy_(_assign(block.att.W_key.bias, k_b, f"b{i}.k.b"))
            block.att.W_value.bias.copy_(_assign(block.att.W_value.bias, v_b, f"b{i}.v.b"))

            block.att.out_proj.weight.copy_(
                _assign(block.att.out_proj.weight, sd[f"{p}.attn.c_proj.weight"].T, f"b{i}.proj.w")
            )
            block.att.out_proj.bias.copy_(
                _assign(block.att.out_proj.bias, sd[f"{p}.attn.c_proj.bias"], f"b{i}.proj.b")
            )

            block.ff.layers[0].weight.copy_(
                _assign(block.ff.layers[0].weight, sd[f"{p}.mlp.c_fc.weight"].T, f"b{i}.fc.w")
            )
            block.ff.layers[0].bias.copy_(
                _assign(block.ff.layers[0].bias, sd[f"{p}.mlp.c_fc.bias"], f"b{i}.fc.b")
            )
            block.ff.layers[2].weight.copy_(
                _assign(block.ff.layers[2].weight, sd[f"{p}.mlp.c_proj.weight"].T, f"b{i}.mlp_proj.w")
            )
            block.ff.layers[2].bias.copy_(
                _assign(block.ff.layers[2].bias, sd[f"{p}.mlp.c_proj.bias"], f"b{i}.mlp_proj.b")
            )

    return model
