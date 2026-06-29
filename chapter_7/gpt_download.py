"""
Load official GPT-2 (124M) weights into our from-scratch GPTModel.

Uses the Hugging Face `openai-community/gpt2` safetensors checkpoint instead of
the book's TensorFlow checkpoint — same weights, no TensorFlow dependency.

HF GPT-2 stores attention/MLP weights as Conv1D layers, whose weight matrices
are the transpose of what nn.Linear expects, so we transpose those on load.
The token embedding (wte) is weight-tied to the output head.
"""

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from chapter04 import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": True,  # GPT-2 has biases on Q/K/V
}


def _assign(target, source):
    if target.shape != source.shape:
        raise ValueError(f"shape mismatch {target.shape} vs {source.shape}")
    return torch.nn.Parameter(source.clone().detach())


def load_gpt2_weights_into(model, sd):
    """sd: state dict from HF gpt2 safetensors."""
    model.tok_emb.weight = _assign(model.tok_emb.weight, sd["wte.weight"])
    model.pos_emb.weight = _assign(model.pos_emb.weight, sd["wpe.weight"])

    for i, block in enumerate(model.trf_blocks):
        p = f"h.{i}."
        # attention: c_attn packs q,k,v together -> [emb, 3*emb] (Conv1D, transposed)
        w = sd[p + "attn.c_attn.weight"]              # [768, 2304]
        b = sd[p + "attn.c_attn.bias"]                # [2304]
        q_w, k_w, v_w = w.split(w.shape[1] // 3, dim=1)
        q_b, k_b, v_b = b.split(b.shape[0] // 3, dim=0)
        block.attn.W_query.weight = _assign(block.attn.W_query.weight, q_w.T)
        block.attn.W_key.weight = _assign(block.attn.W_key.weight, k_w.T)
        block.attn.W_value.weight = _assign(block.attn.W_value.weight, v_w.T)
        block.attn.W_query.bias = _assign(block.attn.W_query.bias, q_b)
        block.attn.W_key.bias = _assign(block.attn.W_key.bias, k_b)
        block.attn.W_value.bias = _assign(block.attn.W_value.bias, v_b)

        block.attn.out_proj.weight = _assign(
            block.attn.out_proj.weight, sd[p + "attn.c_proj.weight"].T
        )
        block.attn.out_proj.bias = _assign(
            block.attn.out_proj.bias, sd[p + "attn.c_proj.bias"]
        )

        # feed forward
        block.ff.layers[0].weight = _assign(
            block.ff.layers[0].weight, sd[p + "mlp.c_fc.weight"].T
        )
        block.ff.layers[0].bias = _assign(
            block.ff.layers[0].bias, sd[p + "mlp.c_fc.bias"]
        )
        block.ff.layers[2].weight = _assign(
            block.ff.layers[2].weight, sd[p + "mlp.c_proj.weight"].T
        )
        block.ff.layers[2].bias = _assign(
            block.ff.layers[2].bias, sd[p + "mlp.c_proj.bias"]
        )

        # layer norms
        block.norm1.scale = _assign(block.norm1.scale, sd[p + "ln_1.weight"])
        block.norm1.shift = _assign(block.norm1.shift, sd[p + "ln_1.bias"])
        block.norm2.scale = _assign(block.norm2.scale, sd[p + "ln_2.weight"])
        block.norm2.shift = _assign(block.norm2.shift, sd[p + "ln_2.bias"])

    model.final_norm.scale = _assign(model.final_norm.scale, sd["ln_f.weight"])
    model.final_norm.shift = _assign(model.final_norm.shift, sd["ln_f.bias"])
    # weight tying: output head shares the token embedding matrix
    model.out_head.weight = _assign(model.out_head.weight, sd["wte.weight"])
    return model


def load_pretrained_gpt2(cfg=GPT_CONFIG_124M):
    path = hf_hub_download(repo_id="openai-community/gpt2", filename="model.safetensors")
    sd = load_file(path)
    model = GPTModel(cfg)
    load_gpt2_weights_into(model, sd)
    model.eval()
    return model


if __name__ == "__main__":
    import tiktoken
    from chapter04 import generate_text_simple

    model = load_pretrained_gpt2()
    tok = tiktoken.get_encoding("gpt2")
    idx = torch.tensor([tok.encode("Every effort moves you")])
    out = generate_text_simple(model, idx, max_new_tokens=20, context_size=1024)
    print(tok.decode(out[0].tolist()))
