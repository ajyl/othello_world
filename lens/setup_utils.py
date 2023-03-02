"""
Module Doc String
"""


def _convert_to_transformer_lens_format(in_sd, n_layers=8, n_heads=8):
    out_sd = {}
    out_sd["pos_embed.W_pos"] = in_sd["pos_emb"].squeeze(0)
    out_sd["embed.W_E"] = in_sd["tok_emb.weight"]

    out_sd["ln_final.w"] = in_sd["ln_f.weight"]
    out_sd["ln_final.b"] = in_sd["ln_f.bias"]
    out_sd["unembed.W_U"] = in_sd["head.weight"].T

    for layer in range(n_layers):
        out_sd[f"blocks.{layer}.ln1.w"] = in_sd[f"blocks.{layer}.ln1.weight"]
        out_sd[f"blocks.{layer}.ln1.b"] = in_sd[f"blocks.{layer}.ln1.bias"]
        out_sd[f"blocks.{layer}.ln2.w"] = in_sd[f"blocks.{layer}.ln2.weight"]
        out_sd[f"blocks.{layer}.ln2.b"] = in_sd[f"blocks.{layer}.ln2.bias"]

        out_sd[f"blocks.{layer}.attn.W_Q"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.query.weight"],
            "(head d_head) d_model -> head d_model d_head",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.b_Q"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.query.bias"],
            "(head d_head) -> head d_head",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.W_K"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.key.weight"],
            "(head d_head) d_model -> head d_model d_head",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.b_K"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.key.bias"],
            "(head d_head) -> head d_head",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.W_V"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.value.weight"],
            "(head d_head) d_model -> head d_model d_head",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.b_V"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.value.bias"],
            "(head d_head) -> head d_head",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.W_O"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.proj.weight"],
            "d_model (head d_head) -> head d_head d_model",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.b_O"] = in_sd[
            f"blocks.{layer}.attn.proj.bias"
        ]

        out_sd[f"blocks.{layer}.mlp.b_in"] = in_sd[
            f"blocks.{layer}.mlp.0.bias"
        ]
        out_sd[f"blocks.{layer}.mlp.W_in"] = in_sd[
            f"blocks.{layer}.mlp.0.weight"
        ].T
        out_sd[f"blocks.{layer}.mlp.b_out"] = in_sd[
            f"blocks.{layer}.mlp.2.bias"
        ]
        out_sd[f"blocks.{layer}.mlp.W_out"] = in_sd[
            f"blocks.{layer}.mlp.2.weight"
        ].T

    return out_sd

