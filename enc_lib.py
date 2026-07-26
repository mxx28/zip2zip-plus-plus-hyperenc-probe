"""Faithful standalone re-implementation of the zip2zip flat HyperEncoder
forward (non-varlen, CPU) + _encode_codebook_with_weights residual.

Bit-exact vs src/zip2zip_core/model.py:HyperEncoder (validated max|Δ|=0 on
v0.5, v0.52, v0.6.4). encoder_dim == model_dim == 3072, so no proj_in/proj_out.
Load a checkpoint's extracted weights with load_pair(path); the returned Enc
objects run the same forward the model uses.
"""
import torch
import torch.nn.functional as F

DIM = 3072
N_HEADS = 32
N_LAYERS = 2
HEAD_DIM = DIM // N_HEADS


class Enc:
    """One hyper-encoder (input or output role) + its embedding matrix.

    encode(ids, mask, residual): the final vector the model uses.
      residual=True  -> E = base_vec[t1] + encoder_out   (models with residual)
      residual=False -> E = encoder_out                  (no_encoder_residual runs)
    """
    def __init__(self, sd, prefix, emb_key):
        self.p = {k[len(prefix) + 1:]: v for k, v in sd.items() if k.startswith(prefix + ".")}
        self.emb = sd[emb_key]

    def _layer(self, li, x, mask):
        p, (B, S, D), pre = self.p, x.shape, f"layers.{li}."
        resid = x
        xn = F.layer_norm(x, (D,), p[pre + "norm1.weight"], p[pre + "norm1.bias"])
        q = (xn @ p[pre + "wq.weight"].T).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = (xn @ p[pre + "wk.weight"].T).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = (xn @ p[pre + "wv.weight"].T).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        am = mask.unsqueeze(1).unsqueeze(2).float()
        am = am.masked_fill(am == 0, float("-inf")).masked_fill(am == 1, 0.0)
        ao = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        ao = ao.transpose(1, 2).contiguous().view(B, S, D)
        x = resid + ao @ p[pre + "wo.weight"].T
        resid = x
        xn = F.layer_norm(x, (D,), p[pre + "norm2.weight"], p[pre + "norm2.bias"])
        return resid + F.gelu(xn @ p[pre + "w1.weight"].T) @ p[pre + "w2.weight"].T

    def _encoder_forward(self, tok_emb, mask):
        p, S = self.p, tok_emb.shape[1]
        x = tok_emb + p["pos_embed.weight"][torch.arange(S)]
        for li in range(N_LAYERS):
            x = self._layer(li, x, mask)
        x = F.layer_norm(x, (DIM,), p["norm.weight"], p["norm.bias"])
        masked = x * mask.unsqueeze(-1)
        return masked.sum(1) / mask.sum(-1, keepdim=True).clamp(min=1)

    def encode(self, ids, mask, residual=True):
        tok_emb = self.emb[ids]
        out = self._encoder_forward(tok_emb, mask)
        return (tok_emb[:, 0, :] + out) if residual else out


def _randomize(sd, seed=0):
    g = torch.Generator().manual_seed(seed)
    new = {}
    for k, v in sd.items():
        if k.startswith("hyper_encoder") or k.startswith("hyper_output"):
            if v.dim() == 2:
                w = torch.empty_like(v); torch.nn.init.xavier_uniform_(w, generator=g); new[k] = w
            elif "norm" in k and k.endswith("weight"): new[k] = torch.ones_like(v)
            elif "pos_embed" in k:
                w = torch.empty_like(v); torch.nn.init.normal_(w, std=0.02, generator=g); new[k] = w
            else: new[k] = torch.zeros_like(v)
        else: new[k] = v
    return new


def load_pair(weights_path, random_init=False, seed=0):
    """Return (input_encoder, output_encoder) for a checkpoint's extracted weights.
    random_init replaces the encoder weights with a fresh init (control baseline);
    the embedding matrices stay real."""
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    if random_init:
        sd = _randomize(sd, seed)
    return Enc(sd, "hyper_encoder", "tok_embeddings.weight"), Enc(sd, "hyper_output", "output.weight")
