"""Standalone Zip2Zip++ hyper-encoder loading and forward pass.

The public path reads a self-contained Zip2Zip++ Hugging Face export. Only the
encoder file and decoder shards containing the two embedding tables are read.
Legacy extracted ``weights/*.pt`` files remain supported for paper results.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


INPUT_EMBEDDING_KEY = "model.embed_tokens.weight"
OUTPUT_EMBEDDING_KEY = "lm_head.weight"


@dataclass
class CheckpointBundle:
    """Weights and metadata required by the probes."""

    state_dict: dict[str, torch.Tensor]
    encoder_config: dict
    tokenizer_name_or_path: str
    tokenizer_revision: str | None
    initial_vocab_size: int
    source: str

    @property
    def residual(self):
        return bool(self.encoder_config.get("residual", True))


class Enc:
    """One hyper-encoder (input or output role) and its embedding matrix."""

    def __init__(self, sd, prefix, emb_key, encoder_config):
        prefix_dot = prefix + "."
        self.p = {
            key[len(prefix_dot):]: value
            for key, value in sd.items()
            if key.startswith(prefix_dot)
        }
        if not self.p:
            raise KeyError(f"checkpoint has no weights with prefix {prefix_dot!r}")
        self.emb = sd[emb_key]
        self.dim = int(encoder_config["hidden_size"])
        self.model_dim = int(encoder_config.get("model_hidden_size") or self.dim)
        self.n_heads = int(encoder_config["num_heads"])
        self.n_layers = int(encoder_config["num_hidden_layers"])
        self.residual = bool(encoder_config.get("residual", True))
        self.causal = bool(encoder_config.get("causal", False))
        if self.dim % self.n_heads:
            raise ValueError(
                f"encoder hidden_size={self.dim} is not divisible by "
                f"num_heads={self.n_heads}"
            )
        self.head_dim = self.dim // self.n_heads
        if self.emb.shape[1] != self.model_dim:
            raise ValueError(
                f"embedding dim {self.emb.shape[1]} does not match configured "
                f"model_hidden_size {self.model_dim}"
            )

    def _layer(self, li, x, mask):
        p, (batch, seq_len, dim), pre = self.p, x.shape, f"layers.{li}."
        residual = x
        x_norm = F.layer_norm(
            x, (dim,), p[pre + "norm1.weight"], p[pre + "norm1.bias"]
        )
        q = F.linear(x_norm, p[pre + "wq.weight"])
        k = F.linear(x_norm, p[pre + "wk.weight"])
        v = F.linear(x_norm, p[pre + "wv.weight"])
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_mask = mask.unsqueeze(1).unsqueeze(2)
        if self.causal:
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
            )
            attn_mask = attn_mask & causal_mask
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        x = residual + F.linear(attn_out, p[pre + "wo.weight"])

        residual = x
        x_norm = F.layer_norm(
            x, (dim,), p[pre + "norm2.weight"], p[pre + "norm2.bias"]
        )
        return residual + F.linear(
            F.gelu(F.linear(x_norm, p[pre + "w1.weight"])),
            p[pre + "w2.weight"],
        )

    def _encoder_forward(self, token_embeddings, mask):
        p, seq_len = self.p, token_embeddings.shape[1]
        if seq_len > p["pos_embed.weight"].shape[0]:
            raise ValueError(
                f"sequence length {seq_len} exceeds encoder max_subtokens "
                f"{p['pos_embed.weight'].shape[0]}"
            )
        x = token_embeddings
        if "proj_in.weight" in p:
            x = F.linear(x, p["proj_in.weight"])
        x = x + p["pos_embed.weight"][torch.arange(seq_len, device=x.device)]
        for li in range(self.n_layers):
            x = self._layer(li, x, mask)
        x = F.layer_norm(x, (self.dim,), p["norm.weight"], p["norm.bias"])

        if self.causal:
            lengths = mask.sum(-1).clamp(min=1)
            result = x[
                torch.arange(x.shape[0], device=x.device), (lengths - 1).long()
            ]
        else:
            masked = x * mask.unsqueeze(-1)
            result = masked.sum(1) / mask.sum(-1, keepdim=True).clamp(min=1)
        if "proj_out.weight" in p:
            result = F.linear(result, p["proj_out.weight"])
        return result

    def encode(self, ids, mask, residual=None):
        """Return the final vector consumed by the decoder."""
        token_embeddings = self.emb[ids]
        out = self._encoder_forward(token_embeddings, mask)
        use_residual = self.residual if residual is None else residual
        return token_embeddings[:, 0, :] + out if use_residual else out


def _randomize(sd, seed=0):
    generator = torch.Generator().manual_seed(seed)
    randomized = {}
    for key, value in sd.items():
        if key.startswith(("hyper_encoder.", "hyper_output.")):
            if value.dim() == 2:
                weight = torch.empty_like(value)
                torch.nn.init.xavier_uniform_(weight, generator=generator)
                randomized[key] = weight
            elif "norm" in key and key.endswith("weight"):
                randomized[key] = torch.ones_like(value)
            else:
                randomized[key] = torch.zeros_like(value)
        else:
            randomized[key] = value
    return randomized


def _legacy_encoder_config(sd, num_heads):
    hidden_size = int(sd["hyper_encoder.norm.weight"].numel())
    layer_ids = {
        int(match.group(1))
        for key in sd
        if (match := re.match(r"hyper_encoder\.layers\.(\d+)\.", key))
    }
    return {
        "hidden_size": hidden_size,
        "model_hidden_size": int(sd["tok_embeddings.weight"].shape[1]),
        "num_hidden_layers": len(layer_ids),
        "intermediate_size": int(sd["hyper_encoder.layers.0.w1.weight"].shape[0]),
        "num_heads": num_heads,
        "causal": False,
        "residual": True,
        "tie_encoders": False,
    }


def _local_or_download(
    source, filename, *, revision, cache_dir, token, local_files_only
):
    local = Path(source).expanduser()
    if local.is_dir():
        path = local / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing {filename} in {local}")
        return path

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=source,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
    )


def _read_decoder_tables(
    source, *, revision, cache_dir, token, local_files_only
):
    """Read only the decoder shard(s) that own the two embedding tables."""
    local = Path(source).expanduser()
    index_path = local / "model.safetensors.index.json" if local.is_dir() else None
    if index_path is None or not index_path.is_file():
        if local.is_dir():
            index_path = None
        else:
            try:
                index_path = _local_or_download(
                    source,
                    "model.safetensors.index.json",
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    local_files_only=local_files_only,
                )
            except Exception as exc:
                from huggingface_hub.utils import EntryNotFoundError

                if not isinstance(exc, EntryNotFoundError):
                    raise
                index_path = None

    if index_path is not None:
        with open(index_path, encoding="utf-8") as file:
            weight_map = json.load(file)["weight_map"]
        missing = [
            key
            for key in (INPUT_EMBEDDING_KEY, OUTPUT_EMBEDDING_KEY)
            if key not in weight_map
        ]
        if missing and missing != [OUTPUT_EMBEDDING_KEY]:
            raise KeyError(f"decoder index is missing required tensors: {missing}")
        filenames = {
            key: weight_map[key]
            for key in (INPUT_EMBEDDING_KEY, OUTPUT_EMBEDDING_KEY)
            if key in weight_map
        }
    else:
        filenames = {
            INPUT_EMBEDDING_KEY: "model.safetensors",
            OUTPUT_EMBEDDING_KEY: "model.safetensors",
        }

    from safetensors import safe_open

    tensors = {}
    for filename in sorted(set(filenames.values())):
        path = _local_or_download(
            source,
            filename,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        wanted = [key for key, owner in filenames.items() if owner == filename]
        with safe_open(path, framework="pt", device="cpu") as file:
            available = set(file.keys())
            for key in wanted:
                if key in available:
                    tensors[key] = file.get_tensor(key)
    if INPUT_EMBEDDING_KEY not in tensors:
        raise KeyError(f"decoder weights are missing {INPUT_EMBEDDING_KEY}")
    tensors.setdefault(OUTPUT_EMBEDDING_KEY, tensors[INPUT_EMBEDDING_KEY])
    return tensors


def load_hf_checkpoint(
    source,
    *,
    revision="hf",
    cache_dir=None,
    token=None,
    local_files_only=False,
):
    """Load a public Zip2Zip++ export from the Hub or a local export folder."""
    config_path = _local_or_download(
        source,
        "zip2zip_config.json",
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    if config.get("encoder_type") != "res_latent_attn":
        raise ValueError(
            "probe supports encoder_type='res_latent_attn', got "
            f"{config.get('encoder_type')!r}"
        )

    encoder_path = _local_or_download(
        source,
        "zip2zip_encoders.safetensors",
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    from safetensors.torch import load_file

    exported_encoders = load_file(str(encoder_path), device="cpu")
    state_dict = {}
    for key, value in exported_encoders.items():
        if key.startswith("input_encoder."):
            state_dict["hyper_encoder." + key.removeprefix("input_encoder.")] = value
        elif key.startswith("output_encoder."):
            state_dict["hyper_output." + key.removeprefix("output_encoder.")] = value

    encoder_config = config["encoder"]
    if encoder_config.get("tie_encoders", False):
        for key, value in list(state_dict.items()):
            if key.startswith("hyper_encoder."):
                state_dict[
                    "hyper_output." + key.removeprefix("hyper_encoder.")
                ] = value
    tables = _read_decoder_tables(
        source,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    state_dict["tok_embeddings.weight"] = tables[INPUT_EMBEDDING_KEY]
    state_dict["output.weight"] = tables[OUTPUT_EMBEDDING_KEY]
    compression = config["compression"]
    return CheckpointBundle(
        state_dict=state_dict,
        encoder_config=encoder_config,
        tokenizer_name_or_path=source,
        tokenizer_revision=None if Path(source).expanduser().is_dir() else revision,
        initial_vocab_size=int(compression["initial_vocab_size"]),
        source=source,
    )


def load_checkpoint(
    source,
    *,
    revision="hf",
    cache_dir=None,
    token=None,
    local_files_only=False,
    legacy_num_heads=32,
    legacy_tokenizer=None,
    legacy_vocab_size=None,
    legacy_residual=True,
):
    """Load either a Hub/local HF export or a legacy extracted ``.pt`` file."""
    source_str = str(source)
    path = Path(source_str).expanduser()
    if path.is_file():
        sd = torch.load(path, map_location="cpu", weights_only=True)
        encoder_config = _legacy_encoder_config(sd, legacy_num_heads)
        encoder_config["residual"] = legacy_residual
        return CheckpointBundle(
            state_dict=sd,
            encoder_config=encoder_config,
            tokenizer_name_or_path=legacy_tokenizer,
            tokenizer_revision=None,
            initial_vocab_size=legacy_vocab_size,
            source=source_str,
        )
    return load_hf_checkpoint(
        source_str,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )


def load_pair(source, random_init=False, seed=0, **load_kwargs):
    """Return input/output probe encoders for a checkpoint or loaded bundle."""
    bundle = (
        source
        if isinstance(source, CheckpointBundle)
        else load_checkpoint(source, **load_kwargs)
    )
    state_dict = (
        _randomize(bundle.state_dict, seed)
        if random_init
        else bundle.state_dict
    )
    return (
        Enc(
            state_dict, "hyper_encoder", "tok_embeddings.weight",
            bundle.encoder_config,
        ),
        Enc(
            state_dict, "hyper_output", "output.weight",
            bundle.encoder_config,
        ),
    )
