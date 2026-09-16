import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

import enc_lib


def encoder_weights(prefix, offset=0.0):
    torch.manual_seed(7)
    weights = {
        "pos_embed.weight": torch.randn(4, 8) + offset,
        "proj_in.weight": torch.randn(8, 12) + offset,
        "proj_out.weight": torch.randn(12, 8) + offset,
        "norm.weight": torch.randn(8) + offset,
        "norm.bias": torch.randn(8) + offset,
    }
    for layer in range(2):
        base = f"layers.{layer}."
        for name in ("wq", "wk", "wv", "wo"):
            weights[base + name + ".weight"] = torch.randn(8, 8) + offset
        weights[base + "w1.weight"] = torch.randn(16, 8) + offset
        weights[base + "w2.weight"] = torch.randn(8, 16) + offset
        for norm in ("norm1", "norm2"):
            weights[base + norm + ".weight"] = torch.randn(8) + offset
            weights[base + norm + ".bias"] = torch.randn(8) + offset
    return {prefix + key: value for key, value in weights.items()}


def write_config(folder, tie_encoders=False):
    config = {
        "format_version": 2,
        "encoder_type": "res_latent_attn",
        "encoder": {
            "hidden_size": 8,
            "model_hidden_size": 12,
            "num_hidden_layers": 2,
            "intermediate_size": 16,
            "num_heads": 2,
            "causal": False,
            "residual": True,
            "tie_encoders": tie_encoders,
        },
        "compression": {
            "initial_vocab_size": 20,
            "max_codebook_size": 8,
            "max_subtokens": 4,
            "disabled_ids": [0],
        },
    }
    (folder / "zip2zip_config.json").write_text(json.dumps(config))


class HfCheckpointTest(unittest.TestCase):
    def test_sharded_untied_export_with_latent_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_config(folder)
            encoders = encoder_weights("input_encoder.")
            encoders.update(encoder_weights("output_encoder.", offset=0.5))
            save_file(encoders, folder / "zip2zip_encoders.safetensors")

            input_table = torch.randn(20, 12)
            output_table = torch.randn(20, 12)
            shard1 = "model-00001-of-00002.safetensors"
            shard2 = "model-00002-of-00002.safetensors"
            save_file({enc_lib.INPUT_EMBEDDING_KEY: input_table}, folder / shard1)
            save_file({enc_lib.OUTPUT_EMBEDDING_KEY: output_table}, folder / shard2)
            index = {
                "metadata": {},
                "weight_map": {
                    enc_lib.INPUT_EMBEDDING_KEY: shard1,
                    enc_lib.OUTPUT_EMBEDDING_KEY: shard2,
                },
            }
            (folder / "model.safetensors.index.json").write_text(json.dumps(index))

            bundle = enc_lib.load_checkpoint(folder)
            self.assertEqual(bundle.initial_vocab_size, 20)
            self.assertIsNone(bundle.tokenizer_revision)
            self.assertTrue(torch.equal(
                bundle.state_dict["tok_embeddings.weight"], input_table
            ))
            self.assertTrue(torch.equal(
                bundle.state_dict["output.weight"], output_table
            ))

            input_encoder, output_encoder = enc_lib.load_pair(bundle)
            ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
            mask = torch.tensor([[True, True, True], [True, True, False]])
            self.assertEqual(input_encoder.encode(ids, mask).shape, (2, 12))
            self.assertEqual(output_encoder.encode(ids, mask).shape, (2, 12))
            self.assertFalse(torch.equal(
                input_encoder.p["norm.weight"], output_encoder.p["norm.weight"]
            ))

    def test_unsharded_tied_export_reuses_input_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_config(folder, tie_encoders=True)
            save_file(
                encoder_weights("input_encoder."),
                folder / "zip2zip_encoders.safetensors",
            )
            table = torch.randn(20, 12)
            save_file(
                {enc_lib.INPUT_EMBEDDING_KEY: table},
                folder / "model.safetensors",
            )

            bundle = enc_lib.load_checkpoint(folder)
            input_encoder, output_encoder = enc_lib.load_pair(bundle)
            self.assertIs(
                bundle.state_dict["tok_embeddings.weight"],
                bundle.state_dict["output.weight"],
            )
            self.assertTrue(torch.equal(
                input_encoder.p["norm.weight"], output_encoder.p["norm.weight"]
            ))

    def test_legacy_load_pair_still_needs_no_tokenizer_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pt"
            state_dict = encoder_weights("hyper_encoder.")
            state_dict.update(encoder_weights("hyper_output.", offset=0.5))
            state_dict["tok_embeddings.weight"] = torch.randn(20, 12)
            state_dict["output.weight"] = torch.randn(20, 12)
            torch.save(state_dict, path)

            input_encoder, output_encoder = enc_lib.load_pair(
                path, legacy_num_heads=2
            )
            self.assertEqual(input_encoder.emb.shape, (20, 12))
            self.assertEqual(output_encoder.emb.shape, (20, 12))



if __name__ == "__main__":
    unittest.main()

