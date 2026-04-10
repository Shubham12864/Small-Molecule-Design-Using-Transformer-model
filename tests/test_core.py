import unittest

import torch

from src.analytics import compute_batch_stats
from src.model import MoleculeTransformer
from src.splits import build_dataset_splits
from src.tokenizer import SmilesTokenizer


class TokenizerTests(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        tokenizer = SmilesTokenizer()
        tokenizer.fit(["CCO", "ClC1=CC=CC=C1", "c1ccccc1"])

        smiles = "ClC1=CC=CC=C1"
        encoded = tokenizer.encode(smiles)
        decoded = tokenizer.decode(encoded)

        self.assertEqual(decoded, smiles)


class SplitTests(unittest.TestCase):
    def test_random_split_has_no_overlap(self):
        smiles_list = [
            "CCO",
            "CCN",
            "CCC",
            "CCCl",
            "CCBr",
            "c1ccccc1",
            "CC(=O)O",
            "CCS",
            "COC",
            "CCF",
        ]

        split_data, metadata = build_dataset_splits(
            smiles_list=smiles_list,
            val_split=0.2,
            test_split=0.2,
            seed=42,
            split_method="random",
            dedup=True,
        )

        self.assertEqual(metadata["overlaps"]["train_val"], 0)
        self.assertEqual(metadata["overlaps"]["train_test"], 0)
        self.assertEqual(metadata["overlaps"]["val_test"], 0)
        self.assertEqual(
            len(split_data["train"]) + len(split_data["val"]) + len(split_data["test"]),
            metadata["total_after_preprocessing"],
        )


class ModelTests(unittest.TestCase):
    def test_model_forward_shape(self):
        model = MoleculeTransformer(
            vocab_size=20,
            d_model=32,
            nhead=4,
            num_layers=2,
            dim_feedforward=64,
            max_len=16,
            dropout=0.1,
            pad_token_id=0,
        )
        batch = torch.randint(0, 20, (3, 12))
        logits = model(batch)

        self.assertEqual(tuple(logits.shape), (3, 12, 20))


class AnalyticsTests(unittest.TestCase):
    def test_batch_stats_reports_uniqueness(self):
        rows = [
            {
                "smiles": "CCO",
                "canonical_smiles": "CCO",
                "mw": 46.07,
                "logp": -0.3,
                "qed": 0.4,
                "lipinski": True,
            },
            {
                "smiles": "OCC",
                "canonical_smiles": "CCO",
                "mw": 46.07,
                "logp": -0.3,
                "qed": 0.4,
                "lipinski": True,
            },
            {
                "smiles": "CCN",
                "canonical_smiles": "CCN",
                "mw": 45.08,
                "logp": -0.1,
                "qed": 0.5,
                "lipinski": True,
            },
        ]

        stats = compute_batch_stats(rows, total_attempts=5)

        self.assertEqual(stats["num_unique"], 2)
        self.assertEqual(stats["duplicate_valid_count"], 1)
        self.assertAlmostEqual(stats["validity_rate"], 60.0)


if __name__ == "__main__":
    unittest.main()
