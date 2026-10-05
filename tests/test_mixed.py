"""Tests for mixed-source splitting and training lineage."""

import contextlib
import io
import json
import unittest

import pandas as pd

from src.improve_data import external_split, CANDIDATE
from src.external_test import main, summarize, trusted_bundle, load_reference
from src.utils import ROOT, sha256


class MixedTests(unittest.TestCase):
    def test_grouped_split_is_deterministic_complete_and_disjoint(self):
        frame = pd.DataFrame(
            [
                {
                    "url": f"https://g{i}.example/p{j}",
                    "group": f"g{i}.example",
                    "label": i % 2,
                    "source_row": i * 2 + j,
                }
                for i in range(50)
                for j in range(2)
            ]
        )
        a, b = external_split(frame), external_split(frame)
        self.assertTrue(a.equals(b))
        self.assertEqual(sorted(a.source_row), list(range(100)))
        self.assertEqual(a.groupby("group").split.nunique().max(), 1)
        self.assertEqual(set(a.split), {"train", "validation", "test"})
        self.assertTrue((a.groupby("split").label.nunique() == 2).all())

    @unittest.skipUnless(
        (ROOT / "models/random_forest_mixed.joblib").exists(),
        "Optional mixed model not trained locally",
    )
    def test_recorded_lineage_models_and_metrics(self):
        protocol = json.loads(
            (ROOT / "results/metrics/mixed_training_protocol.json").read_text()
        )
        splits = pd.read_csv(ROOT / "results/mixed/external_splits.csv.gz")
        self.assertEqual(splits.groupby("group").split.nunique().max(), 1)
        self.assertEqual(
            int(splits.split.eq("train").sum()), protocol["external_train_rows"]
        )
        self.assertEqual(
            protocol["training_rows"],
            protocol["original_train_rows"] + protocol["external_train_rows"],
        )
        reference, training, _ = load_reference()
        self.assertFalse(set(reference.group) & set(splits.group))
        bundle, _ = trusted_bundle(CANDIDATE, training)
        self.assertEqual(bundle["threshold"], 0.5)
        for name, digest in protocol["baseline_model_sha256"].items():
            self.assertEqual(sha256(ROOT / "models" / name), digest)
        comparison = pd.read_csv(ROOT / "results/mixed/model_comparison.csv")
        predictions = pd.read_csv(ROOT / "results/mixed/predictions.csv.gz")
        for row in comparison.itertuples(index=False):
            p = predictions[
                (predictions.dataset == row.dataset)
                & (predictions.split == row.split)
                & (predictions.model == row.model)
            ]
            expected = summarize(
                p.label.to_numpy(), p.phishing_probability.to_numpy(), row.threshold
            )
            for key in ["tn", "fp", "fn", "tp", "rows"]:
                self.assertEqual(getattr(row, key), expected[key])
            for key in ["precision", "recall", "f1", "roc_auc"]:
                self.assertAlmostEqual(getattr(row, key), expected[key])

    @unittest.skipUnless(
        (ROOT / "models/random_forest_mixed.joblib").exists(),
        "Optional mixed model not trained",
    )
    def test_mixed_model_cannot_claim_unseen_hannousse_benchmark(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
            SystemExit
        ) as result:
            main(["--input", "unused.csv", "--models", CANDIDATE])
        self.assertEqual(result.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
