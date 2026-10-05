"""Tests for external-data filtering and metric exports."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.external_test import clean_external, summarize, main
from src.features import domain_group
from src.utils import ROOT, sha256


class ExternalTests(unittest.TestCase):
    def test_filtering_conflicts_duplicates_and_all_known_domain_splits(self):
        reference = pd.DataFrame(
            {
                "url": ["https://known.com/a", "https://val.org", "https://test.net"],
                "group": ["known.com", "val.org", "test.net"],
                "split": ["train", "validation", "test"],
            }
        )
        raw = pd.DataFrame(
            {
                "url": [
                    "https://new.example/help",
                    "HTTPS://NEW.EXAMPLE/help",
                    "https://conflict.example",
                    "https://conflict.example",
                    "https://a.known.com/login",
                    "http://www.val.org/other",
                    "https://test.net",
                    "javascript:alert(1)",
                    "https://unknownlabel.example",
                    "http://other.example/verify",
                ],
                "status": [
                    "legitimate",
                    "legitimate",
                    "legitimate",
                    "phishing",
                    "phishing",
                    "legitimate",
                    "legitimate",
                    "phishing",
                    "safe",
                    "phishing",
                ],
            }
        )
        included, rejected, audit = clean_external(raw, reference)
        self.assertEqual(included.source_row.tolist(), [0, 9])
        self.assertEqual(included.label.tolist(), [0, 1])
        self.assertEqual(
            audit["status_counts"],
            {
                "seen_domain": 3,
                "included": 2,
                "conflicting_label": 2,
                "duplicate_url": 1,
                "invalid_url": 1,
                "invalid_label": 1,
            },
        )
        self.assertEqual(audit["remaining_domain_overlap"], 0)
        self.assertEqual(len(rejected), 8)

    def test_scheme_and_path_bytes_preserved_and_private_suffix_groups(self):
        reference = pd.DataFrame(
            {
                "url": ["https://one.github.io/A"],
                "group": ["one.github.io"],
                "split": ["train"],
            }
        )
        raw = pd.DataFrame(
            {
                "url": [
                    " http://two.github.io/Case?x=%2F#Ab ",
                    "https://sub.one.github.io/other",
                ],
                "status": [" legitimate ", "phishing"],
            }
        )
        included, _, audit = clean_external(raw, reference)
        self.assertEqual(included.url.tolist(), ["http://two.github.io/Case?x=%2F#Ab"])
        self.assertEqual(audit["status_counts"]["seen_domain"], 1)
        with self.assertRaisesRegex(ValueError, "Missing columns"):
            clean_external(pd.DataFrame({"label": [1]}), reference)

    def test_threshold_and_confusion_counts(self):
        values = summarize(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.2, 0.9]), 0.3)
        self.assertEqual([values[k] for k in ["tn", "fp", "fn", "tp"]], [1, 1, 1, 1])
        self.assertEqual(values["precision"], 0.5)
        self.assertEqual(values["recall"], 0.5)
        self.assertEqual(values["f1"], 0.5)

    @unittest.skipUnless(
        (ROOT / "models/random_forest_augmented.joblib").exists(), "Run training first"
    )
    def test_cli_actual_models_offline_and_recomputable_exports(self):
        model_path = ROOT / "models/random_forest_augmented.joblib"
        before = sha256(model_path)

        def forbidden(*args, **kwargs):
            raise AssertionError("External inference attempted a network request")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "synthetic_fixture.csv"
            pd.DataFrame(
                {
                    "url": [
                        "https://fixture-good.example/help",
                        "http://fixture-bad.example/verify?x=123",
                        "https://fixture-good.example/help",
                        "javascript:alert(1)",
                    ],
                    "status": ["legitimate", "phishing", "legitimate", "phishing"],
                }
            ).to_csv(source, index=False)
            output = root / "results"
            with patch("socket.getaddrinfo", forbidden), patch(
                "socket.create_connection", forbidden
            ), patch("urllib.request.urlopen", forbidden), patch(
                "requests.sessions.Session.request", forbidden
            ), contextlib.redirect_stdout(
                io.StringIO()
            ):
                main(["--input", str(source), "--output-dir", str(output)])
            audit = json.loads((output / "audit.json").read_text())
            self.assertEqual(audit["execution_status"], "completed")
            self.assertFalse(audit["retraining"])
            self.assertEqual(audit["input_sha256"], sha256(source))
            self.assertEqual(audit["included_rows"], 2)
            metrics = pd.read_csv(output / "model_comparison.csv")
            rows = pd.read_csv(output / "predictions.csv.gz")
            for item in metrics.itertuples(index=False):
                subset = rows[rows.model.eq(item.model)]
                expected = summarize(
                    subset.label.to_numpy(),
                    subset.phishing_probability.to_numpy(),
                    item.threshold,
                )
                for name in ["tn", "fp", "fn", "tp", "rows"]:
                    self.assertEqual(getattr(item, name), expected[name])
                for name in ["f1", "precision", "recall", "roc_auc"]:
                    self.assertAlmostEqual(getattr(item, name), expected[name])
            for name, digest in audit["output_sha256"].items():
                self.assertEqual(sha256(output / name), digest)
            self.assertTrue((output / "confusion_matrices.png").is_file())
        self.assertEqual(sha256(model_path), before)


if __name__ == "__main__":
    unittest.main()
