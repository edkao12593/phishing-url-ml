"""Tests for URL parsing, data splitting, metrics and the demo."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.features import (
    extract_features,
    canonical_url,
    domain_group,
    feature_frame,
    FEATURE_NAMES,
    parse_url,
)
from src.robustness import perturb, TEST_TRANSFORMS
from src.evaluate import metrics
from src.utils import ROOT, load_processed, sha256


class URLTests(unittest.TestCase):
    def test_exact_features(self):
        f = extract_features("https://a.example.org/login?a=12&b=3")
        self.assertEqual(f["https"], 1)
        self.assertEqual(f["subdomain_count"], 1)
        self.assertEqual(f["path_depth"], 1)
        self.assertEqual(f["query_parameter_count"], 2)
        self.assertEqual(f["digit_count"], 3)
        self.assertEqual(f["token_login"], 1)
        self.assertEqual(f["path_length"], 6)
        self.assertTrue(np.isfinite(list(f.values())).all())

    def test_authority_not_embedded_brand(self):
        f = extract_features("https://bank.example@192.0.2.1:443/Login")
        self.assertEqual(f["ip_host"], 1)
        self.assertEqual(f["has_userinfo"], 1)
        self.assertEqual(f["has_port"], 1)
        self.assertEqual(parse_url("https://[2001:db8::1]/")[2], "2001:db8::1")

    def test_canonicalization_preserves_case_sensitive_path(self):
        self.assertEqual(
            canonical_url("HTTPS://EXAMPLE.ORG/Login?A=B"),
            "https://example.org/Login?A=B",
        )
        self.assertNotEqual(
            canonical_url("https://example.org/Login"),
            canonical_url("https://example.org/login"),
        )
        self.assertEqual(canonical_url("example.org/a"), "http://example.org/a")

    def test_public_and_private_suffix_grouping(self):
        self.assertEqual(domain_group("https://a.b.example.co.uk/"), "example.co.uk")
        self.assertNotEqual(
            domain_group("https://alice.github.io/"),
            domain_group("https://bob.github.io/"),
        )
        self.assertEqual(domain_group("http://192.0.2.1/"), "192.0.2.1")

    def test_invalid_inputs(self):
        for u in [
            "",
            "javascript:alert(1)",
            "ftp://example.org",
            "https://example.org:99999",
            "https://ex ample.org",
            "http://[bad]/",
        ]:
            with self.subTest(u=u), self.assertRaises(ValueError):
                parse_url(u)

    def test_no_network(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("network operation attempted")

        with patch("socket.getaddrinfo", forbidden), patch(
            "urllib.request.urlopen", forbidden
        ), patch("requests.sessions.Session.request", forbidden):
            for kind in TEST_TRANSFORMS:
                url = perturb("https://a.example.org/Hello?x=1#anchor", kind)
                self.assertEqual(
                    url, perturb("https://a.example.org/Hello?x=1#anchor", kind)
                )
                self.assertEqual(len(extract_features(url)), 43)
                self.assertEqual(domain_group(url), "example.org")

    def test_applicable_and_unchanged_cases(self):
        u = "http://192.0.2.1/"
        self.assertEqual(perturb(u, "subdomain_prefix"), u)
        self.assertEqual(perturb(u, "percent_encode_path"), u)
        self.assertEqual(
            perturb("https://example.org/%41B", "percent_encode_path"),
            "https://example.org/%41%42",
        )


class EvaluationTests(unittest.TestCase):
    def test_positive_class_and_counts(self):
        m = metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.3, 0.9]))
        self.assertEqual((m["tn"], m["fp"], m["fn"], m["tp"]), (1, 1, 1, 1))
        for key in ["accuracy", "precision", "recall", "f1"]:
            self.assertEqual(m[key], 0.5)

    @unittest.skipUnless(
        (ROOT / "data/processed/samples.csv").exists(), "Run preprocessing first"
    )
    def test_full_data_alignment_and_domain_disjoint(self):
        d, x = load_processed()
        self.assertEqual(len(d), len(x))
        self.assertEqual(list(x.columns), FEATURE_NAMES)
        self.assertEqual(d.url.nunique(), len(d))
        g = {
            s: set(d.loc[d.split == s, "group"])
            for s in ["train", "validation", "test"]
        }
        self.assertFalse(g["train"] & g["test"])
        self.assertFalse(g["train"] & g["validation"])
        self.assertFalse(g["test"] & g["validation"])
        self.assertTrue(np.allclose(x.iloc[:8], feature_frame(d.url.iloc[:8])))

    def test_stale_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p / "data/processed").mkdir(parents=True)
            (p / "results/metrics").mkdir(parents=True)
            (p / "data/processed/samples.csv").write_text("wrong", encoding="utf-8")
            (p / "data/processed/features.csv.gz").write_text("wrong", encoding="utf-8")
            (p / "results/metrics/data_audit.json").write_text(
                json.dumps({"samples_sha256": "invalid", "features_sha256": "invalid"})
            )
            with patch("src.utils.ROOT", p), self.assertRaisesRegex(
                ValueError, "checksums"
            ):
                load_processed()

    @unittest.skipUnless(
        (ROOT / "results/metrics/test_predictions.csv.gz").exists(),
        "Run evaluation first",
    )
    def test_metrics_match_saved_predictions(self):
        pred = pd.read_csv(ROOT / "results/metrics/test_predictions.csv.gz")
        clean = pd.read_csv(ROOT / "results/metrics/model_comparison.csv")
        for row in clean[clean.split == "test"].itertuples():
            sub = pred[(pred.model == row.model) & (pred.condition == "original")]
            m = metrics(sub.label, sub.probability)
            self.assertAlmostEqual(m["f1"], row.f1, places=12)
            self.assertEqual(m["fn"], row.fn)


@unittest.skipUnless(
    (ROOT / "models/random_forest_augmented.joblib").exists(), "Run training first"
)
class DemoTests(unittest.TestCase):
    def test_demo_valid_and_invalid_input_offline(self):
        from streamlit.testing.v1 import AppTest

        def forbidden(*args, **kwargs):
            raise AssertionError("external request attempted")

        with patch("urllib.request.urlopen", forbidden), patch(
            "requests.sessions.Session.request", forbidden
        ):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            self.assertEqual(len(app.exception), 0)
            app.text_input[0].set_value("https://example.org/help?lang=en")
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.metric), 4)
            app.text_input[0].set_value("javascript:alert(1)")
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.error), 1)


if __name__ == "__main__":
    unittest.main()
