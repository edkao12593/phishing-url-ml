"""Tests for batch prediction and CSV exports."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src.features import feature_frame
from src.predict import predict_lines, csv_bytes, MAX_BATCH_URLS, main
from src.utils import ROOT


@unittest.skipUnless(
    (ROOT / "models/random_forest_augmented.joblib").exists(), "Run training first"
)
class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.name = "random_forest_augmented"
        cls.bundle = joblib.load(ROOT / "models" / f"{cls.name}.joblib")

    def test_alignment_errors_duplicates_and_no_network(self):
        text = (
            "https://example.com\n\njavascript:alert(1)\n"
            " https://example.org/login?a=1 \nhttps://example.com\n"
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("Network access attempted during batch inference")

        with patch("socket.getaddrinfo", forbidden), patch(
            "socket.create_connection", forbidden
        ), patch("urllib.request.urlopen", forbidden), patch(
            "requests.sessions.Session.request", forbidden
        ):
            result = predict_lines(text, self.bundle, self.name)
        self.assertEqual(result.line.tolist(), [1, 3, 4, 5])
        self.assertEqual(result.status.tolist(), ["ok", "invalid", "ok", "ok"])
        self.assertIsNone(result.loc[1, "prediction"])
        self.assertTrue(pd.isna(result.loc[1, "phishing_probability"]))
        valid = result[result.status == "ok"]
        expected = self.bundle["model"].predict_proba(
            feature_frame(valid.url)[self.bundle["features"]]
        )[:, 1]
        np.testing.assert_allclose(valid.phishing_probability.astype(float), expected)
        self.assertEqual(
            valid.prediction.tolist(),
            [
                "Phishing" if p >= self.bundle["threshold"] else "Benign"
                for p in expected
            ],
        )
        self.assertEqual(
            result.loc[0, "phishing_probability"], result.loc[3, "phishing_probability"]
        )

    def test_empty_limits_and_csv_roundtrip(self):
        self.assertTrue(predict_lines("\n  \n", self.bundle, self.name).empty)
        result = predict_lines(
            "javascript:alert(1)\nhttps://example.org/a?x=1&y=2", self.bundle, self.name
        )
        result.loc[0, "url"] = '=HYPERLINK("https://example.org")'
        exported = csv_bytes(result)
        self.assertTrue(exported.startswith(b"\xef\xbb\xbf"))
        loaded = pd.read_csv(io.BytesIO(exported), keep_default_na=False)
        self.assertEqual(loaded.loc[0, "url"], '\'=HYPERLINK("https://example.org")')
        self.assertEqual(loaded.loc[0, "phishing_probability"], "")
        self.assertEqual(loaded.loc[1, "url"], result.loc[1, "url"])
        with self.assertRaisesRegex(ValueError, "At most"):
            predict_lines(
                "https://example.com\n" * (MAX_BATCH_URLS + 1), self.bundle, self.name
            )
        with self.assertRaisesRegex(ValueError, "characters"):
            predict_lines("x" * 2_000_001, self.bundle, self.name)

    def test_cli_writes_every_row(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = (
                Path(directory) / "urls.txt",
                Path(directory) / "results.csv",
            )
            source.write_text(
                "https://example.com\njavascript:alert(1)", encoding="utf-8-sig"
            )
            main(
                ["--input", str(source), "--output", str(target), "--model", self.name]
            )
            result = pd.read_csv(target)
            self.assertEqual(result.status.tolist(), ["ok", "invalid"])

    def test_every_available_model_including_feature_ablation(self):
        for path in sorted((ROOT / "models").glob("*.joblib")):
            with self.subTest(model=path.stem):
                bundle = joblib.load(path)
                result = predict_lines(
                    "https://example.org\nhttp://192.0.2.1/login", bundle, path.stem
                )
                expected = bundle["model"].predict_proba(
                    feature_frame(result.url)[bundle["features"]]
                )[:, 1]
                np.testing.assert_allclose(
                    result.phishing_probability.astype(float), expected
                )
                self.assertEqual(result.model.tolist(), [path.stem, path.stem])
        with patch.object(
            self.bundle["model"],
            "predict_proba",
            side_effect=AssertionError("Invalid URLs reached model"),
        ):
            result = predict_lines(
                "javascript:alert(1)\nhttp://[bad]/", self.bundle, self.name
            )
        self.assertEqual(result.status.tolist(), ["invalid", "invalid"])

    def test_streamlit_batch_and_retained_results(self):
        from streamlit.testing.v1 import AppTest

        def forbidden(*args, **kwargs):
            raise AssertionError("External request attempted")

        with patch("urllib.request.urlopen", forbidden), patch(
            "requests.sessions.Session.request", forbidden
        ):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            app.text_area[0].set_value(
                "https://example.com\njavascript:alert(1)\nhttps://example.com"
            )
            app.button[1].click().run()
            self.assertEqual(len(app.exception), 0)
            result = app.dataframe[0].value
            self.assertEqual(result.status.tolist(), ["ok", "invalid", "ok"])
            self.assertEqual(len(app.get("download_button")), 1)
            # A download/widget rerun must not discard the previous table.
            app.run()
            self.assertEqual(len(app.dataframe[0].value), 3)
            other = next(name for name in app.selectbox[0].options if name != self.name)
            app.selectbox[0].set_value(other).run()
            self.assertEqual(app.session_state["batch_model"], self.name)
            self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
