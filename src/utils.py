from pathlib import Path
import hashlib
import json
import platform
import importlib.metadata
import gzip
import io
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def write_csv_gzip(frame, path):
    """Write CSV without a gzip timestamp or source filename in the header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        with gzip.GzipFile(
            fileobj=output, mode="wb", filename="", mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, lineterminator="\n")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            n: importlib.metadata.version(n)
            for n in [
                "numpy",
                "pandas",
                "scikit-learn",
                "matplotlib",
                "joblib",
                "tldextract",
                "streamlit",
            ]
        },
    }


def load_processed():
    """Load aligned samples and features after checking the stored data hashes."""
    import pandas as pd
    from .features import FEATURE_NAMES

    audit = json.loads(
        (ROOT / "results/metrics/data_audit.json").read_text(encoding="utf-8")
    )
    samples = ROOT / "data/processed/samples.csv"
    features = ROOT / "data/processed/features.csv.gz"
    if (
        sha256(samples) != audit["samples_sha256"]
        or sha256(features) != audit["features_sha256"]
    ):
        raise ValueError(
            "Processed files do not match audit checksums. Rerun preprocessing."
        )
    d, x = pd.read_csv(samples), pd.read_csv(features)
    if (
        len(d) != len(x)
        or len(d) != audit["sample_size"]
        or list(x.columns) != FEATURE_NAMES
    ):
        raise ValueError(
            "Feature cache alignment/schema mismatch. Rerun preprocessing."
        )
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Non-finite features")
    return d, x
