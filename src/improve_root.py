"""Train and evaluate root-URL augmentation."""

from collections import Counter
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from .features import canonical_url, feature_frame, parse_url
from .robustness import TRAIN_TRANSFORMS, TEST_TRANSFORMS, perturb
from .predict import predict_lines, csv_bytes
from .evaluate import metrics
from .train import rf
from .utils import ROOT, SEED, write_json, load_processed, sha256, environment

ROOT_TRANSFORMS = ("root_slash", "remove_www", "remove_www_root_slash")
DIAGNOSTIC_URLS = (
    "google.com",
    "https://google.com",
    "https://www.google.com",
    "https://www.google.com/",
)
BASELINE = "random_forest_augmented"
CANDIDATE = "random_forest_root_augmented"


def root_variant(url, kind):
    if kind not in ROOT_TRANSFORMS:
        raise ValueError("Unknown root transform: " + kind)
    canonical = canonical_url(url)
    _, p, host = parse_url(canonical)
    if kind in ("remove_www", "remove_www_root_slash") and host.startswith("www."):
        host = host[4:]
        authority = host
        if p.port is not None:
            authority += ":" + str(p.port)
        if p.username is not None:
            authority = p.netloc.rsplit("@", 1)[0] + "@" + authority
        p = p._replace(netloc=authority)
    if kind in ("root_slash", "remove_www_root_slash") and p.path == "":
        p = p._replace(path="/")
    return p.geturl()


def training_variants(train):
    """Each parent contributes original and distinct changed variants only."""
    records = []
    for row in train.itertuples():
        seen = {row.url}
        records.append((row.url, int(row.label), int(row.source_row), "original"))
        for kind in (*TRAIN_TRANSFORMS, *ROOT_TRANSFORMS):
            url = (
                perturb(row.url, kind)
                if kind in TRAIN_TRANSFORMS
                else root_variant(row.url, kind)
            )
            if url not in seen:
                records.append((url, int(row.label), int(row.source_row), kind))
                seen.add(url)
    frame = pd.DataFrame(records, columns=["url", "label", "source_row", "transform"])
    labels = frame.groupby("url").label.nunique()
    conflicts = set(labels[labels > 1].index)
    discard = frame.url.isin(conflicts) & frame["transform"].ne("original")
    discarded = int(discard.sum())
    frame = frame.loc[~discard]
    # A generated row may equal another parent's original. Always keep the
    # original, so deduplication cannot erase or relabel original parents.
    frame = pd.concat(
        [
            frame.loc[frame["transform"].eq("original")],
            frame.loc[frame["transform"].ne("original")],
        ],
        ignore_index=True,
    )
    before = len(frame)
    frame = frame.drop_duplicates(["url", "label"]).reset_index(drop=True)
    assert int(frame["transform"].eq("original").sum()) == len(train)
    return frame, {
        "conflicting_generated_rows_removed": discarded,
        "conflicting_urls": len(conflicts),
        "duplicate_rows_removed": before - len(frame),
    }


def main():
    data, _ = load_processed()
    train = data.loc[data.split == "train"].copy()
    training, audit = training_variants(train)
    # Generated records cannot introduce a validation/test parent.
    assert set(training.source_row) <= set(train.source_row)
    start = time.perf_counter()
    model = rf().fit(feature_frame(training.url), training.label)
    bundle = {
        "model": model,
        "features": list(feature_frame(["https://example.org"]).columns),
        "threshold": 0.5,
        "label_mapping": {0: "benign", 1: "phishing"},
        "training": [*TRAIN_TRANSFORMS, *ROOT_TRANSFORMS],
    }
    model_path = ROOT / "models" / f"{CANDIDATE}.joblib"
    joblib.dump(bundle, model_path)
    seconds = time.perf_counter() - start
    bundles = {
        BASELINE: joblib.load(ROOT / "models" / f"{BASELINE}.joblib"),
        CANDIDATE: bundle,
    }
    rows = []
    for split in ["validation", "test"]:
        subset = data.loc[data.split == split].reset_index(drop=True)
        y = subset.label.to_numpy()
        original = {}
        for condition in ["original", *TEST_TRANSFORMS, *ROOT_TRANSFORMS]:
            if condition == "original":
                urls = subset.url.tolist()
            elif condition in TEST_TRANSFORMS:
                urls = [perturb(u, condition) for u in subset.url]
            else:
                urls = [root_variant(u, condition) for u in subset.url]
            features = feature_frame(urls)
            changed = np.asarray(urls) != subset.url.to_numpy()
            for name, current in bundles.items():
                p = current["model"].predict_proba(features[current["features"]])[:, 1]
                if condition == "original":
                    original[name] = p
                flips = (p >= 0.5) != (original[name] >= 0.5)
                row = {
                    "model": name,
                    "split": split,
                    "condition": condition,
                    **metrics(y, p),
                    "changed_rows": int(changed.sum()),
                    "coverage": float(changed.mean()),
                    "flip_rate": float(flips.mean()),
                }
                if changed.any():
                    row["changed_only_flip_rate"] = float(flips[changed].mean())
                    changed_metrics = metrics(y[changed], p[changed])
                    row.update(
                        {
                            f"changed_only_{k}": changed_metrics[k]
                            for k in [
                                "f1",
                                "recall",
                                "precision",
                                "roc_auc",
                                "fp",
                                "fn",
                            ]
                        }
                    )
                rows.append(row)
            print("Evaluated", split, condition, flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(ROOT / "results/metrics/root_improvement.csv", index=False)
    diagnosis = pd.concat(
        [
            predict_lines("\n".join(DIAGNOSTIC_URLS), b, name)
            for name, b in bundles.items()
        ],
        ignore_index=True,
    )
    (ROOT / "results/metrics/root_diagnostic_predictions.csv").write_bytes(
        csv_bytes(diagnosis)
    )
    protocol = {
        "seed": SEED,
        "threshold": 0.5,
        "hyperparameters": model.get_params(),
        "baseline": BASELINE,
        "candidate": CANDIDATE,
        "train_original_rows": len(train),
        "training_rows": len(training),
        "training_label_counts": training.label.value_counts().to_dict(),
        "training_transform_counts": dict(Counter(training["transform"])),
        "generation_audit": audit,
        "parent_sources_train_only": True,
        "fit_seconds": seconds,
        "artifact_sha256": sha256(model_path),
        "samples_sha256": sha256(ROOT / "data/processed/samples.csv"),
        "environment": environment(),
    }
    write_json(ROOT / "results/metrics/root_improvement_protocol.json", protocol)
    clean = results[(results.split == "test") & (results.condition == "original")]
    print(
        clean[["model", "f1", "recall", "fp", "fn"]].to_string(index=False), flush=True
    )
    print(
        diagnosis[["url", "model", "prediction", "phishing_probability"]].to_string(
            index=False
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
