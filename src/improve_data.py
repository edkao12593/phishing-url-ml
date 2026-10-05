"""Train and evaluate a Random Forest on two data sources."""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .external_test import clean_external, load_reference, trusted_bundle, summarize
from .features import feature_frame
from .predict import predict_lines, csv_bytes
from .robustness import TEST_TRANSFORMS, perturb
from .train import rf
from .utils import (
    write_csv_gzip,
    ROOT,
    SEED,
    load_processed,
    sha256,
    environment,
    write_json,
)

CANDIDATE = "random_forest_mixed"
BASELINES = ["random_forest", "random_forest_augmented", "random_forest_root_augmented"]


def external_split(frame):
    """Unchanged fixed 5-fold group rule: test fold0, validation fold1, rest train."""
    frame = frame.copy().reset_index(drop=True)
    if frame.group.nunique() < 5 or frame.label.nunique() != 2:
        raise ValueError("Need at least five domains and both labels")
    folds = np.full(len(frame), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (_, ids) in enumerate(
        splitter.split(frame.url, frame.label, frame.group)
    ):
        folds[ids] = fold
    frame["split"] = np.where(
        folds == 0, "test", np.where(folds == 1, "validation", "train")
    )
    groups = {
        s: set(frame.loc[frame.split.eq(s), "group"])
        for s in ["train", "validation", "test"]
    }
    assert all(
        not groups[a] & groups[b]
        for a, b in [("train", "test"), ("train", "validation"), ("test", "validation")]
    )
    if any(frame.loc[frame.split.eq(s), "label"].nunique() != 2 for s in groups):
        raise ValueError("Both classes must be represented in each split")
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = pd.read_csv(
        args.input, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    reference, training, reference_hash = load_reference()
    external, _, cleanup = clean_external(raw, reference)
    external = external_split(external)
    original, xo = load_processed()
    xe = feature_frame(external.url)
    out = ROOT / "results/mixed"
    out.mkdir(parents=True, exist_ok=True)
    write_csv_gzip(
        external[["source_row", "url", "label", "group", "split"]],
        out / "external_splits.csv.gz",
    )
    # Concatenate the training partitions from both sources.
    ot, et = original.split.eq("train"), external.split.eq("train")
    xtrain = pd.concat([xo.loc[ot], xe.loc[et]], ignore_index=True)
    ytrain = np.concatenate([original.loc[ot, "label"], external.loc[et, "label"]])
    train_groups = set(original.loc[ot, "group"]) | set(external.loc[et, "group"])
    for frame in [original, external]:
        assert not train_groups & set(frame.loc[frame.split.ne("train"), "group"])
    old_model_hashes = {
        p.name: sha256(p)
        for p in (ROOT / "models").glob("*.joblib")
        if p.stem != CANDIDATE
    }
    baselines = {
        n: trusted_bundle(n, training)[0]
        for n in BASELINES
        if (ROOT / "models" / f"{n}.joblib").is_file()
    }
    protocol = {
        "candidate": CANDIDATE,
        "seed": SEED,
        "threshold": 0.5,
        "hyperparameters": rf().get_params(),
        "input_sha256": sha256(args.input),
        "samples_sha256": reference_hash,
        "external_cleanup": cleanup,
        "external_split_rule": "StratifiedGroupKFold(5, seed=42): fold0 test, fold1 validation, rest train",
        "external_splits": {
            s: {
                "rows": len(g),
                "benign": int(g.label.eq(0).sum()),
                "phishing": int(g.label.eq(1).sum()),
                "groups": int(g.group.nunique()),
            }
            for s, g in external.groupby("split")
        },
        "original_train_rows": int(ot.sum()),
        "external_train_rows": int(et.sum()),
        "training_rows": len(xtrain),
        "training_phishing": int(ytrain.sum()),
        "training_benign": int((ytrain == 0).sum()),
        "domain_overlap_with_heldout": 0,
        "hyperparameter_search": False,
        "environment": environment(),
        "baseline_model_sha256": old_model_hashes,
    }
    start = time.perf_counter()
    model = rf().fit(xtrain, ytrain)
    bundle = {
        "model": model,
        "features": list(xtrain.columns),
        "threshold": 0.5,
        "label_mapping": {0: "benign", 1: "phishing"},
        "training": "PhiUSIIL original train + Hannousse domain-disjoint train",
    }
    artifact = ROOT / "models" / f"{CANDIDATE}.joblib"
    joblib.dump(bundle, artifact)
    protocol.update(
        fit_seconds=time.perf_counter() - start, artifact_sha256=sha256(artifact)
    )
    write_json(ROOT / "results/metrics/mixed_training_protocol.json", protocol)
    bundles = {**baselines, CANDIDATE: bundle}
    comparisons, prediction_rows, robustness = [], [], []
    for dataset, frame, x in [
        ("original", original, xo),
        ("hannousse_development", external, xe),
    ]:
        for split in ["validation", "test"]:
            mask = frame.split.eq(split)
            subset = frame.loc[mask].reset_index(drop=True)
            xf = x.loc[mask].reset_index(drop=True)
            y = subset.label.to_numpy()
            probabilities = {}
            for name, b in bundles.items():
                p = b["model"].predict_proba(xf[b["features"]])[:, 1]
                probabilities[name] = p
                comparisons.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "model": name,
                        "threshold": 0.5,
                        **summarize(y, p, 0.5),
                    }
                )
                rows = subset[["source_row", "url", "label", "group"]].copy()
                rows["dataset"], rows["split"], rows["model"] = dataset, split, name
                rows["phishing_probability"] = p
                rows["prediction"] = (p >= 0.5).astype(int)
                prediction_rows.append(rows)
            print("Evaluated", dataset, split, flush=True)
            if split != "test":
                continue
            # Evaluate the same transformations on both test partitions.
            for condition in TEST_TRANSFORMS:
                urls = [perturb(u, condition) for u in subset.url]
                xp = feature_frame(urls)
                changed = np.asarray(urls) != subset.url.to_numpy()
                for name in ["random_forest_augmented", CANDIDATE]:
                    b = bundles[name]
                    p = b["model"].predict_proba(xp[b["features"]])[:, 1]
                    before = summarize(y, probabilities[name], 0.5)
                    row = {
                        "dataset": dataset,
                        "model": name,
                        "condition": condition,
                        **summarize(y, p, 0.5),
                        "coverage": float(changed.mean()),
                        "flip_rate": float(
                            ((p >= 0.5) != (probabilities[name] >= 0.5)).mean()
                        ),
                        **{
                            k + "_before": before[k]
                            for k in ["f1", "recall", "roc_auc"]
                        },
                    }
                    if changed.any():
                        row.update(
                            {
                                "changed_only_" + k: v
                                for k, v in summarize(
                                    y[changed], p[changed], 0.5
                                ).items()
                            }
                        )
                    robustness.append(row)
                print("Stress", dataset, condition, flush=True)
    comparison = pd.DataFrame(comparisons)
    (out / "model_comparison.csv").write_bytes(csv_bytes(comparison))
    write_csv_gzip(
        pd.concat(prediction_rows, ignore_index=True), out / "predictions.csv.gz"
    )
    pd.DataFrame(robustness).to_csv(out / "robustness.csv", index=False)
    diagnosis = pd.concat(
        [
            predict_lines(
                "\n".join(
                    [
                        "google.com",
                        "https://google.com",
                        "https://www.google.com/",
                        "https://example.org/help?lang=en",
                    ]
                ),
                b,
                n,
            )
            for n, b in bundles.items()
        ],
        ignore_index=True,
    )
    (out / "diagnostic_predictions.csv").write_bytes(csv_bytes(diagnosis))
    pd.DataFrame(
        {"feature": xtrain.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False).to_csv(
        out / "feature_importance.csv", index=False
    )
    for name, value in old_model_hashes.items():
        assert sha256(ROOT / "models" / name) == value
    print(
        comparison.loc[
            comparison.split.eq("test"),
            ["dataset", "model", "precision", "recall", "f1", "roc_auc", "fp", "fn"],
        ].to_string(index=False)
    )
    print(
        diagnosis[["url", "model", "prediction", "phishing_probability"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
