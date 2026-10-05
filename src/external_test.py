"""Evaluate trained models on the Hannousse dataset."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from .features import canonical_url, domain_group, feature_frame, FEATURE_NAMES
from .predict import csv_bytes
from .utils import write_csv_gzip, ROOT, sha256, write_json, environment

SOURCE = {
    "dataset": "Web page phishing detection",
    "creators": ["Abdelhakim Hannousse", "Salima Yahiouche"],
    "dataset_page": "https://data.mendeley.com/datasets/c2gw7fy2j4/3",
    "dataset_doi": "10.17632/c2gw7fy2j4.3",
    "paper_doi": "10.1016/j.engappai.2021.104347",
    "license": "CC BY 4.0",
    "collection_period": "May 2020",
    "publisher_reported_rows": 11430,
    "source_labels": {"legitimate": 0, "phishing": 1},
}
COLUMNS = ["source_row", "url", "label", "canonical_url", "group", "status", "error"]


def clean_external(raw, reference):
    """Label validation -> URL validation -> conflict removal -> dedup -> overlap.

    URL strings used for features retain their original scheme/path/query/fragment.
    Canonical strings are used only for duplicate and overlap checks. All known
    train/validation/test domain groups are excluded, conservatively. Rejections
    are recorded separately from included samples.
    """
    missing = {"url", "status"} - set(raw.columns)
    if missing:
        raise ValueError(
            f"Missing columns {sorted(missing)}. Use CSV with url and status "
            "(legitimate/phishing), not a features-only CSV."
        )
    rows = []
    for number, (url, status) in enumerate(zip(raw["url"], raw["status"])):
        row = dict.fromkeys(COLUMNS)
        row.update(source_row=number, url=url, status="included", error="")
        key = status.strip().lower() if isinstance(status, str) else None
        if key not in SOURCE["source_labels"]:
            row.update(status="invalid_label", error="Expected legitimate or phishing")
        else:
            row["label"] = SOURCE["source_labels"][key]
            try:
                row["url"] = url.strip() if isinstance(url, str) else url
                row["canonical_url"] = canonical_url(row["url"])
                row["group"] = domain_group(row["canonical_url"])
            except (ValueError, TypeError, UnicodeError) as error:
                row.update(status="invalid_url", error=str(error))
        rows.append(row)
    records = pd.DataFrame(rows, columns=COLUMNS)
    valid = records.loc[records.status.eq("included")]
    conflicts = valid.groupby("canonical_url")["label"].nunique()
    conflict_keys = set(conflicts[conflicts > 1].index)
    records.loc[records.canonical_url.isin(conflict_keys), "status"] = (
        "conflicting_label"
    )
    valid = records.loc[records.status.eq("included")]
    duplicates = valid.index[valid.duplicated("canonical_url", keep="first")]
    records.loc[duplicates, "status"] = "duplicate_url"
    known_urls = set(reference["url"].map(canonical_url))
    known_groups = set(reference["group"])
    valid_mask = records.status.eq("included")
    url_overlap = valid_mask & records.canonical_url.isin(known_urls)
    domain_overlap = valid_mask & records.group.isin(known_groups)
    overlap_counts = {
        "known_url_rows": int(url_overlap.sum()),
        "known_domain_rows_including_url_overlap": int(domain_overlap.sum()),
    }
    records.loc[valid_mask & (url_overlap | domain_overlap), "status"] = "seen_domain"
    included = records.loc[records.status.eq("included")].copy().reset_index(drop=True)
    included["label"] = included["label"].astype(int)
    rejected = records.loc[~records.status.eq("included")].copy().reset_index(drop=True)
    audit = {
        "input_rows": len(raw),
        "included_rows": len(included),
        "rejected_rows": len(rejected),
        "status_counts": {
            str(k): int(v) for k, v in records.status.value_counts().items()
        },
        "conflicting_url_keys": len(conflict_keys),
        "overlap_before_exclusion": overlap_counts,
        "reference_rows": len(reference),
        "reference_groups": len(known_groups),
        "reference_splits": sorted(reference["split"].unique().tolist()),
        "remaining_domain_overlap": len(set(included.group) & known_groups),
        "label_counts": {
            "benign": int(included.label.eq(0).sum()),
            "phishing": int(included.label.eq(1).sum()),
        },
        "filter_order": [
            "invalid_label",
            "invalid_url",
            "conflicting_label",
            "duplicate_url",
            "seen_domain",
        ],
        "source_row_convention": "zero-based CSV data row (header excluded)",
        "feature_input": "original URL text stripped at ends; no scheme rewriting",
        "sampling": "none; evaluate all valid, unique, unseen-domain rows",
    }
    assert audit["remaining_domain_overlap"] == 0
    return included, rejected, audit


def summarize(y, probabilities, threshold):
    prediction = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "rows": len(y),
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)) if both else None,
        "average_precision": (
            float(average_precision_score(y, probabilities)) if both else None
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": float(fp / (tn + fp)) if tn + fp else None,
        "fnr": float(fn / (fn + tp)) if fn + tp else None,
    }


def load_reference():
    samples = ROOT / "data/processed/samples.csv"
    audit = json.loads(
        (ROOT / "results/metrics/data_audit.json").read_text(encoding="utf-8")
    )
    training = json.loads(
        (ROOT / "results/metrics/training.json").read_text(encoding="utf-8")
    )
    digest = sha256(samples)
    if digest != audit["samples_sha256"] or digest != training["samples_sha256"]:
        raise ValueError("Reference data does not match training/audit checksums.")
    reference = pd.read_csv(samples, usecols=["url", "group", "split"])
    if len(reference) != audit["sample_size"]:
        raise ValueError("Reference row count mismatch")
    if not reference["group"].equals(reference.url.map(domain_group)):
        raise ValueError(
            "Reference domain grouping does not match current feature code"
        )
    return reference, training, digest


def trusted_bundle(name, training):
    path = ROOT / "models" / f"{name}.joblib"
    if name in ["random_forest_root_augmented", "random_forest_mixed"]:
        protocol_name = (
            "root_improvement_protocol.json"
            if name == "random_forest_root_augmented"
            else "mixed_training_protocol.json"
        )
        protocol = json.loads(
            (ROOT / "results/metrics" / protocol_name).read_text(encoding="utf-8")
        )
        if protocol["samples_sha256"] != training["samples_sha256"]:
            raise ValueError("Candidate reference checksum mismatch")
        expected = protocol["artifact_sha256"]
    else:
        expected = training["models"][name]["artifact_sha256"]
    if sha256(path) != expected:
        raise ValueError(f"Model checksum mismatch: {name}")
    # Check model identity against its training record.
    bundle = joblib.load(path)
    if list(bundle["model"].classes_) != [0, 1]:
        raise ValueError(f"Unexpected model classes: {name}")
    if not set(bundle["features"]).issubset(FEATURE_NAMES):
        raise ValueError(f"Unexpected feature schema: {name}")
    threshold = float(bundle["threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f"Invalid threshold: {name}")
    return bundle, expected


def plot_confusions(comparison, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(comparison), figsize=(4.3 * len(comparison), 4), squeeze=False
    )
    for ax, row in zip(axes[0], comparison.itertuples(index=False)):
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > matrix.max() / 2 else "black",
            )
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Benign", "Phishing"],
            yticklabels=["Benign", "Phishing"],
            xlabel="Predicted",
            ylabel="Actual",
            title=row.model.replace("random_forest", "RF"),
        )
    fig.suptitle("External dataset: confusion matrices")
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Official CSV with url and status"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/external/frozen"
    )
    available = sorted(p.stem for p in (ROOT / "models").glob("*.joblib"))
    defaults = [
        n
        for n in [
            "random_forest",
            "random_forest_augmented",
            "random_forest_root_augmented",
        ]
        if n in available
    ]
    parser.add_argument("--models", nargs="+", choices=available, default=defaults)
    args = parser.parse_args(argv)
    try:
        if not args.models or len(args.models) != len(set(args.models)):
            raise ValueError("Select at least one model; no duplicate model names")
        if "random_forest_mixed" in args.models:
            raise ValueError(
                "Hannousse was used to develop random_forest_mixed. Use src.improve_data "
                "for its retrospective split evaluation; do not call this an unseen external benchmark."
            )
        # Prevent accidentally overwriting the input or original project assets.
        output = args.output_dir.resolve()
        if (
            output == ROOT
            or output.is_relative_to(ROOT / "data")
            or output.is_relative_to(ROOT / "models")
        ):
            raise ValueError(
                "Use a separate results directory, not data/models/project root"
            )
        if args.input.resolve().is_relative_to(output):
            raise ValueError("Keep the input CSV outside the output directory")
        raw = pd.read_csv(
            args.input, encoding="utf-8-sig", dtype=str, keep_default_na=False
        )
        reference, training, reference_hash = load_reference()
        included, rejected, audit = clean_external(raw, reference)
        audit.update(
            dataset=SOURCE,
            executed_utc=datetime.now(timezone.utc).isoformat(),
            input_filename=args.input.name,
            input_sha256=sha256(args.input),
            reference_sha256=reference_hash,
            environment=environment(),
            requested_models=args.models,
            frozen_models=True,
            retraining=False,
            models={},
            execution_status="in_progress",
            evaluation_scope="external, deduplicated, disjoint from all known domain groups",
        )
        output.mkdir(parents=True, exist_ok=True)
        write_csv_gzip(rejected, output / "excluded.csv.gz")
        write_json(output / "audit.json", audit)
        if not len(included) or included.label.nunique() != 2:
            raise ValueError(
                "Need both benign and phishing rows after filtering. See audit.json."
            )
        print(
            f"Input: {len(raw):,}; included: {len(included):,}; rejected: {len(rejected):,}",
            flush=True,
        )
        print("Evaluating existing models at their stored thresholds.", flush=True)
        x = feature_frame(included.url)
        y = included.label.to_numpy()
        comparisons, predictions = [], []
        for name in args.models:
            bundle, digest = trusted_bundle(name, training)
            probability = bundle["model"].predict_proba(x[bundle["features"]])[:, 1]
            if not np.isfinite(probability).all():
                raise ValueError(f"Non-finite predictions: {name}")
            threshold = float(bundle["threshold"])
            prediction = (probability >= threshold).astype(int)
            comparisons.append(
                {
                    "model": name,
                    "threshold": threshold,
                    **summarize(y, probability, threshold),
                }
            )
            rows = included[["source_row", "url", "label", "group"]].copy()
            rows["model"] = name
            rows["phishing_probability"] = probability
            rows["threshold"] = threshold
            rows["prediction"] = prediction
            rows["error_type"] = np.where(
                (y == 0) & (prediction == 1),
                "FP",
                np.where((y == 1) & (prediction == 0), "FN", "correct"),
            )
            predictions.append(rows)
            audit["models"][name] = {
                "sha256": digest,
                "threshold": threshold,
                "features": bundle["features"],
                "rows": len(y),
            }
        comparison = pd.DataFrame(comparisons)
        (output / "model_comparison.csv").write_bytes(csv_bytes(comparison))
        write_csv_gzip(
            pd.concat(predictions, ignore_index=True), output / "predictions.csv.gz"
        )
        bias = []
        for label in [0, 1]:
            f = x.loc[included.label.eq(label)]
            bias.append(
                {
                    "label": label,
                    "rows": len(f),
                    "https_rate": float(f.https.mean()),
                    "path_present_rate": float(f.path_length.gt(0).mean()),
                    "query_present_rate": float(f.query_length.gt(0).mean()),
                    "median_url_length": float(f.url_length.median()),
                }
            )
        pd.DataFrame(bias).to_csv(output / "dataset_bias.csv", index=False)
        plot_confusions(comparison, output / "confusion_matrices.png")
        audit["execution_status"] = "completed"
        audit["output_sha256"] = {
            p.name: sha256(p)
            for p in output.iterdir()
            if p.name
            in [
                "model_comparison.csv",
                "predictions.csv.gz",
                "excluded.csv.gz",
                "dataset_bias.csv",
                "confusion_matrices.png",
            ]
        }
        write_json(output / "audit.json", audit)
        print(
            comparison[
                ["model", "precision", "recall", "f1", "roc_auc", "fp", "fn"]
            ].to_string(index=False)
        )
        print(f"Saved: {output}")
    except (
        OSError,
        ValueError,
        KeyError,
        UnicodeError,
        pd.errors.ParserError,
    ) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
