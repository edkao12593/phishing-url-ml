"""Evaluate classifiers, URL transformations and feature importance."""

import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.inspection import permutation_importance
from .features import feature_frame
from .robustness import TEST_TRANSFORMS, TRAIN_TRANSFORMS, HELD_OUT_TRANSFORMS, perturb
from .utils import write_csv_gzip, ROOT, SEED, write_json, load_processed, sha256

BASE_NAMES = ["logistic_regression", "random_forest", "hist_gradient_boosting"]
MODEL_NAMES = [*BASE_NAMES, "random_forest_augmented", "random_forest_no_https"]


def metrics(y, prob):
    pred = np.asarray(prob) >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob) if len(np.unique(y)) == 2 else None,
        "average_precision": average_precision_score(y, prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": float(fp / (tn + fp)) if tn + fp else 0.0,
        "rows": len(y),
    }


def paired_group_f1_ci(y, a, b, groups, repeats=1000):
    """Cluster bootstrap of augmentation-minus-original F1, grouped by domain."""
    g, _ = pd.factorize(groups)
    n = g.max() + 1

    def counts(pred):
        return np.stack(
            [
                np.bincount(g, weights=(pred == 1) & (y == 1), minlength=n),
                np.bincount(g, weights=(pred == 1) & (y == 0), minlength=n),
                np.bincount(g, weights=(pred == 0) & (y == 1), minlength=n),
            ],
            axis=1,
        )

    ca, cb = counts(a >= 0.5), counts(b >= 0.5)
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(repeats):
        idx = rng.integers(0, n, n)
        aa, bb = ca[idx].sum(axis=0), cb[idx].sum(axis=0)

        def f(c):
            return 2 * c[0] / max(2 * c[0] + c[1] + c[2], 1)

        diffs.append(f(bb) - f(aa))
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def savefig(name):
    plt.savefig(ROOT / "results/figures" / name, dpi=170, bbox_inches="tight")
    plt.close()


def plot_results(xt, y, probs, imp, bias):
    """Plot classification metrics, feature importance and URL distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for n in BASE_NAMES:
        p = probs[(n, "original")]
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{n} ({roc_auc_score(y,p):.3f})")
        pre, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(rec, pre, label=f"{n} ({average_precision_score(y,p):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set(
        xlabel="False positive rate",
        ylabel="Recall",
        title="Domain-disjoint test ROC (AUC)",
    )
    axes[1].axhline(y.mean(), ls="--", color="gray", alpha=0.5)
    axes[1].set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-recall (average precision)",
    )
    for a in axes:
        a.legend(fontsize=7)
        a.grid(alpha=0.2)
    savefig("roc_pr_curves.png")
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for a, n in zip(axes.flat, MODEL_NAMES):
        cm = confusion_matrix(y, probs[(n, "original")] >= 0.5, labels=[0, 1])
        a.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            a.text(
                j,
                i,
                str(v),
                ha="center",
                va="center",
                color="white" if v > cm.max() / 2 else "black",
            )
        a.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Benign", "Phishing"],
            yticklabels=["Benign", "Phishing"],
            title=n,
            xlabel="Predicted",
            ylabel="Actual",
        )
    axes.flat[-1].axis("off")
    plt.tight_layout()
    savefig("confusion_matrices.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    top = imp.head(15).iloc[::-1]
    ax.barh(top.feature, top.importance, color="#247b9b")
    ax.set(xlabel="Mean decrease in impurity", title="Random Forest feature importance")
    savefig("feature_importance.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    conditions = ["original", *TEST_TRANSFORMS]
    for n in [*BASE_NAMES, "random_forest_augmented"]:
        for ax, key in zip(axes, ["f1", "recall"]):
            vals = [metrics(y, probs[(n, c)])[key] for c in conditions]
            ax.plot(range(len(conditions)), vals, marker="o", label=n)
            ax.set_xticks(range(len(conditions)), conditions, rotation=40, ha="right")
            ax.set(ylabel=key.upper(), title=f"Offline text stress test: {key.upper()}")
            ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    savefig("robustness_comparison.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for l, color in [(0, "#247b9b"), (1, "#c86648")]:
        v = xt.loc[y == l, "url_length"]
        axes[0].hist(
            v.clip(upper=200),
            bins=35,
            density=True,
            alpha=0.55,
            label=["Benign", "Phishing"][l],
            color=color,
        )
    axes[0].set(
        xlabel="URL length (values >200 capped for plot)",
        ylabel="Density",
        title="Dataset lexical distribution",
    )
    bb = pd.DataFrame(bias)
    bb = bb[bb.split == "test"]
    axes[1].bar(["Benign", "Phishing"], bb.https_rate, color=["#247b9b", "#c86648"])
    axes[1].set(ylim=(0, 1), ylabel="HTTPS fraction", title="HTTPS shortcut diagnostic")
    axes[0].legend()
    savefig("dataset_bias.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bootstrap", action="store_true")
    args = ap.parse_args()
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    d, X = load_processed()
    training = json.loads(
        (ROOT / "results/metrics/training.json").read_text(encoding="utf-8")
    )
    for fname, relative in [
        ("samples", "samples.csv"),
        ("features", "features.csv.gz"),
    ]:
        if sha256(ROOT / "data/processed" / relative) != training[f"{fname}_sha256"]:
            raise ValueError(
                "Models were trained on a different processed dataset; rerun train."
            )
    for name in MODEL_NAMES:
        if (
            sha256(ROOT / "models" / f"{name}.joblib")
            != training["models"][name]["artifact_sha256"]
        ):
            raise ValueError("Model artifact checksum mismatch; rerun train.")
    bundles = {n: joblib.load(ROOT / "models" / f"{n}.joblib") for n in MODEL_NAMES}
    clean, probs = [], {}
    for split in ["validation", "test"]:
        mask = d.split.eq(split)
        y = d.loc[mask, "label"].to_numpy()
        for name, b in bundles.items():
            p = b["model"].predict_proba(X.loc[mask, b["features"]])[:, 1]
            clean.append({"model": name, "split": split, **metrics(y, p)})
            if split == "test":
                probs[(name, "original")] = p
    clean = pd.DataFrame(clean)
    clean.to_csv(ROOT / "results/metrics/model_comparison.csv", index=False)
    candidates = clean[(clean.split == "validation") & clean.model.isin(BASE_NAMES)]
    selected = candidates.sort_values("f1", ascending=False).iloc[0]["model"]
    write_json(
        ROOT / "results/metrics/model_selection.json",
        {
            "best_base_model_by_validation_f1": selected,
            "threshold": 0.5,
            "demo_default": "random_forest_augmented",
            "demo_reason": "shows the prespecified augmentation intervention",
            "selection_split": "validation",
            "selection_metric": "f1",
        },
    )
    test = d.loc[d.split.eq("test")].reset_index(drop=True)
    xt = X.loc[d.split.eq("test")].reset_index(drop=True)
    y = test.label.to_numpy()
    rows = []
    predictions = []
    for kind in TEST_TRANSFORMS:
        urls = [perturb(u, kind) for u in test.url]
        changed = np.array(urls) != test.url.to_numpy()
        xp = feature_frame(urls)
        for name, b in bundles.items():
            original = probs[(name, "original")]
            p = b["model"].predict_proba(xp[b["features"]])[:, 1]
            probs[(name, kind)] = p
            m = metrics(y, p)
            before = metrics(y, original)
            flipped = (p >= 0.5) != (original >= 0.5)
            pos = y == 1
            detected = pos & (original >= 0.5)
            row = {
                "model": name,
                "perturbation": kind,
                "seen_in_augmentation": kind in TRAIN_TRANSFORMS,
                **m,
                "changed_rows": int(changed.sum()),
                "coverage": float(changed.mean()),
                "flip_rate": float(flipped.mean()),
                "changed_only_flip_rate": (
                    float(flipped[changed].mean()) if changed.any() else None
                ),
                "phishing_to_benign_rate": float(
                    ((original >= 0.5) & (p < 0.5))[pos].mean()
                ),
                "detected_phishing_flip_rate": (
                    float((p[detected] < 0.5).mean()) if detected.any() else None
                ),
                "benign_to_phishing_rate": float(
                    ((original < 0.5) & (p >= 0.5))[~pos].mean()
                ),
                **{f"{k}_before": before[k] for k in ["f1", "recall", "roc_auc"]},
                **{f"{k}_delta": m[k] - before[k] for k in ["f1", "recall", "roc_auc"]},
            }
            if changed.any():
                cm = metrics(y[changed], p[changed])
                cb = metrics(y[changed], original[changed])
                row.update(
                    {f"changed_only_{k}": cm[k] for k in ["f1", "recall", "roc_auc"]}
                )
                row.update(
                    {
                        f"changed_only_{k}_before": cb[k]
                        for k in ["f1", "recall", "roc_auc"]
                    }
                )
            rows.append(row)
        print("Evaluated", kind, "coverage", changed.mean(), flush=True)
    rob = pd.DataFrame(rows)
    rob.to_csv(ROOT / "results/metrics/robustness.csv", index=False)
    for (name, condition), p in probs.items():
        predictions.append(
            pd.DataFrame(
                {
                    "source_row": test.source_row,
                    "label": y,
                    "model": name,
                    "condition": condition,
                    "probability": p,
                    "prediction": (p >= 0.5).astype(int),
                }
            )
        )
    write_csv_gzip(
        pd.concat(predictions, ignore_index=True),
        ROOT / "results/metrics/test_predictions.csv.gz",
    )
    improvement = []
    for kind in ["original", *TEST_TRANSFORMS]:
        pa, pb = (
            probs[("random_forest", kind)],
            probs[("random_forest_augmented", kind)],
        )
        lo, hi = (
            paired_group_f1_ci(y, pa, pb, test.group)
            if not args.skip_bootstrap
            else (None, None)
        )
        improvement.append(
            {
                "condition": kind,
                "held_out_transform": kind in HELD_OUT_TRANSFORMS,
                "f1_delta": f1_score(y, pb >= 0.5) - f1_score(y, pa >= 0.5),
                "recall_delta": recall_score(y, pb >= 0.5) - recall_score(y, pa >= 0.5),
                "f1_delta_ci_low": lo,
                "f1_delta_ci_high": hi,
            }
        )
    pd.DataFrame(improvement).to_csv(
        ROOT / "results/metrics/augmentation_comparison.csv", index=False
    )
    write_json(
        ROOT / "results/metrics/robustness_protocol.json",
        {
            "train_transforms": list(TRAIN_TRANSFORMS),
            "held_out_transforms": list(HELD_OUT_TRANSFORMS),
            "label_policy": "reuse_original",
            "threshold": 0.5,
            "paired_bootstrap": {
                "unit": "registrable domain",
                "repeats": 1000,
                "seed": SEED,
                "quantity": "augmented RF F1 minus original RF F1",
                "interval": "percentile",
                "confidence_level": 0.95,
            },
            "include_unchanged_urls": True,
            "include_changed_only_metrics": True,
        },
    )
    # Explain using validation only. Impurity importance is biased by cardinality/correlation.
    rf = bundles["random_forest"]["model"]
    imp = pd.DataFrame(
        {"feature": X.columns, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)
    imp.to_csv(ROOT / "results/metrics/rf_feature_importance.csv", index=False)
    val = d.split.eq("validation")
    vx = X.loc[val].sample(n=min(1500, val.sum()), random_state=SEED)
    perm = permutation_importance(
        rf,
        vx,
        d.loc[vx.index, "label"],
        scoring="f1",
        n_repeats=3,
        random_state=SEED,
        n_jobs=2,
    )
    pd.DataFrame(
        {
            "feature": X.columns,
            "f1_drop_mean": perm.importances_mean,
            "f1_drop_std": perm.importances_std,
        }
    ).sort_values("f1_drop_mean", ascending=False).to_csv(
        ROOT / "results/metrics/rf_permutation_importance.csv", index=False
    )
    lr = bundles["logistic_regression"]["model"].named_steps["logisticregression"]
    pd.DataFrame(
        {"feature": X.columns, "standardized_coefficient": lr.coef_[0]}
    ).sort_values("standardized_coefficient", key=abs, ascending=False).to_csv(
        ROOT / "results/metrics/lr_coefficients.csv", index=False
    )
    bias = []
    for label in [0, 1]:
        for split in ["train", "validation", "test"]:
            f = X.loc[(d.label == label) & (d.split == split)]
            bias.append(
                {
                    "label": label,
                    "split": split,
                    "rows": len(f),
                    "https_rate": f.https.mean(),
                    "root_path_rate": (f.path_depth == 0).mean(),
                    "median_url_length": f.url_length.median(),
                    "median_path_length": f.path_length.median(),
                    "query_present_rate": (f.query_length > 0).mean(),
                }
            )
    pd.DataFrame(bias).to_csv(ROOT / "results/metrics/dataset_bias.csv", index=False)
    plot_results(xt, y, probs, imp, bias)
    print(clean[clean.split == "test"].to_string(index=False), flush=True)
    print("Best base model by validation F1:", selected, flush=True)


if __name__ == "__main__":
    main()
