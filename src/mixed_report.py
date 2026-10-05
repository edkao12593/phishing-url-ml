"""Plot the recorded mixed-source experiment results."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ROOT

NAMES = {
    "random_forest": "RF",
    "random_forest_augmented": "RF augmented",
    "random_forest_root_augmented": "RF root augmented",
    "random_forest_mixed": "RF mixed",
}


def main():
    folder = ROOT / "results/mixed"
    metrics = pd.read_csv(folder / "model_comparison.csv")
    tests = metrics.loc[metrics.split.eq("test")].copy()
    tests["Model"] = tests.model.map(NAMES)
    for col in ["precision", "recall", "f1", "roc_auc", "fp", "fn", "fpr"]:
        tests[col.upper()] = tests[col]
    orig = tests.loc[tests.dataset.eq("original")]
    other = tests.loc[tests.dataset.eq("hannousse_development")]
    stress = pd.read_csv(folder / "robustness.csv")
    figures = folder / "figures"
    figures.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, data, title in zip(
        axes,
        [orig, other],
        ["Original test (n=6,001)", "Hannousse retrospective test (n=2,116)"],
    ):
        positions = np.arange(len(data))
        for i, (metric, label) in enumerate(
            [
                ("f1", "F1"),
                ("recall", "Phishing recall"),
                ("fpr", "Benign false positive rate"),
            ]
        ):
            ax.bar(positions + (i - 1) * 0.24, data[metric], width=0.24, label=label)
        ax.set_xticks(positions, data.Model, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Score / rate (threshold = 0.5)")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Mixed-source Random Forest: test performance")
    fig.tight_layout()
    fig.savefig(figures / "test_comparison.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, dataset, title in zip(
        axes,
        ["original", "hannousse_development"],
        ["Original test", "Hannousse retrospective test"],
    ):
        rows = stress.loc[stress.dataset.eq(dataset)]
        conditions = rows.condition.drop_duplicates().tolist()
        for model, label in [
            ("random_forest_augmented", "RF augmented"),
            ("random_forest_mixed", "RF mixed"),
        ]:
            scores = (
                rows.loc[rows.model.eq(model)]
                .set_index("condition")
                .loc[conditions, "f1"]
            )
            ax.plot(conditions, scores, "o-", label=label)
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Phishing F1")
    fig.suptitle("URL transformations: F1 comparison")
    fig.tight_layout()
    fig.savefig(figures / "robustness_tradeoffs.png", dpi=170)
    plt.close(fig)

    importance = pd.read_csv(folder / "feature_importance.csv").head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(importance.iloc[:, 0], importance.iloc[:, 1], color="#346f9f")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Mixed-source RF feature importance")
    fig.tight_layout()
    fig.savefig(figures / "feature_importance.png", dpi=170)
    plt.close(fig)

    print("Generated 3 mixed-source figures from executed results; no training.")


if __name__ == "__main__":
    main()
