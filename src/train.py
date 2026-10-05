import time
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from .features import feature_frame
from .robustness import TRAIN_TRANSFORMS, perturb
from .utils import ROOT, SEED, write_json, environment, load_processed, sha256


def rf():
    return RandomForestClassifier(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=1,
        random_state=SEED,
        class_weight="balanced_subsample",
    )


def main():
    d, X = load_processed()
    mask = d.split.eq("train")
    xt, yt = X.loc[mask].reset_index(drop=True), d.loc[mask, "label"].to_numpy()
    base = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=SEED
            ),
        ),
        "random_forest": rf(),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED
        ),
    }
    records = {}
    for name, model in base.items():
        start = time.perf_counter()
        model.fit(xt, yt)
        joblib.dump(
            {
                "model": model,
                "features": list(xt.columns),
                "threshold": 0.5,
                "label_mapping": {0: "benign", 1: "phishing"},
                "training": "original only",
            },
            ROOT / "models" / f"{name}.joblib",
        )
        records[name] = {"seconds": time.perf_counter() - start, "train_rows": len(xt)}
        print(name, records[name], flush=True)
    # Equal class/augmentation weighting: each original gets 1 original + 2 variants.
    aug = [
        feature_frame([perturb(u, t) for u in d.loc[mask, "url"]])
        for t in TRAIN_TRANSFORMS
    ]
    xa, ya = pd.concat([xt, *aug], ignore_index=True), np.tile(yt, 1 + len(aug))
    start = time.perf_counter()
    model = rf().fit(xa, ya)
    joblib.dump(
        {
            "model": model,
            "features": list(xa.columns),
            "threshold": 0.5,
            "label_mapping": {0: "benign", 1: "phishing"},
            "training": list(TRAIN_TRANSFORMS),
        },
        ROOT / "models/random_forest_augmented.joblib",
    )
    records["random_forest_augmented"] = {
        "seconds": time.perf_counter() - start,
        "train_rows": len(xa),
    }
    print("random_forest_augmented", records["random_forest_augmented"], flush=True)
    # One fixed ablation probes the potential HTTPS shortcut.
    columns = [c for c in xt.columns if c != "https"]
    start = time.perf_counter()
    model = rf().fit(xt[columns], yt)
    joblib.dump(
        {
            "model": model,
            "features": columns,
            "threshold": 0.5,
            "label_mapping": {0: "benign", 1: "phishing"},
            "training": "original, HTTPS feature removed",
        },
        ROOT / "models/random_forest_no_https.joblib",
    )
    records["random_forest_no_https"] = {
        "seconds": time.perf_counter() - start,
        "train_rows": len(xt),
    }
    for name in records:
        records[name]["artifact_sha256"] = sha256(ROOT / "models" / f"{name}.joblib")
    write_json(
        ROOT / "results/metrics/training.json",
        {
            "seed": SEED,
            "threshold": 0.5,
            "hyperparameters": "fixed before test evaluation; no tuning",
            "models": records,
            "environment": environment(),
            "samples_sha256": sha256(ROOT / "data/processed/samples.csv"),
            "features_sha256": sha256(ROOT / "data/processed/features.csv.gz"),
        },
    )


if __name__ == "__main__":
    main()
