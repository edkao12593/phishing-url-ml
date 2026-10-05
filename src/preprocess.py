"""Clean URLs, sample records and create domain-disjoint splits."""

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from .features import canonical_url, domain_group, feature_frame
from .download_data import CSV_NAME
from .utils import write_csv_gzip, ROOT, SEED, write_json, sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=30000)
    args = ap.parse_args()
    path = ROOT / "data/raw" / CSV_NAME
    provenance = json.loads(
        (ROOT / "data/source_metadata.json").read_text(encoding="utf-8")
    )
    if sha256(path) != provenance["csv_sha256"]:
        raise ValueError("Raw dataset checksum mismatch. Rerun download_data.")
    raw = pd.read_csv(path, usecols=["URL", "label"])
    if not set(raw["label"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Unexpected dataset labels")
    records, invalid = [], 0
    for i, row in raw.iterrows():
        try:
            if pd.isna(row["label"]):
                raise ValueError("missing label")
            u = canonical_url(row["URL"])
            records.append({"source_row": i, "url": u, "label": 1 - int(row["label"])})
        except (ValueError, TypeError, UnicodeError):
            invalid += 1
    d = pd.DataFrame(records)
    conflict_keys = d.groupby("url")["label"].nunique()
    conflict_keys = conflict_keys[conflict_keys > 1].index
    conflicts = int(d.url.isin(conflict_keys).sum())
    d = d[~d.url.isin(conflict_keys)]
    duplicate_count = int(d.duplicated("url").sum())
    d = d.drop_duplicates("url").reset_index(drop=True)
    cleaned_count = len(d)
    if not 1000 <= args.sample_size <= len(d):
        raise ValueError("sample-size must be between 1000 and cleaned data size")
    if args.sample_size < len(d):
        _, d = train_test_split(
            d, test_size=args.sample_size, stratify=d.label, random_state=SEED
        )
    d = d.sort_values("source_row").reset_index(drop=True)
    d["group"] = d.url.map(domain_group)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold = np.full(len(d), -1)
    for k, (_, ids) in enumerate(splitter.split(d.url, d.label, groups=d.group)):
        fold[ids] = k
    d["split"] = np.where(fold == 0, "test", np.where(fold == 1, "validation", "train"))
    group_sets = {
        s: set(d.loc[d.split == s, "group"]) for s in ["train", "validation", "test"]
    }
    assert all(
        not group_sets[a] & group_sets[b]
        for a, b in [("train", "test"), ("train", "validation"), ("validation", "test")]
    )
    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "samples.csv", index=False)
    X = feature_frame(d.url)
    write_csv_gzip(X, out / "features.csv.gz")
    stats = {
        "seed": SEED,
        "raw_count": len(raw),
        "invalid_rows": invalid,
        "conflicting_rows_removed": conflicts,
        "conflicting_url_keys": len(conflict_keys),
        "duplicate_rows_removed": duplicate_count,
        "cleaned_count": cleaned_count,
        "sample_size": len(d),
        "features": list(X.columns),
        "feature_count": len(X.columns),
        "sampling": "stratified without replacement before splitting",
        "split": "StratifiedGroupKFold(5, shuffle=True, seed=42): fold0 test, fold1 validation, rest train",
        "group": "registrable domain using bundled PSL including private suffixes; IP/unknown suffix: hostname",
        "domain_overlap": 0,
        "csv_sha256": sha256(path),
        "samples_sha256": sha256(out / "samples.csv"),
        "features_sha256": sha256(out / "features.csv.gz"),
        "splits": {
            s: {
                "rows": len(g),
                "phishing": int(g.label.sum()),
                "benign": int((g.label == 0).sum()),
                "groups": g.group.nunique(),
            }
            for s, g in d.groupby("split")
        },
    }
    write_json(ROOT / "results/metrics/data_audit.json", stats)
    print(
        {
            k: stats[k]
            for k in [
                "raw_count",
                "invalid_rows",
                "duplicate_rows_removed",
                "cleaned_count",
                "sample_size",
                "splits",
            ]
        }
    )


if __name__ == "__main__":
    main()
