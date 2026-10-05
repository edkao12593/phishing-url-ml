"""Predict labels for newline-separated URL inputs."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .features import extract_features
from .utils import ROOT

MAX_BATCH_URLS = 1000
MAX_BATCH_CHARACTERS = 2_000_000
RESULT_COLUMNS = [
    "line",
    "url",
    "model",
    "prediction",
    "phishing_probability",
    "threshold",
    "status",
    "error",
    "url_length",
    "hostname_length",
    "path_depth",
    "https",
]


def predict_lines(text, bundle, model_name):
    """Preserve order, duplicates and physical line numbers; skip blank lines.

    Invalid rows are returned alongside successful rows without inference.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be multiline text")
    if len(text) > MAX_BATCH_CHARACTERS:
        raise ValueError(
            f"Batch input must contain <= {MAX_BATCH_CHARACTERS:,} characters"
        )
    lines = [
        (number, value.strip())
        for number, value in enumerate(text.splitlines(), start=1)
        if value.strip()
    ]
    if len(lines) > MAX_BATCH_URLS:
        raise ValueError(f"At most {MAX_BATCH_URLS:,} nonempty URL lines per batch")
    rows, features, positions = [], [], []
    threshold = float(bundle["threshold"])
    for number, url in lines:
        row = dict.fromkeys(RESULT_COLUMNS)
        row.update(
            line=number,
            url=url,
            model=model_name,
            threshold=threshold,
            status="invalid",
            error="",
        )
        try:
            f = extract_features(url)
        except (ValueError, UnicodeError) as error:
            row["error"] = str(error)
        else:
            row.update(
                status="ok",
                **{
                    k: f[k]
                    for k in ["url_length", "hostname_length", "path_depth", "https"]
                },
            )
            positions.append(len(rows))
            features.append(f)
        rows.append(row)
    if features:
        frame = pd.DataFrame(features)[bundle["features"]]
        probabilities = bundle["model"].predict_proba(frame)[:, 1]
        for position, probability in zip(positions, probabilities):
            probability = float(probability)
            rows[position].update(
                prediction="Phishing" if probability >= threshold else "Benign",
                phishing_probability=probability,
            )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def csv_bytes(results):
    """UTF-8 BOM CSV for Excel; neutralize formula-like user-controlled cells."""
    safe = results.copy()

    def escape(value):
        if isinstance(value, str) and value.startswith(
            ("=", "+", "-", "@", "\t", "\r", "\n")
        ):
            return "'" + value
        return value

    for column in safe.columns:
        if safe[column].dtype == object:
            safe[column] = safe[column].map(escape)
    return safe.to_csv(index=False, float_format="%.10g").encode("utf-8-sig")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    available = sorted(p.stem for p in (ROOT / "models").glob("*.joblib"))
    parser.add_argument(
        "--input", type=Path, required=True, help="UTF-8 text file: one URL per line"
    )
    parser.add_argument("--output", type=Path, default=Path("batch_predictions.csv"))
    parser.add_argument("--model", choices=available, default="random_forest_augmented")
    args = parser.parse_args(argv)
    try:
        # Limit the read before loading models or processing untrusted input.
        with args.input.open(encoding="utf-8-sig") as source:
            text = source.read(MAX_BATCH_CHARACTERS + 1)
        bundle = joblib.load(ROOT / "models" / f"{args.model}.joblib")
        results = predict_lines(text, bundle, args.model)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(csv_bytes(results))
    except (OSError, ValueError, UnicodeError) as error:
        parser.error(str(error))
    ok = int((results.status == "ok").sum())
    flagged = int((results.prediction == "Phishing").sum())
    print(
        f"Rows: {len(results)}; valid: {ok}; invalid: {len(results)-ok}; "
        f"predicted Phishing: {flagged}. Saved: {args.output}"
    )
    print("Prediction only: ground-truth labels were not supplied.")


if __name__ == "__main__":
    main()
