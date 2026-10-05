"""Download and verify the PhiUSIIL dataset archive."""

import argparse
from pathlib import Path
import urllib.request
import zipfile
from .utils import ROOT, sha256, write_json

SOURCE = (
    "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
)
EXPECTED_ZIP_SHA256 = "0a639fd03aea6308c5b1c10c92aa23c2ce1505447a9137271865cd0badc9a59a"
CSV_NAME = "PhiUSIIL_Phishing_URL_Dataset.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive", type=Path, help="Use an already downloaded official ZIP"
    )
    args = parser.parse_args()
    dest = ROOT / "data/raw"
    dest.mkdir(parents=True, exist_ok=True)
    archive = args.archive or dest / "phiusiil.zip"
    if not args.archive and not archive.exists():
        with urllib.request.urlopen(SOURCE, timeout=60) as r, archive.open("wb") as f:
            while block := r.read(1048576):
                f.write(block)
    digest = sha256(archive)
    if digest != EXPECTED_ZIP_SHA256:
        raise ValueError(
            "Dataset archive changed: checksum mismatch. Review provenance before using."
        )
    with zipfile.ZipFile(archive) as z:
        # Read the expected CSV member.
        with z.open(CSV_NAME) as r, (dest / CSV_NAME).open("wb") as f:
            while block := r.read(1048576):
                f.write(block)
    write_json(
        ROOT / "data/source_metadata.json",
        {
            "source": SOURCE,
            "dataset": "PhiUSIIL Phishing URL (Website)",
            "creators": ["Arvind Prasad", "Shalini Chandra"],
            "year": 2024,
            "license": "CC BY 4.0",
            "dataset_page": "https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset",
            "citation_doi": "10.1016/j.cose.2023.103545",
            "retrieved_utc_date": "2026-10-03",
            "zip_sha256": digest,
            "csv_sha256": sha256(dest / CSV_NAME),
            "source_labels": {"0": "phishing", "1": "legitimate"},
            "project_labels": {"0": "benign", "1": "phishing"},
        },
    )
    print("Verified dataset saved:", dest / CSV_NAME)


if __name__ == "__main__":
    main()
