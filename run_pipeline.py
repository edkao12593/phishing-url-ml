"""Run the ordered experiment from the repository root."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-download", action="store_true", help="Use verified existing CSV"
    )
    parser.add_argument("--sample-size", type=int, default=30000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        }
    )
    commands = [] if args.skip_download else [["src.download_data"]]
    commands += [
        ["src.preprocess", "--sample-size", str(args.sample_size)],
        ["src.train"],
        ["src.evaluate"],
    ]
    for command in commands:
        print("Running:", *command, flush=True)
        subprocess.run([sys.executable, "-m", *command], cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
