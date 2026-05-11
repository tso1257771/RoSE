"""Regenerate the benchmark test-set index files.

These index files (committed under
``application/seisbench-rose-benchmark/data/``) pin exactly which traces make
up each test set, so the benchmark composition is reproducible without the
multi-GB waveform datasets. This script (re)builds them:

  * ``rose_test_index.csv``  — the ``split == "test"`` rows of the repo-root
    ``rose_split_index.csv`` (produced by ``training/build_rose_split_index.py``).
    Derived purely from files in this repo — always regenerated.
  * ``stead_test_index.csv`` / ``stead_noise_index.csv`` — byte copies of
    ``metadata.csv`` / ``metadata_noise.csv`` from the prepared STEAD test
    directory. Only refreshed when ``--stead-dir`` is given (the STEAD
    waveforms / metadata are external; see ``data/README.md``).

Usage:
    python benchmark/build_test_indices.py
    python benchmark/build_test_indices.py --stead-dir /path/to/STEAD/benchmark_stead
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "application" / "seisbench-rose-benchmark" / "data"
SPLIT_INDEX = REPO_ROOT / "rose_split_index.csv"
TEST_SPLIT_NAME = "test"  # SeisBench vocabulary (RED-PAN "test" -> SeisBench "test")


def build_rose_test_index(split_index: Path, out_csv: Path) -> int:
    """Filter ``split == "test"`` rows of ``split_index`` into ``out_csv``.

    Returns the number of test rows written. Streams row-by-row so it does not
    hold the ~220k-row index in memory.
    """
    if not split_index.exists():
        raise FileNotFoundError(
            f"{split_index} not found — run training/build_rose_split_index.py first."
        )
    n_total = n_test = 0
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with split_index.open(newline="") as fh_in, out_csv.open("w", newline="") as fh_out:
        reader = csv.DictReader(fh_in)
        if reader.fieldnames is None or "split" not in reader.fieldnames:
            raise ValueError(f"{split_index}: expected a 'split' column")
        writer = csv.DictWriter(fh_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            n_total += 1
            if row.get("split") == TEST_SPLIT_NAME:
                writer.writerow(row)
                n_test += 1
    print(f"  rose_test_index.csv: {n_test} test rows of {n_total} total -> {out_csv}")
    return n_test


def copy_stead_indices(stead_dir: Path, data_dir: Path) -> None:
    """Copy STEAD ``metadata.csv`` / ``metadata_noise.csv`` into ``data_dir``."""
    pairs = [
        (stead_dir / "metadata.csv", data_dir / "stead_test_index.csv"),
        (stead_dir / "metadata_noise.csv", data_dir / "stead_noise_index.csv"),
    ]
    for src, dst in pairs:
        if not src.exists():
            raise FileNotFoundError(f"{src} not found under --stead-dir {stead_dir}")
        shutil.copyfile(src, dst)
        n_rows = sum(1 for _ in dst.open()) - 1  # minus header
        print(f"  {dst.name}: {n_rows} rows  (copied from {src})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--split-index",
        type=Path,
        default=SPLIT_INDEX,
        help=f"RoSE split index CSV (default: {SPLIT_INDEX}).",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Output directory for the index files (default: {DATA_DIR}).",
    )
    ap.add_argument(
        "--stead-dir",
        type=Path,
        default=None,
        help="Prepared STEAD test directory containing metadata.csv + "
        "metadata_noise.csv. If omitted, the STEAD index files are left as-is.",
    )
    args = ap.parse_args()

    print("Building RoSE test index:")
    build_rose_test_index(args.split_index, args.data_dir / "rose_test_index.csv")

    if args.stead_dir is not None:
        print("Refreshing STEAD index files:")
        copy_stead_indices(args.stead_dir, args.data_dir)
    else:
        print(
            "STEAD index files left unchanged "
            "(pass --stead-dir to refresh stead_test_index.csv / stead_noise_index.csv)."
        )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"error: {exc}")
