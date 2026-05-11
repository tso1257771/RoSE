"""Build a deterministic train/dev/test split index for the RoSE dataset.

Reuses RED-PAN's hash split (`rose.splits.hash_split`, vendored from
`redpan.data.builder.splits`) with the exact salt RED-PAN training used for
ROMPLUS, so SeisBench-trained PhaseNet / EQTransformer models share the same
partition as any RED-PAN checkpoint trained on this corpus.

What it does (per year metadata{year}.csv):
  * Event traces (source_id present) → hash_split(source_id, salt=ROMPLUS-...)
    so all stations of one earthquake land in the same split (no leakage).
  * Noise / source_id-less rows → independent Bernoulli draw (safe because
    these traces have no shared source).
  * Maps RED-PAN's "val" → SeisBench's "dev".

Outputs:
  1. Adds/overwrites a `split` column in each metadata{year}.csv (in place;
     SeisBench reads this directly via WaveformDataset.train_dev_test()).
  2. Writes a standalone audit file rose_split_index.csv next to the metadata
     containing (trace_name, source_id, split) for every row, plus a JSON
     summary rose_split_index.json with per-year counts.

Re-running with the same salt + p_train + p_val is idempotent.
"""
from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the in-tree `rose` package importable when run as a plain script.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rose.splits import (  # noqa: E402
    assert_valid_proportions,
    bernoulli_3way,
    hash_split,
)


# RED-PAN's exact ROMPLUS split parameters.
SALT = "ROMPLUS-singleEQ-v1"
P_TRAIN = 0.70
P_VAL = 0.15
# test = 1 - train - val = 0.15
assert_valid_proportions(P_TRAIN, P_VAL)

# RED-PAN ("val") → SeisBench ("dev") naming.
SPLIT_RENAME = {"train": "train", "val": "dev", "test": "test"}


def assign_split(
    source_id: str | float | None,
    rng: np.random.Generator,
) -> str:
    """Return SeisBench split name for one metadata row."""
    if isinstance(source_id, str) and source_id.strip() and source_id != "nan":
        sp = hash_split(source_id, P_TRAIN, P_VAL, salt=SALT)
    else:
        sp = bernoulli_3way(rng, P_TRAIN, P_VAL)
    return SPLIT_RENAME[sp]


def process_year(
    csv_path: Path,
    rng: np.random.Generator,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    if "source_id" not in df.columns:
        raise KeyError(f"{csv_path}: missing 'source_id' column")

    df["split"] = df["source_id"].apply(lambda sid: assign_split(sid, rng))
    df.to_csv(csv_path, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rose-dir",
        default=os.environ.get("ROSE_DATA_DIR"),
        help="Directory containing metadata{year}.csv + waveforms{year}.hdf5",
    )
    ap.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(range(2014, 2025)),
        help="Years to process (default: 2014..2024 inclusive).",
    )
    ap.add_argument(
        "--noise-seed",
        type=int,
        default=42,
        help="Seed for Bernoulli draws on rows without source_id.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute split counts but do not modify metadata{year}.csv.",
    )
    args = ap.parse_args()

    rose_dir = Path(args.rose_dir)
    rng = np.random.default_rng(args.noise_seed)

    audit_rows: list[pd.DataFrame] = []
    summary: dict[str, dict[str, int]] = {}

    for year in args.years:
        csv_path = rose_dir / f"metadata{year}.csv"
        if not csv_path.exists():
            print(f"[skip] {csv_path} not found")
            continue

        if args.dry_run:
            df = pd.read_csv(csv_path, low_memory=False)
            df["split"] = df["source_id"].apply(
                lambda sid: assign_split(sid, rng)
            )
        else:
            df = process_year(csv_path, rng)

        counts = df["split"].value_counts().to_dict()
        summary[str(year)] = {k: int(v) for k, v in counts.items()}
        print(f"{year}: {summary[str(year)]} (total={len(df)})")

        audit_rows.append(
            df[["trace_name", "source_id", "split"]].assign(year=year)
        )

    if not audit_rows:
        print("No metadata files processed.")
        return

    audit = pd.concat(audit_rows, ignore_index=True)

    totals = audit["split"].value_counts().to_dict()
    summary["_total"] = {k: int(v) for k, v in totals.items()}
    print(f"\nTOTAL: {summary['_total']} (rows={len(audit)})")

    if not args.dry_run:
        audit_path = rose_dir / "rose_split_index.csv"
        json_path = rose_dir / "rose_split_index.json"
        audit.to_csv(audit_path, index=False)
        with json_path.open("w") as fh:
            json.dump(
                {
                    "salt": SALT,
                    "p_train": P_TRAIN,
                    "p_val": P_VAL,
                    "noise_seed": args.noise_seed,
                    "rename": SPLIT_RENAME,
                    "per_year": summary,
                },
                fh,
                indent=2,
            )
        print(f"\nWrote {audit_path}")
        print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
