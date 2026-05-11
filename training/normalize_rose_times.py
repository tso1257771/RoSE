"""Normalize time strings in RoSE metadata{year}.csv files.

SeisBench's WaveformDataset parses time columns with the strict format
``%Y-%m-%d %H:%M:%S.%f`` and refuses rows missing the microsecond fragment.
A small fraction of RoSE rows (notably ``source_origin_time`` for some
events) lack ``.NNNNNN`` and break loading.

Fix: for every time-typed column, append ``.000000`` to any non-empty
string that has no fractional part. Idempotent — re-running is a no-op.

Time columns covered (best-effort, all that contain ``time`` in the name):
    trace_start_time
    trace_p_arrival_time, trace_s_arrival_time
    trace_p_theoretical_time, trace_s_theoretical_time
    trace_p_manual_time, trace_s_manual_time
    source_origin_time, source_origin_time_raw
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd


# Matches "YYYY-MM-DD HH:MM:SS" (with optional 'T' separator) without a
# fractional-second suffix and an optional trailing 'Z'.
_NEEDS_US = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?P<tz>Z?)$"
)


def fix_one(value: object) -> object:
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return value
    m = _NEEDS_US.match(s)
    if m is None:
        return value
    tz = m.group("tz") or ""
    base = s[: len(s) - len(tz)] if tz else s
    return f"{base}.000000{tz}"


def is_time_col(name: str) -> bool:
    return "time" in name.lower()


def process(csv_path: Path, *, dry_run: bool) -> dict[str, int]:
    df = pd.read_csv(csv_path, low_memory=False)
    fixes: dict[str, int] = {}
    for col in df.columns:
        if not is_time_col(col):
            continue
        before = df[col].copy()
        df[col] = df[col].apply(fix_one)
        n = int((before.astype(str) != df[col].astype(str)).sum())
        if n:
            fixes[col] = n
    if fixes and not dry_run:
        df.to_csv(csv_path, index=False)
    return fixes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rose-dir",
        default=os.environ.get("ROSE_DATA_DIR"),
    )
    ap.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(range(2014, 2025)),
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rose_dir = Path(args.rose_dir)
    grand_total = 0
    for year in args.years:
        csv = rose_dir / f"metadata{year}.csv"
        if not csv.exists():
            print(f"[skip] {csv} missing")
            continue
        fixes = process(csv, dry_run=args.dry_run)
        n = sum(fixes.values())
        grand_total += n
        if fixes:
            print(f"{year}: fixed {n} cells {fixes}")
        else:
            print(f"{year}: clean")
    verb = "would fix" if args.dry_run else "fixed"
    print(f"\nTOTAL: {verb} {grand_total} cells")


if __name__ == "__main__":
    main()
