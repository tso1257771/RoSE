#!/usr/bin/env python3
"""Compile the two released tables that go in the data archive.

The event table is the authoritative copy in this repository, which carries
the released magnitudes. The pick table is the merged analyst and RED-PAN
pick set, with the two magnitudes measured for this release joined on so that
a user working pick by pick has them without merging the event table first.

The uncertainty, the quality class and the station counts are not carried
across. They describe the event rather than the pick, and the event table has
them, one join away on ``event_index``.

    python tools/build_release_tables.py /path/to/data
    python tools/build_release_tables.py /path/to/data --check

``--check`` reports what would change and writes nothing. Each file is
written through a temporary file in the same directory, so an interrupted run
leaves the previous copy in place.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rose.convert import CATALOG_SEARCH_PATHS  # noqa: E402

#: The two magnitudes measured for this release, joined onto every pick so a
#: user working pick by pick has them without merging the event table first.
PICK_MAGNITUDES = ["Mw", "ML"]

#: Fields that stay in the event table only. The uncertainty, the quality
#: class and the station counts describe the event, not the pick.
EVENT_ONLY = ["Mw_sigma", "Mw_quality", "Mw_nstations", "Mw_fc_Hz",
              "ML_nstations", "ML_warning", "Mw_ROMPLUS", "ML_ROMPLUS"]


def load_catalog(path: Path | None) -> pd.DataFrame:
    """The released event table, from ``--catalog-csv`` or the repository."""
    for p in ([path] if path else CATALOG_SEARCH_PATHS):
        if p and Path(p).is_file():
            df = pd.read_csv(p, dtype={"event_index": str})
            if df.event_index.duplicated().any():
                raise ValueError(f"{p} repeats event_index")
            missing = [c for c in PICK_MAGNITUDES if c not in df.columns]
            if missing:
                raise ValueError(
                    f"{p} has no {missing} column. This is the catalog from "
                    "before the magnitudes were released. Use the copy in "
                    "data/ of this repository."
                )
            print(f"catalog: {p}")
            return df
    raise FileNotFoundError("no event catalog found; pass --catalog-csv")


def build_picks(picks_csv: Path, cat: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    picks = pd.read_csv(picks_csv, dtype={"event_index": str}, low_memory=False)
    before = list(picks.columns)

    # Drop any previously joined copy so a rerun cannot stack columns.
    picks = picks.drop(columns=[c for c in PICK_MAGNITUDES + EVENT_ONLY
                                if c in picks.columns], errors="ignore")
    if PICK_MAGNITUDES:
        picks = picks.merge(cat[["event_index"] + PICK_MAGNITUDES],
                            on="event_index", how="left")

    orphans = int((~picks.event_index.isin(cat.event_index)).sum())
    stats = {
        "rows": len(picks),
        "columns_before": len(before),
        "columns_after": len(picks.columns),
        "added": [c for c in picks.columns if c not in before],
        "removed": [c for c in before if c not in picks.columns],
        "orphan_picks": orphans,
    }
    return picks, stats


def write(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir", type=Path, help="where the two released tables are written")
    ap.add_argument("--catalog-csv", type=Path, default=None)
    ap.add_argument("--picks-csv", type=Path, default=None,
                    help="source pick table (default: Enhanced_ROMPLUS_picks.csv in outdir)")
    ap.add_argument("--check", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    cat = load_catalog(args.catalog_csv)
    picks_csv = args.picks_csv or args.outdir / "Enhanced_ROMPLUS_picks.csv"
    if not picks_csv.is_file():
        print(f"no pick table at {picks_csv}", file=sys.stderr)
        return 2

    print(f"events : {len(cat):,}  with Mw {int(cat.Mw.notna().sum()):,}  "
          f"with ML {int(cat.ML.notna().sum()):,}")
    picks, s = build_picks(picks_csv, cat)
    print(f"picks  : {s['rows']:,}  columns {s['columns_before']} -> {s['columns_after']}")
    print(f"         joined onto picks: {PICK_MAGNITUDES or 'nothing'}")
    print(f"         event table only : {EVENT_ONLY}")
    if s["added"]:
        print(f"         added  : {s['added']}")
    if s["removed"]:
        print(f"         removed: {s['removed']}")

    if s["orphan_picks"]:
        print(f"\n{s['orphan_picks']:,} picks reference an event that is not in the "
              "catalog, and would carry no magnitude", file=sys.stderr)
        return 1

    if args.check:
        print("\nnothing written (--check)")
        return 0

    args.outdir.mkdir(parents=True, exist_ok=True)
    write(cat, args.outdir / "Enhanced_ROMPLUS_catalog.csv")
    write(picks, picks_csv)
    print(f"\nwrote {args.outdir / 'Enhanced_ROMPLUS_catalog.csv'}")
    print(f"wrote {picks_csv}")
    print("\nRebuild SHA256SUMS before depositing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
