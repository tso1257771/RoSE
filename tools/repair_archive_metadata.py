#!/usr/bin/env python3
"""Bring a built SeisBench archive's metadata up to the released schema.

The archive built before the magnitudes were released carries the NIEP
bulletin local magnitude in ``source_magnitude``, no ``split`` column, and
none of the released magnitude fields. Depositing it would give a permanent
DOI to the one magnitude the data descriptor argues against using.

Only the metadata CSVs are wrong. The waveforms are unchanged, so this
rewrites the eleven ``metadata{YEAR}.csv`` files and leaves the HDF5 files
alone, which is a few hundred megabytes of work rather than rebuilding 35 GB.

    python tools/repair_archive_metadata.py /path/to/data/rose
    python tools/repair_archive_metadata.py /path/to/data/rose --check

``--check`` reports what would change and writes nothing. Without it each
file is rewritten through a temporary file in the same directory, so an
interrupted run leaves the original in place.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rose.convert import CATALOG_SEARCH_PATHS  # noqa: E402
from rose.splits import hash_split  # noqa: E402

#: The salt the released split column was produced with. Changing it moves
#: events between splits and invalidates every published benchmark number.
SPLIT_SALT = "ROMPLUS-singleEQ-v1"

#: Catalog column -> metadata column, for the fields carried verbatim.
_VERBATIM = {
    "Mw": "source_mw",
    "Mw_sigma": "source_mw_sigma",
    "Mw_quality": "source_mw_quality",
    "Mw_nstations": "source_mw_nstations",
    "Mw_fc_Hz": "source_mw_fc_hz",
    "ML": "source_ml",
    "ML_nstations": "source_ml_nstations",
    "ML_warning": "source_ml_warning",
    "ML_ROMPLUS": "source_ml_romplus",
    "Mw_ROMPLUS": "source_mw_romplus",
}

#: Count fields use -1 for "not measured", so they stay integers.
_COUNTS = ("source_mw_nstations", "source_ml_nstations", "source_ml_warning")


def load_catalog(path: Path | None) -> pd.DataFrame:
    for p in ([path] if path else CATALOG_SEARCH_PATHS):
        if p and Path(p).is_file():
            df = pd.read_csv(p, dtype={"event_index": str})
            if df.event_index.duplicated().any():
                raise ValueError(f"{p} repeats event_index")
            return df.set_index("event_index")
    raise FileNotFoundError("no event catalog found; pass --catalog-csv")


def magnitude_frame(cat: pd.DataFrame) -> pd.DataFrame:
    """The released magnitude fields, one row per event.

    The preferred magnitude is Mw where it was measured and ML otherwise, and
    ``source_magnitude_type`` names the scale actually stored, so a reader
    never has to guess which scale a value is on. The bulletin magnitudes are
    kept under their own names and are never preferred.
    """
    out = pd.DataFrame(index=cat.index)
    for src, dst in _VERBATIM.items():
        out[dst] = cat[src] if src in cat.columns else np.nan

    mw, ml = out.source_mw, out.source_ml
    out["source_magnitude"] = mw.where(mw.notna(), ml)
    out["source_magnitude_type"] = np.where(mw.notna(), "mw",
                                            np.where(ml.notna(), "ml", ""))
    out["source_magnitude_uncertainty"] = out.source_mw_sigma.where(mw.notna())

    out["source_mw_quality"] = out.source_mw_quality.fillna("")
    for c in _COUNTS:
        out[c] = out[c].fillna(-1).astype("int64")
    return out


def repair(meta_csv: Path, mag: pd.DataFrame, check: bool) -> dict:
    df = pd.read_csv(meta_csv, dtype={"source_id": str}, low_memory=False)
    before = set(df.columns)

    missing = df.loc[~df.source_id.isin(mag.index), "source_id"].nunique()
    joined = df.drop(columns=[c for c in mag.columns if c in df.columns], errors="ignore")
    joined = joined.join(mag, on="source_id")

    # The split is deterministic in the event id, so it needs no stored index.
    joined["split"] = [hash_split(str(s), salt=SPLIT_SALT) for s in joined.source_id]

    added = [c for c in joined.columns if c not in before]
    stats = {
        "file": meta_csv.name,
        "rows": len(joined),
        "events": int(joined.source_id.nunique()),
        "events_not_in_catalog": int(missing),
        "columns_added": added,
        "type_mw": int((joined.source_magnitude_type == "mw").sum()),
        "type_ml": int((joined.source_magnitude_type == "ml").sum()),
        "type_none": int((joined.source_magnitude_type == "").sum()),
        "split": joined.split.value_counts().to_dict(),
    }
    if not check:
        tmp = meta_csv.with_suffix(".csv.tmp")
        joined.to_csv(tmp, index=False)
        os.replace(tmp, meta_csv)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", type=Path, help="the built archive directory, holding metadata{YEAR}.csv")
    ap.add_argument("--catalog-csv", type=Path, default=None)
    ap.add_argument("--check", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    metas = sorted(args.archive.glob("metadata*.csv"))
    if not metas:
        print(f"no metadata*.csv under {args.archive}", file=sys.stderr)
        return 2

    mag = magnitude_frame(load_catalog(args.catalog_csv))
    print(f"catalog: {len(mag)} events, "
          f"{int(mag.source_mw.notna().sum())} with Mw, {int(mag.source_ml.notna().sum())} with ML")
    print(f"{'checking' if args.check else 'rewriting'} {len(metas)} metadata files\n")

    total = {"rows": 0, "mw": 0, "ml": 0, "none": 0, "orphans": 0}
    for m in metas:
        s = repair(m, mag, args.check)
        total["rows"] += s["rows"]
        total["mw"] += s["type_mw"]
        total["ml"] += s["type_ml"]
        total["none"] += s["type_none"]
        total["orphans"] += s["events_not_in_catalog"]
        print(f"  {s['file']:>20}  {s['rows']:>7} traces  "
              f"mw {s['type_mw']:>6}  ml {s['type_ml']:>5}  none {s['type_none']:>4}  "
              f"split {s['split']}")
        if s["events_not_in_catalog"]:
            print(f"    {s['events_not_in_catalog']} event(s) are not in the catalog "
                  "and keep empty magnitude fields")

    print(f"\n{total['rows']} traces: {total['mw']} on Mw, {total['ml']} on ML, "
          f"{total['none']} with neither")
    if total["orphans"]:
        print(f"{total['orphans']} events were not in the catalog")
    if args.check:
        print("\nnothing written (--check)")
    else:
        print("\nNow re-run the checksums and the dataset README before depositing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
