"""
Build per-event input bundles for ALL ROMPLUS events with magnitude.

Inputs, under $ROMANIA_ROOT:
    outputs/reloc_results_hypoDD3D/Enhanced_ROMPLUS_catalog.csv   origins
    outputs/reloc_results_hypoDD3D/Enhanced_ROMPLUS_picks.csv     P and S picks

Bundles produced (one dir per event_index), where run_sourcespec.py reads them:
    $ROMANIA_ROOT/romania_mw/data/all_events/<event_id>/event.yaml
    $ROMANIA_ROOT/romania_mw/data/all_events/<event_id>/picks.csv

Filter by magnitude with --min-mag.

Usage:
    python extract_all_events.py                  # all 19,206 with mag
    python extract_all_events.py --min-mag 3.5    # 469 events
    python extract_all_events.py --out-dir /elsewhere/m35 --min-mag 3.5
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

# Run from a checkout without installing: the repository root supplies the
# ``rose`` package, ``magnitudes/`` supplies ``_paths``.
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "magnitudes"))
from _paths import work_root  # noqa: E402

ROOT = work_root()
CAT = ROOT / "outputs/reloc_results_hypoDD3D/Enhanced_ROMPLUS_catalog.csv"
PICKS = ROOT / "outputs/reloc_results_hypoDD3D/Enhanced_ROMPLUS_picks.csv"
DEFAULT_OUT = ROOT / "romania_mw" / "data" / "all_events"   # BUNDLES in run_sourcespec.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mag", type=float, default=-99.0)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cat = pd.read_csv(CAT)
    cat["mag_num"] = pd.to_numeric(cat["magnitude"], errors="coerce")
    cat = cat.dropna(subset=["mag_num"])
    if args.min_mag > -99.0:
        cat = cat[cat["mag_num"] >= args.min_mag]
    cat = cat.sort_values("event_index").reset_index(drop=True)
    print(f"Building bundles for {len(cat)} events (min_mag={args.min_mag})")

    # Load picks once and group by event_index for fast lookup
    picks = pd.read_csv(PICKS, low_memory=False)
    picks_by = dict(tuple(picks.groupby("event_index")))

    n_done = 0
    n_skip = 0
    for _, ev in cat.iterrows():
        eid = ev["event_index"]
        ev_dir = out_dir / eid
        if (ev_dir / "event.yaml").exists() and (ev_dir / "picks.csv").exists():
            n_skip += 1
            continue
        ev_dir.mkdir(exist_ok=True)
        meta = {
            "event_id": eid,
            "origin_time": ev["time"],
            "latitude_deg": float(ev["latitude"]),
            "longitude_deg": float(ev["longitude"]),
            "depth_km": float(ev["depth"]),
            "catalog_magnitude": float(ev["magnitude"]),
            "catalog_source": ev["source"],
            "n_stations_used": int(ev["Nsta"]) if pd.notna(ev["Nsta"]) else None,
            "n_phases_used":   int(ev["Npha"]) if pd.notna(ev["Npha"]) else None,
        }
        with (ev_dir / "event.yaml").open("w") as fh:
            yaml.safe_dump(meta, fh, sort_keys=False)
        ev_picks = picks_by.get(eid, picks.iloc[0:0])
        ev_picks.to_csv(ev_dir / "picks.csv", index=False)
        n_done += 1
        if n_done % 500 == 0:
            print(f"  built {n_done} (skipped {n_skip})")

    print(f"\nTotal: built {n_done}, skipped {n_skip} → {out_dir}")


if __name__ == "__main__":
    main()
