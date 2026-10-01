"""Collect the per-station spectral fits into one table.

SourceSpec writes one YAML per event holding the fit of every station
channel. The event magnitude is the weighted mean over those fits, so any
correction for the site has to work on them rather than on the event value.
This reads them once into a single table.

Input
    sourcespec/outputs/<event>/<event>/<event>.ssp.yaml
Output
    outputs/station_fits.csv.gz   one row per station channel fit

`Mw_outlier` is SourceSpec's own rejection of a fit, and those rows are kept
here so that later steps can decide for themselves. `band` is the two-letter
channel band, so that the two instruments of one sensor site (BH and HH, for
example) stay distinguishable while the composed spectrum code is dropped.
"""
import gzip
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

try:
    Loader = yaml.CSafeLoader
except AttributeError:
    Loader = yaml.SafeLoader

# Run from a checkout without installing: the repository root supplies the
# ``rose`` package, ``magnitudes/`` supplies ``_paths``.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "magnitudes"))
from _paths import mw_root  # noqa: E402

ROOT = mw_root()
SSP = ROOT / "sourcespec" / "outputs"
OUT = ROOT / "outputs" / "station_fits.csv.gz"

COLUMNS = ["event_index", "station_id", "net", "sta", "loc", "band", "instrument_type",
           "hypo_dist_km", "azimuth", "Mw", "Mw_unc", "Mw_outlier",
           "fc", "fc_unc", "fc_outlier", "t_star", "t_star_unc", "t_star_outlier"]


def _num(v):
    return "" if v is None else f"{v:.6g}"


def read_event(path: Path):
    """One event's station fits as a list of CSV field lists."""
    try:
        info = yaml.load(path.read_text(), Loader=Loader)
    except Exception:
        return []
    eid = path.name[: -len(".ssp.yaml")]
    rows = []
    for sid, sp in (info.get("station_parameters") or {}).items():
        parts = sid.split(".")
        net, sta, loc, chan = (parts + ["", "", "", ""])[:4]
        p = {k: (sp.get(k) or {}) for k in ("Mw", "fc", "t_star")}
        if p["Mw"].get("value") is None:
            continue
        rows.append([
            eid, sid, net, sta, loc, chan[:2], sp.get("instrument_type") or "",
            _num(sp.get("hypo_dist_in_km")), _num(sp.get("azimuth")),
            _num(p["Mw"]["value"]), _num(p["Mw"].get("uncertainty")),
            str(bool(p["Mw"].get("outlier"))),
            _num(p["fc"].get("value")), _num(p["fc"].get("uncertainty")),
            str(bool(p["fc"].get("outlier"))),
            _num(p["t_star"].get("value")), _num(p["t_star"].get("uncertainty")),
            str(bool(p["t_star"].get("outlier"))),
        ])
    return rows


def main():
    files = sorted(SSP.glob("*/*/*.ssp.yaml"))
    print(f"{len(files)} event files", flush=True)
    n_rows = n_ev = 0
    with gzip.open(OUT, "wt", newline="") as fh:
        fh.write(",".join(COLUMNS) + "\n")
        with ProcessPoolExecutor() as pool:
            for i, rows in enumerate(pool.map(read_event, files, chunksize=64), 1):
                if rows:
                    n_ev += 1
                    n_rows += len(rows)
                    fh.writelines(",".join(r) + "\n" for r in rows)
                if i % 2000 == 0:
                    print(f"  {i}/{len(files)} events, {n_rows} fits", flush=True)
    print(f"{n_rows} station fits from {n_ev} events -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
