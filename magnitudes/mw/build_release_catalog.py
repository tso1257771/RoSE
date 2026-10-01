"""
Add the catalogue magnitudes to the RoSE release event catalogue.

The release file itself is never modified; a copy with extra columns is
written to romania_mw/outputs/.

Inputs
    seisbench_integration/data/Enhanced_ROMPLUS_catalog.csv  release catalogue
    romania_mw/outputs/Mw_catalog.csv, Mw_validation.yaml    Mw (SourceSpec)
    romania_ml/outputs/fit/event_ml.csv, fit/report.json     ML (anchored)
Output
    romania_mw/outputs/Enhanced_ROMPLUS_catalog_with_magnitudes.csv

Columns added
    Mw, Mw_sigma, Mw_quality, Mw_nstations, Mw_fc_Hz, ML, ML_nstations, ML_warning
Each column is a value a reader filters or plots. Anything derivable from
another column is left out: the distance correction is chosen by depth >= 60 km,
and stress drop follows from Mw, Mw_fc_Hz and the velocity model, so both stay
in Mw_catalog.csv rather than here. The methods behind the two magnitudes are
written once to MAGNITUDE_COLUMNS.md next to the catalogue, not on every row.
The catalogue's own ML_ROMPLUS and Mw_ROMPLUS columns are kept as published by
NIEP and are a reference only: they enter neither magnitude.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

# Run from a checkout without installing: the repository root supplies the
# ``rose`` package, ``magnitudes/`` supplies ``_paths``.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "magnitudes"))
from _paths import work_root  # noqa: E402

ROOT = work_root()
MW = ROOT / "romania_mw" / "outputs"
ML = ROOT / "romania_ml" / "outputs"
RELEASE = ROOT / "seisbench_integration" / "data" / "Enhanced_ROMPLUS_catalog.csv"


def atomic_write(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main():
    rel = pd.read_csv(RELEASE)

    mw = pd.read_csv(MW / "Mw_catalog.csv")
    val = yaml.safe_load((MW / "Mw_validation.yaml").read_text())["validation_USGS"]
    mw_desc = (
        "SourceSpec 1.8 S-wave spectral inversion (Brune omega-square, t* per station, 1/r spreading, "
        "stations <= 300 km hypocentral), RED-PAN-Motion S windows with a magnitude-dependent minimum. "
        "No external magnitude enters Mw or its uncertainty, and no depth correction is applied. "
        f"Compared with USGS Mww/Mwr, which is not fitted: USGS minus Mw = "
        f"{val['usgs_minus_Mw']['mean']:+.2f} +- {val['usgs_minus_Mw']['std']:.2f} (n = {val['n']}), "
        "an offset that is significant on that sample and is larger before 2019 than after. "
        "Mw_sigma = sqrt(station term^2 + systematic^2), systematic 0.10-0.18."
    )
    mw_cols = mw[["event_index", "Mw", "Mw_sigma", "Mw_quality", "Mw_nsta", "fc_Hz"]].rename(
        columns={"Mw_nsta": "Mw_nstations", "fc_Hz": "Mw_fc_Hz"})
    mw_cols["Mw_nstations"] = mw_cols["Mw_nstations"].astype("Int64")
    mw_cols["Mw_quality"] = mw_cols["Mw_quality"].where(mw_cols["Mw"].notna())

    ml = pd.read_csv(ML / "fit" / "event_ml.csv")
    rep = yaml.safe_load((ML / "fit" / "report.json").read_text())
    c_cr, c_in = rep["C"]["crustal"], rep["C"]["intermediate"]
    ml_desc = (
        "Local magnitude from Wood-Anderson amplitudes (mean of the two horizontals) with per-station "
        "corrections; separate distance corrections for crustal (depth < 60 km, three segments) and "
        "intermediate-depth events. The baseline constant is anchored to SourceSpec Mw at Mw 4.0 in each "
        f"regime (C = {c_cr:+.2f} crustal, {c_in:+.2f} intermediate-depth; -log A0 at 100 km = "
        f"{3.0 + c_cr:.3f} / {3.0 + c_in:.3f}), so ML minus Mw is zero at Mw 4.0 by construction. "
        "Attenuation shape, station terms and b-value are independent of Mw. Absolute ML carries a "
        "common-mode uncertainty of about 0.10 from the anchor."
    )
    ml_cols = ml[["public_id", "ml", "n_sta", "ml_saturation_risk"]].rename(
        columns={"public_id": "event_index", "ml": "ML", "n_sta": "ML_nstations"})
    ml_cols["ML_nstations"] = ml_cols["ML_nstations"].astype("Int64")
    ml_cols["ML_warning"] = (ml_cols.pop("ml_saturation_risk")
                             .astype("boolean").astype("Int64"))

    out = (rel.merge(mw_cols, on="event_index", how="left", validate="one_to_one")
              .merge(ml_cols, on="event_index", how="left", validate="one_to_one"))
    assert len(out) == len(rel) and (out["event_index"].values == rel["event_index"].values).all()
    # at most 4 decimals, as in the release catalogue's own columns
    out[["Mw", "Mw_sigma", "Mw_fc_Hz", "ML"]] = out[["Mw", "Mw_sigma", "Mw_fc_Hz", "ML"]].round(4)
    atomic_write(out, MW / "Enhanced_ROMPLUS_catalog_with_magnitudes.csv")

    n_warn = int((out["ML_warning"] == 1).sum())
    n_mw = int(out["Mw"].notna().sum())
    n_mw_only = int((out["Mw"].notna() & out["ML"].isna()).sum())
    n_no_bulletin = int((rel["ML_ROMPLUS"].isna() & rel["Mw_ROMPLUS"].isna()
                         & out["Mw"].notna()).sum())
    n_ge4, n_ge45, n_ge5 = (int((out["Mw"] >= x).sum()) for x in (4.0, 4.5, 5.0))
    doc = f"""# Magnitude columns in `Enhanced_ROMPLUS_catalog_with_magnitudes.csv`

The catalogue is the RoSE release event list with two measured magnitudes added,
{out['Mw'].notna().sum():,} events with Mw and {out['ML'].notna().sum():,} with ML out of {len(out):,}.
The two scales are reported separately. ML is anchored to Mw at Mw 4.0, so at that
magnitude the difference between them is zero by construction and is not a check.

## Mw

{mw_desc}

| Column | Meaning |
|---|---|
| `Mw` | moment magnitude, SourceSpec spectral inversion |
| `Mw_sigma` | uncertainty, station term and systematic combined (1 sigma) |
| `Mw_quality` | `A`, `B` or `C`. A: at least 5 distinct sites, few t* at the bound, corner frequency inside the search range, standard error < 0.15. B: at least 3 sites. C: 1 or 2 sites |
| `Mw_nstations` | distinct recording sites used. The two instruments of one site, BH and HH for example, are fitted separately but count once |
| `Mw_fc_Hz` | corner frequency, in Hz. Reported for Mw >= 3.5 only ({out['Mw_fc_Hz'].notna().sum()} events); below that the spectrum is band-limited and fc is not interpreted. Empty therefore means not reported, not unmeasurable |

## ML

{ml_desc}

| Column | Meaning |
|---|---|
| `ML` | local magnitude, Wood-Anderson amplitudes with station corrections |
| `ML_nstations` | stations used, at least 3 |
| `ML_warning` | `0` normal, `1` amplitude quality warning ({n_warn} events): amplitudes clipped or rejected at more than 20 per cent of the stations, or corner frequency below 1.25 Hz |

## Which magnitude to use

Use `Mw` wherever the magnitude stands for the size of the source, including
ground-motion work and machine-learning labels. It is proportional to moment
over the whole range, it does not saturate, and it is measured for
{n_mw:,} events, {n_mw_only:,} of them where no local magnitude could be measured and
{n_no_bulletin:,} where the NIEP bulletin gives no magnitude at all.

`ML` is reported for continuity with the bulletin and for comparison with other
local-magnitude studies. It runs about 1.33 magnitude units per unit of Mw for
crustal events and 1.41 for intermediate-depth ones, because the corner
frequency of a small earthquake lies above the Wood-Anderson band and the
instrument then reads the flat part of the displacement spectrum. Completeness and b-value therefore differ between the
two scales and cannot be carried from one to the other. Do not put both scales
into one frequency-magnitude distribution. To move between them, use the
conversion in `../../romania_ml/outputs/conversion/` within its stated range.

Two limits apply at the ends of the Mw range. Below about Mw 2.5 the corner
frequency is at or beyond the resolvable band, so Mw is biased high by an
amount covered by `Mw_sigma`, which is 0.20 below Mw 2 against 0.11 above
Mw 3.8, and below about Mw 1.0 the smallest events are measured only when
their spectrum clears the noise. At the other end the catalogue holds
{n_ge4} events at Mw 4.0 and above, {n_ge45} at 4.5 and above and {n_ge5} at 5.0 and above, so it
does not constrain the behaviour of any model in the range a warning system
exists for.

## NIEP bulletin columns

`ML_ROMPLUS` and `Mw_ROMPLUS` are the NIEP bulletin magnitudes, kept for
continuity with the published catalogue. {(rel['ML_ROMPLUS'].isna() & rel['Mw_ROMPLUS'].isna()).sum():,} events have neither.
The two are equal in {int((rel['ML_ROMPLUS'] == rel['Mw_ROMPLUS']).sum()):,} rows, so `Mw_ROMPLUS` is not an independent
moment magnitude. Neither column enters `Mw` or `ML`.

## Usual selections

| Selection | Expression | Events |
|---|---|---|
| usable Mw | `Mw_quality in ("A", "B")` | {out['Mw_quality'].isin(['A', 'B']).sum():,} |
| best Mw | `Mw_quality == "A"` | {(out['Mw_quality'] == 'A').sum():,} |
| usable ML | `ML_warning == 0` | {int((out['ML_warning'] == 0).sum()):,} |
| events with a corner frequency | `Mw_fc_Hz` not empty | {out['Mw_fc_Hz'].notna().sum()} |

An empty magnitude column means the event was not measured on that scale:
{int((out['Mw'].notna() & out['ML'].isna()).sum()):,} events have Mw only, {int((out['Mw'].isna() & out['ML'].notna()).sum()):,} have ML only and {int((out['Mw'].isna() & out['ML'].isna()).sum())} have neither.
The counting columns and `ML_warning` are read back as floating point when a row
is empty.

Full method, uncertainties and validity ranges: `../README.md` and
`../../romania_ml/README.md`. Stress drop and t* per event are in
`Mw_catalog.csv`. The ML to Mw conversion is in
`../../romania_ml/outputs/conversion/`.
"""
    (MW / "MAGNITUDE_COLUMNS.md").write_text(doc)
    print(f"{len(out)} events, {len(out.columns)} columns; "
          f"Mw {out['Mw'].notna().sum()}, ML {out['ML'].notna().sum()}; "
          f"methods written to MAGNITUDE_COLUMNS.md")


if __name__ == "__main__":
    main()
