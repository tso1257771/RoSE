#!/usr/bin/env python
"""Assemble the observation table used by the ML fit.

Reads the Wood-Anderson amplitudes and their clipping diagnostics, applies the
catalogue gates and writes the fit and application tables. Blow-ups are limited
by a 3e4 mm ceiling and, in the fit, by their distance-model residual; clipped
observations are kept in the table and excluded by the fit.

Romanian analogue of taiwan_ml.preprocess.build_observations (reference repo):
  catalog gates (fit set): region box, 0 <= depth <= 320 km, gap <= 180 deg,
      Nsta >= 8 (reference nstn_min); the Taiwan quality A/B, ERH/ERZ <= 5 km and
      free-depth gates have no ROMPLUS/hypoDD3D equivalent and are not applied.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from obspy.geodetics.base import gps2dist_azimuth

# Run from a checkout without installing: the repository root supplies the
# ``rose`` package, ``magnitudes/`` supplies ``_paths``.
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "magnitudes"))
from _paths import ml_root, work_root  # noqa: E402

PROJECT = work_root()
ROOT = ml_root()
ROSE = PROJECT / "seisbench_integration/data/rose"
CAT = PROJECT / "seisbench_integration/data/Enhanced_ROMPLUS_catalog.csv"

QC = dict(lat_box=(43.0, 49.0), lon_box=(19.0, 31.0), depth_min=0.0, depth_max=320.0,
          gap_max=180.0, nstn_min=8, wa_floor_mm=1e-5, wa_ceil_mm=3e4, snr_min=1.0, snr_clip=100.0,
          r_min_km=1.0, r_max_km=600.0, min_stations_event=3)


def main():
    amps = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "outputs/amplitudes/wa_20??.csv")))],
                     ignore_index=True)
    md_cols = ["trace_name", "source_id", "station_network_code", "station_code",
               "station_location_code", "station_channel", "station_latitude_deg",
               "station_longitude_deg", "station_elevation_m"]
    md = pd.concat([pd.read_csv(f, usecols=md_cols, low_memory=False).assign(year=int(f[-8:-4]))
                    for f in sorted(glob.glob(str(ROSE / "metadata20??.csv")))], ignore_index=True)
    df = amps.merge(md, on=["trace_name", "year"], how="left", validate="one_to_one")
    stats = {"traces": int(len(df)), "status": df.status.value_counts().to_dict()}

    cat = pd.read_csv(CAT)
    cat = cat.rename(columns={"event_index": "public_id"})
    df = df.merge(cat[["public_id", "time", "latitude", "longitude", "depth", "gap", "Nsta",
                       "ML_ROMPLUS"]],
                  left_on="source_id", right_on="public_id", how="inner")
    stats["traces_in_catalog"] = int(len(df))

    ok = df.status.eq("ok") & np.isfinite(df.wa_mm)
    stats["drop_wa_status"] = int((~ok).sum())
    df = df[ok].copy()
    n0 = len(df)
    df = df[(df.wa_mm >= QC["wa_floor_mm"]) & (df.wa_mm <= QC["wa_ceil_mm"])]
    stats["drop_amp"] = n0 - len(df)
    n0 = len(df)
    snr = df.snr.clip(upper=QC["snr_clip"])
    df = df[~(snr.notna() & (snr < QC["snr_min"]))]
    stats["drop_snr"] = n0 - len(df)
    n0 = len(df)
    df = df[df.epoch_start.fillna("").astype(str).str.len() > 0]
    stats["drop_epoch"] = n0 - len(df)

    df["loc"] = df.station_location_code.fillna("").astype(str)
    df["net"], df["sta"], df["chan_pre"] = df.station_network_code, df.station_code, df.station_channel
    df["epi_km"] = [gps2dist_azimuth(a, b, c, d)[0] / 1000.0 for a, b, c, d in
                    zip(df.latitude, df.longitude, df.station_latitude_deg, df.station_longitude_deg)]
    df["sta_elev_km"] = df.station_elevation_m / 1000.0
    df["depth_km"] = df.depth.astype(float)
    df["R_km"] = np.sqrt(df.epi_km ** 2 + (df.depth_km + df.sta_elev_km) ** 2)
    n0 = len(df)
    df = df[df.R_km.between(QC["r_min_km"], QC["r_max_km"])]
    stats["drop_R"] = n0 - len(df)
    df["log10_A"] = np.log10(df.wa_mm)
    df["station_key"] = (df.net + "." + df.sta + "." + df["loc"] + "." + df.chan_pre + "@"
                         + df.epoch_start.astype(str).str[:10])
    df["ev_time"] = df.time
    keep = ["public_id", "ev_time", "latitude", "longitude", "depth_km", "gap", "Nsta",
            "ML_ROMPLUS", "net", "sta", "loc", "chan_pre",
            "epoch_start", "native_sr", "lowpass_hz", "station_latitude_deg", "station_longitude_deg",
            "sta_elev_km", "epi_km", "R_km", "wa_mm", "log10_A", "snr", "mag_window_truncated",
            "station_key", "trace_name", "clipped", "n_horizontals", "stuck_e", "stuck_n"]
    df = df[keep]

    def min_sta(frame):
        nloc = frame.groupby("public_id")[["net", "sta", "loc"]].apply(
            lambda g: len(g.drop_duplicates()))
        good = nloc[nloc >= QC["min_stations_event"]].index
        return frame[frame.public_id.isin(good)]

    allobs = min_sta(df)
    stats["application_obs"] = int(len(allobs))
    stats["application_events"] = int(allobs.public_id.nunique())

    fit = df[df.latitude.between(*QC["lat_box"]) & df.longitude.between(*QC["lon_box"])
             & df.depth_km.between(QC["depth_min"], QC["depth_max"])
             & (df.gap <= QC["gap_max"]) & (df.Nsta >= QC["nstn_min"])]
    fit = min_sta(fit)
    stats["fit_obs"] = int(len(fit))
    stats["fit_events"] = int(fit.public_id.nunique())
    stats["fit_station_keys"] = int(fit.station_key.nunique())
    stats["clipped_obs"] = dict(all=int(allobs.clipped.sum()), fit=int(fit.clipped.sum()))
    stats["single_horizontal_obs"] = int((allobs.n_horizontals == 1).sum())
    stats["stuck_component_by_station"] = (
        allobs[allobs.n_horizontals == 1].groupby(["net", "sta", "chan_pre"]).size()
        .sort_values(ascending=False).head(15).to_dict())
    stats["stuck_component_by_station"] = {".".join(k): int(v) for k, v in
                                           stats["stuck_component_by_station"].items()}
    stats["qc"] = {k: list(v) if isinstance(v, tuple) else v for k, v in QC.items()}

    allobs.to_csv(ROOT / "outputs/observations_all.csv.gz", index=False)
    fit.to_csv(ROOT / "outputs/observations_fit.csv.gz", index=False)
    (ROOT / "outputs/observations_stats.json").write_text(json.dumps(stats, indent=2, default=int))
    print(json.dumps(stats, indent=2, default=int))


if __name__ == "__main__":
    main()
