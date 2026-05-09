"""Convert native ROMPLUS-derived yearly HDF5 files into the RoSE SeisBench dataset.

Source files (built by `scripts/create_HDF5.py`):
    {YEAR}_{TAG}_counts.h5      raw counts, integer-equivalent float32
    {YEAR}_{TAG}_physical.h5    counts / instrument sensitivity (m/s or m/s^2)

Each event is a top-level HDF5 group with attributes (catalog row) and a
`waveforms` subgroup holding one group per (NET.STA.LOC.BAND) station, with
E/N/Z float32 datasets aligned to a common reference start time at 100 Hz.

The converter writes a SeisBench *chunked* dataset:
    <out_dir>/chunks
    <out_dir>/metadata{YEAR}.csv
    <out_dir>/waveforms{YEAR}.hdf5

One SeisBench trace = one (event, station) tuple, components stacked as ZNE
with shape (3, npts).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable, List, Optional

import h5py
import numpy as np
import pandas as pd
import seisbench.data as sbd

logger = logging.getLogger("rose.convert")

COMPONENT_ORDER = "ZNE"  # canonical SeisBench order; on-disk components stacked Z,N,E
TARGET_SAMPLING_RATE_HZ = 100.0


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return value


def _attr(source: Any, key: str, default: Any = None) -> Any:
    """Read an attribute from either an h5py object or a plain attrs dict."""
    if hasattr(source, "attrs"):
        attrs = source.attrs
    else:
        attrs = source
    if key not in attrs:
        return default
    val = _decode(attrs[key])
    if isinstance(val, str) and val.lower() in ("nan", "none", ""):
        return default
    return val


def _float(value: Any, default: float = np.nan) -> float:
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f


def _int(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _split_station_id(station_id: str) -> tuple[str, str, str, str]:
    """Split 'NET.STA.LOC.BAND' (BAND has 2 chars, channel = BAND + comp)."""
    parts = station_id.split(".")
    if len(parts) != 4:
        return station_id, "", "", ""
    return parts[0], parts[1], parts[2], parts[3]


def _stack_components(station_group, npts: int) -> Optional[np.ndarray]:
    arr = np.zeros((3, npts), dtype=np.float32)
    for i, comp in enumerate(COMPONENT_ORDER):
        if comp not in station_group:
            return None
        data = station_group[comp][:]
        n = min(data.shape[0], npts)
        arr[i, :n] = data[:n].astype(np.float32, copy=False)
    return arr


def _build_metadata(
    event_id: str,
    event_attrs: dict,
    station_id: str,
    station_attrs: dict,
    npts: int,
    stations_df: Optional[pd.DataFrame],
    physical_info: Optional[dict] = None,
) -> dict:
    net, sta, loc, band = _split_station_id(station_id)
    ref_start = _attr(station_attrs, "reference_starttime_utc")

    md = {
        "trace_name": f"{event_id}_{station_id}",
        "trace_start_time": ref_start,
        "trace_sampling_rate_hz": TARGET_SAMPLING_RATE_HZ,
        "trace_dt_s": 1.0 / TARGET_SAMPLING_RATE_HZ,
        "trace_npts": int(npts),
        "trace_category": "earthquake_local",
        "trace_completeness": _attr(station_attrs, "status_counts", "ok"),
        "trace_component_order": COMPONENT_ORDER,
        "trace_units": "counts",
        # Picks (standard SeisBench names)
        "trace_p_arrival_sample": _int(_attr(station_attrs, "p_arrival_sample"), default=-1),
        "trace_s_arrival_sample": _int(_attr(station_attrs, "s_arrival_sample"), default=-1),
        "trace_p_arrival_time": _attr(station_attrs, "p_arrival_utc"),
        "trace_s_arrival_time": _attr(station_attrs, "s_arrival_utc"),
        "trace_p_status": _attr(station_attrs, "pick_source_p"),
        "trace_s_status": _attr(station_attrs, "pick_source_s"),
        "trace_p_snr_db": _float(_attr(station_attrs, "final_snr_p")),
        "trace_s_snr_db": _float(_attr(station_attrs, "final_snr_s")),
        "trace_p_in_window": bool(_attr(station_attrs, "p_arrival_in_trace", False)),
        "trace_s_in_window": bool(_attr(station_attrs, "s_arrival_in_trace", False)),
        # Pick provenance / extras (full superset of native attrs)
        "trace_p_residual_s": _float(_attr(station_attrs, "p_TRes")),
        "trace_s_residual_s": _float(_attr(station_attrs, "s_TRes")),
        "trace_p_theoretical_time": _attr(station_attrs, "p_theo_arrival"),
        "trace_s_theoretical_time": _attr(station_attrs, "s_theo_arrival"),
        "trace_p_manual_time": _attr(station_attrs, "p_manual_arrival"),
        "trace_s_manual_time": _attr(station_attrs, "s_manual_arrival"),
        # Path
        "path_back_azimuth_deg": _float(_attr(station_attrs, "back_azimuth")),
        "path_ep_distance_km": _float(_attr(station_attrs, "epi_dist")),
        "path_hyp_distance_km": _float(_attr(station_attrs, "hyp_dist")),
        # Station
        "station_network_code": net,
        "station_code": sta,
        "station_location_code": loc,
        "station_channel": band,
        # Source (relocated)
        "source_id": str(event_id),
        "source_origin_time": _attr(event_attrs, "time"),
        "source_latitude_deg": _float(_attr(event_attrs, "latitude")),
        "source_longitude_deg": _float(_attr(event_attrs, "longitude")),
        "source_depth_km": _float(_attr(event_attrs, "depth")),
        "source_magnitude": _float(_attr(event_attrs, "magnitude")),
        "source_magnitude_type": "ml",
        "source_catalog": _attr(event_attrs, "source"),
        "source_gap_deg": _float(_attr(event_attrs, "gap")),
        "source_tres_mae_s": _float(_attr(event_attrs, "TRes_MAE")),
        "source_tres_mad_s": _float(_attr(event_attrs, "TRes_MAD")),
        "source_nsta": _int(_attr(event_attrs, "Nsta"), default=-1),
        "source_npha": _int(_attr(event_attrs, "Npha"), default=-1),
        # Original (pre-relocation) ROMPLUS solution
        "source_origin_time_raw": _attr(event_attrs, "raw_time"),
        "source_latitude_raw_deg": _float(_attr(event_attrs, "raw_lat")),
        "source_longitude_raw_deg": _float(_attr(event_attrs, "raw_lon")),
        "source_depth_raw_km": _float(_attr(event_attrs, "raw_depth")),
        # Station enrichment is filled below when stations_df is available;
        # always emit the columns so downstream consumers can rely on them.
        "station_latitude_deg": np.nan,
        "station_longitude_deg": np.nan,
        "station_elevation_m": np.nan,
        "station_region": "",
    }

    if stations_df is not None and sta in stations_df.index:
        srow = stations_df.loc[sta]
        md["station_latitude_deg"] = _float(srow.get("latitude"))
        md["station_longitude_deg"] = _float(srow.get("longitude"))
        md["station_elevation_m"] = _float(srow.get("elevation_m"))
        region = srow.get("region")
        md["station_region"] = "" if region is None else str(region)

    if physical_info is not None:
        md["trace_status_physical"] = physical_info.get("status_physical", "unknown")
        md["trace_missing_response_components"] = physical_info.get(
            "missing_response_components", ""
        )
        md["trace_unit_physical"] = physical_info.get("unit", "")
        sens = physical_info.get("sensitivity", {})
        for comp in COMPONENT_ORDER:
            md[f"trace_sensitivity_{comp.lower()}"] = _float(sens.get(comp), default=np.nan)

    return md


def _load_physical_index(physical_h5: Optional[str]) -> Optional[dict]:
    """Index sensitivity values per (event_id, station_id) from the physical HDF5.

    Returns a dict keyed by (event_id, station_id) with:
        status_physical, missing_response_components, unit,
        sensitivity = {"E": float, "N": float, "Z": float}
    """
    if not physical_h5 or not os.path.exists(physical_h5):
        return None
    index: dict = {}
    with h5py.File(physical_h5, "r") as h5:
        for event_id in h5.keys():
            ev = h5[event_id]
            if "waveforms" not in ev:
                continue
            for station_id, sg in ev["waveforms"].items():
                entry = {
                    "status_physical": _attr(sg, "status_physical", "unknown"),
                    "missing_response_components": _attr(
                        sg, "missing_response_components", ""
                    ),
                    "sensitivity": {},
                    "unit": "",
                }
                for comp in COMPONENT_ORDER:
                    if comp not in sg:
                        continue
                    ds = sg[comp]
                    sens = _attr(ds, "sensitivity_value")
                    if sens is not None:
                        entry["sensitivity"][comp] = float(sens)
                    if not entry["unit"]:
                        unit = _attr(ds, "units")
                        if unit and unit != "counts":
                            entry["unit"] = str(unit)
                index[(event_id, station_id)] = entry
    return index


def _load_stations_df(stations_csv: Optional[str]) -> Optional[pd.DataFrame]:
    if not stations_csv or not os.path.exists(stations_csv):
        return None
    df = pd.read_csv(stations_csv)
    # Normalize a column name for elevation in meters (csv stores km).
    if "elevation" in df.columns and "elevation_m" not in df.columns:
        df["elevation_m"] = pd.to_numeric(df["elevation"], errors="coerce") * 1000.0
    return df.set_index("station")


def convert_year(
    src_h5: str,
    out_dir: str,
    chunk_label: str,
    units: str = "counts",
    stations_csv: Optional[str] = None,
    physical_h5: Optional[str] = None,
    bucket_size: int = 1024,
    overwrite: bool = False,
) -> int:
    """Convert one yearly source HDF5 into a SeisBench chunk.

    :param src_h5: Path to source ROMPLUS HDF5 (counts).
    :param out_dir: Target SeisBench dataset directory.
    :param chunk_label: Chunk name (e.g. "2014").
    :param units: Waveform units stored on disk; always "counts" for the
        published format. Physical units are derived per-trace via the
        ``trace_sensitivity_{e,n,z}`` columns.
    :param stations_csv: Optional station CSV for lat/lon/elev enrichment.
    :param physical_h5: Optional companion ``*_physical.h5`` to harvest
        per-component sensitivity values and response status.
    :param bucket_size: Bucket size for the SeisBench writer.
    :param overwrite: Overwrite existing chunk files.
    :return: Number of traces written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = out_dir / f"metadata{chunk_label}.csv"
    waveforms_path = out_dir / f"waveforms{chunk_label}.hdf5"
    if metadata_path.exists() or waveforms_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Chunk already exists (use overwrite=True): {metadata_path}"
            )
        metadata_path.unlink(missing_ok=True)
        waveforms_path.unlink(missing_ok=True)

    stations_df = _load_stations_df(stations_csv)
    physical_index = _load_physical_index(physical_h5)
    if physical_h5:
        logger.info(
            "Loaded sensitivity index from %s (%d entries)",
            physical_h5,
            0 if physical_index is None else len(physical_index),
        )

    n_traces = 0
    with sbd.WaveformDataWriter(metadata_path, waveforms_path) as writer:
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": COMPONENT_ORDER,
            "sampling_rate": TARGET_SAMPLING_RATE_HZ,
            "measurement": units,
            "unit": units,
            "instrument_response": "raw",
        }
        writer.bucket_size = bucket_size

        with h5py.File(src_h5, "r") as h5:
            for event_id in h5.keys():
                ev_group = h5[event_id]
                if "waveforms" not in ev_group:
                    continue
                wf_group = ev_group["waveforms"]
                event_attrs = dict(ev_group.attrs)

                for station_id in wf_group.keys():
                    sta_group = wf_group[station_id]
                    npts = _int(_attr(sta_group, "npts"), default=0)
                    if npts <= 0:
                        if "Z" in sta_group:
                            npts = int(sta_group["Z"].shape[0])
                        else:
                            continue
                    arr = _stack_components(sta_group, npts)
                    if arr is None:
                        continue
                    phys_info = (
                        physical_index.get((event_id, station_id))
                        if physical_index is not None
                        else None
                    )
                    md = _build_metadata(
                        event_id=event_id,
                        event_attrs=event_attrs,
                        station_id=station_id,
                        station_attrs=dict(sta_group.attrs),
                        npts=npts,
                        stations_df=stations_df,
                        physical_info=phys_info,
                    )
                    md["trace_chunk"] = chunk_label
                    writer.add_trace(md, arr)
                    n_traces += 1

    logger.info("Wrote %d traces -> %s", n_traces, waveforms_path)
    return n_traces


def _update_chunks_file(out_dir: Path, chunk_label: str):
    chunks_file = out_dir / "chunks"
    existing: List[str] = []
    if chunks_file.exists():
        existing = [
            line.strip()
            for line in chunks_file.read_text().splitlines()
            if line.strip()
        ]
    if chunk_label not in existing:
        existing.append(chunk_label)
        chunks_file.write_text("\n".join(sorted(existing)) + "\n")


def convert_all(
    src_dir: str,
    out_dir: str,
    tag: str = "ROMPLUS",
    years: Optional[Iterable[int]] = None,
    stations_csv: Optional[str] = None,
    include_physical: bool = True,
    bucket_size: int = 1024,
    overwrite: bool = False,
) -> int:
    """Convert all yearly ROMPLUS HDF5 files in `src_dir` into one SeisBench dataset.

    Always reads the ``*_counts.h5`` files for waveforms. If
    ``include_physical=True`` (default) also reads the matching
    ``*_physical.h5`` to harvest per-component sensitivity values, so users
    can derive physical units (M/S, M/S**2) on the fly without us shipping
    the data twice.

    :return: Total number of traces written across all chunks.
    """
    src_dir_p = Path(src_dir)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    counts_suffix = f"_{tag}_counts.h5"
    physical_suffix = f"_{tag}_physical.h5"

    counts_available = {}
    physical_available = {}
    for path in sorted(src_dir_p.glob(f"*{counts_suffix}")):
        year = int(path.name.split("_", 1)[0])
        counts_available[year] = path
    if include_physical:
        for path in sorted(src_dir_p.glob(f"*{physical_suffix}")):
            year = int(path.name.split("_", 1)[0])
            physical_available[year] = path

    if not counts_available:
        raise FileNotFoundError(f"No files matching *{counts_suffix} in {src_dir}")

    selected_years = sorted(years) if years else sorted(counts_available.keys())

    total = 0
    for year in selected_years:
        if year not in counts_available:
            logger.warning("Year %s not found in %s, skip", year, src_dir)
            continue
        chunk_label = str(year)
        phys_path = str(physical_available[year]) if year in physical_available else None
        n = convert_year(
            src_h5=str(counts_available[year]),
            out_dir=str(out_dir_p),
            chunk_label=chunk_label,
            units="counts",
            stations_csv=stations_csv,
            physical_h5=phys_path,
            bucket_size=bucket_size,
            overwrite=overwrite,
        )
        total += n
        _update_chunks_file(out_dir_p, chunk_label)
    return total


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Convert ROMPLUS HDF5 to SeisBench.")
    p.add_argument("--src-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--no-physical",
        action="store_true",
        help="Skip ingesting *_physical.h5 sensitivity values.",
    )
    p.add_argument("--tag", default="ROMPLUS")
    p.add_argument(
        "--years",
        default="",
        help="Optional comma list, e.g. '2014,2015,2018-2020'.",
    )
    p.add_argument("--stations-csv", default="")
    p.add_argument("--bucket-size", type=int, default=1024)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _expand_years(spec: str) -> list[int] | None:
    if not spec:
        return None
    out: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            if "-" in chunk:
                a, b = chunk.split("-", 1)
                out.update(range(int(a), int(b) + 1))
            else:
                out.add(int(chunk))
        except ValueError as e:
            raise ValueError(
                f"invalid year token {chunk!r} in --years={spec!r}"
            ) from e
    return sorted(out)


def main():
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    n = convert_all(
        src_dir=args.src_dir,
        out_dir=args.out_dir,
        tag=args.tag,
        years=_expand_years(args.years),
        stations_csv=args.stations_csv or None,
        include_physical=not args.no_physical,
        bucket_size=args.bucket_size,
        overwrite=args.overwrite,
    )
    logger.info("Done. Total traces written: %d", n)


if __name__ == "__main__":
    main()
