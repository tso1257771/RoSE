"""Tests for `rose.convert.convert_year`.

Builds a tiny synthetic native ROMPLUS HDF5 (matching the schema written
by `scripts/create_HDF5.py`) and runs the full converter end-to-end. Then
asserts that:
  - the SeisBench metadata CSV has all documented column groups,
  - per-trace values match the synthetic input,
  - sensitivity / unit / status fields round-trip from the physical HDF5,
  - the ZNE waveform stack is in the correct order on disk,
  - the chunks manifest is updated.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest


def _write_native_counts_h5(path: Path, npts: int = 6000):
    """Write a 2-event, 2-station synthetic native HDF5 (counts file)."""
    with h5py.File(path, "w") as h5:
        for evnum, ev in enumerate(["2099_0000001", "2099_0000002"], start=1):
            g = h5.create_group(ev)
            g.attrs["event_index"] = ev
            g.attrs["time"] = f"2099-01-01T00:00:0{evnum}.000000"
            g.attrs["latitude"] = 45.0 + 0.1 * evnum
            g.attrs["longitude"] = 26.0 + 0.1 * evnum
            g.attrs["depth"] = 80.0 + evnum
            g.attrs["magnitude"] = 3.0 + 0.1 * evnum
            g.attrs["source"] = "hypoDD_3D"
            g.attrs["raw_time"] = f"2099-01-01T00:00:0{evnum}.500000"
            g.attrs["raw_lat"] = 45.0
            g.attrs["raw_lon"] = 26.0
            g.attrs["raw_depth"] = 80.0
            g.attrs["Nsta"] = 6
            g.attrs["Npha"] = 12
            g.attrs["gap"] = 145.0
            g.attrs["TRes_MAE"] = 0.4
            g.attrs["TRes_MAD"] = 0.3

            wf = g.create_group("waveforms")
            for staid, sta in enumerate(["RO.MLR..BH", "RO.BZS..BH"], start=1):
                sg = wf.create_group(sta)
                sg.attrs["reference_starttime_utc"] = (
                    f"2099-01-01T00:00:0{evnum}.000000"
                )
                sg.attrs["status_counts"] = "ok"
                sg.attrs["sampling_rate_hz"] = 100.0
                sg.attrs["npts"] = npts
                sg.attrs["back_azimuth"] = 100.0 * staid
                sg.attrs["epi_dist"] = 50.0 * staid
                sg.attrs["hyp_dist"] = 60.0 * staid
                sg.attrs["p_arrival_sample"] = 1500
                sg.attrs["s_arrival_sample"] = 3000
                sg.attrs["p_arrival_utc"] = (
                    f"2099-01-01T00:00:1{evnum}.000000"
                )
                sg.attrs["s_arrival_utc"] = (
                    f"2099-01-01T00:00:3{evnum}.000000"
                )
                sg.attrs["pick_source_p"] = "manual"
                sg.attrs["pick_source_s"] = "repick"
                sg.attrs["final_snr_p"] = 3.5 + staid
                sg.attrs["final_snr_s"] = 2.8 + staid
                sg.attrs["p_arrival_in_trace"] = True
                sg.attrs["s_arrival_in_trace"] = True
                sg.attrs["p_TRes"] = 0.1
                sg.attrs["s_TRes"] = -0.2
                sg.attrs["p_theo_arrival"] = "2099-01-01T00:00:09.900000"
                sg.attrs["s_theo_arrival"] = "2099-01-01T00:00:30.200000"
                sg.attrs["p_manual_arrival"] = "2099-01-01T00:00:10.000000"
                sg.attrs["s_manual_arrival"] = "2099-01-01T00:00:30.100000"

                # Distinct values per component to verify ZNE stacking
                sg.create_dataset(
                    "Z", data=(np.ones(npts) * (staid * 10 + 1)).astype(np.float32)
                )
                sg.create_dataset(
                    "N", data=(np.ones(npts) * (staid * 10 + 2)).astype(np.float32)
                )
                sg.create_dataset(
                    "E", data=(np.ones(npts) * (staid * 10 + 3)).astype(np.float32)
                )


def _write_physical_h5(path: Path, npts: int = 6000):
    """Companion *_physical.h5 carrying per-component sensitivities."""
    with h5py.File(path, "w") as h5:
        for ev in ["2099_0000001", "2099_0000002"]:
            g = h5.create_group(ev)
            wf = g.create_group("waveforms")
            for sta in ["RO.MLR..BH", "RO.BZS..BH"]:
                sg = wf.create_group(sta)
                sg.attrs["status_physical"] = "ok"
                sg.attrs["missing_response_components"] = ""
                for comp in ("Z", "N", "E"):
                    ds = sg.create_dataset(
                        comp, data=np.zeros(npts, dtype=np.float32)
                    )
                    ds.attrs["sensitivity_value"] = 1.5e8
                    ds.attrs["units"] = "M/S"


def _write_catalog_csv(path: Path):
    """Minimal released-catalog stub: event 1 has Mw, event 2 has ML only."""
    pd.DataFrame(
        [
            {"event_index": "2099_0000001", "Mw": 4.37, "Mw_sigma": 0.12,
             "Mw_quality": "A", "Mw_nstations": 7, "Mw_fc_Hz": 4.2,
             "ML": 2.4, "ML_nstations": 5, "ML_warning": 0,
             "ML_ROMPLUS": 2.5, "Mw_ROMPLUS": 2.7},
            {"event_index": "2099_0000002", "Mw": None, "Mw_sigma": None,
             "Mw_quality": None, "Mw_nstations": None, "Mw_fc_Hz": None,
             "ML": 1.8, "ML_nstations": 4, "ML_warning": 1,
             "ML_ROMPLUS": 1.9, "Mw_ROMPLUS": 2.0},
            # Neither scale measured: the 14 real events like this must come
            # out with an empty type rather than a magnitude on a guessed scale.
            {"event_index": "2099_0000003", "Mw": None, "Mw_sigma": None,
             "Mw_quality": None, "Mw_nstations": None, "Mw_fc_Hz": None,
             "ML": None, "ML_nstations": None, "ML_warning": None,
             "ML_ROMPLUS": None, "Mw_ROMPLUS": None},
        ]
    ).to_csv(path, index=False)


@pytest.fixture
def synthetic_year(tmp_path: Path):
    counts = tmp_path / "2099_TEST_counts.h5"
    physical = tmp_path / "2099_TEST_physical.h5"
    catalog = tmp_path / "catalog.csv"
    _write_native_counts_h5(counts)
    _write_physical_h5(physical)
    _write_catalog_csv(catalog)
    out_dir = tmp_path / "rose_out"
    return counts, physical, out_dir, catalog


def test_convert_year_writes_expected_columns_and_values(synthetic_year):
    from rose import convert_year

    counts, physical, out_dir, catalog = synthetic_year
    n = convert_year(
        src_h5=str(counts),
        out_dir=str(out_dir),
        chunk_label="2099",
        physical_h5=str(physical),
        catalog_csv=str(catalog),
        bucket_size=1024,
        overwrite=True,
    )
    assert n == 4  # 2 events × 2 stations

    csv = out_dir / "metadata2099.csv"
    h5 = out_dir / "waveforms2099.hdf5"
    assert csv.exists() and h5.exists()

    df = pd.read_csv(csv)
    assert len(df) == 4

    # Documented column groups must all be present
    expected = {
        # trace identity
        "trace_name", "trace_chunk", "trace_start_time", "trace_npts",
        "trace_sampling_rate_hz", "trace_dt_s",
        "trace_category", "trace_completeness", "trace_component_order",
        "trace_units",
        # picks
        "trace_p_arrival_sample", "trace_s_arrival_sample",
        "trace_p_arrival_time", "trace_s_arrival_time",
        "trace_p_status", "trace_s_status",
        "trace_p_snr_db", "trace_s_snr_db",
        "trace_p_in_window", "trace_s_in_window",
        "trace_p_residual_s", "trace_s_residual_s",
        "trace_p_theoretical_time", "trace_s_theoretical_time",
        "trace_p_manual_time", "trace_s_manual_time",
        # path
        "path_back_azimuth_deg", "path_ep_distance_km", "path_hyp_distance_km",
        # station
        "station_network_code", "station_code", "station_location_code",
        "station_channel",
        "station_latitude_deg", "station_longitude_deg",
        "station_elevation_m", "station_region",
        # source
        "source_id", "source_origin_time",
        "source_latitude_deg", "source_longitude_deg", "source_depth_km",
        "source_magnitude", "source_magnitude_type",
        "source_magnitude_uncertainty",
        "source_mw", "source_mw_sigma", "source_mw_quality",
        "source_mw_nstations", "source_mw_fc_hz",
        "source_ml", "source_ml_nstations", "source_ml_warning",
        "source_ml_romplus", "source_mw_romplus",
        "source_catalog",
        "source_gap_deg", "source_tres_mae_s", "source_tres_mad_s",
        "source_nsta", "source_npha",
        "source_origin_time_raw", "source_latitude_raw_deg",
        "source_longitude_raw_deg", "source_depth_raw_km",
        # units / response
        "trace_unit_physical", "trace_status_physical",
        "trace_sensitivity_z", "trace_sensitivity_n", "trace_sensitivity_e",
        "trace_missing_response_components",
    }
    missing = expected - set(df.columns)
    assert not missing, f"missing columns: {sorted(missing)}"

    # Spot-check values on the first event/station row
    first = df[df["source_id"] == "2099_0000001"].iloc[0]
    assert first["trace_sampling_rate_hz"] == 100.0
    assert first["trace_npts"] == 6000
    assert first["trace_component_order"] == "ZNE"
    assert first["trace_units"] == "counts"
    assert first["station_network_code"] == "RO"
    assert first["source_catalog"] == "hypoDD_3D"
    # Mw is the preferred scale where it was measured, and the type field
    # names the scale actually stored.
    assert first["source_magnitude"] == pytest.approx(4.37)
    assert first["source_magnitude_type"] == "mw"
    assert first["source_magnitude_uncertainty"] == pytest.approx(0.12)
    assert first["source_mw_quality"] == "A"
    assert first["source_ml"] == pytest.approx(2.4)
    assert first["source_ml_romplus"] == pytest.approx(2.5)
    # Event 2 has no Mw, so it falls back to ML and the type says so.
    second = df[df["source_id"] == "2099_0000002"].iloc[0]
    assert second["source_magnitude"] == pytest.approx(1.8)
    assert second["source_magnitude_type"] == "ml"
    assert second["source_ml_warning"] == 1
    assert first["trace_p_status"] == "manual"
    assert first["trace_s_status"] == "repick"
    assert first["trace_unit_physical"] == "M/S"
    assert first["trace_status_physical"] == "ok"
    assert first["trace_sensitivity_z"] == pytest.approx(1.5e8)


def test_convert_year_empty_magnitude_type_when_neither_scale(synthetic_year, tmp_path):
    """Events with no Mw and no ML get an empty type, never a guessed scale."""
    from rose import convert_year

    counts, physical, out_dir, _catalog = synthetic_year
    bare = tmp_path / "bare_catalog.csv"
    pd.DataFrame([{"event_index": f"2099_000000{i}", "Mw": None, "Mw_sigma": None,
                   "Mw_quality": None, "Mw_nstations": None, "Mw_fc_Hz": None,
                   "ML": None, "ML_nstations": None, "ML_warning": None,
                   "ML_ROMPLUS": None, "Mw_ROMPLUS": None} for i in (1, 2)]
                 ).to_csv(bare, index=False)

    convert_year(
        src_h5=str(counts),
        out_dir=str(out_dir),
        chunk_label="2099",
        physical_h5=str(physical),
        catalog_csv=str(bare),
        bucket_size=1024,
        overwrite=True,
    )
    df = pd.read_csv(out_dir / "metadata2099.csv")
    assert df["source_magnitude"].isna().all()
    assert (df["source_magnitude_type"].fillna("") == "").all()
    assert (df["source_mw_nstations"] == -1).all()
    assert (df["source_ml_warning"] == -1).all()


def test_convert_year_rejects_missing_catalog(synthetic_year, tmp_path):
    """A dataset without magnitudes is not a valid release, so refuse to build one."""
    from rose import convert_year

    counts, physical, out_dir, _catalog = synthetic_year
    with pytest.raises(FileNotFoundError, match="Event catalog not found"):
        convert_year(
            src_h5=str(counts),
            out_dir=str(out_dir),
            chunk_label="2099",
            physical_h5=str(physical),
            catalog_csv=str(tmp_path / "does_not_exist.csv"),
            bucket_size=1024,
            overwrite=True,
        )


def test_convert_year_rejects_catalog_that_covers_nothing(synthetic_year, tmp_path):
    """A catalog matching no event must fail loudly, not silently drop magnitudes."""
    from rose import convert_year

    counts, physical, out_dir, _catalog = synthetic_year
    wrong = tmp_path / "wrong_catalog.csv"
    pd.DataFrame([{"event_index": "1999_0000001", "Mw": 3.0, "Mw_sigma": 0.1,
                   "Mw_quality": "A", "Mw_nstations": 5, "Mw_fc_Hz": 2.0,
                   "ML": 2.5, "ML_nstations": 4, "ML_warning": 0,
                   "ML_ROMPLUS": 2.6, "Mw_ROMPLUS": 2.9}]).to_csv(wrong, index=False)
    with pytest.raises(ValueError, match="absent from the event catalog"):
        convert_year(
            src_h5=str(counts),
            out_dir=str(out_dir),
            chunk_label="2099",
            physical_h5=str(physical),
            catalog_csv=str(wrong),
            bucket_size=1024,
            overwrite=True,
        )


def test_load_catalog_df_rejects_duplicate_event_index(tmp_path):
    """Duplicate keys would make the magnitude lookup ambiguous."""
    from rose.convert import _load_catalog_df

    dup = tmp_path / "dup.csv"
    pd.DataFrame([{"event_index": "2099_0000001", "Mw": 3.0},
                  {"event_index": "2099_0000001", "Mw": 4.0}]).to_csv(dup, index=False)
    with pytest.raises(ValueError, match="Duplicate event_index"):
        _load_catalog_df(str(dup))


def test_convert_year_zne_component_stacking(synthetic_year):
    """Bucket arrays must be stacked Z, N, E in that row order."""
    from rose import convert_year

    counts, physical, out_dir, _catalog = synthetic_year
    convert_year(
        src_h5=str(counts),
        out_dir=str(out_dir),
        chunk_label="2099",
        physical_h5=str(physical),
        catalog_csv=str(_catalog),
        bucket_size=1024,
        overwrite=True,
    )
    df = pd.read_csv(out_dir / "metadata2099.csv")

    # Pull one trace via the SeisBench-style trace_name ("bucket{i}$row,:3,:N").
    row = df.iloc[0]
    name = row["trace_name"]
    # trace_name format: bucket{i}$row,:C,:W
    bucket_id, slc = name.split("$")
    bucket_id = bucket_id.replace("bucket", "")
    slc = slc.replace('"', "").strip()
    parts = slc.split(",")
    row_idx = int(parts[0])

    with h5py.File(out_dir / "waveforms2099.hdf5", "r") as f:
        bucket = f[f"data/bucket{bucket_id}"][:]
    npts = int(row["trace_npts"])
    trace = bucket[row_idx, :, :npts]

    # The fixture filled Z=11, N=12, E=13 for station 1; verify ZNE order.
    if row["station_code"] == "MLR":
        z_val, n_val, e_val = 11.0, 12.0, 13.0
    else:
        z_val, n_val, e_val = 21.0, 22.0, 23.0
    np.testing.assert_allclose(trace[0], z_val)
    np.testing.assert_allclose(trace[1], n_val)
    np.testing.assert_allclose(trace[2], e_val)


def test_convert_year_emits_chunks_manifest_via_convert_all(tmp_path: Path):
    from rose import convert_all

    src = tmp_path / "src"
    src.mkdir()
    counts = src / "2099_TEST_counts.h5"
    physical = src / "2099_TEST_physical.h5"
    _write_native_counts_h5(counts)
    _write_physical_h5(physical)
    catalog = tmp_path / "catalog.csv"
    _write_catalog_csv(catalog)

    out_dir = tmp_path / "out"
    n = convert_all(
        src_dir=str(src),
        out_dir=str(out_dir),
        tag="TEST",
        catalog_csv=str(catalog),
        include_physical=True,
        bucket_size=1024,
        overwrite=True,
    )
    assert n == 4
    chunks = (out_dir / "chunks").read_text().splitlines()
    assert "2099" in chunks


def test_convert_year_overwrite_guard(synthetic_year):
    """Without overwrite=True the second call must fail loudly."""
    from rose import convert_year

    counts, physical, out_dir, _catalog = synthetic_year
    convert_year(
        src_h5=str(counts),
        out_dir=str(out_dir),
        chunk_label="2099",
        physical_h5=str(physical),
        catalog_csv=str(_catalog),
        bucket_size=1024,
        overwrite=True,
    )
    with pytest.raises(FileExistsError):
        convert_year(
            src_h5=str(counts),
            out_dir=str(out_dir),
            chunk_label="2099",
            physical_h5=str(physical),
            bucket_size=1024,
            overwrite=False,
        )
