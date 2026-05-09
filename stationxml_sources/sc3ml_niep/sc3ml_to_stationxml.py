#!/usr/bin/env python3
"""Convert SeisComP inventory XML (SC3ML) to FDSN StationXML using ObsPy.

Splits outputs per-station into ``{dst}/{network}/{station}.xml``. Stations
appearing in multiple source files are merged (channels concatenated).

Usage:
    python sc3ml_to_stationxml.py [--src DIR] [--dst DIR] [--jobs N]

Defaults:
    --src   directory containing this script
    --dst   <src>/stationxml
    --jobs  1
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed

from obspy import read_inventory
from obspy.core.inventory import Inventory, Network


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _patch_sc3ml(src_path: str) -> str:
    """Work around ObsPy 1.5 SC3ML reader quirks. Returns a path ObsPy can read
    (either the original path or a patched temp file)."""
    tree = ET.parse(src_path)
    root = tree.getroot()
    changed = False
    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        if tag == "sensorLocation" and elem.get("code") is None:
            elem.set("code", "")
            changed = True
        if tag == "stream" and elem.get("code") is None:
            elem.set("code", "")
            changed = True
    if not changed:
        return src_path
    tmp = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".xml", delete=False, prefix="sc3ml_patched_"
    )
    tmp_name = tmp.name
    tmp.close()
    try:
        tree.write(tmp_name, encoding="utf-8", xml_declaration=True)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return tmp_name


def _fix_inventory_for_stationxml(inv: Inventory) -> None:
    """Normalize inventory fields so the StationXML writer passes validation."""
    for net in inv.networks:
        for sta in net.stations:
            for ch in sta.channels:
                if ch.response is None:
                    continue
                isens = ch.response.instrument_sensitivity
                if isens is None:
                    continue
                group = [
                    isens.frequency_range_start,
                    isens.frequency_range_end,
                    isens.frequency_range_db_variation,
                ]
                if any(v is not None for v in group) and not all(v is not None for v in group):
                    isens.frequency_range_start = None
                    isens.frequency_range_end = None
                    isens.frequency_range_db_variation = None


def _read_sc3ml(src_path: str) -> Inventory:
    patched = _patch_sc3ml(src_path)
    try:
        inv = read_inventory(patched, format="SC3ML")
    finally:
        if patched != src_path:
            try:
                os.unlink(patched)
            except OSError:
                pass
    _fix_inventory_for_stationxml(inv)
    return inv


def _read_one(src_path: str) -> tuple[str, str, Inventory | None]:
    try:
        return (src_path, "ok", _read_sc3ml(src_path))
    except Exception as e:
        return (src_path, f"error: {e.__class__.__name__}: {e}", None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--dst", default=None, help="output dir (default: <src>/stationxml)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src_dir = os.path.abspath(args.src)
    dst_dir = os.path.abspath(args.dst) if args.dst else os.path.join(src_dir, "stationxml")
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(src_dir, "*.xml")))
    files = [f for f in files if os.path.dirname(os.path.abspath(f)) != dst_dir]
    if not files:
        print(f"No *.xml in {src_dir}", file=sys.stderr)
        return 1

    print(f"Reading {len(files)} SC3ML files  src={src_dir}  jobs={args.jobs}")

    # (net_code, sta_code) -> Network(sta_only) — merged across source files
    bucket: dict[tuple[str, str], Network] = {}
    errors: list[tuple[str, str]] = []

    def ingest(inv: Inventory) -> None:
        for net in inv.networks:
            for sta in net.stations:
                key = (net.code, sta.code)
                existing = bucket.get(key)
                if existing is None:
                    # shallow-copy the network without its other stations
                    net_copy = Network(
                        code=net.code,
                        description=net.description,
                        start_date=net.start_date,
                        end_date=net.end_date,
                        restricted_status=net.restricted_status,
                        alternate_code=net.alternate_code,
                        historical_code=net.historical_code,
                        stations=[sta],
                    )
                    bucket[key] = net_copy
                else:
                    existing.stations.append(sta)

    if args.jobs <= 1:
        for f in files:
            src, status, inv = _read_one(f)
            if inv is None:
                errors.append((src, status))
                print(f"  {os.path.basename(src)}: {status}")
                continue
            ingest(inv)
            print(f"  {os.path.basename(src)}: ok  nets={len(inv.networks)} "
                  f"stas={sum(len(n.stations) for n in inv.networks)}")
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_read_one, f): f for f in files}
            for fut in as_completed(futs):
                src, status, inv = fut.result()
                if inv is None:
                    errors.append((src, status))
                    print(f"  {os.path.basename(src)}: {status}")
                    continue
                ingest(inv)
                print(f"  {os.path.basename(src)}: ok  nets={len(inv.networks)} "
                      f"stas={sum(len(n.stations) for n in inv.networks)}")

    print(f"\nUnique stations: {len(bucket)}")

    # Write per-station files
    n_written = 0
    n_skipped = 0
    for (net_code, sta_code), net in sorted(bucket.items()):
        net_dir = os.path.join(dst_dir, net_code)
        os.makedirs(net_dir, exist_ok=True)
        out_path = os.path.join(net_dir, f"{sta_code}.xml")
        if os.path.exists(out_path) and not args.overwrite:
            n_skipped += 1
            continue
        # de-duplicate channels appearing in multiple source files for this station
        seen: set[tuple] = set()
        for sta in net.stations:
            unique_ch = []
            for ch in sta.channels:
                key = (ch.code, ch.location_code, str(ch.start_date), str(ch.end_date))
                if key in seen:
                    continue
                seen.add(key)
                unique_ch.append(ch)
            sta.channels = unique_ch
        inv_out = Inventory(networks=[net], source="sc3ml_to_stationxml")
        inv_out.write(out_path, format="STATIONXML", validate=False)
        n_written += 1

    dupes = {k: len(v.stations) for k, v in bucket.items() if len(v.stations) > 1}

    print(f"Wrote {n_written} StationXML files  (skipped existing: {n_skipped})")
    if dupes:
        print(f"Stations merged from multiple sources: {len(dupes)}")
        for k, n in sorted(dupes.items())[:10]:
            print(f"  {k[0]}.{k[1]}: {n} source stations merged")
    if errors:
        print(f"\nErrors: {len(errors)}")
        for src, msg in errors:
            print(f"  {os.path.basename(src)}: {msg}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
