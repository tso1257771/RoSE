#!/usr/bin/env python3
"""Offline re-aggregation of the per-trace prediction dump into the
manual-only (M1) and depth-stratified (M2) picking tables, plus a
pick-tolerance comparison (0.5 vs 1.0 s) for the S phase.

Reads the *.preds.csv files written by bench_pickers_rose.py /
bench_redpan_rose.py with --dump-predictions (each row is one
(trace, true-phase): every same-phase predicted pick as
[residual_s, peak_value], tagged with trace name, pick provenance and
source depth).

Re-applies the same greedy-nearest match as match_picks() at a chosen
(threshold, tolerance), so the OVERALL numbers reproduce the published
Tables 2-3 and the manual-only / depth-stratified breakdowns are exactly
consistent with them. Precision/F1 reuse the STEAD-noise false-positive
counts already computed by bench_noise_fp (pick-count convention, the
same as build_rose_final_benchmark.py). For the manual-only subset those
counts are scaled by the subset's share of the labels, so the ratio of noise
to labels, and with it precision, is the same as in the full table.

    python build_subset_tables.py                 # eval/, threshold 0.3, P 0.5 s, S 1.0 s
    python build_subset_tables.py --eval-dir /path/to/eval --threshold 0.5
    python build_subset_tables.py --out-csv ../results/rose_picking_manual_only.csv
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parent
EVAL = BENCH / "eval"  # --eval-dir

# Model display name -> STEAD-noise tally key (matches MODEL_TO_NOISE_KEY).
NOISE_KEY = {
    "EQT-RoSE": "eqt_rose", "PhaseNet-RoSE": "phasenet_rose",
    "EQT-instance": "eqt_instance", "PhaseNet-instance": "phasenet_instance",
    "EQT-ethz": "eqt_ethz", "PhaseNet-ethz": "phasenet_ethz",
    "EQT-stead": "eqt_stead", "PhaseNet-stead": "phasenet_stead",
    "RED-PAN-60s": "redpan",
}
DISPLAY = {"RED-PAN-60s": "RED-PAN", "EQT-instance": "EQT-INSTANCE",
           "EQT-ethz": "EQT-ETHZ", "EQT-stead": "EQT-STEAD",
           "PhaseNet-instance": "PhaseNet-INSTANCE", "PhaseNet-ethz": "PhaseNet-ETHZ",
           "PhaseNet-stead": "PhaseNet-STEAD"}
DEPTH_BINS = [(0.0, 30.0, "shallow 0-30"), (30.0, 70.0, "mid 30-70"),
              (70.0, 180.0, "deep 70-180")]


def match(preds, thr, tol):
    cand = [(r, pv) for r, pv in preds if pv >= thr]
    if not cand:
        return None
    r, _ = min(cand, key=lambda x: abs(x[0]))
    return r if abs(r) <= tol else None


def load_preds():
    files = (sorted(glob.glob(str(EVAL / "bench_rose_full_sweep" / "*.preds.csv")))
             + sorted(glob.glob(str(EVAL / "bench_redpan_rose_full" / "*.preds.csv"))))
    if not files:
        raise SystemExit(f"No *.preds.csv found under {EVAL}. Run bench_pickers_rose.py "
                         "and bench_redpan_rose.py with --dump-predictions first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["preds"] = df["preds"].apply(json.loads)
    return df


def noise_fp(model, phase, thr):
    p = EVAL / "bench_noise_fp" / f"{NOISE_KEY.get(model, '?')}.json"
    if not p.is_file():
        return None
    blk = json.loads(p.read_text()).get(str(thr), {})
    v = blk.get(f"n_picks_{phase}")
    return int(v) if v is not None else None


def stat(sub, thr, tol, model=None, phase=None, with_fp=False, fp_scale=1.0):
    res = [match(pr, thr, tol) for pr in sub["preds"]]
    tp = [r for r in res if r is not None]
    n, ntp = len(res), len(tp)
    rec = ntp / n if n else float("nan")
    mae = float(np.mean(np.abs(tp))) if tp else float("nan")
    med = float(np.median(tp)) if tp else float("nan")
    out = {"n": n, "tp": ntp, "recall": rec, "mae": mae, "median": med}
    if with_fp and model is not None:
        fp = noise_fp(model, phase, thr)
        if fp is not None:
            fp = fp * fp_scale
            prec = ntp / (ntp + fp) if (ntp + fp) else float("nan")
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
            out.update({"fp_noise": fp, "precision": prec, "f1": f1})
    return out


def order_models(models):
    rank = {m: i for i, m in enumerate(
        ["RED-PAN-60s", "EQT-RoSE", "PhaseNet-RoSE", "EQT-instance",
         "PhaseNet-instance", "EQT-stead", "PhaseNet-stead", "EQT-ethz", "PhaseNet-ethz"])}
    return sorted(models, key=lambda m: rank.get(m, 99))


def main():
    global EVAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=str(EVAL),
                    help="Stage (a) output directory: bench_rose_full_sweep/ and "
                         "bench_redpan_rose_full/ with the *.preds.csv pick files, "
                         "and bench_noise_fp/.")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--tol-p", type=float, default=0.5)
    ap.add_argument("--tol-s", type=float, default=1.0,
                    help="S tolerance. Published Tables 2-3 use P=0.5, S=1.0 (asymmetric). "
                         "The dump stores uncapped residuals, so any tolerance re-aggregates "
                         "validly (verified: S@1.0 reproduces the published S table, S@0.5 does not).")
    ap.add_argument("--out-csv", default=None,
                    help="Also write the manual-only table (every model and phase at "
                         "--threshold) to this CSV.")
    args = ap.parse_args()
    EVAL = Path(args.eval_dir)
    thr, tol_p, tol_s = args.threshold, args.tol_p, args.tol_s
    TOL = {"P": tol_p, "S": tol_s}
    df = load_preds()
    models = order_models(df["model"].unique().tolist())
    def disp(m):
        return DISPLAY.get(m, m)

    for phase in ("P", "S"):
        tl = TOL[phase]
        print(f"\n{'='*92}\n  OVERALL (all provenance) {phase}  thr={thr} tol={tl}  "
              f"[sanity-check vs published Tables 2-3]\n{'='*92}")
        print(f"  {'model':16}{'n':>7}{'TP':>7}{'recall':>8}{'prec':>8}{'F1':>8}{'MAE':>8}")
        for m in models:
            sub = df[(df.model == m) & (df.phase == phase)]
            s = stat(sub, thr, tl, m, phase, with_fp=True)
            print(f"  {disp(m):16}{s['n']:>7}{s['tp']:>7}{s['recall']:>8.3f}"
                  f"{s.get('precision', float('nan')):>8.3f}{s.get('f1', float('nan')):>8.3f}{s['mae']:>8.3f}")

    csv_rows = []
    for phase in ("P", "S"):
        tl = TOL[phase]
        print(f"\n{'='*92}\n  MANUAL-ONLY (RED-PAN-independent) {phase}  thr={thr} tol={tl}  (M1)\n{'='*92}")
        print(f"  {'model':16}{'n_man':>7}{'TP':>7}{'recall':>8}{'prec':>8}{'F1':>8}{'MAE':>8}")
        rows = []
        for m in models:
            all_labels = df[(df.model == m) & (df.phase == phase)]
            sub = all_labels[all_labels.status == "manual"]
            s = stat(sub, thr, tl, m, phase, with_fp=True,
                     fp_scale=len(sub) / len(all_labels) if len(all_labels) else 1.0)
            rows.append((m, s))
            csv_rows.append({
                "model": m, "phase": phase, "threshold": thr, "tolerance_s": tl,
                "n_manual": s["n"], "true_pos": s["tp"], "false_neg": s["n"] - s["tp"],
                "false_pos_on_noise_scaled": round(s.get("fp_noise", float("nan")), 1),
                "precision": round(s.get("precision", float("nan")), 4),
                "recall": round(s["recall"], 4), "F1": round(s.get("f1", float("nan")), 4),
                "mae_seconds": round(s["mae"], 4),
                "median_residual_s": round(s["median"], 4),
            })
            print(f"  {disp(m):16}{s['n']:>7}{s['tp']:>7}{s['recall']:>8.3f}"
                  f"{s.get('precision', float('nan')):>8.3f}{s.get('f1', float('nan')):>8.3f}{s['mae']:>8.3f}")
        best = max(rows, key=lambda r: r[1].get("f1", 0) if not np.isnan(r[1].get("f1", float("nan"))) else -1)
        print(f"  -> best manual-only F1: {disp(best[0])} ({best[1].get('f1', float('nan')):.3f})")

    if args.out_csv:
        pd.DataFrame(csv_rows).to_csv(args.out_csv, index=False)
        print(f"\n  manual-only table written to {args.out_csv}")

    for phase in ("P", "S"):
        tl = TOL[phase]
        print(f"\n{'='*92}\n  DEPTH-STRATIFIED {phase}  thr={thr} tol={tl}  (M2)  recall (MAE) by depth bin\n{'='*92}")
        hdr = "  {:16}".format("model") + "".join(f"{lab:>20}" for *_, lab in DEPTH_BINS)
        print(hdr)
        for m in models:
            cells = []
            for lo, hi, _ in DEPTH_BINS:
                sub = df[(df.model == m) & (df.phase == phase)
                         & (df.depth_km >= lo) & (df.depth_km < hi)]
                s = stat(sub, thr, tl)
                cells.append(f"{s['recall']:.3f}({s['mae']:.3f}) n{s['n']}")
            print("  {:16}".format(disp(m)) + "".join(f"{c:>20}" for c in cells))

    # Item 2: S-phase tolerance comparison (0.5 vs 1.0) to settle the caption.
    print(f"\n{'='*92}\n  TOLERANCE CHECK  S overall  thr={thr}  [published S-table claims 1.0 s]\n{'='*92}")
    print(f"  {'model':16}{'rec@0.5':>9}{'F1@0.5':>9}{'rec@1.0':>9}{'F1@1.0':>9}")
    for m in models:
        sub = df[(df.model == m) & (df.phase == "S")]
        a = stat(sub, thr, 0.5, m, "S", with_fp=True)
        b = stat(sub, thr, 1.0, m, "S", with_fp=True)
        print(f"  {disp(m):16}{a['recall']:>9.3f}{a.get('f1', float('nan')):>9.3f}"
              f"{b['recall']:>9.3f}{b.get('f1', float('nan')):>9.3f}")


if __name__ == "__main__":
    main()
