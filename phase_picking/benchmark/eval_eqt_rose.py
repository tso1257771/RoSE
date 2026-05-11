"""Evaluate a trained SeisBench EQTransformer on the RoSE/ROMPLUS test split.

Computes the three Münchmeyer 2022 / TRANSFORM² Section-3 tasks so the
output table slots directly next to NKUA's PhaseNet+EQT-on-CRLBD numbers:

    T1 — Event detection      → F1, AUC
    T2 — Phase identification → MCC (P vs S)
    T3 — Onset time picking   → MAE, RMSE for P and S (seconds)

Implementation notes
--------------------
* SeisBench's GenericGenerator + DataLoader collates only X and y; per-row
  metadata is dropped. Pick positions are recovered from the Gaussian
  ``y`` channels produced by ``ProbabilisticLabeller`` (argmax + max-height
  threshold).
* RoSE has only ``trace_category == 'earthquake_local'``. T1 needs
  negatives — synthesised here from the **post-event coda** window
  ``[s_arrival + post_event_margin, +model_window]``. Pure pre-event noise
  is not viable: typical P arrivals sit at sample ~3000, leaving < 6000
  samples of lead-in. Coda windows still contain seismic energy but
  detection probability ought to drop sharply once the event ends; this is
  the closest in-distribution proxy. Use --skip-t1 to suppress.
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

# Allow `from rose.checkpoint_io import safe_torch_load` when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from rose.checkpoint_io import safe_torch_load  # noqa: E402

import numpy as np
import torch
from scipy.signal import find_peaks
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from torch.utils.data import DataLoader

import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm


logger = logging.getLogger("eval_eqt_rose")


PHASE_DICT = {
    "trace_p_arrival_sample": "P",
    "trace_s_arrival_sample": "S",
}


@dataclass
class EvalConfig:
    ckpt: str
    rose_dir: str
    out_dir: str
    batch_size: int = 64
    num_workers: int = 4
    detection_threshold: float = 0.30
    p_threshold: float = 0.10
    s_threshold: float = 0.10
    pick_tolerance_samples: int = 50      # 0.5 s @ 100 Hz
    post_event_margin: int = 200          # 2 s after S before noise window
    min_peak_distance: int = 100          # 1 s spacing for find_peaks
    pick_present_threshold: float = 0.5   # min y peak height to count as "pick in window"
    skip_t1: bool = False
    max_noise: int = 5000


def load_model(ckpt_path: str, device: torch.device):
    state = safe_torch_load(ckpt_path, map_location=device)
    cfg = state["config"]
    model = sbm.EQTransformer(
        in_samples=cfg["model_window"],
        sampling_rate=cfg["sampling_rate"],
        phases=["P", "S"],
        norm="std",
    )
    model.load_state_dict(state["model"])
    model.to(device).eval()
    logger.info(
        "loaded EQT (epoch=%s, dev_loss=%.4f) from %s",
        state.get("epoch"),
        state.get("dev_loss", float("nan")),
        ckpt_path,
    )
    return model, cfg


def event_window_aug(train_cfg: dict) -> list:
    aug: list = []
    if train_cfg.get("bandpass_low") and train_cfg.get("bandpass_high"):
        aug.append(
            sbg.Filter(
                N=4,
                Wn=[train_cfg["bandpass_low"], train_cfg["bandpass_high"]],
                btype="bandpass",
            )
        )
    aug.extend([
        sbg.WindowAroundSample(
            list(PHASE_DICT.keys()),
            samples_before=train_cfg["model_window"] // 2,
            windowlen=train_cfg["model_window"],
            selection="first",
            strategy="pad",
        ),
        sbg.ChangeDtype(np.float32),
        sbg.ProbabilisticLabeller(
            label_columns=PHASE_DICT,
            model_labels=["P", "S", "N"],
            sigma=train_cfg.get("sigma", 20),
            dim=0,
        ),
        sbg.ChangeDtype(np.float32, key="y"),
    ])
    return aug


@torch.no_grad()
def run_event_inference(
    model: sbm.EQTransformer,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns det_probs, p_probs, s_probs (each (N, T)) and y (N, 3, T)."""
    det_all, p_all, s_all, y_all = [], [], [], []
    for batch in loader:
        x = batch["X"].to(device, non_blocking=True)
        det, p, s = model(x)
        det_all.append(det.cpu().numpy())
        p_all.append(p.cpu().numpy())
        s_all.append(s.cpu().numpy())
        y_all.append(batch["y"].cpu().numpy())
    return (
        np.concatenate(det_all, axis=0),
        np.concatenate(p_all, axis=0),
        np.concatenate(s_all, axis=0),
        np.concatenate(y_all, axis=0),
    )


def true_pick_samples(y: np.ndarray, threshold: float
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Recover window-relative true pick samples from labeller output.

    y has shape (N, 3, T) for [P, S, N]. Returns (p_samples, s_samples),
    each shape (N,), with -1 where the pick is not present in the window.
    """
    p_max = y[:, 0, :].max(axis=-1)
    s_max = y[:, 1, :].max(axis=-1)
    p_arg = y[:, 0, :].argmax(axis=-1)
    s_arg = y[:, 1, :].argmax(axis=-1)
    p_samples = np.where(p_max >= threshold, p_arg, -1)
    s_samples = np.where(s_max >= threshold, s_arg, -1)
    return p_samples, s_samples


def extract_peaks(prob: np.ndarray, threshold: float, distance: int
                  ) -> list[int]:
    peaks, props = find_peaks(prob, height=threshold, distance=distance)
    if len(peaks) == 0:
        return []
    order = np.argsort(-props["peak_heights"])
    return [int(peaks[i]) for i in order]


def match_pick(predicted: list[int], true_sample: int, tolerance: int
               ) -> int | None:
    if not predicted or true_sample < 0:
        return None
    diffs = np.abs(np.asarray(predicted) - true_sample)
    best = int(diffs.argmin())
    return predicted[best] if diffs[best] <= tolerance else None


# --------------------- T2 / T3 ---------------------

def compute_t3(
    p_probs: np.ndarray, s_probs: np.ndarray,
    p_true: np.ndarray, s_true: np.ndarray,
    cfg: EvalConfig, sampling_rate: float,
) -> dict:
    p_residuals: list[float] = []
    s_residuals: list[float] = []
    for i in range(p_probs.shape[0]):
        if p_true[i] >= 0:
            peaks = extract_peaks(p_probs[i], cfg.p_threshold,
                                  cfg.min_peak_distance)
            matched = match_pick(peaks, int(p_true[i]),
                                 cfg.pick_tolerance_samples)
            if matched is not None:
                p_residuals.append((matched - p_true[i]) / sampling_rate)
        if s_true[i] >= 0:
            peaks = extract_peaks(s_probs[i], cfg.s_threshold,
                                  cfg.min_peak_distance)
            matched = match_pick(peaks, int(s_true[i]),
                                 cfg.pick_tolerance_samples)
            if matched is not None:
                s_residuals.append((matched - s_true[i]) / sampling_rate)

    def stats(arr: list[float]) -> dict:
        if not arr:
            return {"n": 0, "mae": float("nan"), "rmse": float("nan")}
        a = np.asarray(arr, dtype=np.float64)
        return {
            "n": int(a.size),
            "mae": float(np.mean(np.abs(a))),
            "rmse": float(np.sqrt(np.mean(a ** 2))),
            "median": float(np.median(a)),
            "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
        }

    return {
        "P": stats(p_residuals),
        "S": stats(s_residuals),
        "p_residuals_s": p_residuals,
        "s_residuals_s": s_residuals,
    }


def compute_t2(
    p_probs: np.ndarray, s_probs: np.ndarray,
    p_true: np.ndarray, s_true: np.ndarray,
    cfg: EvalConfig,
) -> dict:
    """Per-pick phase classification.

    NKUA's "10 s window with exactly one pick" formulation is infeasible for
    RoSE since the typical P-S diff is well under 30 s, so any 60 s window
    centred on P also contains S. Equivalent question, evaluated per-pick:
    *at the location of each true pick, which head fires more strongly?*

    For each true P / S sample t, we read the local maxima of p_probs and
    s_probs in [t - tol, t + tol]. The class whose head is higher in that
    window is the predicted phase. MCC is computed across the resulting
    confusion matrix (P vs S).
    """
    y_true: list[int] = []   # 0 = P, 1 = S
    y_pred: list[int] = []
    R = cfg.pick_tolerance_samples
    T = p_probs.shape[1]
    min_thresh = min(cfg.p_threshold, cfg.s_threshold)

    for i in range(p_probs.shape[0]):
        for true_pos, true_class in (
            (int(p_true[i]), 0),
            (int(s_true[i]), 1),
        ):
            if true_pos < 0:
                continue
            lo = max(0, true_pos - R)
            hi = min(T, true_pos + R + 1)
            p_local = float(p_probs[i, lo:hi].max())
            s_local = float(s_probs[i, lo:hi].max())
            if max(p_local, s_local) < min_thresh:
                continue                          # neither head fired
            pred_label = 0 if p_local >= s_local else 1
            y_true.append(true_class)
            y_pred.append(pred_label)

    if not y_true:
        return {"n": 0, "mcc": float("nan")}
    return {
        "n": len(y_true),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "n_p": int(sum(1 for v in y_true if v == 0)),
        "n_s": int(sum(1 for v in y_true if v == 1)),
        "method": "per_pick_local_max",
    }


# --------------------- T1 ---------------------

def build_noise_array(
    rose_dir: str, train_cfg: dict, cfg: EvalConfig,
) -> np.ndarray:
    """Synthesize negatives from post-event coda.

    For traces where (trace_npts - s_arrival) >= post_event_margin +
    model_window, slice [s_arrival + margin : s_arrival + margin + W].
    """
    data = sbd.WaveformDataset(
        path=rose_dir,
        sampling_rate=train_cfg["sampling_rate"],
        component_order=train_cfg["component_order"],
        cache=None,
    )
    test = data.test()
    md = test.metadata.reset_index(drop=True)

    s_col = "trace_s_arrival_sample"
    n_col = "trace_npts"
    if s_col not in md.columns or n_col not in md.columns:
        return np.zeros((0, 3, train_cfg["model_window"]), dtype=np.float32)

    W = train_cfg["model_window"]
    needed = cfg.post_event_margin + W
    s_arr = md[s_col].fillna(-1).astype(float).to_numpy()
    npts = md[n_col].fillna(0).astype(int).to_numpy()
    eligible_idx = np.where((s_arr >= 0) & (npts - s_arr >= needed))[0]
    logger.info(
        "noise eligibility: %d / %d traces have post-S coda >= %d samples",
        len(eligible_idx), len(md), needed,
    )
    if len(eligible_idx) == 0:
        return np.zeros((0, 3, W), dtype=np.float32)
    if cfg.max_noise and len(eligible_idx) > cfg.max_noise:
        rng = np.random.default_rng(42)
        eligible_idx = rng.choice(eligible_idx, size=cfg.max_noise,
                                  replace=False)

    out = np.zeros((len(eligible_idx), 3, W), dtype=np.float32)
    valid = np.zeros(len(eligible_idx), dtype=bool)
    for k, idx in enumerate(eligible_idx):
        try:
            wf, _ = test.get_sample(int(idx))      # (channels, samples)
        except Exception:
            continue
        start = int(s_arr[idx]) + cfg.post_event_margin
        if wf.shape[1] < start + W:
            continue
        out[k] = wf[:, start : start + W].astype(np.float32)
        valid[k] = True
    out = out[valid]
    if out.size and train_cfg.get("bandpass_low") and train_cfg.get("bandpass_high"):
        from scipy.signal import butter, sosfiltfilt
        sos = butter(
            4, [train_cfg["bandpass_low"], train_cfg["bandpass_high"]],
            btype="bandpass", fs=train_cfg["sampling_rate"], output="sos",
        )
        out = sosfiltfilt(sos, out, axis=-1).astype(np.float32)
    logger.info("noise array: %s", out.shape)
    return out


@torch.no_grad()
def run_noise_inference(model, noise_array: np.ndarray, device,
                        cfg: EvalConfig) -> np.ndarray:
    if noise_array.size == 0:
        return np.zeros((0, model.in_samples), dtype=np.float32)
    out = []
    for i in range(0, noise_array.shape[0], cfg.batch_size):
        x = torch.from_numpy(noise_array[i : i + cfg.batch_size]).to(device)
        det, _, _ = model(x)
        out.append(det.cpu().numpy())
    return np.concatenate(out, axis=0)


def compute_t1(det_event: np.ndarray, det_noise: np.ndarray,
               cfg: EvalConfig) -> dict:
    score_event = det_event.max(axis=-1) if det_event.size else np.array([])
    score_noise = det_noise.max(axis=-1) if det_noise.size else np.array([])
    if score_event.size == 0 or score_noise.size == 0:
        return {
            "n_event": int(score_event.size),
            "n_noise": int(score_noise.size),
            "f1": float("nan"),
            "auc": float("nan"),
            "note": "T1 not evaluated: RoSE has no noise traces. "
                    "Post-event coda (within EQT's S+1.4*(S-P) detection-positive "
                    "label range) is not a valid negative; pure pre-event lead-in "
                    "is too short for a 60 s window. Pass --skip-t1 to suppress.",
        }
    y_true = np.concatenate([
        np.ones_like(score_event, dtype=np.int32),
        np.zeros_like(score_noise, dtype=np.int32),
    ])
    scores = np.concatenate([score_event, score_noise])
    auc = float(roc_auc_score(y_true, scores))
    y_pred = (scores >= cfg.detection_threshold).astype(np.int32)
    f1 = float(f1_score(y_true, y_pred))
    return {
        "n_event": int(score_event.size),
        "n_noise": int(score_noise.size),
        "f1": f1,
        "auc": auc,
        "threshold": cfg.detection_threshold,
        "noise_source": "post_event_coda",
        "warning": (
            "AUC may dip below 0.5 because EQT's training label marks "
            "samples up to S + 1.4*(S-P) as event-positive, so the first "
            "half of each coda window is still in-event by definition. "
            "Treat T1 numbers as advisory only; consider --skip-t1."
        ),
    }


def build_event_loader(rose_dir: str, train_cfg: dict, cfg: EvalConfig
                       ) -> DataLoader:
    data = sbd.WaveformDataset(
        path=rose_dir,
        sampling_rate=train_cfg["sampling_rate"],
        component_order=train_cfg["component_order"],
        cache=None,
    )
    test = data.test()
    logger.info("RoSE test split: %d traces", len(test))
    gen = sbg.GenericGenerator(test)
    gen.add_augmentations(event_window_aug(train_cfg))
    return DataLoader(
        gen, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rose-dir", default=os.environ.get("ROSE_DATA_DIR"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--detection-threshold", type=float, default=0.30)
    ap.add_argument("--p-threshold", type=float, default=0.10)
    ap.add_argument("--s-threshold", type=float, default=0.10)
    ap.add_argument("--pick-tolerance-samples", type=int, default=50)
    ap.add_argument("--post-event-margin", type=int, default=200)
    ap.add_argument("--max-noise", type=int, default=5000)
    ap.add_argument("--skip-t1", action="store_true",
                    help="Skip T1; report N/A in metrics.json.")
    args = ap.parse_args()
    if args.rose_dir is None:
        ap.error("--rose-dir is required (or set the ROSE_DATA_DIR environment variable)")

    cfg = EvalConfig(
        ckpt=args.ckpt, rose_dir=args.rose_dir, out_dir=args.out_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
        detection_threshold=args.detection_threshold,
        p_threshold=args.p_threshold, s_threshold=args.s_threshold,
        pick_tolerance_samples=args.pick_tolerance_samples,
        post_event_margin=args.post_event_margin,
        skip_t1=args.skip_t1,
        max_noise=args.max_noise,
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(out_dir / "eval.log")],
        force=True,
    )
    logger.info("eval cfg = %s", json.dumps(asdict(cfg), indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_cfg = load_model(cfg.ckpt, device)

    logger.info("--- event-window inference ---")
    loader = build_event_loader(cfg.rose_dir, train_cfg, cfg)
    det_event, p_event, s_event, y_event = run_event_inference(
        model, loader, device,
    )
    logger.info("event arrays: det=%s p=%s s=%s y=%s",
                det_event.shape, p_event.shape, s_event.shape, y_event.shape)

    p_true, s_true = true_pick_samples(y_event, cfg.pick_present_threshold)
    n_p = int((p_true >= 0).sum())
    n_s = int((s_true >= 0).sum())
    logger.info("true picks in window: P=%d/%d  S=%d/%d",
                n_p, y_event.shape[0], n_s, y_event.shape[0])

    if cfg.skip_t1:
        logger.info("--- T1 skipped (--skip-t1) ---")
        det_noise = np.zeros((0, train_cfg["model_window"]), dtype=np.float32)
    else:
        logger.info("--- post-event noise inference ---")
        noise_arr = build_noise_array(cfg.rose_dir, train_cfg, cfg)
        det_noise = run_noise_inference(model, noise_arr, device, cfg)

    t1 = compute_t1(det_event, det_noise, cfg)
    t2 = compute_t2(p_event, s_event, p_true, s_true, cfg)
    t3 = compute_t3(p_event, s_event, p_true, s_true, cfg,
                    train_cfg["sampling_rate"])

    metrics = {
        "checkpoint": cfg.ckpt,
        "rose_dir": cfg.rose_dir,
        "n_test": int(y_event.shape[0]),
        "T1_event_detection": t1,
        "T2_phase_identification": t2,
        "T3_onset_time_picking": {
            k: v for k, v in t3.items() if not k.endswith("_residuals_s")
        },
    }

    summary_path = out_dir / "metrics.json"
    with summary_path.open("w") as fh:
        json.dump(metrics, fh, indent=2)
    np.savez(
        out_dir / "residuals.npz",
        p_residuals_s=np.asarray(t3["p_residuals_s"]),
        s_residuals_s=np.asarray(t3["s_residuals_s"]),
    )
    logger.info("wrote %s", summary_path)
    logger.info("metrics = %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
