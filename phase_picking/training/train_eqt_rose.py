"""Fine-tune SeisBench EQTransformer on the RoSE/ROMPLUS dataset.

By default the model is initialised from the INSTANCE pretrained weights
(`--init-weights instance`); pass `--init-weights scratch` for random init.
Train + dev only. Uses the split column injected by build_rose_split_index.py
(salt "ROMPLUS-singleEQ-v1") so the train/dev partition matches RED-PAN.

EQTransformer outputs three sequence-prediction heads:
  - Detection (per-sample event probability)
  - P-pick probability
  - S-pick probability

Loss is a weighted BCE on each head; weights default to the original EQT
paper values (Mousavi et al. 2020): detection=0.05, P=0.40, S=0.55.

Usage:
    export ROSE_DATA_DIR=/path/to/rose          # or pass --rose-dir
    export ROSE_TRAIN_OUT_DIR=checkpoints/eqt   # or pass --out-dir
    python training/train_eqt_rose.py --epochs 30 --batch-size 128 --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import timedelta
from pathlib import Path

# Allow `from rose.checkpoint_io import safe_torch_load` when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from rose.checkpoint_io import safe_torch_load  # noqa: E402

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding


logger = logging.getLogger("train_eqt_rose")


PHASE_DICT = {
    "trace_p_arrival_sample": "P",
    "trace_s_arrival_sample": "S",
}


@dataclass
class Config:
    rose_dir: str
    out_dir: str
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-4
    num_workers: int = 8
    sigma: int = 10
    sampling_rate: int = 100
    component_order: str = "ZNE"
    model_window: int = 6000
    candidate_window: int = 12000
    samples_before: int = 6000
    seed: int = 42
    bandpass_low: float | None = 1.0
    bandpass_high: float | None = 45.0
    loss_w_det: float = 0.05
    loss_w_p: float = 0.40
    loss_w_s: float = 0.55
    detection_factor: float = 1.4
    cache: str | None = None
    smoke_test: int = 0
    # init_weights: "scratch" → fresh random init,
    # otherwise a SeisBench pretrained tag (e.g. "instance", "ethz", "stead").
    init_weights: str = "instance"
    # norm: "peak" matches pick-benchmark and all SeisBench pretrained weights.
    norm: str = "peak"


def build_dataset(cfg: Config) -> tuple[sbd.WaveformDataset, sbd.WaveformDataset]:
    data = sbd.WaveformDataset(
        path=cfg.rose_dir,
        sampling_rate=cfg.sampling_rate,
        component_order=cfg.component_order,
        cache=cfg.cache,
    )
    train, dev, _ = data.train_dev_test()

    if cfg.smoke_test and cfg.smoke_test > 0:
        n_train = min(cfg.smoke_test, len(train))
        n_dev = min(max(int(cfg.smoke_test * 0.1), 200), len(dev))
        rng = np.random.default_rng(cfg.seed)
        train_keep = np.zeros(len(train), dtype=bool)
        train_keep[rng.choice(len(train), size=n_train, replace=False)] = True
        dev_keep = np.zeros(len(dev), dtype=bool)
        dev_keep[rng.choice(len(dev), size=n_dev, replace=False)] = True
        train.filter(train_keep, inplace=True)
        dev.filter(dev_keep, inplace=True)

    logger.info("RoSE loaded: train=%d dev=%d (cache=%s smoke=%d)",
                len(train), len(dev), cfg.cache, cfg.smoke_test)
    return train, dev


def build_augmentations(cfg: Config, model: sbm.EQTransformer) -> list:
    """Augmentation pipeline mirrors seisbench/pick-benchmark for EQT.

    Order: bandpass → window → random window → ChangeDtype → Normalize →
    ProbabilisticLabeller → DetectionLabeller. Normalize uses
    amp_norm_type='peak' (per-channel max-amplitude scaling) to match the
    SeisBench pretrained convention; std-normalisation drifts at random
    crops where window content varies between epochs.
    """
    aug: list = []
    if cfg.bandpass_low is not None and cfg.bandpass_high is not None:
        aug.append(
            sbg.Filter(
                N=4,
                Wn=[cfg.bandpass_low, cfg.bandpass_high],
                btype="bandpass",
            )
        )
    aug.extend([
        sbg.WindowAroundSample(
            list(PHASE_DICT.keys()),
            samples_before=cfg.samples_before,
            windowlen=cfg.candidate_window,
            selection="random",
            strategy="variable",
        ),
        sbg.RandomWindow(windowlen=cfg.model_window, strategy="pad"),
        sbg.ChangeDtype(np.float32),
        sbg.Normalize(
            detrend_axis=-1,
            amp_norm_axis=-1,
            amp_norm_type="peak",
        ),
        sbg.ProbabilisticLabeller(
            label_columns=PHASE_DICT,
            model_labels=["P", "S", "N"],
            sigma=cfg.sigma,
            dim=0,
        ),
        sbg.DetectionLabeller(
            p_phases="trace_p_arrival_sample",
            s_phases="trace_s_arrival_sample",
            factor=cfg.detection_factor,
            key=("X", "detections"),
        ),
        sbg.ChangeDtype(np.float32, key="y"),
        sbg.ChangeDtype(np.float32, key="detections"),
    ])
    return aug


def eqt_loss(
    det_pred: torch.Tensor, p_pred: torch.Tensor, s_pred: torch.Tensor,
    det_true: torch.Tensor, p_true: torch.Tensor, s_true: torch.Tensor,
    w_det: float, w_p: float, w_s: float,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Weighted binary cross-entropy across the three EQT heads.

    All `*_pred` tensors are model sigmoid outputs in (0, 1).
    All `*_true` tensors are Gaussian / boxcar targets in [0, 1].
    """
    def bce(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        p = p.clamp(eps, 1.0 - eps)
        return -(t * torch.log(p) + (1.0 - t) * torch.log(1.0 - p)).mean()

    return (
        w_det * bce(det_pred, det_true)
        + w_p * bce(p_pred, p_true)
        + w_s * bce(s_pred, s_true)
    )


def setup_distributed() -> tuple[bool, int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank < 0:
        return False, 0, 1, 0
    dist.init_process_group(backend="nccl",
                            timeout=timedelta(minutes=60))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def reduce_loss(loss: torch.Tensor, world_size: int) -> float:
    if world_size > 1:
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss = loss / world_size
    return float(loss.detach())


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def run_epoch(model, loader, optimizer, device, cfg: Config,
              *, train: bool, world_size: int) -> float:
    model.train(train)
    total_loss = torch.zeros((), device=device)
    n_batches = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            x = batch["X"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)            # (B, 3, T) P/S/N
            d = batch["detections"].to(device, non_blocking=True)   # (B, 1, T)

            det_pred, p_pred, s_pred = model(x)  # each (B, T)

            det_true = d.squeeze(1)
            p_true = y[:, 0, :]
            s_true = y[:, 1, :]

            loss = eqt_loss(
                det_pred, p_pred, s_pred,
                det_true, p_true, s_true,
                cfg.loss_w_det, cfg.loss_w_p, cfg.loss_w_s,
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.detach()
            n_batches += 1
    avg = total_loss / max(1, n_batches)
    return reduce_loss(avg, world_size)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rose-dir",
                    default=os.environ.get("ROSE_DATA_DIR"))
    ap.add_argument("--out-dir",
                    default=os.environ.get("ROSE_TRAIN_OUT_DIR", "checkpoints/eqt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--sigma", type=int, default=20)
    ap.add_argument("--bandpass-low", type=float, default=1.0,
                    help="Set to negative to disable.")
    ap.add_argument("--bandpass-high", type=float, default=45.0)
    ap.add_argument("--w-det", type=float, default=0.05)
    ap.add_argument("--w-p", type=float, default=0.40)
    ap.add_argument("--w-s", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--cache", type=str, default="none",
                    choices=["none", "trace", "full"],
                    help="WaveformDataset cache mode (default: none, RAM-safe).")
    ap.add_argument("--smoke-test", type=int, default=0,
                    help="If > 0, subsample N train rows for a fast sanity run.")
    ap.add_argument("--init-weights", type=str, default="instance",
                    help="SeisBench pretrained weight tag (e.g. 'instance', 'ethz', "
                         "'stead') or 'scratch' for random init.")
    ap.add_argument("--norm", type=str, default="peak",
                    choices=["peak", "std"],
                    help="Model normalization mode (only used when init=scratch).")
    ap.add_argument("--detection-factor", type=float, default=1.4,
                    help="DetectionLabeller factor: labelled event box runs from "
                         "P to S + factor*(S - P). Default 1.4 (Mousavi 2020 / "
                         "INSTANCE-style). Use 0.7 for RoSE local-event coda "
                         "physics (avoids the long-tail detection overfit "
                         "diagnosed in v2).")
    args = ap.parse_args()
    if args.rose_dir is None:
        ap.error("--rose-dir is required (or set the ROSE_DATA_DIR environment variable)")

    cfg = Config(
        rose_dir=args.rose_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        sigma=args.sigma,
        seed=args.seed,
        bandpass_low=(None if args.bandpass_low is not None
                      and args.bandpass_low < 0 else args.bandpass_low),
        bandpass_high=(None if args.bandpass_high is not None
                       and args.bandpass_high < 0 else args.bandpass_high),
        loss_w_det=args.w_det,
        loss_w_p=args.w_p,
        loss_w_s=args.w_s,
        cache=None if args.cache == "none" else args.cache,
        smoke_test=args.smoke_test,
        init_weights=args.init_weights,
        norm=args.norm,
        detection_factor=args.detection_factor,
    )

    is_distributed, rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    out_dir = Path(cfg.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    log_level = logging.INFO if is_main else logging.WARNING
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if is_main:
        handlers.append(logging.FileHandler(out_dir / "train.log"))
    logging.basicConfig(
        level=log_level,
        format=f"%(asctime)s %(levelname)s [r{rank}] %(name)s :: %(message)s",
        handlers=handlers,
        force=True,
    )
    if is_main:
        logger.info("world_size=%d  distributed=%s", world_size, is_distributed)
        logger.info("config = %s", json.dumps(asdict(cfg), indent=2))

    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)

    train_ds, dev_ds = build_dataset(cfg)

    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if cfg.init_weights and cfg.init_weights != "scratch":
        if is_main:
            logger.info("loading pretrained EQT weights '%s' (transfer learning)",
                        cfg.init_weights)
        # wait_for_file=True: under DDP, ranks > 0 wait for the rank that
        # is actually writing the cache file instead of erroring on the
        # `.partial` lock. Pre-fetch in the launcher avoids the race
        # entirely; this is the belt-and-suspenders.
        model = sbm.EQTransformer.from_pretrained(
            cfg.init_weights, wait_for_file=True,
        )
        if is_main:
            logger.info("pretrained config: in_samples=%d sr=%s norm=%s",
                        model.in_samples, model.sampling_rate, model.norm)
    else:
        if is_main:
            logger.info("training EQT from scratch (no pretrained init)")
        model = sbm.EQTransformer(
            in_samples=cfg.model_window,
            sampling_rate=cfg.sampling_rate,
            phases=["P", "S"],
            norm=cfg.norm,
        )
    model = model.to(device)

    if args.resume:
        if is_main:
            logger.info("resuming from %s", args.resume)
        state = safe_torch_load(args.resume, map_location=device)
        unwrap(model).load_state_dict(
            state["model"] if "model" in state else state
        )

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    aug = build_augmentations(cfg, unwrap(model))
    train_gen = sbg.GenericGenerator(train_ds)
    train_gen.add_augmentations(aug)
    dev_gen = sbg.GenericGenerator(dev_ds)
    dev_gen.add_augmentations(aug)

    train_sampler = (
        DistributedSampler(train_gen, shuffle=True, drop_last=True,
                           seed=cfg.seed)
        if is_distributed else None
    )
    dev_sampler = (
        DistributedSampler(dev_gen, shuffle=False, drop_last=False)
        if is_distributed else None
    )

    train_loader = DataLoader(
        train_gen,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        worker_init_fn=worker_seeding,
        pin_memory=False,
        drop_last=True,
        persistent_workers=False,
    )
    dev_loader = DataLoader(
        dev_gen,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=dev_sampler,
        num_workers=cfg.num_workers,
        worker_init_fn=worker_seeding,
        pin_memory=False,
        persistent_workers=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history: list[dict] = []
    best_dev = float("inf")
    best_path = out_dir / "eqt_rose_best.pt"
    last_path = out_dir / "eqt_rose_last.pt"

    for epoch in range(1, cfg.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, device, cfg,
                               train=True, world_size=world_size)
        dev_loss = run_epoch(model, dev_loader, optimizer, device, cfg,
                             train=False, world_size=world_size)
        dt = time.time() - t0

        if is_main:
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_loss": dev_loss,
                "secs": dt,
            }
            history.append(row)
            logger.info("epoch %02d  train=%.4f  dev=%.4f  (%.1fs)",
                        epoch, train_loss, dev_loss, dt)

            torch.save(
                {"model": unwrap(model).state_dict(),
                 "config": asdict(cfg),
                 "epoch": epoch, "dev_loss": dev_loss},
                last_path,
            )
            if dev_loss < best_dev:
                best_dev = dev_loss
                torch.save(
                    {"model": unwrap(model).state_dict(),
                     "config": asdict(cfg),
                     "epoch": epoch, "dev_loss": dev_loss},
                    best_path,
                )
                logger.info("  ↳ new best dev=%.4f → %s", dev_loss, best_path)

            with (out_dir / "history.json").open("w") as fh:
                json.dump(history, fh, indent=2)

    if is_main:
        logger.info("training done. best dev_loss=%.4f at %s",
                    best_dev, best_path)
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
