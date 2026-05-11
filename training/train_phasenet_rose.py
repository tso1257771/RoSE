"""Fine-tune SeisBench PhaseNet on the RoSE/ROMPLUS dataset.

By default the model is initialised from the INSTANCE pretrained weights
(`--init-weights instance`); pass `--init-weights scratch` for random init.
Train + dev only (no test usage during training). The split column was
injected by build_rose_split_index.py using RED-PAN's hash_split with salt
"ROMPLUS-singleEQ-v1", so this matches any RED-PAN checkpoint trained on
the same corpus.

Usage:
    export ROSE_DATA_DIR=/path/to/rose                  # or pass --rose-dir
    export ROSE_TRAIN_OUT_DIR=checkpoints/phasenet      # or pass --out-dir
    python training/train_phasenet_rose.py --epochs 30 --batch-size 256 --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from datetime import timedelta
from pathlib import Path

# Allow `from rose.checkpoint_io import safe_torch_load` when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
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


logger = logging.getLogger("train_phasenet_rose")


# RoSE has trace_p_arrival_sample / trace_s_arrival_sample only.
PHASE_DICT = {
    "trace_p_arrival_sample": "P",
    "trace_s_arrival_sample": "S",
}


@dataclass
class Config:
    rose_dir: str
    out_dir: str
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-4
    num_workers: int = 8
    sigma: int = 10
    sampling_rate: int = 100
    component_order: str = "ZNE"
    candidate_window: int = 6000
    samples_before: int = 3000
    model_window: int = 3001
    seed: int = 42
    bandpass_low: float | None = None
    bandpass_high: float | None = None
    # "full" caches all waveforms in RAM (fast but memory-hungry: ~37 GB
    # for full RoSE). "trace" caches per-trace lazily. None reads HDF5 on
    # every access (slowest, lowest RAM). Default None for safety.
    cache: str | None = None
    # When > 0, randomly subsample at most this many train rows and
    # max(0.1*N, 200) dev rows for a fast pipeline smoke test.
    smoke_test: int = 0
    init_weights: str = "instance"
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


def build_augmentations(cfg: Config, model: sbm.PhaseNet) -> list:
    """Augmentation pipeline mirrors seisbench/pick-benchmark for PhaseNet.

    Order: (bandpass) → window → random window → ChangeDtype → Normalize →
    ProbabilisticLabeller. Normalize uses amp_norm_type='peak' to match
    the SeisBench pretrained convention; std-normalisation drifts at
    random crops where window content varies between epochs.
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
            demean_axis=-1,
            amp_norm_axis=-1,
            amp_norm_type="peak",
        ),
        sbg.ProbabilisticLabeller(
            label_columns=PHASE_DICT,
            model_labels=model.labels,
            sigma=cfg.sigma,
            dim=0,
        ),
        sbg.ChangeDtype(np.float32, key="y"),
    ])
    return aug


def vector_cross_entropy(y_pred: torch.Tensor, y_true: torch.Tensor,
                         eps: float = 1e-5) -> torch.Tensor:
    """Per Münchmeyer et al. 2022 / SeisBench 03a tutorial."""
    h = y_true * torch.log(y_pred + eps)
    h = h.mean(-1).sum(-1)
    return -h.mean()


def setup_distributed() -> tuple[bool, int, int, int]:
    """Initialise the process group when launched via torchrun.

    Returns (is_distributed, rank, world_size, local_rank). When not
    launched under torchrun all values default to single-process.
    """
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


def run_epoch(model, loader, optimizer, device, *, train: bool,
              world_size: int) -> float:
    model.train(train)
    total_loss = torch.zeros((), device=device)
    n_batches = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            x = batch["X"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            y_pred = model(x)
            loss = vector_cross_entropy(y_pred, y)
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
                    default=os.environ.get("ROSE_TRAIN_OUT_DIR"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--sigma", type=int, default=30,
                    help="ProbabilisticLabeller Gaussian sigma (samples).")
    ap.add_argument("--bandpass-low", type=float, default=None,
                    help="Optional pre-augment bandpass low (Hz). e.g. 0.1")
    ap.add_argument("--bandpass-high", type=float, default=None,
                    help="Optional pre-augment bandpass high (Hz). e.g. 45")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to a .pt to resume from.")
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
    args = ap.parse_args()

    cfg = Config(
        rose_dir=args.rose_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        sigma=args.sigma,
        seed=args.seed,
        bandpass_low=args.bandpass_low,
        bandpass_high=args.bandpass_high,
        cache=None if args.cache == "none" else args.cache,
        smoke_test=args.smoke_test,
        init_weights=args.init_weights,
        norm=args.norm,
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
            logger.info("loading pretrained PhaseNet weights '%s' (transfer learning)",
                        cfg.init_weights)
        # wait_for_file=True: under DDP, ranks > 0 wait for the rank that
        # is actually writing the cache file instead of erroring on the
        # `.partial` lock. Pre-fetch in the launcher avoids the race
        # entirely; this is the belt-and-suspenders.
        model = sbm.PhaseNet.from_pretrained(
            cfg.init_weights, wait_for_file=True,
        )
        if is_main:
            logger.info("pretrained config: norm=%s labels=%s",
                        model.norm, model.labels)
    else:
        if is_main:
            logger.info("training PhaseNet from scratch (no pretrained init)")
        model = sbm.PhaseNet(
            phases="PSN",
            norm=cfg.norm,
            default_args={"blinding": (200, 200)},
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
    best_path = out_dir / "phasenet_rose_best.pt"
    last_path = out_dir / "phasenet_rose_last.pt"

    for epoch in range(1, cfg.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, device,
                               train=True, world_size=world_size)
        dev_loss = run_epoch(model, dev_loader, optimizer, device,
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
