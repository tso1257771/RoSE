# SeisBench-RoSE phase-picker benchmark

> **Not the same as the repo's top-level `benchmark/` directory** (the names
> are similar; the directories are not). This — `application/seisbench-rose-benchmark/`
> — is the **self-contained release bundle**: the 3 bundled checkpoints +
> `SHA256SUMS`, unified loaders, **pre-computed** `results/*.csv`, pinned
> test-index CSVs, and a quick `scripts/reproduce_all.sh` sanity-re-scorer. It
> imports **nothing** from the rest of the RoSE repo, so it can be distributed
> on its own. The repo's `benchmark/` directory is the full *internal* pipeline
> (all 9 pickers, the RoSE dataset, the `rose` package) and is what actually
> **produces** the `results/*.csv` shipped here — `bash benchmark/regenerate_results.sh`;
> see [`benchmark/README.md`](../../benchmark/README.md).

Self-contained release of three deep-learning earthquake phase pickers
benchmarked on the **RoSE** (Romanian local-earthquake) and **STEAD** test
sets — bundled checkpoints, unified loaders, scoring code, and the
pre-computed result tables reported in the paper.

| Model              | Framework          | What it does                                                                                       |
|---|---|---|
| **EQT-RoSE**       | PyTorch (SeisBench) | EQTransformer fine-tuned on the RoSE training split (released 2026-04-30)                          |
| **PhaseNet-RoSE**  | PyTorch (SeisBench) | PhaseNet fine-tuned on the RoSE training split                                                     |
| **RED-PAN-60s**    | TensorFlow / Keras  | RED-PAN multitask attention (R2U-Net), 60 s window — retrained outside SeisBench (Liao et al. 2022 recipe) on Taiwan + STEAD + INSTANCE + **RoSE**, warm-started from the published RED-PAN(60 s) weights |

Per-model details (architecture, pretraining, fine-tuning recipe, dev loss):
see [`models/README.md`](models/README.md).

---

## Use the models in five lines

```python
from benchmarks.models import load_eqt_rose, load_phasenet_rose, load_redpan_tf60

model = load_eqt_rose()                                              # 1.6 MB checkpoint
out   = model.classify(stream,                                       # ObsPy Stream, ZNE order
                       P_threshold=0.3, S_threshold=0.3,
                       detection_threshold=0.3)
```

All three loaders return a SeisBench-style object with `.classify(stream, ...)`.
RED-PAN-60s reorders ZNE → ENZ internally. PyTorch checkpoints are loaded
with `weights_only=True` (safe pickle); see the security note in
[`models/README.md#verifying-the-bundled-weights`](models/README.md#verifying-the-bundled-weights).

---

## Reproduce the paper

```bash
# 1. install (separate envs for the two stacks — CUDA/cuDNN don't co-exist cleanly)
conda create -n picker-pt python=3.10 -y && conda activate picker-pt
pip install -r env/requirements.txt -r env/requirements-pytorch.txt
#                  (or for RED-PAN: -r env/requirements-tf.txt in a sibling env)

# 2. fetch the test data — see data/README.md
export ROSE_DIR=/path/to/rose                            # SeisBench format
export STEAD_DIR=/path/to/STEAD/benchmark_stead          # flat .npy layout

# 3. run the three bundled RoSE checkpoints on both pools
bash scripts/reproduce_all.sh --rose-dir "$ROSE_DIR" --stead-dir "$STEAD_DIR"
```

This writes `results/runs/{rose,stead}_{picking,detection}.csv` — an
**independent re-score** of the three bundled checkpoints using this release's
own self-contained `pickerbench/` code. It is **not** a bit-for-bit copy of
the committed `results/*.csv`: those were produced by the main repo's full
benchmark suite (`benchmark/build_*`), which uses a richer schema, a separate
dedicated-noise pass, and the "FP on dedicated-noise traces only" convention
for the RoSE pool (RoSE event labels are incomplete, so unmatched picks on
event traces are kept only as a diagnostic, not counted as FP — `pickerbench`
counts them, so its RoSE precision reads lower; and without the separate noise
pass the RoSE detection precision/F1 here are `nan`). Use `reproduce_all.sh`
to confirm the checkpoints load and pick sanely and to get ballpark numbers —
the STEAD rows track the committed values closely at full scale, the RoSE
precision/FP figures differ by construction. The committed CSVs also carry the
off-the-shelf `instance` / `ethz` / `stead` baselines and a
`*_residual_stats.csv`, which `reproduce_all.sh` doesn't re-run. It writes only
under `results/runs/` and never overwrites the committed reference files.

To run a single model on a single pool:

```bash
python benchmarks/run_benchmark.py --model eqt_rose --dataset rose \
    --rose-dir "$ROSE_DIR" --out-dir results/runs/eqt_rose_rose
```

---

## Verify the bundled weights

```bash
cd models && sha256sum -c SHA256SUMS
# eqt_rose/eqt_rose.pt: OK
# phasenet_rose/phasenet_rose.pt: OK
# redpan_tf60/train.hdf5: OK
```

---

## Layout

```
seisbench-rose-benchmark/
├── README.md                       # this file
├── env/                            # per-framework requirement files
│   ├── requirements.txt            #   core (numpy, obspy, sklearn, …)
│   ├── requirements-pytorch.txt    #   PyTorch + seisbench (EQT/PhaseNet)
│   └── requirements-tf.txt         #   TensorFlow (RED-PAN-60s)
├── models/                         # checkpoints + SHA256SUMS + model cards
│   ├── eqt_rose/eqt_rose.pt                  (1.6 MB, PyTorch state dict)
│   ├── phasenet_rose/phasenet_rose.pt        (1.1 MB, PyTorch state dict)
│   ├── redpan_tf60/train.hdf5                (5.8 MB, Keras model)
│   ├── SHA256SUMS                            (verify with `sha256sum -c`)
│   └── README.md                             (per-model cards)
├── data/                           # test-set index files + waveform-download notes
│   ├── rose_test_index.csv         (32 374 rows, the RoSE test split)
│   ├── stead_test_index.csv        (103 040 rows, STEAD events)
│   ├── stead_noise_index.csv       (23 526 rows, STEAD noise)
│   └── README.md
├── benchmarks/                     # unified loaders + runners
│   ├── models.py                   #   load_eqt_rose, load_phasenet_rose, load_redpan_tf60
│   ├── run_benchmark.py            #   one model × one pool
│   ├── run_stead.py                #   STEAD inner loop
│   └── build_tables.py             #   aggregate JSONs → publication CSVs
├── pickerbench/                    # scoring (pure-Python module)
│   ├── matching.py  residual_stats.py  trace_io.py  leaderboard.py
├── redpan_inference/               # bundled minimal RED-PAN inference subset
│   ├── core.py  picks.py  utils.py
├── results/                        # canonical numbers (rose_*.csv, stead_*.csv)
└── scripts/reproduce_all.sh        # one-shot reproducer
```

Total size: ~16 MB (3 model checkpoints + index CSVs + inference code +
results CSVs).

---

## Pre-computed result files

The `results/` directory contains the canonical benchmark numbers reported in
the paper:

| File | What it contains |
|---|---|
| `rose_picking.csv`        | RoSE per-phase TP/FP/FN, precision/recall/F1, MAE, MAD, std, median, RMSE, IQR; 9 models × 6 thresholds = 54 rows. **FP counted on STEAD-noise traces only** because RoSE labels are incomplete (an unlabelled aftershock would otherwise be miscounted as a false positive). |
| `rose_detection.csv`      | RoSE trace-level event-vs-noise TP/FN/FP/TN, precision/recall/F1, Matthews correlation, AUC, plus detection-box IoU and start/end MAE for the EQT family and RED-PAN-60s. |
| `rose_residual_stats.csv` | Detailed residual statistics at threshold 0.30 — 18 rows = 9 models × 2 phases. |
| `stead_picking.csv`       | Same shape as `rose_picking.csv` but on the full STEAD test pool (103 040 events + 23 526 noise). FP uses canonical pick-benchmark convention (events + noise pooled) because STEAD labels are clean. |
| `stead_detection.csv`     | Same shape as `rose_detection.csv` but on STEAD. |

Headline numbers and threshold-sweep details: [`results/README.md`](results/README.md).

---

## Confusion-matrix conventions

### Phase picking (per-phase, TP/FP/FN)

```
                                  Predicted pick of this phase exists
                                  within the time tolerance?
                              ┌─────────────────────┬─────────────────────┐
On an EVENT trace, the        │  TRUE POSITIVE      │  FALSE NEGATIVE     │
labelled pick of this phase   │  (recovered)        │  (missed)           │
exists                        │                     │                     │
                              ├─────────────────────┼─────────────────────┤
On a NOISE trace, no          │  FALSE POSITIVE     │  TRUE NEGATIVE      │
labelled pick exists          │  (hallucination)    │  (silence; correct) │
                              └─────────────────────┴─────────────────────┘
```

Tolerance: P = 0.5 s, S = 1.0 s (RED-PAN paper convention).
False positives are counted on **dedicated noise traces only** for RoSE
(label-incompleteness) and on the full pool (events + noise) for STEAD.

### Earthquake detection (trace-level, event-vs-noise)

```
                                  Did the model emit any pick at threshold T?
                              ┌─────────────────────┬─────────────────────┐
This trace is an EVENT        │  TRUE POSITIVE      │  FALSE NEGATIVE     │
                              ├─────────────────────┼─────────────────────┤
This trace is NOISE           │  FALSE POSITIVE     │  TRUE NEGATIVE      │
                              └─────────────────────┴─────────────────────┘
```

For models with detection heads (the EQT family and RED-PAN-60s) the true
window for detection-box IoU follows Mousavi 2020:
`[T_P, T_S + 1.4 · (T_S − T_P)]`.

---

## Citation

* This release: *(insert DOI)*
* RED-PAN: Liao et al., 2022 — *RED-PAN: Real-time Earthquake Detection and Phase-picking with multi-task Attention Network*
* SeisBench: Woollam et al., 2022 — *SeisBench — A Toolbox for Machine Learning in Seismology*
* PhaseNet: Zhu & Beroza, 2019
* EQTransformer: Mousavi et al., 2020

## Licence

MIT (see `LICENSE`).
