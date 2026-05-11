# SeisBench-RoSE phase-picker benchmark

Standalone release for three published deep-learning earthquake phase pickers
benchmarked on the **RoSE** (Romanian local-earthquake) and **STEAD** test sets:

| Model | Framework | Description | Tag |
|---|---|---|---|
| **EQT-RoSE-v3**       | PyTorch (SeisBench) | EQTransformer fine-tuned on the RoSE training split | v3 |
| **PhaseNet-RoSE-v2**  | PyTorch (SeisBench) | PhaseNet fine-tuned on the RoSE training split | v2 |
| **RED-PAN-60s**       | TensorFlow / Keras  | RED-PAN multi-task attention R2U-Net (60 s window, TaiwanCWB-trained) | 240107 |

The RED-PAN-60s model lives in a separate Keras pipeline and **is not
SeisBench-compatible**. To keep this release self-contained, a minimal
inference subset of the upstream `redpan` package is bundled under
`redpan_inference/`. The two RoSE-fine-tuned models are stock SeisBench
checkpoints loaded with custom inference-norm settings.

## Directory layout

```
seisbench-rose-benchmark/
├── README.md                       # this file
├── env/
│   ├── requirements.txt            # core deps (numpy, obspy, sklearn, ...)
│   ├── requirements-pytorch.txt    # PyTorch + seisbench (EQT/PhaseNet)
│   └── requirements-tf.txt         # TensorFlow (RED-PAN-60s)
├── models/
│   ├── eqt_rose_v3/eqt_rose_v3.pt          (1.6 MB, PyTorch state dict)
│   ├── phasenet_rose_v2/phasenet_rose_v2.pt (1.1 MB, PyTorch state dict)
│   ├── redpan_tf60/train.hdf5               (6.0 MB, Keras model)
│   └── README.md                            # model cards
├── data/
│   └── README.md                   # download instructions for RoSE + STEAD
├── pickerbench/                    # Python module: matching, scoring, residuals
│   ├── matching.py
│   ├── residual_stats.py
│   ├── trace_io.py
│   └── leaderboard.py
├── redpan_inference/               # bundled minimal RED-PAN inference subset
│   ├── core.py
│   ├── picks.py
│   └── utils.py
├── benchmarks/
│   ├── models.py                   # unified model loader
│   ├── run_benchmark.py            # main entrypoint (RoSE / STEAD / noise)
│   ├── run_stead.py                # STEAD-format inner loop
│   └── build_tables.py             # aggregate JSONs → publication CSVs
├── results/                        # pre-computed CSVs (ground truth)
│   ├── rose_picking.csv
│   ├── rose_detection.csv
│   ├── rose_residual_stats.csv
│   ├── stead_picking.csv
│   └── stead_detection.csv
└── scripts/
    └── reproduce_all.sh            # top-level reproducer
```

Total size: ~9 MB (3 model checkpoints + inference code + results CSVs).

## Quick start

### 1 · Install dependencies

The PyTorch (SeisBench) and TensorFlow (RED-PAN) stacks are best installed
in **separate conda environments** to avoid CUDA/cuDNN conflicts:

```bash
# SeisBench environment (EQT-RoSE-v3, PhaseNet-RoSE-v2)
conda create -n picker-pt python=3.10 -y
conda activate picker-pt
pip install -r env/requirements.txt -r env/requirements-pytorch.txt

# RED-PAN-60s environment (TensorFlow / Keras)
conda create -n picker-tf python=3.10 -y
conda activate picker-tf
pip install -r env/requirements.txt -r env/requirements-tf.txt
```

### 2 · Download test datasets

Follow `data/README.md` to fetch the RoSE SeisBench dataset and the STEAD test
split. Set `ROSE_DIR=/path/to/rose` and `STEAD_DIR=/path/to/STEAD/benchmark_stead`.

### 3 · Reproduce the benchmark

```bash
bash scripts/reproduce_all.sh --rose-dir "$ROSE_DIR" --stead-dir "$STEAD_DIR"
```

This produces `results/rose_{picking,detection}.csv` and `results/stead_{picking,detection}.csv`.
Compare against the pre-computed `results/*.csv` to confirm reproduction (numbers
should match to floating-point precision).

To run a single model on a single dataset:

```bash
python benchmarks/run_benchmark.py \
    --model eqt_rose_v3 --dataset rose --rose-dir "$ROSE_DIR" \
    --out-dir results/runs/eqt_rose_v3_rose
```

## Pre-computed result files

The `results/` directory contains the canonical benchmark numbers reported in
the paper, identical to the upstream development repository's outputs:

| File | What it contains |
|---|---|
| `rose_picking.csv`        | RoSE per-phase TP/FP/FN, precision/recall/F1, MAE, MAD, std, median, RMSE, IQR; 9 models × 6 thresholds = 54 rows. FP counted on STEAD-noise traces only because RoSE labels are incomplete (an unlabelled aftershock would otherwise be miscounted as a false positive). |
| `rose_detection.csv`      | RoSE trace-level detection (event-vs-noise) — TP/FN/FP/TN, precision/recall/F1, Matthews correlation coefficient, AUC, plus detection-box IoU + start/end MAE for the EQT family and RED-PAN-60s. |
| `rose_residual_stats.csv` | Detailed residual statistics at threshold 0.30, 18 rows = 9 models × 2 phases. |
| `stead_picking.csv`       | Same as `rose_picking.csv` but on the full STEAD test pool (103 040 events + 23 526 noise). FP uses canonical pick-benchmark convention (events + noise pooled) because STEAD labels are clean. |
| `stead_detection.csv`     | Same as `rose_detection.csv` but on STEAD. |

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
False positives counted on **dedicated noise traces only** for RoSE
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

For models with detection heads (the EQT family and RED-PAN-60s), the true
window for detection-box IoU follows Mousavi 2020:
`[T_P, T_S + 1.4 · (T_S − T_P)]`.

## Models published with this release

The two RoSE-fine-tuned PyTorch checkpoints (`eqt_rose_v3.pt`, `phasenet_rose_v2.pt`)
contain the model state dict plus a small training-config dict. They are
loaded via SeisBench's `EQTransformer` / `PhaseNet` constructors with
`norm="peak"` and `default_args={"blinding": (200, 200)}` — see
`benchmarks/models.py` for the exact loader.

The `redpan_tf60/train.hdf5` file is the published RED-PAN-60s Keras model
(20240107 retraining checkpoint, ~200 K parameters). It is loaded via
`tf.keras.models.load_model(..., compile=False)` and wrapped with the
sliding-window `REDPAN` class from `redpan_inference/`.

## Citation

If you use these models or benchmarks, please cite:

* This release: *(insert DOI)*
* RED-PAN: Liao et al., 2022 — *RED-PAN: Real-time Earthquake Detection and Phase-picking with multi-task Attention Network*
* SeisBench: Woollam et al., 2022 — *SeisBench — A Toolbox for Machine Learning in Seismology*
* PhaseNet: Zhu & Beroza, 2019
* EQTransformer: Mousavi et al., 2020

## Licence

MIT (see `LICENSE`).
