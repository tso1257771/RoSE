# Model cards

Three published pickers, all evaluated on the RoSE test split (Romanian local
earthquakes) and on STEAD. The checkpoints live here (`eqt_rose/`,
`phasenet_rose/`, `redpan_tf60/train.hdf5`); the loaders live in `rose.pickers`
(`load_eqt_rose`, `load_phasenet_rose`, `load_redpan_tf60`, also re-exported as
`rose.load_*`). For the benchmark numbers see `../results/` and `../benchmark/`.

## Pick a model

| If you want…                                            | Choose          | Has a detection head? | Window     |
|---|---|---|---|
| Best phase-pick F1 on the RoSE test set                 | **RED-PAN-60s**  | yes                    | 60 s |
| Best earthquake-detection F1 / SeisBench-native PyTorch | **EQT-RoSE**     | yes                    | 60 s |
| Smaller / faster, picks-only                            | **PhaseNet-RoSE**| no                     | 30 s |

| Model | Framework | Init / warm-start | Training data | Input | Heads | Checkpoint |
|---|---|---|---|---|---|---|
| **EQT-RoSE**       | PyTorch / SeisBench `EQTransformer` | INSTANCE (Italy)            | RoSE train split                     | ZNE, 6000 @ 100 Hz (60 s) | P, S, detection                | `eqt_rose/eqt_rose.pt`           (1.6 MB) |
| **PhaseNet-RoSE**  | PyTorch / SeisBench `PhaseNet`      | INSTANCE (Italy)            | RoSE train split                     | ZNE, 3001 @ 100 Hz (30 s) | P, S, Noise                    | `phasenet_rose/phasenet_rose.pt` (1.1 MB) |
| **RED-PAN-60s**    | TensorFlow / Keras                  | published RED-PAN(60 s)     | Taiwan + STEAD + INSTANCE + **RoSE** | ENZ, 6000 @ 100 Hz (60 s) | P / S / "Others" + event mask  | `redpan_tf60/train.hdf5`         (5.8 MB) |

Each PyTorch checkpoint is a dict `{model: state_dict, config: {...}, epoch, dev_loss}`;
`redpan_tf60/train.hdf5` is a full Keras model.

---

## EQT-RoSE

* **Architecture** — EQTransformer (Mousavi et al., 2020): bidirectional
  attention encoder with three sequence-prediction heads (P, S, detection),
  ~370 K parameters.
* **Pre-training** — `seisbench.models.EQTransformer.from_pretrained("instance")`
  (Italian INSTANCE catalogue).
* **Fine-tuning** — RoSE training split. 30 epochs, Adam (lr 1e-4), 1–45 Hz
  bandpass, per-channel peak normalisation (`norm="peak"`), weighted BCE per head
  (detection 0.05 / P 0.40 / S 0.55), detection-label width factor 0.7.
* **Checkpoint** — `eqt_rose.pt`, selected at epoch 30 (dev loss 0.016).
* **Released** — 2026-04-30.

## PhaseNet-RoSE

* **Architecture** — PhaseNet (Zhu & Beroza, 2019): U-Net with 3-class output
  (P, S, Noise), ~270 K parameters.
* **Pre-training** — `seisbench.models.PhaseNet.from_pretrained("instance")`
  (Italian INSTANCE catalogue).
* **Fine-tuning** — RoSE training split. 30 epochs, Adam (lr 1e-4), per-channel
  peak normalisation (`norm="peak"`), Gaussian label width σ = 10 samples,
  blinding `(200, 200)` samples at inference.
* **Checkpoint** — `phasenet_rose.pt`, selected at epoch 29 (dev loss 0.032).

## RED-PAN-60s

* **Architecture** — multitask attention network with a recurrent-residual
  U-Net (R2U-Net) backbone and two task-specific attention sub-networks (Liao
  et al., 2022), **unchanged**. Input: a Z-score-standardised raw
  three-component seismogram, 6000 samples (60 s @ 100 Hz). Output: a
  softmax-normalised three-channel phase-picking map (P, S, "Others") plus a
  two-channel earthquake-detection mask. ~200 K parameters.
* **Training** — retrained **outside SeisBench** with the original RED-PAN
  recipe (Liao et al., 2022), modified only by (i) **augmenting the training
  set with RoSE** alongside the regional and global catalogues used in the
  original work (Taiwan, STEAD, INSTANCE), and (ii) **warm-starting from the
  published RED-PAN(60 s) weights** rather than from random initialisation.
  Targets follow the original paper: truncated-Gaussian P/S phase-time
  functions (σ = 0.2 s / 0.3 s), `"Others" = 1 − target(P) − target(S)`, and a
  boxcar detection mask wrapping the P–S window with half-Gaussian edge
  roll-offs. Each task is trained with categorical cross-entropy; the two task
  losses are combined by Dynamic Weight Averaging (DWA, temperature H = 2),
  with per-sample category-balancing weights equalising exposure across data
  categories. The original goal-oriented augmentation suite is retained —
  Mosaic Waveform Augmentation (MWA: superimpose 2–4 co-recorded, randomly
  scaled earthquake waveforms in-window), Marching MWA (MMWA:
  backward/centre/forward triplet with the centre always carrying paired P and
  S), and Earthquake-Early-Warning Augmentation (EEWA: only a P arrival near
  the end of the window) — plus random within-window relocation of single-event
  traces and random zero-padding / channel-drop for single-component-style
  sources; both raw and 1–45 Hz band-passed inputs are presented. Distributed
  over 4× NVIDIA V100 (32 GB), per-GPU batch 64 (effective 256), Adam
  (lr 1e-4); early-stopped after 20 consecutive epochs without dev-loss
  improvement, lowest-dev-loss checkpoint kept.
* **Inference** — sliding-window `REDPAN` class (`rose/redpan_inference/core.py`)
  with spectrum-matched noise padding for time alignment; expects ENZ order (the
  loader reorders from ZNE automatically).
* **Checkpoint** — `redpan_tf60/train.hdf5`, a full Keras model (build tag `240107`).

---

## Loading the models

`rose.pickers` has one loader per model (also re-exported from the top-level
`rose` package); each returns a SeisBench-style object with
`.classify(stream, ...)`. Streams are passed in ZNE order (SeisBench
convention); RED-PAN-60s reorders to ENZ internally.

```python
from rose import load_eqt_rose, load_phasenet_rose, load_redpan_tf60

# EQT-RoSE — SeisBench EQTransformer (has a detection head)
model = load_eqt_rose()
out = model.classify(stream, P_threshold=0.3, S_threshold=0.3,
                     detection_threshold=0.3)

# PhaseNet-RoSE — SeisBench PhaseNet (picks only)
model = load_phasenet_rose()
out = model.classify(stream, P_threshold=0.3, S_threshold=0.3)

# RED-PAN-60s — TF/Keras, sliding-window wrapper (has a detection head)
model = load_redpan_tf60()
out = model.classify(stream, P_threshold=0.3, S_threshold=0.3,
                     detection_threshold=0.3)
```

---

## Verifying the bundled weights

A `SHA256SUMS` file lives alongside the checkpoints. Verify your checkout
matches the published artifacts before loading:

```bash
cd models
sha256sum -c SHA256SUMS
# eqt_rose/eqt_rose.pt: OK
# phasenet_rose/phasenet_rose.pt: OK
# redpan_tf60/train.hdf5: OK
```

### Security notes

* The two PyTorch checkpoints are loaded via `rose.checkpoint_io.safe_torch_load`
  (used by `rose.pickers`), which forces `torch.load(weights_only=True)` — the
  restricted unpickler that only accepts tensors and plain Python scalars. The
  training scripts emit checkpoints in the form `{"model": state_dict,
  "config": {...}, "epoch": int, "dev_loss": float}`, all of which the safe
  loader accepts; a checkpoint that fails to load under `weights_only=True` is
  one whose provenance you should inspect before trusting.
* `redpan_tf60/train.hdf5` is a Keras model. **`tf.keras.models.load_model()`
  executes any Python code embedded in the HDF5** (e.g. Lambda layers, custom
  `call()` implementations). This is intrinsic to the Keras serialization
  format and cannot be mitigated at load time — verify the SHA above and only
  load HDF5s from sources you trust.
