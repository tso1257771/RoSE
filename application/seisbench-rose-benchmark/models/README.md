# Model cards

Three published pickers, all evaluated on the RoSE test split (Romanian local
earthquakes) and on STEAD. Loaders live in `benchmarks/models.py`
(`load_eqt_rose`, `load_phasenet_rose`, `load_redpan_tf60`).

## Pick a model

| If you want…                                      | Choose          | Has a detection head? | Window     |
|---|---|---|---|
| Best phase-pick F1 on local Romanian earthquakes  | **EQT-RoSE**     | yes                    | 60 s |
| Smaller / faster, picks-only                      | **PhaseNet-RoSE**| no                     | 30 s |
| Cross-region baseline trained on TaiwanCWB only   | **RED-PAN-60s**  | yes                    | 60 s |

| Model | Framework | Pre-training | Fine-tuning | Input | Heads | Checkpoint |
|---|---|---|---|---|---|---|
| **EQT-RoSE**       | PyTorch / SeisBench `EQTransformer` | INSTANCE (Italy)         | RoSE train split          | ZNE, 6000 @ 100 Hz (60 s) | P, S, detection                | `eqt_rose/eqt_rose.pt`           (1.6 MB) |
| **PhaseNet-RoSE**  | PyTorch / SeisBench `PhaseNet`      | INSTANCE (Italy)         | RoSE train split          | ZNE, 3001 @ 100 Hz (30 s) | P, S, Noise                    | `phasenet_rose/phasenet_rose.pt` (1.1 MB) |
| **RED-PAN-60s**    | TensorFlow / Keras                  | —                        | TaiwanCWB (no STEAD/RoSE) | ENZ, 6000 @ 100 Hz (60 s) | P / S / Noise + event mask     | `redpan_tf60/train.hdf5`         (5.8 MB) |

Each PyTorch checkpoint is a dict `{model: state_dict, config: {...}, epoch, dev_loss}`.

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

## RED-PAN-60s (TaiwanCWB-trained)

* **Architecture** — Multi-task attention R2U-Net (Liao et al., 2022): picker
  head (P / S / Noise) and detector head (event mask), ~200 K parameters.
* **Training** — TaiwanCWB catalogue only; no STEAD or RoSE exposure
  (`240107` retraining).
* **Inference** — sliding-window `REDPAN` class (`redpan_inference/core.py`) with
  spectrum-matched noise padding for time alignment; expects ENZ order (the
  loader reorders from ZNE automatically).
* **Checkpoint** — `redpan_tf60/train.hdf5`, full Keras model.
* **Tag** — `240107` (the `train.hdf5` build date).

---

## Loading the models

`benchmarks/models.py` is the canonical loader for each; every model exposes a
SeisBench-style `.classify(stream, ...)`. Streams are passed in ZNE order
(SeisBench convention); RED-PAN-60s reorders to ENZ internally.

```python
from benchmarks.models import (
    load_eqt_rose, load_phasenet_rose, load_redpan_tf60,
)

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
cd application/seisbench-rose-benchmark/models
sha256sum -c SHA256SUMS
# eqt_rose/eqt_rose.pt: OK
# phasenet_rose/phasenet_rose.pt: OK
# redpan_tf60/train.hdf5: OK
```

### Security notes

* The two PyTorch checkpoints are loaded via `benchmarks/models.py`'s
  `_safe_torch_load`, which forces `torch.load(weights_only=True)` — the
  restricted unpickler that only accepts tensors and plain Python scalars.
  (It mirrors `rose.checkpoint_io.safe_torch_load` from the main repo; this
  release is self-contained and deliberately doesn't import the `rose`
  package.) The training scripts emit checkpoints in the form `{"model":
  state_dict, "config": {...}, "epoch": int, "dev_loss": float}`, all of which
  the safe loader accepts; a checkpoint that fails to load under
  `weights_only=True` is one whose provenance you should inspect before
  trusting.
* `redpan_tf60/train.hdf5` is a Keras model. **`tf.keras.models.load_model()`
  executes any Python code embedded in the HDF5** (e.g. Lambda layers, custom
  `call()` implementations). This is intrinsic to the Keras serialization
  format and cannot be mitigated at load time — verify the SHA above and only
  load HDF5s from sources you trust.
