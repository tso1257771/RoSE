# `phase_picking/` — phase pickers + detection on RoSE

An **extension of the RoSE dataset**: three published deep-learning phase
pickers, the recipe that trained two of them, and the benchmark that scores all
nine. Everything here builds on the `rose` package (the dataset API) and the
SeisBench model API — it is *not* part of the dataset itself.

```
phase_picking/
├── models/      ── the 3 published checkpoints + SHA256SUMS + per-model cards
│                   (loaded via `from rose import load_eqt_rose / load_phasenet_rose / load_redpan_tf60`)
├── results/     ── the pre-computed benchmark CSVs (the published numbers) + README
├── training/    ── fine-tune SeisBench EQT / PhaseNet on the RoSE training split
│                   (build_rose_split_index.py · train_eqt_rose.py · train_phasenet_rose.py · cloud/)
└── benchmark/   ── the benchmark pipeline: run_inference.py (a) + build_leaderboard.py (b)
                    + config.json + the bench_*/build_* stage scripts + data/ (pinned test indices)
                    + the standalone tools (eval_eqt_rose.py · bench_joint_rose.py · viz_models_rose.py)
```

- **Use a picker** on your own data → `from rose import load_eqt_rose, ...`
  (see `models/README.md`); a runnable demo is `examples/04_picker_inference.py`.
- **Re-train** EQT / PhaseNet on RoSE → `training/` (see the top-level README's
  "Training & benchmarking" section).
- **Re-run the benchmark** → `bash benchmark/regenerate_results.sh` (needs the
  RoSE + STEAD test datasets — see `benchmark/README.md` and `benchmark/data/README.md`).

The three pickers: **EQT-RoSE** / **PhaseNet-RoSE** (SeisBench EQTransformer /
PhaseNet fine-tuned on the RoSE training split from INSTANCE init) and
**RED-PAN-60s** (RED-PAN MTAN R2U-Net retrained on Taiwan + STEAD + INSTANCE +
RoSE, warm-started from the published RED-PAN(60 s) weights). Cards + references:
`models/README.md`; the leaderboard tables: `results/README.md`.
