# SeisBench compatibility

Bridge that turns the RoSE-fine-tuned PhaseNet and EQTransformer checkpoints
into the artifact layout SeisBench's `from_pretrained` expects.

```
seisbench_compat/
├── convert.py        # RoSE .pt  → SeisBench (.pt.v1, .json.v1) pair
├── smoke_test.py     # Stage into temp cache, call from_pretrained, diff state_dicts
└── weights/          # Generated artifacts (gitignored)
    ├── eqtransformer/
    │   ├── rose.pt.v1     # state_dict only (torch.save)
    │   └── rose.json.v1   # docstring, model_args, default_args, version
    └── phasenet/
        ├── rose.pt.v1
        └── rose.json.v1
```

Both pairs round-trip bit-identically against
`rose.pickers.load_eqt_rose` / `load_phasenet_rose` — that is what
`smoke_test.py` checks (249/111 state_dict tensors equal, forward pass
shape OK).

## Regenerating

```bash
PYTHONPATH=. python seisbench_compat/convert.py            # writes weights/
PYTHONPATH=. python seisbench_compat/smoke_test.py         # verifies round-trip
sha256sum seisbench_compat/weights/*/rose.pt.v1            # capture hashes
```

## Submitting to SeisBench upstream

Weights-only contributions need **no Python change in `seisbench/seisbench`** —
SeisBench discovers weight tags by listing the remote WebDAV root at
`https://hifis-storage.desy.de/.../SeisBench/models/v3/<model>/`. The
maintainers host that server, so the flow is:

1. Open an issue on <https://github.com/seisbench/seisbench/issues> with
   label `model extension`. Suggested title: *"Add RoSE-fine-tuned weights
   for PhaseNet and EQTransformer"*. Include:
   - Dataset paragraph (RoSE, 80/10/10 split, Taiwan CWA stations, 2024-04-03
     Hualien aftershocks, etc.) and training recipe (warm start from
     INSTANCE, optimizer, sigma, bandpass, loss weights — same wording as
     the JSON `docstring` field).
   - Zenodo DOI: `10.5281/zenodo.20250670`.
   - SHA-256 of each `.pt.v1` so the maintainers can verify the upload.
2. Attach the four files (both `.pt.v1` and `.json.v1` are <2 MB; well
   under GitHub's 25 MB cap).
3. Maintainers upload to the HIFIS WebDAV server. Once visible, anyone on
   SeisBench ≥0.7 can call
   `seisbench.models.PhaseNet.from_pretrained("rose")` and
   `seisbench.models.EQTransformer.from_pretrained("rose")` without
   waiting for a SeisBench release.

### Versioning

- Bump `--version 2` only when the weights change in a way users should
  opt into explicitly (different training data, different normalization).
- Otherwise overwrite v1 and ask the maintainers to replace the file
  in-place — the JSON's `"version"` field stays `"1"`.

### Optional follow-up PR

Only required if a maintainer requests a `_weight_warnings` entry or a
docs example. The spots are:

- `seisbench/models/phasenet.py` → `PhaseNet._weight_warnings`
- `seisbench/models/eqtransformer.py` → `EQTransformer._weight_warnings`
- `docs/pages/models/pretrained_models.rst` (generic; only edit for
  a code-block example)
