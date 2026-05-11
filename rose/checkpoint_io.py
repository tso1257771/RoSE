"""Safe checkpoint loading helpers.

The training and benchmark scripts in this repo load PyTorch checkpoints from
arbitrary user-supplied paths. ``torch.load`` with ``weights_only=False`` (the
historical default) calls the full ``pickle`` unpickler, which can execute
arbitrary code embedded in a malicious or corrupted checkpoint — a real risk
for a public repo that ships ``.pt`` artifacts and accepts ``--ckpt`` flags.

:func:`safe_torch_load` wraps ``torch.load`` with ``weights_only=True`` (a
restricted unpickler that only accepts tensors, plain Python scalars, and a
small allowlist of container types) and falls back transparently for older
PyTorch versions that don't accept the keyword. All checkpoints written by
``training/train_eqt_rose.py`` and ``training/train_phasenet_rose.py`` (and the
bundled release weights under ``models/``)
are plain ``state_dict`` + scalars + a config dict, so the safe loader is
sufficient.

If a future checkpoint contains non-allowlisted pickled objects, the safe load
will raise ``UnpicklingError`` rather than silently executing the payload —
that is the intended behavior. To intentionally load such a checkpoint, set
``trusted=True`` (which logs a warning and defers to the unrestricted loader).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def safe_torch_load(
    path: str | Path,
    *,
    map_location: Any = "cpu",
    trusted: bool = False,
) -> Any:
    """Load a torch checkpoint with ``weights_only=True`` by default.

    Parameters
    ----------
    path
        Filesystem path to the checkpoint (``.pt``/``.pth``); ``str`` or
        ``pathlib.Path``.
    map_location
        Forwarded to ``torch.load`` (e.g. ``"cpu"``, a ``torch.device``,
        or a function).
    trusted
        If ``True``, fall back to the unrestricted ``torch.load`` (full
        pickle) — equivalent to ``weights_only=False``. Use only for
        checkpoints whose provenance you have independently verified.
        A warning is logged when this path is taken.

    Returns
    -------
    The loaded checkpoint (typically ``OrderedDict`` for a bare ``state_dict``,
    or ``dict`` for a wrapped ``{"model": state_dict, "config": ..., ...}``).

    Raises
    ------
    pickle.UnpicklingError
        If ``weights_only=True`` rejects a non-allowlisted object in the
        checkpoint. This is the intended security behaviour — the offending
        checkpoint is not from one of this repo's training scripts; pass
        ``trusted=True`` only if you have independently verified its source.
    """
    import torch  # local import: keeps `rose` importable without torch

    if trusted:
        logger.warning(
            "safe_torch_load(trusted=True) — using unrestricted pickle for %s",
            path,
        )
        try:
            # Be explicit: PyTorch >= 2.6 defaults `weights_only` to True, so a
            # bare torch.load() would *not* be the unrestricted loader the
            # caller asked for.
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            # PyTorch < 1.13 does not accept the `weights_only` keyword.
            return torch.load(path, map_location=map_location)

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # PyTorch < 1.13 does not accept the `weights_only` keyword.
        logger.warning(
            "torch.load does not support weights_only=True on this PyTorch "
            "version; falling back to unrestricted pickle for %s",
            path,
        )
        return torch.load(path, map_location=map_location)
