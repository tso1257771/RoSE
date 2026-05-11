"""Deterministic train/dev/test split helpers.

Vendored from the RED-PAN data-builder (`redpan.data.builder.splits`) so the
RoSE training/benchmark pipeline is self-contained. The RoSE SeisBench split
column was produced with :func:`hash_split` using the salt
``"ROMPLUS-singleEQ-v1"`` (see ``phase_picking/training/build_rose_split_index.py``); keeping
the implementation here guarantees the assignment can be reproduced bit-for-bit
without depending on the RED-PAN repository.

Three helpers:

  - :func:`bernoulli_3way` — per-row Bernoulli draw, fast and key-free.
    OK only when samples are independent (e.g. noise traces with no shared
    source event).
  - :func:`hash_split` — group samples that share ``key`` (e.g. ``source_id``)
    into the SAME split. Stable across runs given the same salt.
  - :func:`year_split` — explicit year assignment for catalogues with
    year-based partitioning.

Default proportions: 70 % train / 15 % val / 15 % test. These helpers return the
RED-PAN split vocabulary (``"train"`` / ``"val"`` / ``"test"``); SeisBench uses
``"dev"`` for the validation split, so callers must remap ``"val" -> "dev"``
(``phase_picking/training/build_rose_split_index.py`` does this).
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Literal

import numpy as np

Split = Literal["train", "val", "test"]

_DEFAULT_TRAIN = 0.70
_DEFAULT_VAL = 0.15
# test = 1 - train - val


def bernoulli_3way(
    rng: np.random.Generator,
    p_train: float = _DEFAULT_TRAIN,
    p_val: float = _DEFAULT_VAL,
) -> Split:
    """Single-sample Bernoulli draw from ``rng``. Use only when samples are
    independent (e.g. noise traces with no shared event source).

    Returns the RED-PAN split vocabulary; remap ``"val" -> "dev"`` for SeisBench.
    """
    u = float(rng.random())
    if u < p_train:
        return "train"
    if u < p_train + p_val:
        return "val"
    return "test"


def hash_split(
    key: str,
    p_train: float = _DEFAULT_TRAIN,
    p_val: float = _DEFAULT_VAL,
    salt: str = "redpan-builder",
) -> Split:
    """Deterministic split: every sample with the same ``key`` lands in the
    same split. Use for event-bearing categories where multiple traces share a
    source event (pass e.g. ``source_id`` as ``key``).

    The hash is BLAKE2s of ``f"{salt}|{key}"`` taken as a big-endian uint64,
    divided by ``2**64`` to land in ``[0, 1]`` — the all-ones digest divides to
    exactly ``1.0`` (probability ``2**-64``) and falls into ``"test"``.

    .. warning::
       ``salt`` and ``key`` are joined with a literal ``"|"``; if either can
       contain ``"|"`` the join is ambiguous (``salt="a|b", key="c"`` hashes
       the same as ``salt="a", key="b|c"``). The RoSE salt
       (``"ROMPLUS-singleEQ-v1"``) and its numeric ``source_id`` keys are
       delimiter-free, so the published RoSE split is unaffected — but pick a
       different separator before reusing this with arbitrary keys.

    Returns the RED-PAN split vocabulary; remap ``"val" -> "dev"`` for SeisBench.
    """
    h = hashlib.blake2s(f"{salt}|{key}".encode("utf-8"), digest_size=8).digest()
    u = int.from_bytes(h, "big") / 2**64  # in [0, 1]; all-ones digest -> 1.0 -> test
    if u < p_train:
        return "train"
    if u < p_train + p_val:
        return "val"
    return "test"


def year_split(
    year: int,
    train_years: Iterable[int],
    val_years: Iterable[int],
    test_years: Iterable[int],
) -> Split | None:
    """Return the split for a given year, or ``None`` if the year is unlisted.

    Each ``*_years`` argument is materialised to a ``frozenset`` once up front,
    so passing one-shot generators is safe and lookups are O(1). If a year
    appears in more than one set, train wins, then val (first-match order).

    Returns the RED-PAN split vocabulary; remap ``"val" -> "dev"`` for SeisBench.
    """
    s_train = frozenset(train_years)
    s_val = frozenset(val_years)
    s_test = frozenset(test_years)
    if year in s_train:
        return "train"
    if year in s_val:
        return "val"
    if year in s_test:
        return "test"
    return None


def assert_valid_proportions(p_train: float, p_val: float) -> None:
    """Sanity-check split proportions; raise ``ValueError`` if they're degenerate.

    Requires ``0 < p_train < 1``, ``p_val >= 0``, and ``p_train + p_val < 1``
    (so the implied test fraction ``1 - p_train - p_val`` is strictly positive).
    The sum check is done directly rather than comparing ``p_val`` to
    ``1.0 - p_train`` — the latter is fragile under floating-point rounding
    (e.g. ``1.0 - 0.85 == 0.15000000000000002``, which would wrongly accept
    ``(0.85, 0.15)`` and leave nothing for the test split).
    """
    if not 0.0 < p_train < 1.0:
        raise ValueError(f"p_train must be in (0, 1); got {p_train}")
    if p_val < 0.0:
        raise ValueError(f"p_val must be >= 0; got {p_val}")
    if p_train + p_val >= 1.0:
        raise ValueError(
            f"p_train + p_val must be < 1.0 (the rest is the test split); "
            f"got {p_train} + {p_val} = {p_train + p_val}"
        )
