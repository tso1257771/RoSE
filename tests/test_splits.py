"""Tests for `rose.splits` deterministic train/dev/test split helpers.

The golden values here pin the exact partition that the published RoSE
`split` column (and therefore every benchmark number) is built on — if the
hashing scheme ever changes, these break loudly.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest

from rose import splits

# The salt RED-PAN used for the ROMPLUS / RoSE single-earthquake corpus
# (see training/build_rose_split_index.py).
ROSE_SALT = "ROMPLUS-singleEQ-v1"


# --------------------------------------------------------------------------- #
# hash_split — determinism & golden values
# --------------------------------------------------------------------------- #
def test_hash_split_is_deterministic():
    a = splits.hash_split("2014_0000123", 0.70, 0.15, salt=ROSE_SALT)
    b = splits.hash_split("2014_0000123", 0.70, 0.15, salt=ROSE_SALT)
    assert a == b
    assert a in {"train", "val", "test"}


@pytest.mark.parametrize(
    "source_id, expected",
    [
        # Real ROMPLUS source_ids, taken from the published rose_split_index.csv.
        ("2014_0000001", "train"),
        ("2014_0000010", "test"),
    ],
)
def test_hash_split_reproduces_published_rose_partition(source_id, expected):
    assert splits.hash_split(source_id, 0.70, 0.15, salt=ROSE_SALT) == expected


def test_hash_split_salt_changes_assignment_space():
    # Different salt -> independent stream. Over many keys the two streams must
    # disagree somewhere (they would only ever fully agree by 1/3**N chance).
    keys = [str(i) for i in range(500)]
    with_salt_a = [splits.hash_split(k, 0.70, 0.15, salt="A") for k in keys]
    with_salt_b = [splits.hash_split(k, 0.70, 0.15, salt="B") for k in keys]
    assert with_salt_a != with_salt_b


def test_hash_split_proportions_are_close_to_targets():
    keys = [str(i) for i in range(20_000)]
    counts = collections.Counter(
        splits.hash_split(k, 0.70, 0.15, salt=ROSE_SALT) for k in keys
    )
    n = len(keys)
    assert counts["train"] / n == pytest.approx(0.70, abs=0.02)
    assert counts["val"] / n == pytest.approx(0.15, abs=0.02)
    assert counts["test"] / n == pytest.approx(0.15, abs=0.02)
    assert sum(counts.values()) == n  # every key landed somewhere


# --------------------------------------------------------------------------- #
# bernoulli_3way
# --------------------------------------------------------------------------- #
def test_bernoulli_3way_is_seeded_and_reproducible():
    s1 = [
        splits.bernoulli_3way(np.random.default_rng(7), 0.70, 0.15) for _ in range(50)
    ]
    s2 = [
        splits.bernoulli_3way(np.random.default_rng(7), 0.70, 0.15) for _ in range(50)
    ]
    assert s1 == s2
    assert set(s1) <= {"train", "val", "test"}


def test_bernoulli_3way_proportions():
    rng = np.random.default_rng(0)
    counts = collections.Counter(
        splits.bernoulli_3way(rng, 0.70, 0.15) for _ in range(20_000)
    )
    n = sum(counts.values())
    assert counts["train"] / n == pytest.approx(0.70, abs=0.02)
    assert counts["val"] / n == pytest.approx(0.15, abs=0.02)


# --------------------------------------------------------------------------- #
# year_split
# --------------------------------------------------------------------------- #
def test_year_split_assigns_listed_years_and_returns_none_otherwise():
    train_years = range(2014, 2022)
    val_years = (2022, 2023)
    test_years = (2024,)
    assert splits.year_split(2015, train_years, val_years, test_years) == "train"
    assert splits.year_split(2023, train_years, val_years, test_years) == "val"
    assert splits.year_split(2024, train_years, val_years, test_years) == "test"
    assert splits.year_split(1999, train_years, val_years, test_years) is None


# --------------------------------------------------------------------------- #
# assert_valid_proportions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p_train, p_val", [(0.70, 0.15), (0.5, 0.0), (0.999, 0.0)])
def test_assert_valid_proportions_accepts_valid(p_train, p_val):
    splits.assert_valid_proportions(p_train, p_val)  # must not raise


@pytest.mark.parametrize(
    "p_train, p_val",
    [
        (0.0, 0.15),  # p_train not in (0, 1)
        (1.0, 0.0),  # p_train not in (0, 1)
        (1.5, 0.0),  # p_train > 1
        (0.70, -0.1),  # p_val negative
        (0.70, 0.40),  # p_train + p_val > 1
    ],
)
def test_assert_valid_proportions_rejects_invalid(p_train, p_val):
    with pytest.raises(ValueError):
        splits.assert_valid_proportions(p_train, p_val)
