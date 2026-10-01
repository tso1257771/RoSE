"""Reading the published magnitude calibration tables.

The three tables under ``magnitudes/calibration/`` are what the released
magnitudes were computed with. Loading them is the only supported way to
reproduce a magnitude, because no coefficient is hard coded in this package:

``parameter_table.csv``
    every scalar of the scale, with its standard error and the method that
    produced the error.
``station_terms.csv``
    one correction per station and response epoch, ``S_station_term``.
``conversion_coefficients.csv``
    the ML to Mw relation per depth regime, with its validity range.

The lower bound of the conversion needs care. Tables written before 2026-10
record ``ml_fit_min`` as the smallest ML the fit happened to see, 2.0002 say,
while the rule the released flags apply is the round number published as
``conversion_fit_lower_bound_ml_<regime>`` in ``parameter_table.csv``.
:func:`load_calibration` resolves this: ``ml_fit_min`` of the table it returns
is always the published rule, and the range the fit saw is kept beside it as
``ml_fit_set_min``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .attenuation import NAMES
from .conversion import REGIMES

__all__ = ["DEFAULT_CALIBRATION_DIR", "CALIBRATION_SEARCH_PATHS", "Calibration", "load_calibration"]

_PKG = Path(__file__).resolve().parent
DEFAULT_CALIBRATION_DIR = _PKG.parent.parent / "magnitudes" / "calibration"
# Repository layout first, then the copy shipped inside an installed wheel.
CALIBRATION_SEARCH_PATHS = (
    DEFAULT_CALIBRATION_DIR,
    _PKG / "calibration",
)

_FILES = ("parameter_table.csv", "station_terms.csv", "conversion_coefficients.csv")


@dataclass(frozen=True)
class Calibration:
    """The three calibration tables, plus the views the code needs."""

    parameters: pd.Series          #: value per parameter name
    parameter_table: pd.DataFrame  #: the full table, with standard errors
    station_terms: pd.DataFrame    #: indexed by ``station_key``
    conversion: pd.DataFrame       #: indexed by regime
    directory: Path                #: where it was read from

    @property
    def atten(self) -> dict[str, float]:
        """The five ``-log A0`` coefficients, keyed as :data:`~.attenuation.NAMES`."""
        return {n: float(self.parameters[n]) for n in NAMES}

    @property
    def anchor(self) -> dict[str, float]:
        """The anchor constant per depth regime."""
        return {r: float(self.parameters[f"C_{r}"]) for r in REGIMES}

    def station_term(self, station_key: str, default: float | None = None) -> float:
        """Correction of one station and response epoch.

        A station with no fitted term has no correction to apply, which is not
        the same as a correction of zero: its amplitudes were never used in the
        fit. Pass ``default=0.0`` to treat it as uncorrected on purpose.
        """
        try:
            return float(self.station_terms.at[station_key, "S_station_term"])
        except KeyError:
            if default is None:
                raise KeyError(
                    f"no station term for {station_key!r}; "
                    f"{len(self.station_terms)} station epochs are calibrated"
                ) from None
            return float(default)


def load_calibration(directory: str | Path | None = None) -> Calibration:
    """Load the calibration tables.

    ``directory`` overrides the search. Without it the repository layout is
    tried first, then the copy inside an installed wheel.
    """
    if directory is not None:
        candidates = (Path(directory),)
    else:
        candidates = CALIBRATION_SEARCH_PATHS

    for d in candidates:
        if all((d / f).is_file() for f in _FILES):
            break
    else:
        tried = "\n  ".join(str(d) for d in candidates)
        raise FileNotFoundError(
            "no magnitude calibration directory holding "
            f"{', '.join(_FILES)}. Tried:\n  {tried}"
        )

    ptab = pd.read_csv(d / "parameter_table.csv")
    dup = ptab.parameter[ptab.parameter.duplicated()].tolist()
    if dup:
        raise ValueError(f"{d / 'parameter_table.csv'} repeats parameter(s): {dup}")
    params = ptab.set_index("parameter").value.astype(float)

    missing = [n for n in list(NAMES) + [f"C_{r}" for r in REGIMES] if n not in params.index]
    if missing:
        raise ValueError(f"{d / 'parameter_table.csv'} is missing {missing}")

    st = pd.read_csv(d / "station_terms.csv", dtype={"location": str}).set_index("station_key")
    if st.index.has_duplicates:
        raise ValueError(f"{d / 'station_terms.csv'} repeats station keys")

    conv = pd.read_csv(d / "conversion_coefficients.csv").set_index("regime")
    unknown = set(conv.index) - set(REGIMES)
    if unknown:
        raise ValueError(f"{d / 'conversion_coefficients.csv'} has unknown regime(s) {unknown}")

    # Older tables put the smallest fitted ML in ml_fit_min. Keep it under a
    # name that says so, and take the bound the flags are defined by from the
    # parameter table, so mw_from_ml applies the published rule.
    conv = conv.rename(columns={"ml_fit_min": "ml_fit_set_min",
                                "ml_fit_max": "ml_fit_set_max"})
    bound = {}
    for r in conv.index:
        key = f"conversion_fit_lower_bound_ml_{r}"
        if key not in params.index:
            raise ValueError(f"{d / 'parameter_table.csv'} is missing {key}, "
                             "which is the lower bound mw_from_ml flags against")
        bound[r] = float(params[key])
    conv["ml_fit_min"] = pd.Series(bound)
    over = [r for r in conv.index
            if "ml_fit_set_min" in conv.columns
            and conv.at[r, "ml_fit_min"] > conv.at[r, "ml_fit_set_min"] + 1e-9]
    if over:
        raise ValueError(
            f"{d.name}: the published lower bound is above the smallest fitted ML "
            f"for {over}, so events inside the fit would be flagged below_range")

    return Calibration(parameters=params, parameter_table=ptab, station_terms=st,
                       conversion=conv, directory=d)
