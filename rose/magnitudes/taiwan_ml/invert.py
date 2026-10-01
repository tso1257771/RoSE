# Vendored unchanged from taiwan-local-magnitude commit 9d645e4, src/taiwan_ml/invert.py
# (MIT, Copyright (c) 2026 Wu-Yu Liao). Provenance: rose/magnitudes/taiwan_ml/__init__.py
"""Huber-IRLS solver for the joint ML inversion.

Solves G x = d for x = [M_i | S_j | attenuation coeffs] with a sum-to-zero
constraint on station terms to break the M-S degeneracy.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsmr

from .design import attenuation_columns, build_design

# ---- publication-quality thresholds for station terms (C4) ----
# Minimum observations for a S_j to be treated as publishable (cf. Perron et al.
# 2022 >=10-event rule; we use a stricter per-observation floor).
MIN_PUBLISH_NOBS = 20
# |S_j| beyond this is physically implausible (~30x amplitude) and almost
# certainly a response/metadata error; flag for manual review.
ABS_S_FLAG = 1.5
# Stations whose large |S_j| has been manually reviewed and accepted. Empty by
# default: TW.NLKH.00.HH@ (S ~ +2.45) is NOT here — it is flagged pending an
# instrument-response review (see REVIEW_FINDINGS C4).
S_FLAG_EXCEPTIONS: frozenset = frozenset()


# 1.4826 converts MAD to Gaussian-equivalent sigma
def _robust_scale(resid: np.ndarray) -> float:
    mad = np.median(np.abs(resid - np.median(resid)))
    return max(1.4826 * mad, 1e-6)


def _attach_station_flags(stations: pd.DataFrame, obs: pd.DataFrame,
                          station_col: str) -> pd.DataFrame:
    """Add QC flag columns to a station-term table (C4).

    Columns added:
        n_obs           observation count backing each S_j
        flag_low_nobs   True if n_obs < MIN_PUBLISH_NOBS (S_j not publishable —
                        prefer interpolation/0 downstream)
        flag_outlier_S  True if |S| > ABS_S_FLAG and not on the accepted list
    """
    counts = obs[station_col].value_counts()
    out = stations.copy()
    out["n_obs"] = out["station_key"].map(counts).fillna(0).astype(int)
    out["flag_low_nobs"] = out["n_obs"] < MIN_PUBLISH_NOBS
    out["flag_outlier_S"] = ((out["S"].abs() > ABS_S_FLAG)
                             & (~out["station_key"].isin(S_FLAG_EXCEPTIONS)))
    return out


def _sum_zero_columns(meta: dict, n_e: int, n_s: int,
                      sum_zero_keys: list[str] | None) -> np.ndarray:
    """Column indices spanned by the ΣS_j = 0 gauge constraint.

    With ``sum_zero_keys`` the constraint is applied ONLY over the given station
    keys (e.g. the step-1 surface reference set), so new stations added in a
    later borehole step is expressed in the step-1 gauge via
    shared events rather than shifting the magnitude datum. ``None`` reproduces
    the historical behaviour of constraining every station column.
    """
    if sum_zero_keys is None:
        return np.arange(n_e, n_e + n_s)
    stations = list(meta.get("stations", []))
    key_to_idx = {k: i for i, k in enumerate(stations)}
    sel = [key_to_idx[k] for k in sum_zero_keys if k in key_to_idx]
    if not sel:
        raise ValueError(
            "sum_zero_keys matched none of the station columns; cannot anchor "
            "the ΣS_j=0 gauge over an empty reference set")
    missing = [k for k in sum_zero_keys if k not in key_to_idx]
    if missing:
        shown = ", ".join(missing[:10])
        more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        warnings.warn(
            f"{len(missing)}/{len(sum_zero_keys)} sum_zero_keys are absent "
            f"from the observation set and were skipped — the ΣS_j=0 gauge "
            f"spans only the {len(sel)} present reference keys: {shown}{more}",
            stacklevel=3,
        )
    return n_e + np.asarray(sorted(sel), dtype=int)


def _warn_disconnected(G: sp.csr_matrix, meta: dict, n_e: int, n_s: int,
                       ref_cols: np.ndarray) -> tuple[int, int]:
    """Check connectivity on the sparse event–station bipartite graph.

    The ΣS_j=0 gauge fixes the magnitude datum only within the connected
    component that contains the reference stations; islands float on their own
    datum. Warn (do not drop) so the caller can decide what to do.
    """
    inc = G[:, :n_e + n_s].tocsr()
    inc.sort_indices()
    row_counts = np.diff(inc.indptr)
    if not np.all(row_counts == 2):
        raise ValueError(
            "each observation must connect exactly one event and one station"
        )
    left = inc.indices[inc.indptr[:-1]]
    right = inc.indices[inc.indptr[:-1] + 1]
    if np.any(left >= n_e) or np.any(right < n_e):
        raise ValueError(
            "event–station design columns do not follow the declared layout"
        )
    rows = np.concatenate([left, right])
    columns = np.concatenate([right, left])
    adj = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, columns)),
        shape=(n_e + n_s, n_e + n_s),
    ).tocsr()
    n_comp, labels = connected_components(adj, directed=False)
    if n_comp <= 1:
        return int(n_comp), 0
    ref_labels = set(labels[ref_cols].tolist())
    island_nodes = int(np.sum(~np.isin(labels, list(ref_labels))))
    warnings.warn(
        f"event–station graph is disconnected: {n_comp} components; "
        f"{island_nodes} node(s) lie in island(s) not containing the reference "
        f"set — their magnitude/station datum is not tied to the gauge",
        stacklevel=2,
    )
    return int(n_comp), island_nodes


def solve(
    G: sp.csr_matrix,
    d: np.ndarray,
    meta: dict,
    base_w: np.ndarray | None = None,
    sum_zero: bool = True,
    sum_zero_keys: list[str] | None = None,
    n_irls: int = 6,
    huber_delta: float = 1.345,  # 95% efficiency for Gaussian
    sumzero_weight: float = 1e3,
    lsmr_iter: int = 5000,
    irls_tol: float = 1e-4,
    warn_disconnected: bool = True,
) -> tuple[np.ndarray, dict]:
    """Huber-IRLS solve. Returns (x, info).

    ``sum_zero_keys`` restricts the ΣS_j=0 constraint to the given station keys
    (defaults to None = all stations, i.e. current behaviour). The IRLS loop
    stops early once ‖Δx‖/‖x‖ < ``irls_tol`` (max ``n_irls`` iterations). LSMR
    status is captured and a warning is emitted if it hits ``lsmr_iter``.
    """
    if n_irls < 1:
        raise ValueError("n_irls must be at least 1")
    n_e, n_s = meta["n_e"], meta["n_s"]
    ncol = G.shape[1]
    n_obs = len(d)
    base_w = np.ones(n_obs) if base_w is None else np.asarray(base_w, float)
    graph_components: int | None = None
    ungauged_graph_nodes: int | None = None

    if sum_zero:
        cols = _sum_zero_columns(meta, n_e, n_s, sum_zero_keys)
        constraint = sp.csr_matrix(
            (np.ones(len(cols)), (np.zeros(len(cols)), cols)), shape=(1, ncol))
        g_full = sp.vstack([G, constraint]).tocsr()
        d_full = np.append(d, 0.0)
        w_extra = np.array([sumzero_weight])
        if warn_disconnected:
            graph_components, ungauged_graph_nodes = _warn_disconnected(
                G,
                meta,
                n_e,
                n_s,
                cols,
            )
    else:
        g_full, d_full, w_extra = G, d, np.array([])

    x: np.ndarray | None = None
    istop = itn = -1
    lsmr_normr = float("nan")
    irls_iters = 0
    dx_rel = float("nan")
    column_norm_min = float("nan")
    column_norm_max = float("nan")
    for it in range(n_irls):
        irls_iters = it + 1
        if x is None:
            w_data = base_w
        else:
            resid = G @ x - d
            scale = _robust_scale(resid)
            # Huber weight: 1.0 inside |r| <= delta*scale, tapers as delta*scale/|r| outside
            hub = np.minimum(1.0, huber_delta * scale / np.maximum(np.abs(resid), 1e-9))
            w_data = base_w * hub
        w = np.concatenate([w_data, w_extra]) if len(w_extra) else w_data
        sqrt_w = np.sqrt(w)
        G_weighted = g_full.multiply(sqrt_w[:, None]).tocsr()
        # Right-precondition the weighted system by its column L2 norms. This
        # is an exact change of variables, not a change to the objective:
        #     (G D) z = d,  x = D z,  D_jj = 1 / ||G[:, j]||_2.
        # Event-balanced weights and sparsely populated channel families can
        # otherwise produce column norms spanning several orders of
        # magnitude, causing LSMR to stop at its condition or iteration limit.
        column_norms = np.sqrt(
            np.asarray(G_weighted.power(2).sum(axis=0)).ravel()
        )
        positive = np.isfinite(column_norms) & (column_norms > 0.0)
        inverse_column_norms = np.ones_like(column_norms)
        inverse_column_norms[positive] = 1.0 / column_norms[positive]
        column_norm_min = float(np.min(column_norms[positive]))
        column_norm_max = float(np.max(column_norms[positive]))
        G_scaled = G_weighted.multiply(inverse_column_norms).tocsr()
        sol = lsmr(
            G_scaled,
            d_full * sqrt_w,
            atol=1e-10,
            btol=1e-10,
            maxiter=lsmr_iter,
        )
        x_new = inverse_column_norms * sol[0]
        istop, itn, lsmr_normr = int(sol[1]), int(sol[2]), float(sol[3])
        if istop == 7 or itn >= lsmr_iter:
            warnings.warn(
                f"LSMR reached the iteration limit (itn={itn}, istop={istop}); "
                "the normal equations may be ill-conditioned",
                stacklevel=2)
        if x is not None:
            dx_rel = float(np.linalg.norm(x_new - x)) / max(
                float(np.linalg.norm(x_new)),
                1e-12,
            )
            x = x_new
            if dx_rel < irls_tol:
                break
        else:
            x = x_new

    if x is None:  # pragma: no cover - guarded by n_irls validation
        raise RuntimeError("IRLS returned no solution")
    resid = G @ x - d
    info = dict(resid_std=float(np.std(resid)), resid_mad=float(_robust_scale(resid)),
                n_obs=int(n_obs), n_events=int(n_e), n_stations=int(n_s),
                sum_S=float(np.sum(x[n_e:n_e + n_s])),
                lsmr_istop=istop, lsmr_itn=itn, lsmr_normr=lsmr_normr,
                irls_iters=irls_iters, irls_dx_rel=dx_rel,
                graph_components=graph_components,
                ungauged_graph_nodes=ungauged_graph_nodes,
                column_scaling="unit_l2_norm",
                weighted_column_norm_min=column_norm_min,
                weighted_column_norm_max=column_norm_max)
    return x, info


def assemble(x: np.ndarray, meta: dict):
    """Split x into coeffs dict + event-magnitude and station-term DataFrames.

    The attenuation columns are centered in ``build_design`` (``atten_mean``),
    so the fitted event magnitudes carry a uniform ``theta . atten_mean`` shift.
    We export that shift as ``centering_offset`` (and the raw ``atten_mean``) so
    residuals and event ML can be de-biased. By construction:
        event_ml(anchor_c=0) - M_i == centering_offset
    """
    n_e, n_s = meta["n_e"], meta["n_s"]
    atten_names = list(meta["atten_names"])
    atten_coef = x[n_e + n_s:]
    coeffs = dict(zip(atten_names, atten_coef.tolist(), strict=False))
    coeffs.update(
        rref=meta["rref"],
        h_break=meta["h_break"],
        n_cheb=meta["n_cheb"],
        depth_term=meta["depth_term"],
    )

    atten_mean = np.asarray(meta.get("atten_mean", np.zeros(len(atten_names))), float)
    if atten_mean.size == len(atten_coef) and atten_mean.size:
        # M_i_fit = M_i_true - theta.atten_mean  =>  offset applied to event ML
        centering_offset = float(-np.dot(atten_coef, atten_mean))
        coeffs["atten_mean"] = dict(zip(atten_names, atten_mean.tolist(), strict=False))
    else:
        centering_offset = 0.0
        coeffs["atten_mean"] = {}
    coeffs["centering_offset"] = centering_offset

    events = pd.DataFrame({"public_id": meta["events"], "M_i": x[:n_e]})
    stations = pd.DataFrame({"station_key": meta["stations"], "S": x[n_e:n_e + n_s]})
    return coeffs, events, stations


def neg_log_a0(R: np.ndarray, h: np.ndarray, coeffs: dict) -> np.ndarray:
    """Compute the released attenuation correction from fitted coefficients."""

    if int(coeffs.get("n_cheb", 0)) != 0:
        raise ValueError("the released attenuation requires n_cheb=0")
    if not bool(coeffs.get("depth_term", True)):
        raise ValueError("the released attenuation requires the depth-hinge term")
    R, h = np.broadcast_arrays(np.asarray(R, float), np.asarray(h, float))
    output_shape = R.shape
    basis, names = attenuation_columns(
        R.ravel(),
        h.ravel(),
        rref=float(coeffs["rref"]),
        h_break=float(coeffs.get("h_break", 40.0)),
    )
    atten = np.zeros(R.size, dtype=float)
    for index, name in enumerate(names):
        atten += float(coeffs[name]) * basis[:, index]
    return atten.reshape(output_shape)


def fit_station_terms(obs: pd.DataFrame, frozen_coeffs: dict,
                      base_w: np.ndarray | None = None,
                      n_irls: int = 6, huber_delta: float = 1.345,
                      station_col: str = "station_key",
                      sum_zero_keys: list[str] | None = None,
                      warn_disconnected: bool = True) -> dict:
    """Two-step inversion: freeze attenuation, solve only for M_i and S_j.

    Use when attenuation (n, K, dK) is already determined from the surface
    network and station coverage is then extended with the combined surface
    and borehole observations.

    ``sum_zero_keys`` (e.g. the step-1 surface reference set) applies the
    ΣS_j=0 gauge over only those station keys, so newly added stations are
    expressed in the step-1 gauge instead of shifting the magnitude datum.

    ``warn_disconnected`` forwards to :func:`solve`. The check operates on the
    sparse event–station edge list and therefore remains suitable for the full
    observation archive.
    """
    R = obs["R_km"].to_numpy(float)
    h = obs["depth_km"].to_numpy(float)
    # d_corrected = log10(A) + frozen attenuation → model is just M_i - S_j
    atten = neg_log_a0(R, h, frozen_coeffs)
    d_corr = obs["log10_A"].to_numpy(float) + atten

    ev_cat = pd.Categorical(obs["public_id"])
    sta_cat = pd.Categorical(obs[station_col])
    n_e, n_s = len(ev_cat.categories), len(sta_cat.categories)
    n_obs = len(obs)

    row_idx = np.repeat(np.arange(n_obs), 2)
    col_idx = np.empty(n_obs * 2, dtype=int)
    col_idx[0::2] = ev_cat.codes
    col_idx[1::2] = n_e + sta_cat.codes
    G = sp.coo_matrix((np.ones(n_obs * 2), (row_idx, col_idx)),
                      shape=(n_obs, n_e + n_s)).tocsr()

    meta = dict(
        events=list(ev_cat.categories),
        stations=list(sta_cat.categories),
        atten_names=[],
        atten_mean=np.array([]),
        n_e=n_e,
        n_s=n_s,
        rref=frozen_coeffs["rref"],
        h_break=frozen_coeffs["h_break"],
        n_cheb=0,
        depth_term=True,
    )

    x, info = solve(G, d_corr, meta, base_w=base_w, n_irls=n_irls,
                    huber_delta=huber_delta, sum_zero_keys=sum_zero_keys,
                    warn_disconnected=warn_disconnected)

    events = pd.DataFrame({"public_id": meta["events"], "M_i": x[:n_e]})
    stations = pd.DataFrame({"station_key": meta["stations"], "S": x[n_e:n_e + n_s]})
    stations = _attach_station_flags(stations, obs, station_col)

    sj_lookup = stations.set_index("station_key")["S"]
    ml_before = d_corr
    ml_after = ml_before - obs[station_col].map(sj_lookup).to_numpy(float)
    ml_per_obs = pd.DataFrame({"public_id": obs["public_id"].to_numpy(),
                                "before": ml_before, "after": ml_after})
    by_event = ml_per_obs.groupby("public_id")
    scatter_before = by_event["before"].std().median()
    scatter_after = by_event["after"].std().median()

    return dict(coeffs=frozen_coeffs, events=events, stations=stations, info=info,
                scatter_before=float(scatter_before), scatter_after=float(scatter_after),
                scatter_reduction=float(1.0 - scatter_after / scatter_before)
                if scatter_before else float("nan"))


def fit_ml(obs: pd.DataFrame, base_w: np.ndarray | None = None,
           solver_kw: dict | None = None, **design_kw):
    """Build design, solve, assemble, and compute within-event scatter."""
    G, data, meta = build_design(obs, **design_kw)
    x, info = solve(G, data, meta, base_w=base_w, **(solver_kw or {}))
    coeffs, events, stations = assemble(x, meta)
    stations = _attach_station_flags(stations, obs, "station_key")

    atten = neg_log_a0(obs["R_km"].to_numpy(float), obs["depth_km"].to_numpy(float), coeffs)
    ml_before = obs["log10_A"].to_numpy(float) + atten
    sj_lookup = stations.set_index("station_key")["S"]
    ml_after = ml_before - obs["station_key"].map(sj_lookup).to_numpy(float)

    ml_per_obs = pd.DataFrame({"public_id": obs["public_id"].to_numpy(),
                                "before": ml_before, "after": ml_after})
    by_event = ml_per_obs.groupby("public_id")
    scatter_before = by_event["before"].std().median()
    scatter_after = by_event["after"].std().median()
    return dict(coeffs=coeffs, events=events, stations=stations, info=info,
                scatter_before=float(scatter_before), scatter_after=float(scatter_after),
                scatter_reduction=float(1.0 - scatter_after / scatter_before)
                if scatter_before else float("nan"))
