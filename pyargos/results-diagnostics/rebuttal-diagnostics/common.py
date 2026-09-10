"""Shared helpers for the rebuttal diagnostics (Referee 1, Points 6-7;
Referee 2, minor point 2).

Everything here re-uses the exact simulator and library builder that produced
Fig. 3 of the submitted manuscript (``utils.argos_simulator`` and
``utils.argos_utils.build_design_matrix``), so the new numbers are directly
comparable with the published VIF and residual panels.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYARGOS = os.path.abspath(os.path.join(_HERE, "..", ".."))
_REPO = os.path.abspath(os.path.join(_PYARGOS, ".."))
for _p in (_PYARGOS, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import utils.argos_simulator as ags  # noqa: E402
import utils.argos_utils as agu  # noqa: E402
from src.argos_sparse_regression import (  # noqa: E402
    argos_alasso,
    shrink_design_matrix_based_on_estimate,
)

DT = 0.01
LIBRARY_DEGREE = 5
SG_POLY_ORDER = 4

# --------------------------------------------------------------------------
# Benchmark systems exactly as in the submitted experiments
# --------------------------------------------------------------------------
SYSTEMS = {
    "aizawa": dict(
        coeff=[
            [-3.5, -0.7, 1],
            [3.5, -0.7, 1],
            [0.95, 0.65, 0.1, -1 / 3, -0.25, -1, -0.25, -1],
        ],
        names=[
            ["x2", "x1", "x1x3"],
            ["x1", "x2", "x2x3"],
            ["x3", "", "x1^3x3", "x3^3", "x1^2x3", "x1^2", "x2^2x3", "x2^2"],
        ],
        ic_ranges=((-2, 2), (-2, 2), (-1, 2)),
        # ground-truth support per equation in sklearn PolynomialFeatures naming
        truth={
            1: dict(intercept=False, terms=["x1", "x2", "x1 x3"]),
            2: dict(intercept=False, terms=["x1", "x2", "x2 x3"]),
            3: dict(
                intercept=True,
                terms=[
                    "x3",
                    "x1^2",
                    "x2^2",
                    "x1^2 x3",
                    "x2^2 x3",
                    "x1^3 x3",
                    "x3^3",
                ],
            ),
        },
    ),
    "rossler": dict(
        coeff=[[-1, -1], [1, 0.2], [0.2, 1, -5.7]],
        names=[["x2", "x3"], ["x1", "x2"], ["", "x1x3", "x3"]],
        ic_ranges=((-10, 10), (-10, 10), (0, 20)),
        truth={
            1: dict(intercept=False, terms=["x2", "x3"]),
            2: dict(intercept=False, terms=["x1", "x2"]),
            3: dict(intercept=True, terms=["x3", "x1 x3"]),
        },
    ),
}


def rhs(system: str, x: np.ndarray) -> np.ndarray:
    """True right-hand side f(x) evaluated row-wise on states x (n x 3)."""
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    if system == "aizawa":
        return np.column_stack(
            [
                -3.5 * x2 - 0.7 * x1 + x1 * x3,
                3.5 * x1 - 0.7 * x2 + x2 * x3,
                0.95 * x3
                + 0.65
                + 0.1 * x1**3 * x3
                - x3**3 / 3
                - 0.25 * x1**2 * x3
                - x1**2
                - 0.25 * x2**2 * x3
                - x2**2,
            ]
        )
    if system == "rossler":
        return np.column_stack(
            [-x2 - x3, x1 + 0.2 * x2, 0.2 + x1 * x3 - 5.7 * x3]
        )
    raise ValueError(system)


def initial_conditions(system: str, num: int = 100, seed: int = 100):
    """Same seeded pool of initial conditions as the benchmark experiments."""
    r = SYSTEMS[system]["ic_ranges"]
    return ags.generate_initial_value_df(
        seed=seed,
        num_init_samples=num,
        x1_range=r[0],
        x2_range=r[1],
        x3_range=r[2],
        num_columns=3,
    )


def simulate(system: str, n: int, snr: float, ic, noise_seed: int):
    """Clean and noisy trajectories with a reproducible noise draw.

    Returns (x_clean, x_noisy). Noise follows the benchmark convention:
    additive Gaussian per coordinate, sd = 10^(-SNR/20) * sd(clean coordinate).
    """
    s = SYSTEMS[system]
    t = np.arange(0, float(n) * DT, DT)
    x_clean = ags.solve_ode_odeint(s["coeff"], s["names"], np.asarray(ic), t)
    rng = np.random.default_rng(noise_seed)
    if np.isinf(snr):
        return x_clean, x_clean.copy()
    snr_volt = 10 ** (-(snr / 20))
    noise = rng.normal(size=x_clean.shape) * (snr_volt * x_clean.std(axis=0))
    return x_clean, x_clean + noise


def library(x_noisy: np.ndarray):
    """Savitzky-Golay smoothing/differentiation + polynomial library, as in
    the manuscript pipeline (degree 5, SG polynomial order 4)."""
    return agu.build_design_matrix(
        x_t=x_noisy,
        dt=DT,
        library_degree=LIBRARY_DEGREE,
        sg_poly_order=SG_POLY_ORDER,
        library_type="poly",
    )


def poly_features(x: np.ndarray, degree: int = LIBRARY_DEGREE):
    """Polynomial library evaluated on arbitrary state samples (no SG)."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    theta = poly.fit_transform(x)
    names = list(poly.get_feature_names_out(["x1", "x2", "x3"]))
    return theta, names


def monomial_degree(name: str) -> int:
    deg = 0
    for tok in name.split():
        deg += int(tok.split("^")[1]) if "^" in tok else 1
    return deg


# --------------------------------------------------------------------------
# Screening (the two adaptive-lasso passes, without the HMC stage)
# --------------------------------------------------------------------------
def screen(design_matrix: dict, eq: int, n_threads: int = 2):
    """Reproduce ``multi_stage_regression``: pass 1 (ridge weights) sets the
    degree cut, pass 2 (OLS weights) selects the support.

    Returns dict with the degree-trimmed library names (what the published
    VIF panel was computed on) and the final selected support.
    """
    theta = design_matrix["sorted_theta"]
    xdot = design_matrix["xdot_filtered"][:, eq - 1]
    data = np.column_stack([xdot, theta])
    init = argos_alasso(data, weights_method="ridge", ols_ps=True, n_threads=n_threads)
    data2, dm2, _ = shrink_design_matrix_based_on_estimate(
        design_matrix=design_matrix, data=data, initial_estimate=init
    )
    final = argos_alasso(data2, weights_method="ols", ols_ps=True, n_threads=n_threads)
    final = np.nan_to_num(final)
    trimmed_names = list(dm2["sorted_feature_names"])
    nz = final != 0
    selected = [nm for nm, keep in zip(trimmed_names, nz[1:]) if keep]
    return dict(
        trimmed_names=trimmed_names,
        trimmed_theta=dm2["sorted_theta"],
        selected_intercept=bool(nz[0]),
        selected_terms=selected,
        final_estimate=final,
    )


# --------------------------------------------------------------------------
# Collinearity diagnostics
# --------------------------------------------------------------------------
def vif(X: np.ndarray, names, method: str = "corr") -> pd.Series:
    """Centered VIF_j = 1 / (1 - R_j^2), R_j^2 from regressing column j on
    all other columns plus an intercept. Identical to
    ``statsmodels.variance_inflation_factor`` applied after ``add_constant``
    (the convention used for Fig. 3b), but computed from the inverse of the
    correlation matrix, which is exact and fast."""
    X = np.asarray(X, dtype=float)
    sd = X.std(axis=0)
    keep = sd > 0
    out = pd.Series(np.nan, index=list(names), dtype=float)
    if keep.sum() < 2:
        out[keep] = 1.0
        return out
    Z = (X[:, keep] - X[:, keep].mean(axis=0)) / sd[keep]  # standardised columns
    if method == "corr":
        # inverse of the correlation matrix; exact but unstable when p ~ n
        C = Z.T @ Z / Z.shape[0]
        try:
            v = np.diag(np.linalg.inv(C))
        except np.linalg.LinAlgError:
            v = np.diag(np.linalg.pinv(C))
        if np.any(v < 1) or not np.all(np.isfinite(v)):
            method = "lstsq"
    if method == "lstsq":
        # one least-squares regression per column via lstsq (rank-revealing),
        # R^2 = 1 - SS_res / SS_tot on the standardised column; VIF >= 1 always
        v = []
        n = Z.shape[0]
        for j in range(Z.shape[1]):
            others = np.delete(Z, j, axis=1)
            coef, *_ = np.linalg.lstsq(others, Z[:, j], rcond=None)
            ss_res = float(np.sum((Z[:, j] - others @ coef) ** 2))
            r2 = 1.0 - ss_res / n
            v.append(1.0 / max(1.0 - r2, 1e-16))
        v = np.array(v)
    out[np.array(list(names))[keep]] = v
    return out


def sg_params(x_col: np.ndarray):
    """(polyorder, window) the pipeline selects for one noisy state column."""
    comb = agu.sg_optimal_combination(x_col, DT, polyorder=SG_POLY_ORDER)
    return int(comb["sg_order_wl"][0]), int(comb["sg_order_wl"][1])


def sg_derivative(x_col: np.ndarray, order: int, window: int):
    from scipy.signal import savgol_filter

    return savgol_filter(x_col, window_length=window, polyorder=order, deriv=1, delta=DT)


def belsley(X: np.ndarray, names, intercept: bool = True):
    """Belsley-Kuh-Welsch collinearity diagnostics.

    Columns (plus an intercept column if requested) are scaled to unit length;
    the condition indices eta_k = sigma_max / sigma_k and the
    variance-decomposition proportions phi[k, j] are returned. A near
    dependency is a row k with large eta_k in which two or more columns carry
    proportion > 0.5 -- this exhibits the *collective* structure that a
    per-term VIF cannot show.
    """
    X = np.asarray(X, dtype=float)
    cols = list(names)
    if intercept:
        X = np.column_stack([np.ones(X.shape[0]), X])
        cols = ["Intercept"] + cols
    Xs = X / np.linalg.norm(X, axis=0)
    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    V = Vt.T
    eta = s.max() / s
    phi_raw = (V**2) / (s**2)[None, :]  # j x k
    phi = (phi_raw / phi_raw.sum(axis=1, keepdims=True)).T  # k x j
    return dict(
        condition_number=float(eta.max()),
        condition_indices=eta,
        proportions=pd.DataFrame(phi, columns=cols),
    )


def near_dependencies(bel: dict, eta_min: float = 30.0, phi_min: float = 0.5):
    """List the column groups involved in each near dependency."""
    groups = []
    for k, eta in enumerate(bel["condition_indices"]):
        if eta < eta_min:
            continue
        row = bel["proportions"].iloc[k]
        members = row[row > phi_min].index.tolist()
        if len(members) >= 2:
            groups.append(dict(condition_index=float(eta), members=members))
    return groups


# --------------------------------------------------------------------------
# Heteroscedasticity / residual-structure diagnostics
# --------------------------------------------------------------------------
def ols_fit(y: np.ndarray, X: np.ndarray, intercept: bool):
    Xd = sm.add_constant(X, has_constant="add") if intercept else X
    return sm.OLS(y, Xd).fit(), Xd


def residual_stats(y: np.ndarray, X: np.ndarray, intercept: bool, n_bins: int = 10):
    """Quantitative residual diagnostics for one fitted equation.

    * Breusch-Pagan LM statistic and p-value with the fitted value as the
      auxiliary regressor (the test that matches a residual-vs-fitted plot);
    * White's general test on the full design;
    * binned variance ratio: residual variance in the highest-variance decile
      of fitted values over that in the lowest;
    * lag-1 autocorrelation and Durbin-Watson of residuals in time order --
      a direct measure of "coherent structure" (i.i.d. noise gives ~0 and ~2).
    """
    fit, Xd = ols_fit(y, X, intercept)
    e = fit.resid
    fitted = fit.fittedvalues
    bp_lm, bp_p, _, _ = het_breuschpagan(e, sm.add_constant(fitted))
    try:
        w_lm, w_p, _, _ = het_white(e, Xd if intercept else sm.add_constant(Xd))
    except Exception:
        w_lm, w_p = np.nan, np.nan
    order = np.argsort(fitted)
    bins = np.array_split(e[order], n_bins)
    v = np.array([b.var(ddof=1) for b in bins])
    rho1 = np.corrcoef(e[:-1], e[1:])[0, 1]
    return dict(
        sigma=float(e.std(ddof=1)),
        bp_lm=float(bp_lm),
        bp_p=float(bp_p),
        bp_r2=float(bp_lm / len(e)),
        white_lm=float(w_lm),
        white_p=float(w_p),
        var_ratio=float(v.max() / v.min()),
        rho1=float(rho1),
        durbin_watson=float(durbin_watson(e)),
        residuals=e,
        fitted=fitted,
    )


def design_for_terms(dm: dict, terms):
    names = list(dm["sorted_feature_names"])
    idx = [names.index(t) for t in terms]
    return dm["sorted_theta"][:, idx]


def theta_columns(theta: np.ndarray, names, terms):
    names = list(names)
    return theta[:, [names.index(t) for t in terms]]


def log_grid(start=2.0, stop=5.0, step=0.2):
    k = np.round(np.arange(start, stop + 1e-9, step), 1)
    return [(float(kk), int(round(10**kk))) for kk in k]


SNR_GRID = [10, 20, 30, 40, 45, 50, 55, 57, 60, np.inf]


def snr_label(snr) -> str:
    return "inf" if np.isinf(snr) else str(int(snr))
