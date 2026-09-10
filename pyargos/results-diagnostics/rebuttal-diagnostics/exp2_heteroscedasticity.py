"""Experiment 2 -- quantify the residual structure behind Fig. 3d / Fig. 4c.

Referee 1 Point 6 asks for a quantitative or qualitative measure of
heteroscedasticity, "such as the emergence of coherent structures in the
residual at high SNR". For every trial and SNR we fit the ground-truth
support by OLS (the posterior mean under the weakly informative priors used
in the HMC stage coincides with OLS to well within Monte-Carlo error; this is
verified in exp3) and report

    bp_lm, bp_p    Breusch-Pagan LM test, auxiliary regressor = fitted value
    white_lm, _p   White's general heteroscedasticity test
    var_ratio      max/min residual variance across deciles of fitted value
    rho1, DW       lag-1 autocorrelation and Durbin-Watson in time order
                   (coherent structure: rho1 -> 1, DW -> 0)

and a variance decomposition of the residual e into
    d = xdot_SG - f(x_clean)         derivative-estimation error
    g = f(x_clean) - Theta(x_SG) b*   state-smoothing error in the library
    h = Theta(x_SG) (b* - b_hat)      coefficient-estimation error
which shows what the residual *is* when the noise vanishes.

Systems: Aizawa x3-dot (Fig. 3d) and Rossler x1-dot (Fig. 4c), n = 5000.
Outputs (results/): exp2_stats.csv, exp2_examples.npz
"""

from __future__ import annotations

import argparse
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import common as C

N_FIXED = 5000
CASES = {
    "aizawa": dict(eq=3, snrs=[10, 20, 30, 40, 45, 50, 55, 57, 60, np.inf]),
    "rossler": dict(eq=1, snrs=[20, 30, 40, 50, 55, 58, 61, np.inf]),
}
EXAMPLE_TRIAL = 0
EXAMPLE_SNRS = {"aizawa": [50, 60, np.inf], "rossler": [61, np.inf]}


def true_beta(system: str, eq: int, terms):
    """Ground-truth coefficient vector in the order of ``terms``."""
    lookup = {
        ("aizawa", 3): {"x3": 0.95, "x1^2": -1, "x2^2": -1, "x1^2 x3": -0.25,
                        "x2^2 x3": -0.25, "x1^3 x3": 0.1, "x3^3": -1 / 3, "Intercept": 0.65},
        ("rossler", 1): {"x2": -1, "x3": -1, "Intercept": 0.0},
    }
    tab = lookup[(system, eq)]
    return np.array([tab[t] for t in terms]), tab["Intercept"]


def one(system: str, trial: int, ic, snr: float):
    eq = CASES[system]["eq"]
    tr = C.SYSTEMS[system]["truth"][eq]
    terms, intercept = tr["terms"], tr["intercept"]
    x_clean, x_noisy = C.simulate(system, N_FIXED, snr, ic, 20_000 + 100 * trial + (99 if np.isinf(snr) else int(snr)))
    dm = C.library(x_noisy)
    y = dm["xdot_filtered"][:, eq - 1]
    X = C.design_for_terms(dm, terms)
    st = C.residual_stats(y, X, intercept)
    e = st["residuals"]

    f_clean = C.rhs(system, x_clean)[:, eq - 1]
    b_star, b0 = true_beta(system, eq, terms)
    theta_bstar = X @ b_star + (b0 if intercept else 0.0)
    d = y - f_clean
    g = f_clean - theta_bstar
    h = theta_bstar - st["fitted"]
    # deterministic part of d: the SG derivative of the *clean* signal with the
    # window the pipeline chose on the noisy signal, minus the true derivative
    order, window = C.sg_params(x_noisy[:, eq - 1])
    d_trunc = C.sg_derivative(x_clean[:, eq - 1], order, window) - f_clean
    d_noise = d - d_trunc

    # the support the pipeline actually fits (screening passes 1+2, before HMC)
    scr = C.screen(dm, eq, n_threads=1)
    sel_terms, sel_int = scr["selected_terms"], scr["selected_intercept"]
    if len(sel_terms) >= 1:
        st_sel = C.residual_stats(y, C.design_for_terms(dm, sel_terms), sel_int)
    else:
        st_sel = {k: np.nan for k in st}
    row = dict(system=system, eq=eq, trial=trial, snr=snr, n=N_FIXED,
               sg_order=order, sg_window=window,
               **{k: v for k, v in st.items() if k not in ("residuals", "fitted")},
               **{f"sel_{k}": v for k, v in st_sel.items() if k not in ("residuals", "fitted")},
               selected=";".join(sel_terms), selected_intercept=sel_int,
               screen_exact_truth=set(sel_terms) == set(terms) and sel_int == intercept,
               truth_in_selected=set(terms) <= set(sel_terms),
               var_e=float(e.var()), var_d=float(d.var()), var_g=float(g.var()), var_h=float(h.var()),
               var_dtrunc=float(d_trunc.var()), var_dnoise=float(d_noise.var()),
               corr_e_d=float(np.corrcoef(e, d)[0, 1]),
               corr_e_dtrunc=float(np.corrcoef(e, d_trunc)[0, 1]),
               corr_e_g=float(np.corrcoef(e, g)[0, 1]),
               r2_e_on_d=float(np.corrcoef(e, d)[0, 1] ** 2),
               rho1_d=float(np.corrcoef(d[:-1], d[1:])[0, 1]),
               rho1_dtrunc=float(np.corrcoef(d_trunc[:-1], d_trunc[1:])[0, 1]),
               noise_sd_state=float(0.0 if np.isinf(snr) else 10 ** (-snr / 20) * x_clean[:, eq - 1].std()))
    example = None
    if trial == EXAMPLE_TRIAL and any(np.isclose(snr, s) or (np.isinf(snr) and np.isinf(s)) for s in EXAMPLE_SNRS[system]):
        example = dict(residuals=e, fitted=st["fitted"], d=d, d_trunc=d_trunc, g=g, x_clean=x_clean, y=y,
                       sel_residuals=st_sel.get("residuals", np.full_like(e, np.nan)),
                       sel_fitted=st_sel.get("fitted", np.full_like(e, np.nan)))
    return row, (system, snr, example)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tasks = []
    for system, cfg in CASES.items():
        ics = C.initial_conditions(system)
        for trial in range(args.trials):
            for snr in cfg["snrs"]:
                tasks.append((system, trial, ics.iloc[trial].values, snr))
    print(f"{len(tasks)} conditions on {args.jobs} workers")
    t0 = time.time()
    res = Parallel(n_jobs=args.jobs, backend="multiprocessing", verbose=5)(delayed(one)(*t) for t in tasks)
    pd.DataFrame([r for r, _ in res]).to_csv(os.path.join(args.out, "exp2_stats.csv"), index=False)
    examples = {}
    for _, (system, snr, ex) in res:
        if ex is not None:
            for k, v in ex.items():
                examples[f"{system}_{C.snr_label(snr)}_{k}"] = v
    np.savez_compressed(os.path.join(args.out, "exp2_examples.npz"), **examples)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
