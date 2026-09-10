"""Experiment 3 -- full Bayesian-ARGOS runs for the Fig. 3d cases.

Runs the complete pipeline (screening + HMC with the manuscript's custom
priors) for Aizawa x3-dot at SNR = 50, 60 and infinity on the same trial-0
data used in exp2, and checks that

  1. residuals about the posterior-predictive mean coincide with the OLS
     residuals on the same support (so exp2's OLS-based statistics describe
     the HMC fit as well), and
  2. the quantitative residual diagnostics of exp2 reproduce on the actual
     HMC output; PSIS-LOO Pareto k-hat counts are recorded too.

Outputs (results/): exp3_hmc_check.csv, exp3_hmc_residuals.npz
"""

from __future__ import annotations

import argparse
import os
import time

import arviz as az
import numpy as np
import pandas as pd

import common as C
from src.argos_bayesian_argos import BayesianArgos

SYSTEM, EQ, N_FIXED, TRIAL = "aizawa", 3, 5000, 0
SNRS = [50, 60, np.inf]


def run_case(snr):
    ic = C.initial_conditions(SYSTEM).iloc[TRIAL].values
    seed = 20_000 + 100 * TRIAL + (99 if np.isinf(snr) else int(snr))  # identical to exp2
    x_clean, x_noisy = C.simulate(SYSTEM, N_FIXED, snr, ic, seed)
    dm = C.library(x_noisy)
    t0 = time.time()
    model = BayesianArgos(design_matrix=dm, custom_prior=True, accelerator=False)
    res = model.run_straight(state_var_deriv=EQ, ci_level=0.90)
    secs = time.time() - t0
    idata = res["results"]
    res["model"].predict(idata, kind="response")
    y_obs = idata.observed_data["target"].values
    pp_mean = idata.posterior_predictive["target"].mean(dim=("chain", "draw")).values
    e_hmc = y_obs - pp_mean

    # OLS on the same (selected) support and on the ground-truth support
    sel_terms = [t for t in res["identified_model_df"]["Variable"] if t != "Intercept"]
    sel_int = "Intercept" in set(res["identified_model_df"]["Variable"])
    y = dm["xdot_filtered"][:, EQ - 1]
    st_sel = C.residual_stats(y, C.design_for_terms(dm, sel_terms), sel_int)
    truth = C.SYSTEMS[SYSTEM]["truth"][EQ]
    st_truth = C.residual_stats(y, C.design_for_terms(dm, truth["terms"]), truth["intercept"])
    # residual stats computed directly on the HMC residuals
    st_hmc = _stats_from_residuals(e_hmc, pp_mean)
    loo = az.loo(idata, pointwise=True)
    khat = np.asarray(loo.pareto_k)
    row = dict(
        snr=snr, hmc_seconds=secs,
        identified=";".join(res["identified_model_df"]["Variable"]),
        truth_recovered=set(sel_terms) == set(truth["terms"]) and sel_int,
        corr_hmc_vs_ols_selected=float(np.corrcoef(e_hmc, st_sel["residuals"])[0, 1]),
        max_abs_diff_hmc_vs_ols_selected=float(np.max(np.abs(e_hmc - st_sel["residuals"]))),
        rel_diff_hmc_vs_ols_selected=float(np.std(e_hmc - st_sel["residuals"]) / np.std(e_hmc)),
        n_khat_gt_0p7=int((khat > 0.7).sum()), khat_max=float(khat.max()),
        **{f"hmc_{k}": v for k, v in st_hmc.items()},
        **{f"ols_sel_{k}": v for k, v in st_sel.items() if k not in ("residuals", "fitted")},
        **{f"ols_truth_{k}": v for k, v in st_truth.items() if k not in ("residuals", "fitted")},
    )
    arrays = {f"{C.snr_label(snr)}_residuals": e_hmc, f"{C.snr_label(snr)}_pp_mean": pp_mean,
              f"{C.snr_label(snr)}_khat": khat, f"{C.snr_label(snr)}_x_clean": x_clean}
    return row, arrays


def _stats_from_residuals(e, fitted, n_bins=10):
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.stattools import durbin_watson

    bp_lm, bp_p, _, _ = het_breuschpagan(e, sm.add_constant(fitted))
    order = np.argsort(fitted)
    v = np.array([b.var(ddof=1) for b in np.array_split(e[order], n_bins)])
    return dict(sigma=float(e.std(ddof=1)), bp_lm=float(bp_lm), bp_p=float(bp_p),
                var_ratio=float(v.max() / v.min()),
                rho1=float(np.corrcoef(e[:-1], e[1:])[0, 1]),
                durbin_watson=float(durbin_watson(e)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows, arrays = [], {}
    for snr in SNRS:
        row, arr = run_case(snr)
        print({k: v for k, v in row.items() if not k.startswith("ols_")})
        rows.append(row)
        arrays.update(arr)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "exp3_hmc_check.csv"), index=False)
    np.savez_compressed(os.path.join(args.out, "exp3_hmc_residuals.npz"), **arrays)


if __name__ == "__main__":
    main()
