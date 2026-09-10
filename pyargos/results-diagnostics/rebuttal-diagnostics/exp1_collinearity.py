"""Experiment 1 -- where does the Aizawa x3-dot collinearity come from?

Answers Referee 1 Point 6 ("why per-term VIF?", "system or library?") and
Referee 2 minor point 2 ("VIF on the original or the reduced library?",
"identifiability on the attractor rather than classical multicollinearity").

Part A (sweeps, as in Fig. 3b but over many trials): for every trial and every
(n, SNR) we build the library exactly as the pipeline does and compute
per-term VIF and the Belsley condition number on four nested designs:

    full      : the 55-term degree-5 candidate library
    trimmed   : the degree-truncated library after screening pass 1
                (this is what the published Fig. 3b was computed on)
    selected  : the support returned by screening pass 2 (what enters HMC)
    truth     : the 7 ground-truth non-constant terms of x3-dot

Part B (geometry, n = 5000): the *same* monomials are evaluated on
    trajectory_sg     : SG-smoothed noisy trajectory (the pipeline's design)
    trajectory_clean  : the clean trajectory, no noise, no smoothing
    box_uniform       : uniform samples in the attractor's bounding box
    multi_transient   : K short clean trajectories from random initial
                        conditions (transients included), concatenated
    multi_transient_sg: the same K segments, noisy (49 dB) and SG-filtered
                        segment by segment
so that attractor geometry, library size and smoothing can be separated.

Outputs (results/): exp1_vif_long.csv, exp1_kappa.csv, exp1_geometry.csv,
exp1_near_dependencies.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker; parallelism is over trials

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import common as C

EQ = 3
SYSTEM = "aizawa"
N_GRID = C.log_grid(2.0, 5.0, 0.2)
SNR_GRID = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, np.inf]
SNR_FIXED = 49
N_FIXED = 5000
K_SEGMENTS = 50


def designs_for(dm: dict, scr: dict, truth_terms):
    names = list(dm["sorted_feature_names"])
    theta = dm["sorted_theta"]
    out = {
        "full": (theta, names),
        "trimmed": (scr["trimmed_theta"], scr["trimmed_names"]),
        "truth": (C.theta_columns(theta, names, truth_terms), list(truth_terms)),
    }
    sel = scr["selected_terms"]
    if len(sel) >= 2:
        out["selected"] = (C.theta_columns(theta, names, sel), sel)
    return out


def one_condition(trial: int, ic, k: float, n: int, snr: float, sweep: str):
    t0 = time.time()
    noise_seed = 10_000 * trial + int(round(k * 10)) + (0 if sweep == "n" else 5000 + int(0 if np.isinf(snr) else snr))
    _, x_noisy = C.simulate(SYSTEM, n, snr, ic, noise_seed)
    dm = C.library(x_noisy)
    scr = C.screen(dm, EQ, n_threads=1)
    truth = C.SYSTEMS[SYSTEM]["truth"][EQ]["terms"]
    vif_rows, kappa_rows = [], []
    for design, (X, names) in designs_for(dm, scr, truth).items():
        v = C.vif(X, names)
        bel = C.belsley(X, names, intercept=True)
        for term, val in v.items():
            vif_rows.append(
                dict(trial=trial, sweep=sweep, log10n=k, n=n, snr=snr,
                     design=design, term=term, degree=C.monomial_degree(term),
                     is_truth=term in truth, vif=val)
            )
        kappa_rows.append(
            dict(trial=trial, sweep=sweep, log10n=k, n=n, snr=snr, design=design,
                 p=len(names), kappa=bel["condition_number"],
                 vif_max=float(np.nanmax(v.values)), vif_median=float(np.nanmedian(v.values)),
                 n_vif_gt10=int((v.values > 10).sum()))
        )
    meta = dict(trial=trial, sweep=sweep, log10n=k, n=n, snr=snr,
                trimmed_p=len(scr["trimmed_names"]),
                trimmed_max_degree=max(C.monomial_degree(t) for t in scr["trimmed_names"]),
                selected=";".join(scr["selected_terms"]),
                selected_intercept=scr["selected_intercept"],
                # pass-2 support before the HMC stage prunes it (not the final model)
                screen_exact_truth=set(scr["selected_terms"]) == set(truth) and scr["selected_intercept"],
                truth_in_selected=set(truth) <= set(scr["selected_terms"]),
                seconds=time.time() - t0)
    return vif_rows, kappa_rows, meta


def clean_vs_n(trial: int, ic):
    """Clean trajectory (no noise, no smoothing) over the n grid: separates
    the trajectory-length effect from the smoothing effect on collinearity."""
    truth = C.SYSTEMS[SYSTEM]["truth"][EQ]["terms"]
    n_max = N_GRID[-1][1]
    x_clean, _ = C.simulate(SYSTEM, n_max, np.inf, ic, 0)
    rows = []
    for k, n in N_GRID:
        th, nm = C.poly_features(x_clean[:n])
        for design, (X, names) in {"full": (th, nm), "truth": (C.theta_columns(th, nm, truth), truth)}.items():
            v = C.vif(X, names)
            rows.append(dict(trial=trial, log10n=k, n=n, design=design,
                             kappa=C.belsley(X, names)["condition_number"],
                             vif_max=float(np.nanmax(v.values)), vif_median=float(np.nanmedian(v.values))))
    return rows


def geometry(trial: int, ic, rng_seed: int):
    """Part B: same monomials, different sampling of state space."""
    truth = C.SYSTEMS[SYSTEM]["truth"][EQ]["terms"]
    rng = np.random.default_rng(rng_seed)
    rows = []

    def record(label, theta, names):
        for design, (X, nm) in {
            "full": (theta, names),
            "truth": (C.theta_columns(theta, names, truth), truth),
        }.items():
            v = C.vif(X, nm)
            bel = C.belsley(X, nm, intercept=True)
            rows.append(dict(trial=trial, sampling=label, design=design, p=len(nm),
                             kappa=bel["condition_number"],
                             vif_max=float(np.nanmax(v.values)),
                             vif_median=float(np.nanmedian(v.values)),
                             vif_truth_json=json.dumps({t: float(v[t]) for t in truth if t in v.index})))

    x_clean, x_noisy = C.simulate(SYSTEM, N_FIXED, SNR_FIXED, ic, 777 + trial)
    dm = C.library(x_noisy)
    record("trajectory_sg", dm["sorted_theta"], list(dm["sorted_feature_names"]))
    th, nm = C.poly_features(x_clean)
    record("trajectory_clean", th, nm)

    lo, hi = x_clean.min(axis=0), x_clean.max(axis=0)
    box = rng.uniform(lo, hi, size=(N_FIXED, 3))
    th, nm = C.poly_features(box)
    record("box_uniform", th, nm)

    r = C.SYSTEMS[SYSTEM]["ic_ranges"]
    seg_len = N_FIXED // K_SEGMENTS
    segs_clean, segs_sg = [], []
    for j in range(K_SEGMENTS):
        ic_j = [rng.uniform(*r[0]), rng.uniform(*r[1]), rng.uniform(*r[2])]
        xc, xn = C.simulate(SYSTEM, seg_len, SNR_FIXED, ic_j, 9_000 + 100 * trial + j)
        segs_clean.append(xc)
        segs_sg.append(C.library(xn)["sorted_theta"])
    th, nm = C.poly_features(np.vstack(segs_clean))
    record("multi_transient", th, nm)
    record("multi_transient_sg", np.vstack(segs_sg), nm)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20, help="trials for the SNR sweep and Part B")
    ap.add_argument("--trials-n", type=int, default=10, help="trials for the n sweep (n = 1e5 screening is slow)")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ics = C.initial_conditions(SYSTEM)
    tasks = []
    for trial in range(max(args.trials, args.trials_n)):
        ic = ics.iloc[trial].values
        if trial < args.trials_n:
            for k, n in N_GRID:
                tasks.append((trial, ic, k, n, SNR_FIXED, "n"))
        if trial < args.trials:
            for snr in SNR_GRID:
                tasks.append((trial, ic, np.log10(N_FIXED), N_FIXED, snr, "snr"))
    tasks.sort(key=lambda t: -t[3])  # longest jobs first for better packing
    print(f"Part A: {len(tasks)} conditions on {args.jobs} workers", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=args.jobs, backend="multiprocessing", verbose=5)(delayed(one_condition)(*t) for t in tasks)
    vif_long = pd.DataFrame([r for v, _, _ in res for r in v])
    kappa = pd.DataFrame([r for _, kk, _ in res for r in kk])
    meta = pd.DataFrame([m for _, _, m in res])
    vif_long.to_csv(os.path.join(args.out, "exp1_vif_long.csv"), index=False)
    kappa.to_csv(os.path.join(args.out, "exp1_kappa.csv"), index=False)
    meta.to_csv(os.path.join(args.out, "exp1_screening_meta.csv"), index=False)
    print(f"Part A done in {time.time() - t0:.0f}s")

    print("Part B: geometry")
    t0 = time.time()
    geo = Parallel(n_jobs=args.jobs, backend="multiprocessing", verbose=5)(
        delayed(geometry)(trial, ics.iloc[trial].values, 5_000 + trial) for trial in range(args.trials)
    )
    pd.DataFrame([r for g in geo for r in g]).to_csv(os.path.join(args.out, "exp1_geometry.csv"), index=False)
    cvn = Parallel(n_jobs=args.jobs, backend="multiprocessing", verbose=5)(
        delayed(clean_vs_n)(trial, ics.iloc[trial].values) for trial in range(args.trials)
    )
    pd.DataFrame([r for g in cvn for r in g]).to_csv(os.path.join(args.out, "exp1_clean_vs_n.csv"), index=False)
    print(f"Part B done in {time.time() - t0:.0f}s", flush=True)

    # Near-dependency structure for one representative trial (Belsley)
    truth = C.SYSTEMS[SYSTEM]["truth"][EQ]["terms"]
    ic = ics.iloc[0].values
    _, x_noisy = C.simulate(SYSTEM, N_FIXED, SNR_FIXED, ic, 10_000)
    dm = C.library(x_noisy)
    scr = C.screen(dm, EQ)
    report = {}
    for design, (X, names) in designs_for(dm, scr, truth).items():
        bel = C.belsley(X, names, intercept=True)
        report[design] = dict(
            p=len(names), condition_number=bel["condition_number"],
            near_dependencies=C.near_dependencies(bel),
            vif={t: float(v) for t, v in C.vif(X, names).items()},
        )
    with open(os.path.join(args.out, "exp1_near_dependencies.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("wrote near-dependency report")


if __name__ == "__main__":
    main()
