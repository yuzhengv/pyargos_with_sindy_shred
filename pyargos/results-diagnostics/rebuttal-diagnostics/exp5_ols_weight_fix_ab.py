"""Experiment 5 -- does the OLS-weight change made in AgenticARGOS explain the
weaker Python screening on Aizawa x3-dot?  (Answer: no.)

The ``weights_method="ols"`` branch of ``argos_alasso`` fits raw X without an
intercept and without rescaling.  This is the intended ARGOS design: the R
reference implementation that produced the manuscript's benchmarks does the
same (``lsfit(intercept = FALSE)`` in ``R/argos_files.R``).  AgenticARGOS
(Experiments-reports/2026-08-10_argos_ols_weights_fix.md) replaced it with a
standardised fit with intercept because its LLM-designed libraries contain
mixed-scale terms such as exp(x); that situation does not arise with the
polynomial libraries used here.

This script applies that alternative in memory only (the repository code is
NOT modified and should not be) and runs the two-pass screening with and
without it on identical Aizawa data (20 trials; n = 10^3.7 and 10^4 at 49 dB,
10^3.7 at 60 dB).

Result (2026-09-07): the alternative shrinks the support from ~11.5 to ~6 terms
but admits x1^3x3 in only 10 % of trials (40 % with the original branch;
88-97 % in the stored R results).  So it neither explains nor closes the
R/Python gap, which must lie in the lasso path / cross-validation /
post-selection steps (glmnet vs adelie).

Output (results/): exp5_ols_weight_fix_ab.csv
"""
import os, sys, types, importlib.util
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, common as C
import src.argos_sparse_regression  # ensures the "src" package is importable for the relative import
from joblib import Parallel, delayed

SRC = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "argos_sparse_regression.py"))
code = open(SRC).read()
old = "ols_model = LinearRegression(fit_intercept=False).fit(X, y)\n                weight = ols_model.coef_"
new = "ols_model = LinearRegression(fit_intercept=True).fit(X_matrix, y)\n                weight = np.true_divide(ols_model.coef_, scale)"
assert old in code
fixed_code = code.replace(old, new)
spec = importlib.util.spec_from_loader("asr_fixed", loader=None, origin=SRC)
mod = types.ModuleType("asr_fixed"); mod.__file__ = SRC; mod.__package__ = "src"
sys.modules["asr_fixed"] = mod
exec(compile(fixed_code, SRC, "exec"), mod.__dict__)

def screen_with(module, dm, eq):
    theta = dm["sorted_theta"]; xdot = dm["xdot_filtered"][:, eq-1]
    data = np.column_stack([xdot, theta])
    init = module.argos_alasso(data, weights_method="ridge", ols_ps=True, n_threads=1)
    data2, dm2, _ = module.shrink_design_matrix_based_on_estimate(design_matrix=dm, data=data, initial_estimate=init)
    final = np.nan_to_num(module.argos_alasso(data2, weights_method="ols", ols_ps=True, n_threads=1))
    names = list(dm2["sorted_feature_names"]); nz = final != 0
    return bool(nz[0]), [nm for nm, k in zip(names, nz[1:]) if k]

import src.argos_sparse_regression as buggy  # the original (design-faithful) implementation
TRUTH = set(C.SYSTEMS["aizawa"]["truth"][3]["terms"])

def one(trial, k, snr):
    ic = C.initial_conditions("aizawa").iloc[trial].values
    n = int(round(10**k))
    _, xn = C.simulate("aizawa", n, snr, ic, 30_000 + trial)
    dm = C.library(xn)
    out = dict(trial=trial, log10n=k, snr=snr)
    for label, m in [("original", buggy), ("alt", mod)]:
        b0, sel = screen_with(m, dm, 3)
        out[f"{label}_p"] = len(sel); out[f"{label}_truth_in"] = TRUTH <= set(sel); out[f"{label}_exact"] = (set(sel) == TRUTH) and b0
        out[f"{label}_has_x13x3"] = "x1^3 x3" in sel; out[f"{label}_sel"] = ";".join(sel)
    return out

tasks = [(t, k, s) for t in range(20) for k, s in [(3.7, 49), (3.7, 60), (4.0, 49)]]
res = pd.DataFrame(Parallel(n_jobs=16, backend="multiprocessing")(delayed(one)(*t) for t in tasks))
res.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "exp5_ols_weight_fix_ab.csv"), index=False)
pd.set_option("display.width", 200)
print(res.groupby(["log10n","snr"])[["original_p","original_truth_in","original_exact","original_has_x13x3","alt_p","alt_truth_in","alt_exact","alt_has_x13x3"]].mean().round(2).to_string())
print("\nexample supports with the alternative weights (n=10^3.7, 49 dB):")
for s in res[(res.log10n==3.7)&(res.snr==49)].alt_sel.head(5): print("  ", s)
