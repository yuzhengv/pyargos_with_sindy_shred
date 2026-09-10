"""Experiment 4 -- where in the pipeline are true terms lost, and does VIF
predict it?  (Aizawa x3-dot, stored R benchmark results)

Reads the per-trial supports exported by
``code-repos/Bayesian-ARGOS/ResultsAnalysis/rebuttal-diagnostics/aizawa_screening_vs_hmc.R``
(``aizawa_eq3_supports.csv``) and, for the n sweep at 49 dB and the SNR sweep at
n = 5000, tabulates for every ground-truth term

    admitted   fraction of trials in which the adaptive-lasso pass-2 support
               contains the term (what enters HMC)
    kept       fraction of trials in which the final model contains it

so that losses can be attributed to the screening stage or to the 90 %
credible-interval rule.  For the trials that lose a true term it lists the
spurious terms that were selected instead ("substitutes"), and for the two
extreme n it computes the VIF of every true term *inside the admitted support*
(clean-trajectory design; VIF is a property of the design only) together with
the survival of the term, to test whether within-support VIF predicts loss.

Outputs (results/): exp4_term_rates.csv, exp4_substitutes.csv,
exp4_vif_in_support.csv
"""

from __future__ import annotations

import argparse
import collections
import os
import re

import numpy as np
import pandas as pd

import common as C

TRUTH_R = ["(Intercept)", "x3", "x1^2", "x2^2", "x1^2x3", "x2^2x3", "x3^3", "x1^3x3"]
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SUPPORTS = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "..", "Bayesian-ARGOS", "ResultsAnalysis", "rebuttal-diagnostics", "aizawa_eq3_supports.csv"))


def to_sklearn(term: str) -> str:
    """R naming 'x1^2x3' -> sklearn naming 'x1^2 x3'."""
    return " ".join(re.findall(r"x\d(?:\^\d)?", term))


def term_rates(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sweep, value), g in d.groupby(["sweep", "value"]):
        p2 = [set(s.split("|")) for s in g.pass2]
        fi = [set(s.split("|")) for s in g.final]
        for t in TRUTH_R:
            rows.append(dict(sweep=sweep, value=value, trials=len(g), term=t,
                             admitted=np.mean([t in s for s in p2]), kept=np.mean([t in s for s in fi])))
        rows.append(dict(sweep=sweep, value=value, trials=len(g), term="<exact truth>",
                         admitted=np.mean([s == set(TRUTH_R) for s in p2]), kept=np.mean([s == set(TRUTH_R) for s in fi])))
    return pd.DataFrame(rows)


def substitutes(d: pd.DataFrame, levels) -> pd.DataFrame:
    rows = []
    for sweep, value in levels:
        g = d[(d.sweep == sweep) & (d.value == value)]
        fi = [set(s.split("|")) for s in g.final]
        for t in ["x1^2x3", "x2^2x3", "x1^3x3", "x3^3"]:
            lost = [s for s in fi if t not in s]
            comp = collections.Counter(u for s in lost for u in s - set(TRUTH_R))
            for u, c in comp.most_common(6):
                rows.append(dict(sweep=sweep, value=value, lost_term=t, n_lost=len(lost), substitute=u,
                                 frac_of_lost_trials=c / len(lost)))
    return pd.DataFrame(rows)


def vif_in_support(d: pd.DataFrame, levels) -> pd.DataFrame:
    ic = C.initial_conditions("aizawa").iloc[0].values
    n_max = max(int(round(10 ** v)) for s, v in levels if s == "n")
    x_all, _ = C.simulate("aizawa", n_max, np.inf, ic, 0)
    rows = []
    for sweep, value in levels:
        n = int(round(10 ** value)) if sweep == "n" else 5000
        th, nm = C.poly_features(x_all[:n])
        g = d[(d.sweep == sweep) & (d.value == value)]
        for _, r in g.iterrows():
            terms = [to_sklearn(t) for t in r.pass2.split("|") if t != "(Intercept)"]
            terms = [t for t in terms if t in nm]
            if len(terms) < 2:
                continue
            v = C.vif(C.theta_columns(th, nm, terms), terms)
            final = set(r.final.split("|"))
            for t in TRUTH_R[1:]:
                ts = to_sklearn(t)
                if ts in v.index:
                    rows.append(dict(sweep=sweep, value=value, trial=r.trial, p_admitted=len(terms), term=t,
                                     vif_in_support=float(v[ts]), kept=t in final))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--supports", default=DEFAULT_SUPPORTS)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = pd.read_csv(args.supports, keep_default_na=False).drop_duplicates(["sweep", "value", "trial"])  # empty final model -> ""
    d["value"] = d["value"].astype(float)

    rates = term_rates(d)
    rates.to_csv(os.path.join(args.out, "exp4_term_rates.csv"), index=False)
    levels = [("n", 3.7), ("n", 5.0), ("snr", 60.0), ("snr", np.inf)]
    subs = substitutes(d, levels)
    subs.to_csv(os.path.join(args.out, "exp4_substitutes.csv"), index=False)
    vs = vif_in_support(d, [("n", 3.7), ("n", 5.0)])
    vs.to_csv(os.path.join(args.out, "exp4_vif_in_support.csv"), index=False)

    pd.set_option("display.width", 200)
    print(rates[rates.sweep == "n"].pivot(index="value", columns="term", values=["admitted", "kept"]).round(2).to_string())
    print(subs.to_string(index=False))
    for (sweep, value), g in vs.groupby(["sweep", "value"]):
        print(f"\n{sweep}={value}: median p admitted {g.p_admitted.median():.0f}")
        print(g.groupby("term").agg(vif_median=("vif_in_support", "median"), vif_q75=("vif_in_support", lambda x: x.quantile(.75)),
                                    kept=("kept", "mean"), trials=("kept", "size")).round(2).to_string())


if __name__ == "__main__":
    main()
