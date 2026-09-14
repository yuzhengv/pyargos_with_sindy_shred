"""Aggregate results/exp_*/decoder_ablation.json with the identified-model structure from tables/experiments.csv.

Writes tables/decoder_ablation.csv and appends a section to tables/summary.md: for the stage-1 & valid
Bayesian-ARGOS runs, the decoded-field degradation when each coordinate is frozen, and whether the
coordinate with the smallest dynamical role (the one nearest the near-zero eigenvalue's direction is not
identified here; we use the coordinate absent from the most equations, else the one with the smallest
own-equation coefficients) is the one the decoder needs least.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "results", "exp_*", "decoder_ablation.json")), key=lambda p: int(p.split("exp_")[-1].split(os.sep)[0])):
        r = json.load(open(f))
        rows.append({"iteration": r["iteration"], "rrmse_full": r["full"]["rrmse"],
                     **{f"rrmse_z{k}_frozen": r[f"z{k}_frozen"]["rrmse"] for k in (1, 2, 3)},
                     **{f"rmse_degC_z{k}_frozen": r[f"z{k}_frozen"]["rmse_degC"] for k in (1, 2, 3)},
                     "rmse_degC_full": r["full"]["rmse_degC"]})
    d = pd.DataFrame(rows)
    e = pd.read_csv(os.path.join(HERE, "tables", "experiments.csv"))
    ba = e[e.model == "ba_deg1"].set_index("iteration")
    t = pd.read_csv(os.path.join(HERE, "tables", "terms.csv")); t = t[t.model == "ba_deg1"]
    # coordinate absent from the most equations in the identified model ("slaved" candidate)
    absent = {}
    for it, g in t.groupby("iteration"):
        cnt = {f"z{k}": sum(1 for eq in ["dz1/dt", "dz2/dt", "dz3/dt"] if not ((g.equation == eq) & (g.term == f"x{k}")).any()) for k in (1, 2, 3)}
        absent[it] = max(cnt, key=cnt.get) if max(cnt.values()) > 0 else None
    d["stage1"] = d.iteration.map(ba.stage1); d["valid"] = d.iteration.map(ba.valid)
    d["most_absent_coord"] = d.iteration.map(absent)
    for k in (1, 2, 3):
        d[f"ratio_z{k}"] = d[f"rrmse_z{k}_frozen"] / d.rrmse_full
    d["least_needed_coord"] = d[[f"ratio_z{k}" for k in (1, 2, 3)]].idxmin(axis=1).str.replace("ratio_", "")
    d["min_ratio"] = d[[f"ratio_z{k}" for k in (1, 2, 3)]].min(axis=1)
    d["max_ratio"] = d[[f"ratio_z{k}" for k in (1, 2, 3)]].max(axis=1)
    d.to_csv(os.path.join(HERE, "tables", "decoder_ablation.csv"), index=False)
    g = d[d.stage1.fillna(False) & d.valid.fillna(False)]
    L = ["\n## Decoder ablation: is the third latent coordinate needed for the field?\n",
         f"Stage1 & valid Bayesian-ARGOS runs: n = {len(g)}. Decoded test-field rRMSE with one coordinate frozen at its training mean, relative to the full decode:",
         f"- least-needed coordinate: median ratio {g.min_ratio.median():.2f} (5-95 %: {g.min_ratio.quantile(.05):.2f}-{g.min_ratio.quantile(.95):.2f}); runs where freezing it costs < 10 %: {int((g.min_ratio < 1.1).sum())}; < 25 %: {int((g.min_ratio < 1.25).sum())}",
         f"- most-needed coordinate: median ratio {g.max_ratio.median():.2f}",
         f"- runs in which the identified model leaves one coordinate out of some equation: {int(g.most_absent_coord.notna().sum())}; in those, that coordinate is also the decoder's least-needed one in {int((g.most_absent_coord == g.least_needed_coord).sum())}"]
    open(os.path.join(HERE, "tables", "summary.md"), "a").write("\n".join(L) + "\n")
    print(f"{len(d)} experiments -> tables/decoder_ablation.csv; summary appended")


if __name__ == "__main__":
    main()
