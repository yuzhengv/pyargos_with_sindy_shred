"""Paired comparison of the r = 2 retrainings with the existing r = 3 networks.

Reads results/train_<it>_r<r>.json (reconstruction), results/identification_r<r>/exp_<it>/summary.json
(identification and forecast on the r = 2 latent trajectory) and the r = 3 rows of
../rebuttal-latent-models/tables/experiments.csv. Writes tables/latent_dim_comparison.csv and
tables/summary.md (paired counts and medians, Wilcoxon signed-rank p-values where n allows).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latent-dim", type=int, default=2)
    a = p.parse_args(); r = a.latent_dim
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "results", f"train_*_r{r}.json"))):
        t = json.load(open(f)); it = t["iteration"]
        row = {"iteration": it, f"rrmse_r{r}": t["recon_rrmse_test_eval"], f"stage1_r{r}": t["stage1_pass_eval"], f"train_seconds_r{r}": t["seconds"]}
        s = os.path.join(HERE, "results", f"identification_r{r}", f"exp_{it}", "summary.json")
        if os.path.exists(s):
            m = json.load(open(s))["models"]
            for tag, key in [("ba_deg1", "ba"), ("sindy_deg1", "sindy")]:
                if tag in m:
                    rec = m[tag]
                    row[f"{key}_valid_r{r}"] = rec["valid"]; row[f"{key}_rmse_r{r}"] = rec["forecast"].get("RMSE_0_275", np.nan)
                    row[f"{key}_latent_mse_r{r}"] = rec["latent_mse_train"]["overall"]
                    md = rec.get("modal")
                    if md:
                        osc = [mo for mo in md["modes"] if mo["kind"] == "oscillatory"]
                        row[f"{key}_period_r{r}"] = osc[0]["period_yr"] if osc else np.nan
                        row[f"{key}_max_real_eig_r{r}"] = md["max_real_eig"]
        rows.append(row)
    d = pd.DataFrame(rows).set_index("iteration")
    e = pd.read_csv(os.path.join(HERE, "..", "rebuttal-latent-models", "tables", "experiments.csv"))
    for tag, key in [("ba_deg1", "ba"), ("sindy_deg1", "sindy")]:
        g = e[e.model == tag].set_index("iteration")
        d[f"{key}_valid_r3"] = g.valid.reindex(d.index); d[f"{key}_rmse_r3"] = g.RMSE_0_275.reindex(d.index)
        d[f"{key}_latent_mse_r3"] = g.latent_mse_train.reindex(d.index); d[f"{key}_period_r3"] = g.period_yr.reindex(d.index)
    g = e[e.model == "ba_deg1"].set_index("iteration")
    d["rrmse_r3"] = g.recon_rrmse_eval.reindex(d.index); d["stage1_r3"] = g.stage1.reindex(d.index)
    os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
    d.to_csv(os.path.join(HERE, "tables", "latent_dim_comparison.csv"))

    from scipy.stats import wilcoxon
    L = [f"# Latent dimension {r} versus 3: paired comparison on {len(d)} sensor sets\n"]
    def paired(col, label, lower_better=True, valid_key=None):
        x, y = d[f"{col}_r{r}"], d[f"{col}_r3"]; ok = x.notna() & y.notna()
        if valid_key:  # restrict to pairs where both identified models are valid
            ok &= d[f"{valid_key}_valid_r{r}"].fillna(False).astype(bool) & d[f"{valid_key}_valid_r3"].fillna(False).astype(bool)
        x, y = x[ok], y[ok]
        if len(x) < 5:
            return f"- {label}: n = {len(x)} (too few for a test)"
        pval = wilcoxon(x, y).pvalue if (x != y).any() else 1.0
        better = int((x < y).sum()) if lower_better else int((x > y).sum())
        return (f"- {label}: median r={r} {x.median():.3f} vs r=3 {y.median():.3f}; r={r} better in {better} of {len(x)} pairs; "
                f"median paired difference {(x - y).median():+.3f}; Wilcoxon p = {pval:.3f}")
    L.append(paired("rrmse", "test reconstruction rRMSE (eval mode)"))
    L.append(f"- stage-1 pass (rRMSE <= 0.05): r={r} {int(d[f'stage1_r{r}'].sum())} of {len(d)}, r=3 {int(d.stage1_r3.sum())} of {len(d)}; both {int((d[f'stage1_r{r}'] & d.stage1_r3).sum())}, r={r} only {int((d[f'stage1_r{r}'] & ~d.stage1_r3).sum())}, r=3 only {int((~d[f'stage1_r{r}'] & d.stage1_r3).sum())}")
    for key, name in [("ba", "Bayesian-ARGOS"), ("sindy", "SINDy")]:
        if f"{key}_valid_r{r}" in d.columns:
            L.append(f"- {name} valid model: r={r} {int(d[f'{key}_valid_r{r}'].fillna(False).sum())}, r=3 {int(d[f'{key}_valid_r3'].fillna(False).sum())} of {len(d)}")
            L.append(paired(f"{key}_rmse", f"{name} decoded-forecast RMSE 0-275 wk (degC), pairs valid for both", valid_key=key))
            L.append(paired(f"{key}_rmse", f"{name} decoded-forecast RMSE 0-275 wk (degC), all pairs with a finite forecast"))
            L.append(paired(f"{key}_latent_mse", f"{name} training latent MSE"))
            L.append(paired(f"{key}_period", f"{name} identified period (yr)", lower_better=False).replace("better in", "larger in"))
    open(os.path.join(HERE, "tables", "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
