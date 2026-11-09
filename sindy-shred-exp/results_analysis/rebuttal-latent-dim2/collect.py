"""Paired comparison of the r = 2 retrainings with the existing r = 3 networks.

Reads results/train_<it>_r<r>.json (reconstruction), results/identification_r<r>/exp_<it>/summary.json
(identification and forecast on the r = 2 latent trajectory) and the r = 3 rows of
../rebuttal-latent-models/tables/experiments.csv. Writes tables/latent_dim_comparison.csv and
tables/summary.md (paired counts and medians, Wilcoxon signed-rank p-values where n allows).

Two conventions are reported. "All 20 networks": the latent model is identified for every
network whether or not it passes the stage-1 reconstruction filter (the original design of this
folder). "Filter-passing convention": validity is counted among the networks that pass the
filter and forecast errors and periods are summarized over the valid models, as the 150-experiment
batch does; the Supplementary Information (Table S4) reports this convention since 2026-11-09.
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
    L = [f"# Latent dimension {r} versus 3: paired comparison on {len(d)} sensor sets\n", "## All 20 networks (identification run whether or not the network passes the filter)\n"]
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
    # Filter-passing convention (as in ../rebuttal-latent-models: validity among filter-passing networks)
    from scipy.stats import mannwhitneyu
    p3, pr = d.stage1_r3.astype(bool), d[f"stage1_r{r}"].astype(bool)
    L.append(f"\n## Filter-passing convention (SI Table S4): validity among networks passing the filter, medians over the valid models\n")
    L.append(f"- networks passing the filter (rRMSE <= 0.05): r={r} {int(pr.sum())} of {len(d)}, r=3 {int(p3.sum())} of {len(d)}")
    for key, name in [("ba", "Bayesian-ARGOS"), ("sindy", "SINDy")]:
        if f"{key}_valid_r{r}" not in d.columns:
            continue
        v3, vr = d[f"{key}_valid_r3"].fillna(False).astype(bool), d[f"{key}_valid_r{r}"].fillna(False).astype(bool)
        L.append(f"- {name} valid model among passing networks: r={r} {int((pr & vr).sum())} of {int(pr.sum())}, r=3 {int((p3 & v3).sum())} of {int(p3.sum())}")
        a, b = d.loc[p3 & v3, f"{key}_rmse_r3"].dropna(), d.loc[pr & vr, f"{key}_rmse_r{r}"].dropna()
        pv = mannwhitneyu(a, b).pvalue if len(a) and len(b) else float("nan")
        L.append(f"- {name} decoded-forecast RMSE 0-275 wk (degC), medians over the valid passing models: r={r} {b.median():.3f} (n = {len(b)}) vs r=3 {a.median():.3f} (n = {len(a)}); Mann-Whitney U p = {pv:.3f}")
        ok = p3 & pr & v3 & vr
        x, y = d.loc[ok, f"{key}_rmse_r{r}"], d.loc[ok, f"{key}_rmse_r3"]
        L.append(f"- {name} decoded-forecast RMSE, pairs passing and valid for both: n = {int(ok.sum())}" + (f", median r={r} {x.median():.3f} vs r=3 {y.median():.3f}, r={r} lower in {int((x < y).sum())}" if ok.sum() else ""))
        q3, qr = d.loc[p3 & v3, f"{key}_period_r3"].dropna(), d.loc[pr & vr, f"{key}_period_r{r}"].dropna()
        L.append(f"- {name} identified period (yr), valid passing models: r={r} median {qr.median():.3f}, max |period - 1| {float((qr - 1).abs().max()):.3f} (n = {len(qr)}); r=3 median {q3.median():.3f}, max |period - 1| {float((q3 - 1).abs().max()):.3f} (n = {len(q3)})")
    open(os.path.join(HERE, "tables", "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
