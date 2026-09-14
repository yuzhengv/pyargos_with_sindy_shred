"""Aggregate results/exp_*/summary.json into tables with the manuscript's selection flags.

Writes to tables/:
  experiments.csv   one row per experiment x model: flags, term counts, latent MSE, forecast RMSE,
                    modal quantities (period, decays, stability) and their posterior quantiles (BA)
  terms.csv         one row per experiment x model x equation x term
  summary.md        counts per selection subset and distributions of the Point 8 quantities

Selection (as in results_analysis/performance_assessment.py):
  stage1  : network test reconstruction rRMSE <= 0.05 (eval mode here; the submitted batch used a dropout draw)
  valid   : per model, training latent MSE <= 1 on every dimension (and the simulation finite)
Subsets reported: all runs; stage1; stage1 & valid (per model); stage1 & valid for both BA and SINDy
(the paired subset). Nothing is dropped from the CSVs; the flags are columns.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    files = sorted(glob.glob(os.path.join(HERE, "results", "exp_*", "summary.json")),
                   key=lambda p: int(p.split("exp_")[-1].split(os.sep)[0]))
    rows, terms = [], []
    for f in files:
        s = json.load(open(f))
        it = s["iteration"]
        for tag, rec in s["models"].items():
            r = {"iteration": it, "model": tag, "method": rec["method"], "degree": rec["degree"],
                 "recon_rrmse_eval": s["recon_rrmse_test_eval"],
                 "recon_rrmse_dropout_draw": s.get("recon_rrmse_test_dropout_draw", np.nan),
                 "stage1": s["stage1_pass_eval"], "valid": rec["valid"],
                 "n_terms": rec["n_terms"], "n_nonlinear_terms": rec["n_nonlinear_terms"],
                 "affine_linear": rec["affine_linear"],
                 "latent_mse_train": rec["latent_mse_train"]["overall"],
                 "latent_mse_train_max_dim": max(rec["latent_mse_train"]["per_dim"]),
                 "latent_mse_test": rec["latent_mse_test"]["overall"],
                 "fit_seconds": rec["fit_seconds"]}
            r.update({k: v for k, v in rec["forecast"].items()})
            md = rec.get("modal")
            if md:
                osc = [m for m in md["modes"] if m["kind"] == "oscillatory"]
                real = [m for m in md["modes"] if m["kind"] == "real"]
                r.update(max_real_eig=md["max_real_eig"], spectrally_stable=md["max_real_eig"] < 0,
                         n_oscillatory_pairs=len(osc), n_real_modes=len(real),
                         period_yr=osc[0]["period_yr"] if osc else np.nan,
                         slow_decay_rate=osc[0]["decay_rate"] if osc else np.nan,
                         slow_half_life_yr=osc[0]["half_life_yr"] if osc else np.nan,
                         fast_decay_rate=real[0]["decay_rate"] if real else np.nan,
                         fast_e_folding_yr=real[0]["e_folding_yr"] if real else np.nan)
                if md.get("equilibrium"):
                    r.update({f"equilibrium_z{i + 1}": v for i, v in enumerate(md["equilibrium"])})
            mp = rec.get("modal_posterior")
            if mp:
                for k in ["osc_period_yr", "osc_decay_rate", "real_decay_rate"]:
                    r[f"{k}_q05"], r[f"{k}_q95"] = mp[k]["q05"], mp[k]["q95"]
                r["p_stable"], r["p_slow_mode_decaying"] = mp["p_stable"], mp["p_slow_mode_decaying"]
            rows.append(r)
            for i, (ts, cs) in enumerate(zip(rec["terms"], rec["coefficients"])):
                for t, c in zip(ts, cs):
                    terms.append({"iteration": it, "model": tag, "equation": f"dz{i + 1}/dt",
                                  "term": t or "1", "coefficient": c, "stage1": s["stage1_pass_eval"], "valid": rec["valid"]})
    df, tdf = pd.DataFrame(rows), pd.DataFrame(terms)
    os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
    df.to_csv(os.path.join(HERE, "tables", "experiments.csv"), index=False)
    tdf.to_csv(os.path.join(HERE, "tables", "terms.csv"), index=False)

    # ---- summary.md ----
    L = [f"# Identified latent models: {df.iteration.nunique()} experiments\n"]
    n1 = df.drop_duplicates("iteration").stage1.sum()
    L.append(f"Stage 1 (rRMSE <= 0.05, eval mode): {n1} of {df.iteration.nunique()} experiments pass.\n")
    L.append("| model | runs | stage1 | stage1 & valid | spectrally stable (stage1 & valid) | affine-linear | mean terms | runs with nonlinear terms |")
    L.append("|---|---|---|---|---|---|---|---|")
    for tag, g in df.groupby("model", sort=False):
        s1 = g[g.stage1]; sv = s1[s1.valid]
        L.append(f"| {tag} | {len(g)} | {len(s1)} | {len(sv)} | {int(sv.get('spectrally_stable', pd.Series(dtype=bool)).fillna(False).sum())} "
                 f"| {int(sv.affine_linear.sum())} | {sv.n_terms.mean():.2f} | {int((sv.n_nonlinear_terms > 0).sum())} |")
    # paired subset for degree 1
    ba = df[(df.model == "ba_deg1")].set_index("iteration"); si = df[(df.model == "sindy_deg1")].set_index("iteration")
    common = ba.index.intersection(si.index)
    paired = [i for i in common if ba.loc[i, "stage1"] and ba.loc[i, "valid"] and si.loc[i, "valid"]]
    L.append(f"\nPaired subset (stage1, both degree-1 models valid): {len(paired)} experiments.\n")
    one = df.drop_duplicates("iteration")
    if "recon_rrmse_dropout_draw" in one.columns and one.recon_rrmse_dropout_draw.notna().any():
        L.append(f"Stage 1 evaluated with one dropout-active draw instead (as the submitted batch did): "
                 f"{int((one.recon_rrmse_dropout_draw <= 0.05).sum())} pass; passing under both: "
                 f"{int((one.stage1 & (one.recon_rrmse_dropout_draw <= 0.05)).sum())}.\n")
    L.append("**Validity and spectral stability.** Invalid = training latent MSE > 1 on some dimension "
             "(or a non-finite simulation); an affine-linear model with an eigenvalue of positive real part grows exponentially.\n")
    for tag, g in df.groupby("model", sort=False):
        inv = g[g.stage1 & ~g.valid]
        if "max_real_eig" in g.columns and inv.max_real_eig.notna().any():
            L.append(f"- {tag}: {len(inv)} invalid stage1 runs; max real eigenvalue > 0 in {int((inv.max_real_eig > 0).sum())} of them "
                     f"(median {inv.max_real_eig.median():+.3f} /yr); non-finite simulations {int(inv.latent_mse_train.isna().sum())}")
        elif len(inv):
            L.append(f"- {tag}: {len(inv)} invalid stage1 runs; non-finite simulations {int(inv.latent_mse_train.isna().sum())}")
    for tag in [t for t in df.model.unique() if "deg3" in t]:
        g = df[(df.model == tag) & df.stage1 & df.valid]
        if len(g):
            L.append(f"- {tag}: of {len(g)} valid runs, test-period forecast non-finite in {int(g.RMSE_0_275.isna().sum())}, "
                     f"test latent MSE > 1 in {int((g.latent_mse_test > 1).sum())}, decoded RMSE > 5 degC in {int((g.RMSE_0_275 > 5).sum())} "
                     f"(validity is judged on the training period only)")
    L.append("\n**Mode structure** of the stage1 & valid affine-linear models:\n")
    for tag, g in df.groupby("model", sort=False):
        g = g[g.stage1 & g.valid & g.affine_linear.fillna(False)]
        if len(g) and "n_oscillatory_pairs" in g.columns:
            fd = g.fast_decay_rate.dropna()
            L.append(f"- {tag} (n = {len(g)}): one oscillatory pair + one real mode in {int(((g.n_oscillatory_pairs == 1) & (g.n_real_modes == 1)).sum())} runs; "
                     f"period within 1.00 +/- 0.05 yr in {int((abs(g.period_yr - 1) <= 0.05).sum())}; "
                     f"annual mode decaying (real part < 0) in {int((g.slow_decay_rate > 0).sum())}; "
                     f"real mode decaying in {int((fd > 0).sum())}, fast (e-folding < 2 yr) in {int(((fd > 0) & (1 / fd < 2)).sum())}, "
                     f"near-neutral (|rate| < 0.05 /yr) in {int((fd.abs() < 0.05).sum())}")
    if "p_slow_mode_decaying" in df.columns:
        g = df[(df.model == "ba_deg1") & df.stage1 & df.valid]
        L.append(f"- ba_deg1 posterior: annual mode decaying with P > 0.95 in {int((g.p_slow_mode_decaying > 0.95).sum())} runs, growing with P > 0.95 in "
                 f"{int((g.p_slow_mode_decaying < 0.05).sum())}; real mode decaying with P > 0.95 (q05 > 0) in {int((g.real_decay_rate_q05 > 0).sum())} of {len(g)}")
    for col, label in [("period_yr", "period (yr)"), ("slow_decay_rate", "annual-mode decay rate (/yr; negative = growing)"),
                       ("fast_decay_rate", "real-mode decay rate (/yr; negative = growing)"), ("RMSE_0_275", "decoded-forecast RMSE 0-275 (degC)")]:
        if col not in df.columns:
            continue
        L.append(f"\n**{label}**, median [5 %, 95 %] over stage1 & valid runs:\n")
        for tag, g in df.groupby("model", sort=False):
            v = g[g.stage1 & g.valid][col].dropna()
            if len(v):
                L.append(f"- {tag}: {v.median():.4f} [{v.quantile(0.05):.4f}, {v.quantile(0.95):.4f}] (n = {len(v)})")
    if "p_slow_mode_decaying" in df.columns:
        v = df[(df.model == "ba_deg1") & df.stage1 & df.valid]["p_slow_mode_decaying"].dropna()
        if len(v):
            L.append(f"\nBayesian-ARGOS degree 1: posterior probability that the annual mode is decaying, median over runs {v.median():.2f}; "
                     f"runs with P > 0.95: {(v > 0.95).sum()} of {len(v)}.")
    if len(paired):
        L.append(f"\n**Paired comparison** on the {len(paired)} experiments where both degree-1 models are valid: "
                 f"decoded-forecast RMSE 0-275 median Bayesian-ARGOS {ba.loc[paired, 'RMSE_0_275'].median():.3f} vs SINDy {si.loc[paired, 'RMSE_0_275'].median():.3f} degC; "
                 f"Bayesian-ARGOS lower in {int((ba.loc[paired, 'RMSE_0_275'] < si.loc[paired, 'RMSE_0_275']).sum())}. "
                 f"Stage1 runs valid for Bayesian-ARGOS only: {int((ba.stage1 & ba.valid & ~si.valid.reindex(ba.index).fillna(False)).sum())}; "
                 f"valid for SINDy only: {int((ba.stage1 & ~ba.valid & si.valid.reindex(ba.index).fillna(False)).sum())}.")
    L.append("\n**Term frequency** (share of stage1 & valid runs in which the term is identified, per equation):\n")
    for tag, g in tdf.groupby("model", sort=False):
        g = g[g.stage1 & g.valid]
        nrun = g.iteration.nunique()
        if nrun == 0:
            continue
        freq = g.groupby(["equation", "term"]).iteration.nunique().div(nrun).unstack(fill_value=0).round(2)
        L.append(f"\n{tag} (n = {nrun}):\n")
        L.append(freq.to_markdown())
    open(os.path.join(HERE, "tables", "summary.md"), "w").write("\n".join(L) + "\n")
    print(f"{len(files)} experiments -> tables/experiments.csv, tables/terms.csv, tables/summary.md")


if __name__ == "__main__":
    main()
