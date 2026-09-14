"""Log the identified latent models of one SINDy-SHRED experiment — Referee 1, Point 8.

For one trained latent space (experiment <it>) this script follows the batch
scripts `sindy-shred-exp/analysis_script_ba.py` / `analysis_script_sindy.py`
(same split, scaler, network, latent normalisation, simulation horizons and
metric definitions) and additionally records everything needed to answer the
Point 8 questions across the 150 experiments:

* the identified equations of Bayesian-ARGOS and of the SINDy baseline
  (STLSQ, threshold 0.8, alpha 0.05, finite differences), each with the
  manuscript's degree-1 library and with a degree-3 polynomial library;
* for Bayesian-ARGOS, the posterior mean, sd and credible interval of every
  candidate term (screened-out terms recorded as absent);
* the E-SINDy coefficients learnt inside the network (the latent model the
  SINDy-SHRED training itself converged to);
* for every affine-linear model: matrix form, equilibrium, eigenvalues, and per
  mode the period, decay rate and half-life; for Bayesian-ARGOS also the
  posterior distribution of those quantities over the HMC draws;
* the selection quantities of the manuscript: network test reconstruction rRMSE
  (stage 1, threshold 0.05; in eval mode and, for reference, one dropout-active
  draw as the submitted batch computed it), and per model the training latent
  MSE per dimension (stage 2, "valid" if all <= 1), plus the test latent MSE and
  the decoded-forecast RMSE over the Fig. 8a horizons.

Outputs go to results/exp_<it>/: summary.json (everything scalar), one CSV per
model with terms and coefficients, posterior_<tag>.csv for Bayesian-ARGOS,
modes_<tag>.csv with the modal analysis, and modes_draws_<tag>.csv with the
per-draw modal quantities. Nothing is filtered here; `collect.py` applies the
selection flags when it aggregates.
"""
import argparse
import json
import os
import random
import sys
import time
import warnings

import arviz as az
import numpy as np
import pandas as pd
import pysindy as ps
import torch
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.dirname(os.path.abspath(__file__))
SHRED_EXP = os.path.dirname(os.path.dirname(HERE))
sys.path.append(SHRED_EXP)
sys.path.append(os.path.dirname(SHRED_EXP))

import src.sindy as sindy  # noqa: E402
import src.sindy_shred as sindy_shred  # noqa: E402
from utils.processdata import TimeSeriesDataset, load_data_with_path  # noqa: E402
import pyargos.utils.argos_utils as agu  # noqa: E402
from pyargos.src.argos_bayesian_argos import BayesianArgos  # noqa: E402
from pyargos.utils.argos_simulator import solve_ode_odeint  # noqa: E402

LAGS = 52
LATENT_DIM = 3
DT = 1 / 52.0
N_SIM_TRAIN = 1040          # analysis scripts: BA simulate(n=1040); SINDy t = arange(0, 20, 1/52)
N_SIM_TEST = 317
FORECAST_RANGES = [(0, 100), (100, 200), (200, 275), (0, 275)]
VALID_LATENT_MSE = 1.0      # analysis-functions: valid if every training latent-MSE dimension <= 1
STAGE1_RRMSE = 0.05


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--exp-root", default=SHRED_EXP, help="sindy-shred-exp folder holding Data/ and exp-1/")
    p.add_argument("--data", default=None)
    p.add_argument("--num-sensors", type=int, default=250)
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    p.add_argument("--degrees", default="1,3", help="library degrees to run for both methods")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-dropout-rrmse", action="store_true")
    return p.parse_args()


# ----------------------------------------------------------------------------- helpers
def to_matrix(terms, coeffs, d=LATENT_DIM):
    """(A, b) of an affine-linear model given per-equation term names; None if any term is nonlinear."""
    A, b = np.zeros((d, d)), np.zeros(d)
    for i, (ts, cs) in enumerate(zip(terms, coeffs)):
        for t, c in zip(ts, cs):
            if t in ("", "1"):
                b[i] += c
            elif t.startswith("x") and t[1:].isdigit():
                A[i, int(t[1:]) - 1] += c
            else:
                return None, None
    return A, b


def modal_analysis(A, b):
    """Equilibrium, eigenvalues and per-mode period / decay / half-life (years)."""
    out = {"max_real_eig": float(np.max(np.linalg.eigvals(A).real))}
    try:
        out["equilibrium"] = (-np.linalg.solve(A, b)).tolist()
    except np.linalg.LinAlgError:
        out["equilibrium"] = None
    modes = []
    seen = set()
    for lam in np.linalg.eigvals(A):
        key = (round(lam.real, 10), round(abs(lam.imag), 10))
        if key in seen:
            continue
        seen.add(key)
        m = {"real": float(lam.real), "imag": float(abs(lam.imag)), "decay_rate": float(-lam.real),
             "half_life_yr": float(np.log(2) / -lam.real) if lam.real < 0 else np.inf,
             "e_folding_yr": float(1 / -lam.real) if lam.real < 0 else np.inf}
        if abs(lam.imag) > 1e-9:
            m.update(kind="oscillatory", omega_rad_yr=float(abs(lam.imag)), period_yr=float(2 * np.pi / abs(lam.imag)))
        else:
            m.update(kind="real", omega_rad_yr=0.0, period_yr=np.inf)
        modes.append(m)
    out["modes"] = modes
    return out


def latent_mse(z, z_sim):
    n = min(len(z), len(z_sim))
    per_dim = [float(np.mean((z[:n, i] - z_sim[:n, i]) ** 2)) for i in range(z.shape[1])]
    return {"per_dim": per_dim, "overall": float(np.mean(per_dim)), "n_compared": int(n),
            "finite": bool(np.all(np.isfinite(z_sim[:n])))}


def decode(shred, latent):
    z = torch.FloatTensor(latent).to(next(shred.parameters()).device)
    out = shred.linear1(z); out = shred.dropout(out); out = torch.nn.functional.relu(out)
    out = shred.linear2(out); out = shred.dropout(out); out = torch.nn.functional.relu(out)
    out = shred.linear3(out)
    return out.detach().cpu().numpy()


def forecast_metrics(shred, sc, z_sim_test_n, lo, hi, truth_degC):
    """Decode a simulated test latent trajectory and score it as the batch scripts do."""
    if not np.all(np.isfinite(z_sim_test_n)):
        return {f"RMSE_{s}_{e}": np.nan for s, e in FORECAST_RANGES} | {f"RMSE_normalized_{s}_{e}": np.nan for s, e in FORECAST_RANGES}
    z = (np.asarray(z_sim_test_n) + 1) / 2 * (hi - lo) + lo
    pred = sc.inverse_transform(decode(shred, z))
    res = {}
    for s, e in FORECAST_RANGES:
        if e > len(pred) or e > len(truth_degC):
            continue
        d = pred[s:e] - truth_degC[s:e]
        res[f"RMSE_{s}_{e}"] = float(np.sqrt(np.mean(d ** 2)))
        res[f"RMSE_normalized_{s}_{e}"] = float(np.linalg.norm(d) / np.linalg.norm(truth_degC[s:e]))
    return res


def ba_posterior_table(ba, candidate_names, ci_level):
    """Posterior summary for every candidate term of every equation (absent = screened out)."""
    rows = []
    for k, eq in enumerate(sorted(ba.results.keys()), start=1):
        idata = ba.results[eq]["results"]
        post = az.extract(idata)
        hdi = az.hdi(idata, hdi_prob=ci_level)
        ident = ba.results[eq]["identified_model_df"]
        ident = dict(zip(ident["Variable"], ident["Coefficient"])) if ident is not None else {}
        names = [v for v in idata.posterior.data_vars if not v.startswith("sigma")]
        for nm in ["Intercept"] + list(candidate_names):
            key = nm.replace(" ", "") if nm != "Intercept" else nm
            hit = [v for v in names if v.replace("`", "").replace(" ", "") == key]
            if hit:
                v = hit[0]
                s = post[v].values
                rows.append({"equation": f"dz{k}/dt", "term": "1" if nm == "Intercept" else nm.replace(" ", ""),
                             "in_posterior": True, "mean": float(s.mean()), "sd": float(s.std()),
                             "hdi_low": float(hdi[v].values[0]), "hdi_high": float(hdi[v].values[1]),
                             "identified": bool(v in ident and ident[v] != 0)})
            else:
                rows.append({"equation": f"dz{k}/dt", "term": "1" if nm == "Intercept" else nm.replace(" ", ""),
                             "in_posterior": False, "mean": np.nan, "sd": np.nan, "hdi_low": np.nan,
                             "hdi_high": np.nan, "identified": False})
    return pd.DataFrame(rows)


def ba_modes_over_draws(ba, terms, coeffs, n_draws=1000, seed=0):
    """Modal quantities for posterior draws of an affine-linear Bayesian-ARGOS model."""
    A0, _ = to_matrix(terms, coeffs)
    if A0 is None:
        return None
    eqs = sorted(ba.results.keys())
    posts = [az.extract(ba.results[eq]["results"]) for eq in eqs]
    n_avail = min(int(p.sizes["sample"]) for p in posts)
    rng = np.random.RandomState(seed)
    idx = rng.choice(n_avail, size=min(n_draws, n_avail), replace=False)
    rows = []
    for j in idx:
        A = np.zeros((LATENT_DIM, LATENT_DIM))
        for i, p in enumerate(posts):
            for v in p.data_vars:
                nm = v.replace("`", "").replace(" ", "")
                if nm.startswith("x") and nm[1:].isdigit():
                    A[i, int(nm[1:]) - 1] = float(p[v].values[j])
        lam = np.linalg.eigvals(A)
        osc = [l for l in lam if abs(l.imag) > 1e-9]
        real = [l for l in lam if abs(l.imag) <= 1e-9]
        rows.append({"draw": int(j), "max_real_eig": float(np.max(lam.real)),
                     "osc_omega_rad_yr": float(abs(osc[0].imag)) if osc else np.nan,
                     "osc_period_yr": float(2 * np.pi / abs(osc[0].imag)) if osc else np.nan,
                     "osc_decay_rate": float(-osc[0].real) if osc else np.nan,
                     "real_decay_rate": float(-real[0].real) if real else np.nan,
                     "n_oscillatory_pairs": len(osc) // 2})
    return pd.DataFrame(rows)


def sindy_fit(z_train_n, degree):
    model = ps.SINDy(optimizer=ps.STLSQ(threshold=0.8, alpha=0.05),
                     differentiation_method=ps.differentiation.FiniteDifference(),
                     feature_library=ps.PolynomialLibrary(degree=degree))
    model.fit(z_train_n, t=DT)
    names = [n.replace(" ", "").replace("x0", "X1").replace("x1", "X2").replace("x2", "X3").lower()
             for n in model.get_feature_names()]           # pysindy 'x0 x1', 'x0^2' -> 'x1x2', 'x1^2'
    names = ["1" if n == "1" else n for n in names]
    C = model.coefficients()
    terms = [[names[j] for j in range(len(names)) if C[i, j] != 0] for i in range(C.shape[0])]
    coeffs = [[float(C[i, j]) for j in range(len(names)) if C[i, j] != 0] for i in range(C.shape[0])]
    return model, terms, coeffs


def safe_sim(fn):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = np.asarray(fn(), dtype=float)
        return x
    except Exception as e:  # divergence, LSODA failure, overflow
        return np.full((1, LATENT_DIM), np.nan)


def load_experiment(a, summary):
    """Data split, scaler, network (eval mode), stage-1 rRMSE and normalised latent trajectories, as in the batch scripts.
    Fills summary with the stage-1 quantities; returns (shred, sc, z_train_n, z_test_n, lo, hi, truth)."""
    it = a.iteration
    data_path = a.data or os.path.join(a.exp_root, "Data", "SST_data.mat")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ---- data, split, scaler, network (as in the batch scripts) ----
    load_X = load_data_with_path("SST", data_path)
    n, m = load_X.shape
    sensor_locations = np.load(os.path.join(a.exp_root, "exp-1", "sensor-location", f"sensor_location_{it}.npy"))
    train_idx = np.arange(0, 1000)
    mask = np.ones(n - LAGS); mask[train_idx] = 0
    rest = np.arange(0, n - LAGS)[np.where(mask != 0)[0]]
    test_idx = rest[30:]
    sc = MinMaxScaler().fit(load_X[train_idx])
    X = sc.transform(load_X)
    all_in = np.zeros((n - LAGS, LAGS, a.num_sensors))
    for i in range(len(all_in)):
        all_in[i] = X[i:i + LAGS, sensor_locations]
    t32 = lambda arr: torch.tensor(arr, dtype=torch.float32).to(device)  # noqa: E731
    train_ds = TimeSeriesDataset(t32(all_in[train_idx]), t32(X[train_idx + LAGS - 1]))
    test_ds = TimeSeriesDataset(t32(all_in[test_idx]), t32(X[test_idx + LAGS - 1]))
    library_dim = sindy.library_size(LATENT_DIM, 3, False, True)
    shred = sindy_shred.SINDy_SHRED(
        a.num_sensors, m, hidden_size=LATENT_DIM, hidden_layers=2, l1=350, l2=400, dropout=0.1,
        library_dim=library_dim, poly_order=3, include_sine=False, dt=DT * 0.1, layer_norm=False).to(device)
    shred.load_state_dict(torch.load(os.path.join(a.exp_root, "exp-1", "saved-model", f"model_{it}.pth")))

    # ---- stage-1 quantity: network test reconstruction rRMSE ----
    truth = sc.inverse_transform(test_ds.Y.detach().cpu().numpy())
    with torch.no_grad():
        if not a.skip_dropout_rrmse:
            shred.train()
            rec = sc.inverse_transform(shred(test_ds.X).detach().cpu().numpy())
            summary["recon_rrmse_test_dropout_draw"] = float(np.linalg.norm(rec - truth) / np.linalg.norm(truth))
        shred.eval()
        rec = sc.inverse_transform(shred(test_ds.X).detach().cpu().numpy())
        summary["recon_rrmse_test_eval"] = float(np.linalg.norm(rec - truth) / np.linalg.norm(truth))
        summary["stage1_pass_eval"] = bool(summary["recon_rrmse_test_eval"] <= STAGE1_RRMSE)
        # ---- latent trajectories, normalised to [-1, 1] with the training min/max ----
        z_train = shred.gru_outputs(train_ds.X, sindy=True)[0][:, 0, :].cpu().numpy()
        z_test = shred.gru_outputs(test_ds.X, sindy=True)[0][:, 0, :].cpu().numpy()
    lo, hi = z_train.min(axis=0), z_train.max(axis=0)
    z_train_n = 2 * (z_train - lo) / (hi - lo) - 1
    z_test_n = 2 * (z_test - lo) / (hi - lo) - 1
    summary["latent_range_train"] = (hi - lo).tolist()

    return shred, sc, z_train_n, z_test_n, lo, hi, truth


# ----------------------------------------------------------------------------- main
def main():
    a = parse_args()
    it = a.iteration
    t0 = time.time()
    degrees = [int(v) for v in a.degrees.split(",")]
    out_dir = os.path.join(a.out, f"exp_{it}")
    os.makedirs(out_dir, exist_ok=True)
    data_path = a.data or os.path.join(a.exp_root, "Data", "SST_data.mat")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = {"iteration": it, "seed": a.seed, "degrees": degrees}

    shred, sc, z_train_n, z_test_n, lo, hi, truth = load_experiment(a, summary)
    data_path = a.data or os.path.join(a.exp_root, "Data", "SST_data.mat")

    # ---- E-SINDy coefficients learnt inside the network ----
    C = (shred.e_sindy.coefficients * shred.e_sindy.coefficient_mask).detach().cpu().numpy()
    np.save(os.path.join(out_dir, "esindy_coefficients.npy"), C)   # (replicates, library terms, latent dims)
    C_mean = C.mean(axis=0) if C.ndim == 3 else C
    summary["esindy_shape"] = list(C.shape)
    summary["esindy_nonzero_terms_mean"] = int(np.count_nonzero(np.abs(C_mean) > 1e-8))
    np.savetxt(os.path.join(out_dir, "esindy_coefficients_mean.csv"), C_mean, delimiter=",",
               header="ensemble mean; rows = library terms (degree-3 polynomial, network order), cols = latent dims")

    models = {}
    for deg in degrees:
        # ------------------------------------------------ Bayesian-ARGOS
        tag = f"ba_deg{deg}"
        t1 = time.time()
        dm = agu.build_design_matrix(x_t=z_train_n, dt=DT, library_degree=deg, sg_poly_order=4, library_type="poly")
        ba = BayesianArgos(design_matrix=dm, accelerator=True)
        ba.run(mode="straight", parallel="yes", ncpus=None, ci_level=0.90)
        coeffs, terms = ba.expressions_for_simulation()
        rec = {"method": "Bayesian-ARGOS", "degree": deg, "fit_seconds": round(time.time() - t1, 1),
               "n_candidates": int(len(dm["sorted_feature_names"])) + 1,
               "terms": terms, "coefficients": coeffs,
               "n_terms": int(sum(len(t) for t in terms)),
               "n_nonlinear_terms": int(sum(1 for ts in terms for t in ts if t not in ("", "1") and not (t.startswith("x") and t[1:].isdigit())))}
        ba_posterior_table(ba, dm["sorted_feature_names"], 0.90).to_csv(os.path.join(out_dir, f"posterior_{tag}.csv"), index=False)
        z_sim_tr = safe_sim(lambda: ba.simulate(n=N_SIM_TRAIN, dt=DT, init_conditions=z_train_n[0], variable_coeff=coeffs, variable_names=terms))
        z_sim_te = safe_sim(lambda: ba.simulate(n=N_SIM_TEST, dt=DT, init_conditions=z_test_n[0], variable_coeff=coeffs, variable_names=terms))
        rec["latent_mse_train"] = latent_mse(z_train_n, z_sim_tr)
        rec["latent_mse_test"] = latent_mse(z_test_n, z_sim_te)
        rec["valid"] = bool(rec["latent_mse_train"]["finite"] and all(v <= VALID_LATENT_MSE for v in rec["latent_mse_train"]["per_dim"]))
        rec["forecast"] = forecast_metrics(shred, sc, z_sim_te, lo, hi, truth)
        A, b = to_matrix(terms, coeffs)
        rec["affine_linear"] = A is not None
        if A is not None:
            rec["A"], rec["b"] = A.tolist(), b.tolist()
            rec["modal"] = modal_analysis(A, b)
            draws = ba_modes_over_draws(ba, terms, coeffs)
            if draws is not None:
                draws.to_csv(os.path.join(out_dir, f"modes_draws_{tag}.csv"), index=False)
                q = draws.quantile([0.05, 0.5, 0.95])
                rec["modal_posterior"] = {c: {"q05": float(q.loc[0.05, c]), "median": float(q.loc[0.5, c]), "q95": float(q.loc[0.95, c])}
                                          for c in ["osc_period_yr", "osc_decay_rate", "real_decay_rate", "max_real_eig"]}
                rec["modal_posterior"]["p_stable"] = float((draws["max_real_eig"] < 0).mean())
                rec["modal_posterior"]["p_slow_mode_decaying"] = float((draws["osc_decay_rate"] > 0).mean())
        models[tag] = rec
        # ------------------------------------------------ SINDy baseline
        tag = f"sindy_deg{deg}"
        t1 = time.time()
        model, terms, coeffs = sindy_fit(z_train_n, deg)
        rec = {"method": "SINDy", "degree": deg, "fit_seconds": round(time.time() - t1, 2),
               "n_candidates": int(len(model.get_feature_names())),
               "terms": terms, "coefficients": coeffs,
               "n_terms": int(sum(len(t) for t in terms)),
               "n_nonlinear_terms": int(sum(1 for ts in terms for t in ts if t != "1" and not (t.startswith("x") and t[1:].isdigit())))}
        z_sim_tr = safe_sim(lambda: model.simulate(z_train_n[0], np.arange(0, 20, DT)))
        z_sim_te = safe_sim(lambda: model.simulate(z_test_n[0], np.arange(0, N_SIM_TEST * DT, DT)))
        rec["latent_mse_train"] = latent_mse(z_train_n, z_sim_tr)
        rec["latent_mse_test"] = latent_mse(z_test_n, z_sim_te)
        rec["valid"] = bool(rec["latent_mse_train"]["finite"] and all(v <= VALID_LATENT_MSE for v in rec["latent_mse_train"]["per_dim"]))
        rec["forecast"] = forecast_metrics(shred, sc, z_sim_te, lo, hi, truth)
        A, b = to_matrix(terms, coeffs)
        rec["affine_linear"] = A is not None
        if A is not None:
            rec["A"], rec["b"] = A.tolist(), b.tolist()
            rec["modal"] = modal_analysis(A, b)
        models[tag] = rec

    # ---- write ----
    for tag, rec in models.items():
        rows = [{"equation": f"dz{i + 1}/dt", "term": t or "1", "coefficient": c}
                for i, (ts, cs) in enumerate(zip(rec["terms"], rec["coefficients"])) for t, c in zip(ts, cs)]
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"model_{tag}.csv"), index=False)
        if "modal" in rec:
            pd.DataFrame(rec["modal"]["modes"]).to_csv(os.path.join(out_dir, f"modes_{tag}.csv"), index=False)
    summary["models"] = models
    summary["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else str(o))
    # console digest
    print(f"exp {it}: rRMSE eval {summary['recon_rrmse_test_eval']:.4f} (stage1 {'pass' if summary['stage1_pass_eval'] else 'FAIL'})", end="")
    if "recon_rrmse_test_dropout_draw" in summary:
        print(f", dropout draw {summary['recon_rrmse_test_dropout_draw']:.4f}", end="")
    print()
    for tag, rec in models.items():
        eqs = "; ".join(" ".join(f"{c:+.2f}{'' if t in ('', '1') else t}" for t, c in zip(ts, cs)) for ts, cs in zip(rec["terms"], rec["coefficients"]))
        lm = rec["latent_mse_train"]["overall"]
        md = rec.get("modal", {})
        osc = [mo for mo in md.get("modes", []) if mo["kind"] == "oscillatory"]
        real = [mo for mo in md.get("modes", []) if mo["kind"] == "real"]
        extra = f" | period {osc[0]['period_yr']:.3f} yr, slow decay {osc[0]['decay_rate']:+.4f}/yr" if osc else ""
        extra += f", fast e-fold {real[0]['e_folding_yr']:.2f} yr" if real else ""
        print(f"  {tag:10s} valid={rec['valid']!s:5s} latentMSE_train={lm:8.4f} terms={rec['n_terms']:2d} nonlin={rec['n_nonlinear_terms']:2d}{extra}")
        print(f"             {eqs}")
    print(f"  done in {summary['seconds']} s")


if __name__ == "__main__":
    main()
