"""Propagate the Bayesian-ARGOS coefficient posterior through the identified
latent system and the SINDy-SHRED decoder for one SST experiment, and draw the
credible bands the revised Fig. 8b, Fig. S9 and the per-horizon RMSE need.
Referee 1, Point 8 ("error bars").

Pipeline (identical to `results_analysis/analysis_script_ba.py` up to the
forecast): data split, scaler, network in eval mode, latent normalisation,
Bayesian-ARGOS on the training latent trajectory (degree-1 library, 90 %
credible intervals). Then, instead of simulating the posterior-mean model only:

* `n_draws` joint draws of the coefficients are taken from the HMC posterior of
  each equation (the same draw index across equations is irrelevant because the
  three equations are fitted independently, so draws are paired at random);
* each draw is simulated over the training period (Fig. 8b band) and over the
  test period from the test initial condition, decoded to the field, and scored
  by its global RMSE at every lead time (RMSE band) and at the 18 held-out
  sensors of Fig. S9 (sensor bands);
* two draw sets are evaluated: `identified` (terms whose 90 % interval excludes
  zero, i.e. the structure of Eq. 5, with the other terms fixed at zero) and
  `allterms` (every term that survived the screening step, whatever its
  interval). The figures use `identified`; `allterms` is a sensitivity check.

The bands carry coefficient uncertainty only, conditional on the screened
support: encoder/decoder weights are fixed, and the regression noise `sigma` is
not injected as process noise. The SST data, sensor locations and weights are
git-ignored; point --exp-root at a checkout that has them.

Outputs (results/, figures/; <it> = iteration):
  identified_model_<it>.csv           posterior-mean model (as in Fig. 8)
  latent_bands_<it>.npz               gru/point/q05/q50/q95 latent trajectories, train and test
  sensor_bands_<it>.npz               obs/point/q05/q95 at the Fig. S9 sensors (normalised units)
  rmse_lead_time_<it>.csv             per-week global RMSE: point model, draw quantiles, mean-of-draws forecast
  rmse_horizons_<it>.csv              the same at t = 50, 100, 150, 200, 250 plus field coverage of the 90 % band
  sensor_stats_<it>.csv               per-panel RMSE and coverage (Fig. S9)
  rmse_draws_<it>_<set>.csv           per-draw RMSE at the five horizons
  figures/comparison_plot_ba_<it>_bands.pdf, multi_sensor_forecast_<it>_bands.pdf, rmse_lead_time_<it>.pdf
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SHRED_EXP = os.path.dirname(os.path.dirname(HERE))
sys.path.append(SHRED_EXP)
sys.path.append(os.path.dirname(SHRED_EXP))
sys.path.append(os.path.join(os.path.dirname(HERE), "rebuttal-latent-models"))

import pyargos.utils.argos_utils as agu  # noqa: E402
from pyargos.src.argos_bayesian_argos import BayesianArgos  # noqa: E402
from run_experiment import load_experiment, decode  # noqa: E402  (data split, network, latents; decoder pass)
from plot_bands import plot_latent_bands, plot_sensor_bands, plot_rmse_lead_time  # noqa: E402

DT = 1 / 52.0
N_SIM_TRAIN = 1040
HORIZONS = [50, 100, 150, 200, 250]
CI = 0.90
Q = (0.05, 0.50, 0.95)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-root", default=SHRED_EXP, help="sindy-shred-exp folder holding Data/ and exp-1/")
    p.add_argument("--data", default=None)
    p.add_argument("--iteration", type=int, default=91)
    p.add_argument("--num-sensors", type=int, default=250)
    p.add_argument("--n-draws", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sensors-from", default=os.path.join(os.path.dirname(HERE), "rebuttal-sst-forecast-figs", "results"),
                   help="folder with sensor_trajectories_<it>.npz, whose 18-sensor sample Fig. S9 uses")
    p.add_argument("--out", default=HERE)
    return p.parse_args()


def _norm(name):
    return name.replace("`", "").replace(" ", "")


def posterior_draw_sets(ba, n_draws, seed):
    """Return {set_name: (coeff_draws, names)} where coeff_draws[eq] is (n_draws, n_terms) and names[eq] the
    variable names in ba.simulate's format ('' = intercept, 'x2', ...)."""
    eqs = sorted(ba.results.keys())
    posts = [az.extract(ba.results[eq]["results"]) for eq in eqs]
    n_avail = min(int(p.sizes["sample"]) for p in posts)
    rng = np.random.RandomState(seed)
    idx = [rng.choice(n_avail, size=min(n_draws, n_avail), replace=False) for _ in eqs]   # independent equations
    out = {}
    for set_name in ("identified", "allterms"):
        coeffs, names = [], []
        for i, eq in enumerate(eqs):
            post = posts[i]
            pvars = {_norm(v): v for v in post.data_vars if not v.startswith("sigma")}
            ident = ba.results[eq]["identified_model_df"]
            keep = [v for v, c in zip(ident["Variable"], ident["Coefficient"]) if set_name == "allterms" or c != 0]
            keep = [_norm(v) for v in keep if _norm(v) in pvars]
            draws = np.column_stack([post[pvars[v]].values[idx[i]] for v in keep]) if keep else np.zeros((len(idx[i]), 0))
            coeffs.append(draws)
            names.append(["" if v == "Intercept" else v for v in keep])
        out[set_name] = (coeffs, names, idx)
    return out


def simulate(ba, coeffs, names, n, z0):
    z = np.asarray(ba.simulate(variable_coeff=coeffs, variable_names=names, n=n, dt=DT, init_conditions=z0))
    return z[:n]                                        # np.arange(0, n*dt, dt) can return n+1 points


def main():
    a = parse_args()
    it = a.iteration
    res_dir, fig_dir = os.path.join(a.out, "results"), os.path.join(a.out, "figures")
    os.makedirs(res_dir, exist_ok=True); os.makedirs(fig_dir, exist_ok=True)
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    t0 = time.time()

    # ---- experiment (data split, scaler, eval-mode network, normalised latents), as in the batch scripts ----
    ns = SimpleNamespace(iteration=it, data=a.data, exp_root=a.exp_root, num_sensors=a.num_sensors,
                         skip_dropout_rrmse=True, latent_dim=3, model_path=None)
    summary = {}
    shred, sc, z_train_n, z_test_n, lo, hi, truth = load_experiment(ns, summary)
    n_test = min(len(z_test_n), len(truth))             # GRU output and target differ by one row at the end
    z_test_n, truth = z_test_n[:n_test], truth[:n_test]
    truth_sea = truth                                   # the loader already keeps only the ocean points
    print(f"loaded exp {it}: test weeks {n_test}, ocean points {truth.shape[1]}, "
          f"stage-1 rRMSE {summary['recon_rrmse_test_eval']:.4f}  [{time.time() - t0:.0f}s]")

    # ---- Bayesian-ARGOS on the training latent trajectory ----
    dm = agu.build_design_matrix(x_t=z_train_n, dt=DT, library_degree=1, sg_poly_order=4, library_type="poly")
    ba = BayesianArgos(design_matrix=dm, accelerator=True)
    ba.run(mode="straight", parallel="yes", ncpus=None, ci_level=CI)
    coeffs0, names0 = ba.expressions_for_simulation()
    pd.DataFrame([{"equation": f"dz{i + 1}/dt", "term": nm or "1", "coefficient": c}
                  for i, (cs, ns_) in enumerate(zip(coeffs0, names0)) for c, nm in zip(cs, ns_)]
                 ).to_csv(os.path.join(res_dir, f"identified_model_{it}.csv"), index=False)
    for i, (cs, ns_) in enumerate(zip(coeffs0, names0)):
        print(f"dz{i + 1}/dt = " + " ".join(f"{c:+.3f}{(' ' + nm) if nm else ''}" for c, nm in zip(cs, ns_)))
    print(f"Bayesian-ARGOS done [{time.time() - t0:.0f}s]")

    # ---- point forecast (posterior-mean model), as in Fig. 8 ----
    def to_latent_units(z):
        return (z + 1) / 2 * (hi - lo) + lo

    def forecast_field(z_sim_n):
        pred_n = decode(shred, to_latent_units(z_sim_n))          # normalised units
        return pred_n, sc.inverse_transform(pred_n)             # degC on the ocean points

    z_train_pt = simulate(ba, coeffs0, names0, N_SIM_TRAIN, z_train_n[0])
    z_test_pt = simulate(ba, coeffs0, names0, n_test, z_test_n[0])
    pred_n_pt, pred_pt = forecast_field(z_test_pt)
    rmse_pt = np.sqrt(np.mean((truth_sea - pred_pt) ** 2, axis=1))
    sensors = np.load(os.path.join(a.sensors_from, f"sensor_trajectories_{it}.npz"))["sensors"]
    obs_n = sc.transform(truth)[:, sensors]
    print("S9 sensors:", sensors.tolist())

    # ---- posterior draws ----
    sets = posterior_draw_sets(ba, a.n_draws, a.seed)
    summary_rows = []
    for set_name, (cdraws, names, idx) in sets.items():
        nd = len(idx[0])
        print(f"[{set_name}] {nd} draws; terms per equation: {[len(n) for n in names]}")
        z_tr = np.empty((nd, N_SIM_TRAIN, 3)); z_te = np.empty((nd, n_test, 3))
        rmse_week = np.empty((nd, n_test)); fields_h = np.empty((nd, len(HORIZONS), truth.shape[1]), dtype=np.float32)
        sens = np.empty((nd, n_test, len(sensors))); pred_sum = np.zeros_like(pred_pt)
        failed = 0
        for j in range(nd):
            cj = [list(c[j]) for c in cdraws]
            try:
                z_tr[j] = simulate(ba, cj, names, N_SIM_TRAIN, z_train_n[0])
                z_te[j] = simulate(ba, cj, names, n_test, z_test_n[0])
            except Exception as e:  # noqa: BLE001
                failed += 1; z_tr[j] = np.nan; z_te[j] = np.nan; rmse_week[j] = np.nan; fields_h[j] = np.nan; sens[j] = np.nan
                print(f"  draw {j} failed: {e}"); continue
            pn, p = forecast_field(z_te[j])
            rmse_week[j] = np.sqrt(np.mean((truth_sea - p) ** 2, axis=1))
            fields_h[j] = p[HORIZONS]; sens[j] = pn[:, sensors]; pred_sum += p
            if (j + 1) % 100 == 0:
                print(f"  {j + 1}/{nd} draws [{time.time() - t0:.0f}s]")
        ok = ~np.isnan(rmse_week[:, 0]); nd_ok = int(ok.sum())
        pred_mean = pred_sum / max(nd_ok, 1)
        rmse_mean_fc = np.sqrt(np.mean((truth_sea - pred_mean) ** 2, axis=1))
        q_week = np.nanquantile(rmse_week, Q, axis=0)
        f_lo, f_hi = np.nanquantile(fields_h, [0.05, 0.95], axis=0)
        cover_h = [float(np.mean((truth_sea[t] >= f_lo[k]) & (truth_sea[t] <= f_hi[k]))) for k, t in enumerate(HORIZONS)]
        width_h = [float(np.mean(f_hi[k] - f_lo[k])) for k in range(len(HORIZONS))]
        pd.DataFrame(rmse_week[ok][:, HORIZONS], columns=[f"t{t}" for t in HORIZONS]
                     ).to_csv(os.path.join(res_dir, f"rmse_draws_{it}_{set_name}.csv"), index_label="draw")
        for k, t in enumerate(HORIZONS):
            summary_rows.append({"set": set_name, "t": t, "rmse_point_model": float(rmse_pt[t]),
                                 "rmse_mean_of_draws_forecast": float(rmse_mean_fc[t]),
                                 "rmse_draws_q05": float(q_week[0, t]), "rmse_draws_q50": float(q_week[1, t]),
                                 "rmse_draws_q95": float(q_week[2, t]), "field_coverage_90": cover_h[k],
                                 "field_band_width_90_degC": width_h[k], "n_draws_ok": nd_ok, "n_draws_failed": failed})
        if set_name == "identified":
            s_lo, s_hi = np.nanquantile(sens, [0.05, 0.95], axis=0)
            ztr_q = np.nanquantile(z_tr, Q, axis=0); zte_q = np.nanquantile(z_te, Q, axis=0)
            np.savez(os.path.join(res_dir, f"latent_bands_{it}.npz"), gru_train=z_train_n, point_train=z_train_pt,
                     q05_train=ztr_q[0], q50_train=ztr_q[1], q95_train=ztr_q[2],
                     gru_test=z_test_n, point_test=z_test_pt, q05_test=zte_q[0], q50_test=zte_q[1], q95_test=zte_q[2])
            np.savez(os.path.join(res_dir, f"sensor_bands_{it}.npz"), obs=obs_n, point=pred_n_pt[:, sensors],
                     q05=s_lo, q95=s_hi, sensors=sensors)
            pd.DataFrame({"week": np.arange(n_test), "rmse_point_model": rmse_pt, "rmse_mean_of_draws_forecast": rmse_mean_fc,
                          "rmse_draws_q05": q_week[0], "rmse_draws_q50": q_week[1], "rmse_draws_q95": q_week[2]}
                         ).to_csv(os.path.join(res_dir, f"rmse_lead_time_{it}.csv"), index=False)
            plot_latent_bands(z_train_n, z_train_pt, ztr_q[0], ztr_q[2],
                              os.path.join(fig_dir, f"comparison_plot_ba_{it}_bands.pdf"))
            plot_latent_bands(z_test_n, z_test_pt, zte_q[0], zte_q[2],
                              os.path.join(fig_dir, f"latent_forecast_{it}_bands.pdf"), panel="")
            stats = plot_sensor_bands(obs_n, pred_n_pt[:, sensors], s_lo, s_hi,
                                      os.path.join(fig_dir, f"multi_sensor_forecast_{it}_bands.pdf"))
            st = pd.DataFrame(stats); st["sensor_column"] = sensors[:len(st)]
            st.to_csv(os.path.join(res_dir, f"sensor_stats_{it}.csv"), index=False)
            plot_rmse_lead_time(np.arange(n_test), rmse_pt, q_week[0], q_week[1], q_week[2],
                                os.path.join(fig_dir, f"rmse_lead_time_{it}.pdf"), marks=HORIZONS)
            print(f"S9 mean RMSE (point) {st['rmse_normalised'].mean():.4f}, mean coverage {st['coverage_90'].mean():.3f}")
    out = pd.DataFrame(summary_rows)
    out.to_csv(os.path.join(res_dir, f"rmse_horizons_{it}.csv"), index=False)
    print(out.to_string(index=False))
    with open(os.path.join(res_dir, f"run_{it}.json"), "w") as f:
        json.dump({"iteration": it, "n_draws": a.n_draws, "seed": a.seed, "ci": CI, "seconds": time.time() - t0,
                   "stage1_rrmse_eval": summary["recon_rrmse_test_eval"], "sensors": sensors.tolist()}, f, indent=1)
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
