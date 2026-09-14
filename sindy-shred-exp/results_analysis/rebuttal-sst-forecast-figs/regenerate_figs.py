"""Regenerate the SST forecast figures for one SINDy-SHRED experiment — Fig. 8b,
Fig. 8c (full globe) and Fig. S9 — from the fixed pipeline. Referee 1, Point 8.

Follows `results_analysis/analysis_script_ba.py` step by step (data split,
scaler, network, latent normalisation, Bayesian-ARGOS design matrix, forecast,
decoder pass, plotting functions), so the outputs are directly comparable with
the submitted figures. Two repository fixes it relies on (2026-09-12):

* the analysis scripts now call `shred.eval()` after loading the weights, so
  the decoder's dropout (p = 0.1) is off at inference and the decoded fields are
  deterministic (`--dropout-active` reproduces the submitted behaviour);
* `extract_identified_model` in pyargos drops every `sigma*` site, so the
  sampler's transformed `sigma_log__` (exposed by bambi >= 0.16) no longer
  leaks into the identified model and `simulate()` works.

Outputs (in --out, `_dropout` suffix with --dropout-active):
  results/identified_model_<it>.csv     identified latent model
  results/latent_train_<it>.npz         GRU latent (training period) and its simulation   -> Fig. 8b
  results/rmse_<it>.csv                 per-horizon RMSE on the 0-180E crop and the globe  -> Fig. 8c
  results/fields_<it>.npz               observed and decoded fields at the plotted horizons
  results/sensor_trajectories_<it>.npz  observed and forecast series at 18 held-out sensors -> Fig. S9
  results/sensor_rmse_<it>.csv          per-sensor RMSE (normalised units)
  figures/comparison_plot_ba_<it>.pdf, global_comparison_<it>.pdf, crop_comparison_<it>.pdf,
  figures/multi_sensor_forecast_<it>.pdf

The SST data, sensor locations and trained weights are git-ignored; point
--exp-root at a checkout that has them.
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
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

from plot_fields import plot_zoomed_comparison  # noqa: E402
from plot_latent import plot_latent_comparison  # noqa: E402
from plot_sensors import plot_multiple_sensor_predictions  # noqa: E402

LAGS = 52
LATENT_DIM = 3
TIMESTEPS = [50, 100, 150, 200, 250]
N_SENSORS_S9 = 18


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-root", default=SHRED_EXP, help="sindy-shred-exp folder holding Data/ and exp-1/")
    p.add_argument("--data", default=None, help="SST_data.mat (default: <exp-root>/Data/SST_data.mat)")
    p.add_argument("--iteration", type=int, default=18)
    p.add_argument("--num-sensors", type=int, default=250)
    p.add_argument("--out", default=HERE)
    p.add_argument("--dropout-active", action="store_true", help="decode in train mode, as the submitted scripts did")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sensor-offset", type=int, default=1,
                   help="skip this many global-RNG draws before sampling the S9 sensors; 1 reproduces the submitted sample")
    return p.parse_args()


def decode(shred, latent):
    """Decoder pass exactly as in analysis_script_ba.py (dropout layers included; inert in eval mode)."""
    z = torch.FloatTensor(latent).to(next(shred.parameters()).device)
    out = shred.linear1(z)
    out = shred.dropout(out)
    out = torch.nn.functional.relu(out)
    out = shred.linear2(out)
    out = shred.dropout(out)
    out = torch.nn.functional.relu(out)
    out = shred.linear3(out)
    return out.detach().cpu().numpy()


def main():
    a = parse_args()
    it = a.iteration
    tag = "_dropout" if a.dropout_active else ""
    data_path = a.data or os.path.join(a.exp_root, "Data", "SST_data.mat")
    res_dir, fig_dir = os.path.join(a.out, "results"), os.path.join(a.out, "figures")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- data and split (identical to analysis_script_ba.py) ----
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

    # ---- network ----
    library_dim = sindy.library_size(LATENT_DIM, 3, False, True)
    shred = sindy_shred.SINDy_SHRED(
        a.num_sensors, m, hidden_size=LATENT_DIM, hidden_layers=2, l1=350, l2=400, dropout=0.1,
        library_dim=library_dim, poly_order=3, include_sine=False, dt=1 / 52.0 * 0.1, layer_norm=False,
    ).to(device)
    shred.load_state_dict(torch.load(os.path.join(a.exp_root, "exp-1", "saved-model", f"model_{it}.pth")))
    shred.train() if a.dropout_active else shred.eval()
    print(f"dropout active: {shred.training}")

    # ---- latent trajectories, normalised with the training min/max to [-1, 1] ----
    with torch.no_grad():
        z_train = shred.gru_outputs(train_ds.X, sindy=True)[0][:, 0, :].cpu().numpy()
        z_test = shred.gru_outputs(test_ds.X, sindy=True)[0][:, 0, :].cpu().numpy()
    lo, hi = z_train.min(axis=0), z_train.max(axis=0)
    z_train_n = 2 * (z_train - lo) / (hi - lo) - 1
    z_test_n = 2 * (z_test - lo) / (hi - lo) - 1

    # ---- Bayesian-ARGOS on the training latent trajectory ----
    dm = agu.build_design_matrix(x_t=z_train_n, dt=1 / 52.0, library_degree=1, sg_poly_order=4, library_type="poly")
    ba = BayesianArgos(design_matrix=dm, accelerator=True)
    ba.run(mode="straight", parallel="yes", ncpus=None, ci_level=0.90)
    coeffs, names = ba.expressions_for_simulation()
    rows = [{"equation": f"dz{i + 1}/dt", "term": nm or "1", "coefficient": c}
            for i, (cs, ns) in enumerate(zip(coeffs, names)) for c, nm in zip(cs, ns)]
    pd.DataFrame(rows).to_csv(os.path.join(res_dir, f"identified_model_{it}{tag}.csv"), index=False)
    for r in rows:
        print(f"{r['equation']}: {r['coefficient']:+.3f} {r['term']}")

    # ---- Fig. 8b: training-period latent trajectory vs simulation ----
    z_sim_train = np.asarray(ba.simulate(n=1040, dt=1 / 52.0, init_conditions=z_train_n[0]))
    np.savez(os.path.join(res_dir, f"latent_train_{it}{tag}.npz"), gru=z_train_n, sim=z_sim_train)
    plot_latent_comparison(z_train_n, z_sim_train, os.path.join(fig_dir, f"comparison_plot_ba_{it}{tag}.pdf"))

    # ---- forecast the test latent trajectory and decode it ----
    z_sim = np.asarray(ba.simulate(n=len(z_test_n), dt=1 / 52.0, init_conditions=z_test_n[0]))
    z_sim = (z_sim + 1) / 2 * (hi - lo) + lo
    pred_n = decode(shred, z_sim)                       # normalised units, as the network sees them
    real_n = test_ds.Y.detach().cpu().numpy()
    pred = sc.inverse_transform(pred_n)                 # degrees Celsius
    real = sc.inverse_transform(real_n)
    mean_X = np.mean(loadmat(data_path)["Z"].T, axis=0)
    sst_locs = np.where(mean_X != 0)[0]

    # ---- Fig. 8c: RMSE on the submitted crop (0-180E) and on the globe ----
    def field(v):
        f = np.full(180 * 360, np.nan); f[sst_locs] = v
        return f.reshape(180, 360)
    rmse = []
    for t in TIMESTEPS:
        d = field(real[t]) - field(pred[t])
        rmse.append({"t": t, "rmse_crop_0_180E": np.sqrt(np.nanmean(d[:, :180] ** 2)),
                     "rmse_global": np.sqrt(np.nanmean(d ** 2))})
    rmse = pd.DataFrame(rmse)
    rmse.to_csv(os.path.join(res_dir, f"rmse_{it}{tag}.csv"), index=False)
    print(rmse.to_string(index=False))
    np.savez(os.path.join(res_dir, f"fields_{it}{tag}.npz"), real=real[TIMESTEPS], pred=pred[TIMESTEPS],
             sst_locs=sst_locs, timesteps=np.array(TIMESTEPS))
    for lon, name in [((0, 360), f"global_comparison_{it}{tag}.pdf"), ((0, 180), f"crop_comparison_{it}{tag}.pdf")]:
        plot_zoomed_comparison(real, pred, sst_locs, TIMESTEPS, (0, 180), lon,
                               diff_scale=1, figsize=(12, 5.6) if lon[1] == 360 else None,
                               font_scale=0.8 if lon[1] == 360 else 1.0, save=True, save_dir=fig_dir, filename=name)

    # ---- Fig. S9: 18 held-out sensors. The original draws them from the global RNG (seeded 0 at
    # the top of the script) after the Bayesian-ARGOS run, and the sampler consumes that RNG, so a
    # fresh seed-0 draw gives a different set. The submitted figure's sample is this stream advanced
    # by one more draw than this script reaches (checked panel by panel against the submitted
    # per-sensor RMSEs), hence --sensor-offset 1 by default.
    stream = np.random.randint(1, 40000, size=N_SENSORS_S9 + a.sensor_offset)
    sensors = stream[a.sensor_offset:]
    sensors_direct = np.random.RandomState(a.seed).randint(1, 40000, size=N_SENSORS_S9)
    print(f"S9 sensors (global RNG after BA run, offset {a.sensor_offset}):", sensors.tolist())
    print("S9 sensors (fresh seed-0 draw, for reference)  :", sensors_direct.tolist())
    obs, fore = real_n[:, sensors], pred_n[:, sensors]
    np.savez(os.path.join(res_dir, f"sensor_trajectories_{it}{tag}.npz"), obs=obs, fore=fore, sensors=sensors)
    _, _, stats = plot_multiple_sensor_predictions(
        obs, fore, np.arange(N_SENSORS_S9), list(range(N_SENSORS_S9)), rows=3, cols=6,
        save_path=os.path.join(fig_dir, f"multi_sensor_forecast_{it}{tag}.pdf"))
    pd.DataFrame({"panel": list(stats["per_sensor_rmse"].keys()), "sensor_column": sensors,
                  "rmse_normalised": list(stats["per_sensor_rmse"].values())}
                 ).to_csv(os.path.join(res_dir, f"sensor_rmse_{it}{tag}.csv"), index=False)
    print(f"S9 mean per-sensor RMSE (normalised units): {stats['mean_rmse']:.4f}")
    print("done")


if __name__ == "__main__":
    main()
