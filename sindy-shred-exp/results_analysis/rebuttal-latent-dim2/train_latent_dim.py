"""Retrain SINDy-SHRED with a chosen latent dimension on an existing experiment's sensor set.

Referee 1, Point 8 ("why a 3-D latent system ... why not find a 2-D latent system
directly?"). Reproduces `sindy-shred-exp/training_script.py` step by step (same
split, scaler, sensors, network widths, training schedule, seed 0) with the latent
dimension as an argument, so that a network with r = 2 can be compared with the
existing r = 3 network trained on the same sensor set. Nothing is written into
the experiment folders; outputs go to --out:
  models/model_<it>_r<r>.pth, results/train_<it>_r<r>.json (test-set reconstruction
  rRMSE in eval mode, and in train mode as the original script computed it;
  validation-error trace; wall time).
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.dirname(os.path.abspath(__file__))
SHRED_EXP = os.path.dirname(os.path.dirname(HERE))
sys.path.append(SHRED_EXP)
import src.sindy as sindy  # noqa: E402
import src.sindy_shred as sindy_shred  # noqa: E402
from utils.processdata import TimeSeriesDataset, load_data_with_path  # noqa: E402

LAGS = 52


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iteration", type=int, required=True, help="experiment whose sensor set is reused")
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--exp-root", default=SHRED_EXP, help="sindy-shred-exp folder holding Data/ and exp-1/sensor-location/")
    p.add_argument("--out", default=HERE)
    p.add_argument("--num-sensors", type=int, default=250)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--seed", type=int, default=0, help="training_script.py seeds 0 after drawing the sensors")
    a = p.parse_args()
    it, r = a.iteration, a.latent_dim
    t0 = time.time()
    os.makedirs(os.path.join(a.out, "models"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "results"), exist_ok=True)

    load_X = load_data_with_path("SST", os.path.join(a.exp_root, "Data", "SST_data.mat"))
    n, m = load_X.shape
    sensor_locations = np.load(os.path.join(a.exp_root, "exp-1", "sensor-location", f"sensor_location_{it}.npy"))
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed(a.seed)
    train_idx = np.arange(0, 1000)
    mask = np.ones(n - LAGS); mask[train_idx] = 0
    rest = np.arange(0, n - LAGS)[np.where(mask != 0)[0]]
    valid_idx, test_idx = rest[:30], rest[30:]
    sc = MinMaxScaler().fit(load_X[train_idx])
    X = sc.transform(load_X)
    all_in = np.zeros((n - LAGS, LAGS, a.num_sensors))
    for i in range(len(all_in)):
        all_in[i] = X[i:i + LAGS, sensor_locations]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t32 = lambda arr: torch.tensor(arr, dtype=torch.float32).to(device)  # noqa: E731
    train_ds = TimeSeriesDataset(t32(all_in[train_idx]), t32(X[train_idx + LAGS - 1]))
    valid_ds = TimeSeriesDataset(t32(all_in[valid_idx]), t32(X[valid_idx + LAGS - 1]))
    test_ds = TimeSeriesDataset(t32(all_in[test_idx]), t32(X[test_idx + LAGS - 1]))

    library_dim = sindy.library_size(r, 3, False, True)
    shred = sindy_shred.SINDy_SHRED(
        a.num_sensors, m, hidden_size=r, hidden_layers=2, l1=350, l2=400, dropout=0.1,
        library_dim=library_dim, poly_order=3, include_sine=False, dt=1 / 52.0 * 0.1, layer_norm=False,
    ).to(device)
    val_errors = sindy_shred.fit(
        shred, train_ds, valid_ds, batch_size=128, num_epochs=a.epochs, lr=1e-3, verbose=False,
        threshold=0.25, patience=5, sindy_regularization=10.0, optimizer="AdamW", thres_epoch=100,
    )
    torch.save(shred.state_dict(), os.path.join(a.out, "models", f"model_{it}_r{r}.pth"))

    truth = sc.inverse_transform(test_ds.Y.detach().cpu().numpy())
    def rrmse():
        rec = sc.inverse_transform(shred(test_ds.X).detach().cpu().numpy())
        return float(np.linalg.norm(rec - truth) / np.linalg.norm(truth))
    with torch.no_grad():
        shred.train(); rr_train_mode = rrmse()      # as training_script.py computed it
        shred.eval(); rr_eval = rrmse()
    res = {"iteration": it, "latent_dim": r, "seed": a.seed, "epochs": a.epochs,
           "recon_rrmse_test_eval": rr_eval, "recon_rrmse_test_train_mode_draw": rr_train_mode,
           "stage1_pass_eval": rr_eval <= 0.05,
           "validation_errors": [float(v) for v in np.asarray([e.detach().cpu() if hasattr(e, "detach") else e for e in val_errors]).ravel()],
           "seconds": round(time.time() - t0, 1)}
    json.dump(res, open(os.path.join(a.out, "results", f"train_{it}_r{r}.json"), "w"), indent=1)
    print(f"exp {it} r={r}: test rRMSE eval {rr_eval:.4f} (train-mode draw {rr_train_mode:.4f}), stage1 {'pass' if res['stage1_pass_eval'] else 'FAIL'}, {res['seconds']} s")


if __name__ == "__main__":
    main()
