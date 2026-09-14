"""Does the decoder need the third latent coordinate? — Referee 1, Point 8 ("why not 2D").

For one experiment: decode the network's own test latent trajectory through the
decoder (eval mode), then again with each latent coordinate in turn frozen at its
training-period mean, and record the decoded-field error (rRMSE and RMSE in degC)
for each. A coordinate whose freezing barely changes the field is redundant for
the representation; one whose freezing degrades it is load-bearing regardless of
how trivial its identified dynamics are. Also records each coordinate's range on
the test period. Output: results/exp_<it>/decoder_ablation.json.
"""
import argparse
import json
import os

import numpy as np

from run_experiment import HERE, SHRED_EXP, decode, load_experiment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--exp-root", default=SHRED_EXP)
    p.add_argument("--data", default=None)
    p.add_argument("--num-sensors", type=int, default=250)
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    a = p.parse_args(); a.seed = 0; a.skip_dropout_rrmse = True
    summ = {}
    shred, sc, z_train_n, z_test_n, lo, hi, truth = load_experiment(a, summ)
    z_test = (z_test_n + 1) / 2 * (hi - lo) + lo
    z_train = (z_train_n + 1) / 2 * (hi - lo) + lo
    def score(z):
        pred = sc.inverse_transform(decode(shred, z)); tr = truth[:len(pred)]
        return {"rrmse": float(np.linalg.norm(pred - tr) / np.linalg.norm(tr)), "rmse_degC": float(np.sqrt(np.mean((pred - tr) ** 2)))}
    res = {"iteration": a.iteration, "recon_rrmse_test_eval": summ["recon_rrmse_test_eval"], "full": score(z_test)}
    for k in range(3):
        zz = z_test.copy(); zz[:, k] = z_train[:, k].mean()
        res[f"z{k + 1}_frozen"] = score(zz)
    res["test_range_normalised"] = (z_test_n.max(0) - z_test_n.min(0)).tolist()
    out_dir = os.path.join(a.out, f"exp_{a.iteration}"); os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, "decoder_ablation.json"), "w"), indent=1)
    print(f"exp {a.iteration}: full {res['full']['rrmse']:.3f}; frozen z1 {res['z1_frozen']['rrmse']:.3f}, z2 {res['z2_frozen']['rrmse']:.3f}, z3 {res['z3_frozen']['rrmse']:.3f}")


if __name__ == "__main__":
    main()
