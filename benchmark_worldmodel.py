"""E2. Algebraic unit tests for learned world models.

Two questions, and the second is the one that matters.

First, does the consistency term of Section 5.1 do what it claims? That is the
easy half and the answer is the expected one.

Second, and this is the point: one-step validation error is the number everyone
reports for a dynamics model, and it is a poor predictor of whether the model
survives a long rollout. The algebraic residual of the model's *own* imagined
trajectories is a better one, and unlike long-horizon error against ground
truth it can be computed without a simulator to compare against. That is what
makes it a test you could actually run in a training loop on a system you do
not have a reference for.

The experiment deliberately trains a *spread* of models, varying capacity, data
budget, regularisation and seed, so that the correlation between predictors and
long-horizon fidelity is measured across a range of model qualities rather than
asserted from two points.

Writes Results/worldmodel.csv.
"""

import os
import csv
import itertools

import numpy as np
import sympy as sp
import torch

import bugs
import envs
import ideal_diff
import recover
import worldmodel as wm

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
os.makedirs(RESULTS, exist_ok=True)

V = list(bugs.ACRO_SYMS)
G_REF, NAMES, KINDS = bugs.acrobot_reference()
DT = 0.05
HORIZON = 100

#: Errors at which a rollout counts as having diverged from the truth.
DIVERGENCE_THRESHOLDS = [0.05, 0.1, 0.25, 0.5, 1.0]
THRESH = 0.25


#: Transitions per trajectory in `make_dataset`, and the unit `worldmodel.train`
#: holds validation out by. Rows arrive in trajectory order, so the length is
#: all the splitter needs to keep a trajectory whole.
TRAJ_LEN = 200


def make_dataset(n_traj=60, n_steps=TRAJ_LEN, dt=DT, seed=0, torque_scale=1.0):
    """One-step transitions (s, a, s') under a random torque policy."""
    rng = np.random.default_rng(seed)
    S, A, S2 = [], [], []
    for _ in range(n_traj):
        s = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi),
                      rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)])
        for _ in range(n_steps):
            a = rng.uniform(-1.0, 1.0) * torque_scale
            nxt = envs.acrobot_rollout(s, dt, 1, torque=a)[-1]
            S.append(envs.acrobot_obs(s))
            A.append([a])
            S2.append(envs.acrobot_obs(nxt))
            s = nxt
    return np.array(S), np.array(A), np.array(S2)


def true_rollout(s0_obs, actions, dt=DT):
    """Ground-truth rollout from the same start and actions, for comparison."""
    th1 = np.arctan2(s0_obs[1], s0_obs[0])
    th2 = np.arctan2(s0_obs[3], s0_obs[2])
    s = np.array([th1, th2, s0_obs[4], s0_obs[5]])
    out = [envs.acrobot_obs(s)]
    for a in actions:
        s = envs.acrobot_rollout(s, dt, 1, torque=float(a))[-1]
        out.append(envs.acrobot_obs(s))
    return np.array(out)


def evaluate(model, seed=0, n_eval=12, horizon=HORIZON):
    """Long-horizon fidelity, and the algebraic residuals of the model's rollouts.

    The algebraic side never looks at `true`; that asymmetry is the result.
    """
    rng = np.random.default_rng(1000 + seed)
    model_trajs, errs = [], []
    for _ in range(n_eval):
        s0 = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi),
                       rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)])
        s0_obs = envs.acrobot_obs(s0)
        actions = rng.uniform(-1.0, 1.0, (horizon, 1))
        pred = wm.rollout(model, s0_obs, actions)
        true = true_rollout(s0_obs, actions[:, 0])
        model_trajs.append(pred)
        errs.append(np.sqrt(np.mean((pred - true) ** 2, axis=1)))
    errs = np.array(errs)
    resid = {n: ideal_diff.generator_residual(g, model_trajs, V, k)
             for n, g, k in zip(NAMES, G_REF, KINDS)}
    # Divergence horizon rather than terminal error. Every model of a system
    # this sensitive eventually diverges, so terminal error saturates and stops
    # separating a good model from a bad one; how long fidelity lasts does not.
    #
    # The horizon depends on the error a rollout is allowed before it counts as
    # diverged, and the whole comparison rests on it, so it is reported at
    # several thresholds rather than one. A conclusion that holds at 0.25 and
    # nowhere else is a conclusion about 0.25.
    out = {}
    for thresh in DIVERGENCE_THRESHOLDS:
        horizons = []
        for e in errs:
            over = np.where(e > thresh)[0]
            horizons.append(float(over[0]) if len(over) else float(len(e)))
        out[f"divergence_horizon_{thresh:g}"] = float(np.mean(horizons))
    return dict(rollout_rmse_final=float(errs[:, -1].mean()),
                rollout_rmse_h25=float(errs[:, min(25, errs.shape[1] - 1)].mean()),
                divergence_horizon=out[f"divergence_horizon_{THRESH:g}"],
                **out,
                **{f"resid_{n.replace(' ', '_')}": v
                   for n, v in resid.items()},
                algebraic_score=float(sum(resid.values()))), model_trajs


def normalised_generators(data):
    """Rescale each generator so that one lambda means the same thing for all.

    Without this the energy, whose values are of order ten, contributes a
    squared drift term some eight orders of magnitude above a one-step MSE of
    1e-4, and lambda stops being a weight and becomes a switch. Vanishing
    generators are divided by their term scale on the data; the conserved one is
    divided by its own standard deviation, which is the scale of the quantity
    whose *changes* are being penalised.
    """
    out = []
    for g, kind in zip(G_REF, KINDS):
        if kind == ideal_diff.CONSERVED:
            vals = ideal_diff.eval_poly(g, data, V)
            s = float(np.std(vals))
        else:
            s = ideal_diff.term_scale(g, data, V)
        out.append(sp.expand(g / max(s, 1e-12)))
    return out


def geomean(x):
    """Geometric mean of a set of ratios.

    A ratio of 0.5 and a ratio of 2.0 are the same effect in opposite
    directions, and their arithmetic mean is 1.25 rather than 1. The geometric
    mean is the summary that treats them symmetrically, which is what a column
    of "x times" figures needs.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float("nan")


def bootstrap_ci(d, n_boot=10000, seed=0, alpha=0.05):
    """Percentile bootstrap interval for the mean of a paired difference."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def spearman(x, y):
    """Rank correlation, computed here to avoid a scipy version dependency.

    Ties take their average rank. The divergence horizon is a mean of integers
    over twelve rollouts and repeats across the sweep, and breaking those ties
    by sort order makes the correlation depend on which model happened to be
    trained first.
    """
    def rank(v):
        v = np.asarray(v, float)
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        s = v[order]
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
            i = j + 1
        return r
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    S0, _, _ = make_dataset(n_traj=20, seed=99)
    G_N = normalised_generators(S0)
    vanishing = wm.polys_to_torch(G_N[:2], V)
    conserved = wm.polys_to_torch([G_N[2]], V)

    # A spread of model qualities, on purpose. If every model were good the
    # correlation between predictors and long-horizon fidelity would be
    # measured over no range at all.
    # The regulariser has to be pushed until it engages before an absence of
    # effect on rollout fidelity says anything. The upper end of this range is
    # where the algebraic residual has fallen by orders of magnitude and the
    # one-step validation error has started to pay for it, which is the point
    # past which nothing further is learned.
    grid = dict(
        lam=([0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0] if not args.quick
             else [0.0, 1e-2]),
        hidden=[32, 128] if not args.quick else [64],
        n_traj=[10, 60] if not args.quick else [30],
        seed=[0, 1, 2] if not args.quick else [0],
    )
    epochs = 60 if args.quick else 250

    rows, srows, prows = [], [], []
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    print("=" * 104)
    print(f"Algebraic unit tests for learned world models: {len(combos)} "
          f"models")
    print("=" * 104)
    print(f"{'lam':>6}{'hid':>5}{'traj':>6}{'seed':>5}"
          f"{'val MSE':>11}{'div.horizon':>12}"
          f"{'unit-norm':>11}{'energy':>11}{'alg score':>11}")
    print("-" * 104)

    for combo in combos:
        cfg = dict(zip(keys, combo))
        S, A, S2 = make_dataset(n_traj=cfg["n_traj"], seed=cfg["seed"])
        model, info = wm.train(
            S, A, S2, n_obs=6, n_act=1, hidden=cfg["hidden"], depth=2,
            epochs=epochs, lam=cfg["lam"], vanishing=vanishing,
            conserved=conserved, seed=cfg["seed"], traj_len=TRAJ_LEN)
        ev, _ = evaluate(model, seed=cfg["seed"])
        row = dict(**cfg, val_mse=info["val_mse"], **ev)
        rows.append(row)
        print(f"{cfg['lam']:>6g}{cfg['hidden']:>5}{cfg['n_traj']:>6}"
              f"{cfg['seed']:>5}{info['val_mse']:>11.2e}"
              f"{ev['divergence_horizon']:>12.1f}"
              f"{ev['resid_unit-norm_th1']:>11.2e}"
              f"{ev['resid_energy']:>11.2e}"
              f"{ev['algebraic_score']:>11.2e}")

    # ------------------------------------------------------- the two claims ---
    print("\n" + "=" * 104)
    print("Claim 1. The consistency term reduces both the algebraic residual "
          "and long-horizon error")
    print("=" * 104)
    for lam in sorted({r["lam"] for r in rows}):
        sub = [r for r in rows if r["lam"] == lam]
        print(f"  lambda={lam:<7g} val MSE "
              f"{np.mean([r['val_mse'] for r in sub]):.2e}"
              f"   divergence horizon "
              f"{np.mean([r['divergence_horizon'] for r in sub]):6.1f}"
              f"   algebraic {np.mean([r['algebraic_score'] for r in sub]):.2e}")

    print("\n" + "=" * 104)
    print("Claim 2. Which predictor of long-horizon fidelity is better, and "
          "which needs a reference?")
    print("=" * 104)
    # Negated so that for both predictors "larger is worse", making the two
    # correlations directly comparable in sign as well as magnitude.
    #
    # Reported over three pools, and the pool matters. Most of the sweep was
    # trained with the algebraic score *in the loss*, so over all models the
    # comparison is not like-for-like: lambda inflates one-step error while
    # deflating the very quantity the algebraic predictor reads, which flatters
    # the first predictor and Goodharts the second. The unregularised pool is
    # the clean comparison between two predictors of a model's fidelity, and
    # the benign pool shows where the picture starts to move. The direction of
    # the difference runs against the algebraic score, which is why the whole
    # sweep is the one quoted.
    pools = [("all models", lambda r: True),
             ("lambda = 0 only", lambda r: r["lam"] == 0),
             ("lambda <= 0.1", lambda r: r["lam"] <= 0.1)]
    print("  Spearman rho against loss of long-horizon fidelity "
          "(negated divergence horizon)")
    print(f"    {'pool':<18}{'n':>5}{'val MSE':>12}{'algebraic':>12}")
    print("    " + "-" * 47)
    rho_mse = rho_alg = float("nan")
    for label, keep in pools:
        sub = [r for r in rows if keep(r)]
        if len(sub) < 3:
            continue
        ys = [-r["divergence_horizon"] for r in sub]
        rm = spearman([r["val_mse"] for r in sub], ys)
        ra = spearman([r["algebraic_score"] for r in sub], ys)
        if label == "all models":
            rho_mse, rho_alg = rm, ra
        print(f"    {label:<18}{len(sub):>5}{rm:>+12.3f}{ra:>+12.3f}")
        prows.append(dict(pool=label, n_models=len(sub),
                          rho_val_mse=rm, rho_algebraic=ra))
    print("\n  One-step validation MSE needs held-out ground-truth "
          "transitions; the\n  algebraic score needs no reference at all, "
          "being computed from the model's\n  own imagined rollouts. That is "
          "what makes it usable on a system where the\n  long-horizon ground "
          "truth is precisely what is unavailable.")

    # ------------------------------------------------------------- claim 3 ---
    # The global correlation above is dominated by capacity and data budget,
    # which move one-step error and long-horizon error together. The sharper
    # question is whether one-step error can see an improvement that does not
    # come from either, and the consistency term is exactly such an
    # improvement: it is a change to what the model respects, not to how well
    # it fits the next step.
    print("\n" + "=" * 104)
    print("Claim 3. Paired within architecture and data budget: what does the "
          "consistency term move?")
    print("=" * 104)
    lams = sorted({r["lam"] for r in rows})
    lo = lams[0]
    print(f"{'lambda':>9}{'val MSE':>12}{'alg. residual':>16}"
          f"{'horizon':>12}{'improved':>11}{'95% CI on horizon':>26}")
    print("-" * 104)
    for hi in lams[1:]:
        pairs = []
        for r in rows:
            if r["lam"] != hi:
                continue
            for b in rows:
                if (b["lam"] == lo and b["hidden"] == r["hidden"]
                        and b["n_traj"] == r["n_traj"]
                        and b["seed"] == r["seed"]):
                    pairs.append((b, r))
        if not pairs:
            continue
        d_mse = np.array([p[1]["val_mse"] / p[0]["val_mse"] for p in pairs])
        d_hor = np.array([p[1]["divergence_horizon"]
                          - p[0]["divergence_horizon"] for p in pairs])
        d_alg = np.array([p[1]["algebraic_score"] / p[0]["algebraic_score"]
                          for p in pairs])
        wins = int(np.sum(d_hor > 0))
        lo_ci, hi_ci = bootstrap_ci(d_hor)
        print(f"{hi:>9g}{geomean(d_mse):>12.3f}{geomean(d_alg):>16.3e}"
              f"{d_hor.mean():>+12.1f}{wins:>7}/{len(pairs):<3}"
              f"{f'[{lo_ci:+.1f}, {hi_ci:+.1f}]':>26}")
        srows.append(dict(lam_lo=lo, lam_hi=hi, n_pairs=len(pairs),
                          val_mse_geomean_ratio=geomean(d_mse),
                          algebraic_geomean_ratio=geomean(d_alg),
                          horizon_mean_delta=float(d_hor.mean()),
                          horizon_improved=wins,
                          horizon_ci_lo=lo_ci, horizon_ci_hi=hi_ci,
                          **{f"horizon_delta_{t:g}": float(np.mean(
                              [p[1][f"divergence_horizon_{t:g}"]
                               - p[0][f"divergence_horizon_{t:g}"]
                               for p in pairs]))
                             for t in DIVERGENCE_THRESHOLDS}))

    print("\n  Ratios are summarised by the geometric mean, since a ratio of "
          "0.5 and one of\n  2.0 are the same size of effect in opposite "
          "directions and their arithmetic\n  mean is not 1. The interval is a "
          "bootstrap over the matched pairs; where it\n  straddles zero the "
          "horizon has not been shown to move at all.")

    # The horizon depends on the error threshold that defines divergence, so
    # the same paired comparison is reported across the whole range of them.
    print("\n" + "=" * 104)
    print("Claim 4. Does the conclusion survive the choice of divergence "
          "threshold?")
    print("=" * 104)
    hi = lams[-1]
    print(f"{'threshold':>11}" + "".join(f"{f'lam={l:g}':>14}"
                                         for l in lams[1:]))
    print("-" * 104)
    for t in DIVERGENCE_THRESHOLDS:
        cells = []
        for l in lams[1:]:
            base = [r[f"divergence_horizon_{t:g}"] for r in rows
                    if r["lam"] == lo]
            got = [r[f"divergence_horizon_{t:g}"] for r in rows
                   if r["lam"] == l]
            cells.append(f"{np.mean(got) - np.mean(base):+.1f}")
        print(f"{t:>11g}" + "".join(f"{c:>14}" for c in cells))

    out = os.path.join(RESULTS, "worldmodel.csv")
    keys_all = []
    for r in rows:
        for k in r:
            if k not in keys_all:
                keys_all.append(k)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys_all + ["rho_val_mse",
                                                     "rho_algebraic"])
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow(dict(r, rho_val_mse=rho_mse if i == 0 else "",
                            rho_algebraic=rho_alg if i == 0 else ""))
    print(f"\nWrote {out}")

    if srows:
        out2 = os.path.join(RESULTS, "worldmodel_paired.csv")
        keys2 = []
        for r in srows:
            for k in r:
                if k not in keys2:
                    keys2.append(k)
        with open(out2, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys2)
            w.writeheader()
            w.writerows(srows)
        print(f"Wrote {out2}")

    if prows:
        # The predictor comparison over each pool, so that the paper quotes a
        # row of this file rather than one correlation computed over a sweep
        # that manipulates one of the two predictors being compared.
        out3 = os.path.join(RESULTS, "worldmodel_predictors.csv")
        with open(out3, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(prows[0].keys()))
            w.writeheader()
            w.writerows(prows)
        print(f"Wrote {out3}")


if __name__ == "__main__":
    main()
