"""Tier C negative controls: environments with no exact polynomial invariant.

Hopper and HalfCheetah are actuated, dissipative and contact-rich, and their
observation omits the root x-coordinate, so it is not even a state. The correct
behaviour of a discovery procedure is to return nothing. A method that always
finds something is worse than useless in a safety pipeline.

Also runs the synthetic spurious-variable stress test: a coordinate that is
correlated with the state on the training trajectories but not causally
constrained, which cross-trajectory validation should reject.

Writes Results/negative_controls.csv.
"""

import os
import csv
import numpy as np
import sympy as sp

import deflation
import discover

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
os.makedirs(RESULTS, exist_ok=True)


def collect(env_id, n_steps=4000, seed=0, n_episodes=8):
    import gymnasium as gym
    env = gym.make(env_id)
    rng = np.random.default_rng(seed)
    obs_all = []
    per_ep = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_obs = [obs]
        for _ in range(n_steps // n_episodes):
            a = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(a)
            ep_obs.append(obs)
            if term or trunc:
                obs, _ = env.reset()
        per_ep.append(np.array(ep_obs))
        obs_all.append(np.array(ep_obs))
    env.close()
    return np.concatenate(obs_all, axis=0), per_ep


def nullspace_dims(data, var_names, degree, tols=(1e-10, 1e-8, 1e-6, 1e-4)):
    sym = list(sp.symbols(" ".join(var_names), real=True))
    monos = deflation.monomials(sym, degree)
    if len(monos) > data.shape[0]:
        return None, len(monos), {}
    Phi = discover._eval_monomials(data, sym, monos)
    colnorm = np.linalg.norm(Phi, axis=0)
    colnorm[colnorm == 0] = 1.0
    s = np.linalg.svd(Phi / colnorm, compute_uv=False)
    return s, len(monos), {t: int(np.sum(s <= t * s[0])) for t in tols}


def main():
    rows = []
    print("=" * 88)
    print("Tier C negative controls: the procedure should return nothing")
    print("=" * 88)

    for env_id in ("Hopper-v5", "HalfCheetah-v5"):
        try:
            data, per_ep = collect(env_id)
        except Exception as e:
            print(f"  {env_id}: SKIPPED ({e})")
            continue
        n = data.shape[1]
        print(f"\n{env_id}: n = {n}, N = {data.shape[0]} samples, "
              f"{len(per_ep)} episodes")
        for d in (1, 2):
            s, M, dims = nullspace_dims(data, [f"x{i}" for i in range(n)], d)
            if s is None:
                print(f"  degree {d}: M = {M} exceeds N, skipped")
                continue
            print(f"  degree {d}: M = {M:>5}  smallest sv/largest = "
                  f"{s[-1]/s[0]:.3e}   nullspace dim by tolerance: "
                  + "  ".join(f"{t:g}->{v}" for t, v in dims.items()))
            rows.append(dict(env=env_id, n=n, N=data.shape[0], degree=d, M=M,
                             cond_ratio=float(s[-1] / s[0]),
                             **{f"dim_tol_{t:g}": v for t, v in dims.items()}))

        # cross-episode validation: a genuine invariant holds on every episode
        d = 2
        sym = list(sp.symbols(" ".join(f"x{i}" for i in range(n)), real=True))
        monos = deflation.monomials(sym, d)
        if len(monos) <= per_ep[0].shape[0]:
            Phi0 = discover._eval_monomials(per_ep[0], sym, monos)
            V0, _ = discover.nullspace_of(Phi0, tol_rel=1e-6)
            survived = 0
            if V0.shape[0]:
                for other in per_ep[1:]:
                    Ph = discover._eval_monomials(other, sym, monos)
                    resid = np.linalg.norm(Ph @ V0.T, axis=0) / \
                        max(np.linalg.norm(Ph), 1e-300)
                    survived = int(np.sum(resid < 1e-6))
            print(f"  per-episode nullspace dim = {V0.shape[0]}, "
                  f"directions surviving cross-episode validation = {survived}")
            rows.append(dict(env=env_id, degree=d, M=len(monos),
                             single_episode_nullspace=V0.shape[0],
                             cross_episode_survivors=survived))

    # ------------------------------------------- spurious-variable stress test
    print("\n" + "=" * 88)
    print("Synthetic spurious-variable stress test")
    print("=" * 88)
    rng = np.random.default_rng(0)
    N = 2000
    th = rng.uniform(0, 2 * np.pi, N)
    x, y = np.cos(th), np.sin(th)
    # z is correlated with x on the training set but not causally constrained
    z_train = 2 * x + 0.0 * rng.normal(size=N)
    z_test = 2 * x + 0.5 * rng.normal(size=N)
    names = ["x", "y", "z"]
    symv = list(sp.symbols("x y z", real=True))
    monos = deflation.monomials(symv, 2)

    for label, z in [("train (spurious correlation present)", z_train),
                     ("held-out trajectory (correlation absent)", z_test)]:
        data = np.stack([x, y, z], axis=1)
        Phi = discover._eval_monomials(data, symv, monos)
        V, _ = discover.nullspace_of(Phi, tol_rel=1e-8)
        print(f"  {label:<42} nullspace dim = {V.shape[0]}")
        rows.append(dict(env="spurious_variable", label=label,
                         nullspace_dim=V.shape[0]))

    # does cross-trajectory validation reject the spurious relation z - 2x?
    spurious = symv[2] - 2 * symv[0]
    sv = deflation.coeff_vector(spurious, monos, symv)
    data_test = np.stack([x, y, z_test], axis=1)
    Ph = discover._eval_monomials(data_test, symv, monos)
    resid = np.linalg.norm(Ph @ sv) / np.linalg.norm(Ph)
    print(f"  residual of the spurious relation z - 2x on held-out data: "
          f"{resid:.3e}   -> {'REJECTED' if resid > 1e-6 else 'accepted'}")
    true_rel = symv[0]**2 + symv[1]**2 - 1
    tv = deflation.coeff_vector(true_rel, monos, symv)
    residt = np.linalg.norm(Ph @ tv) / np.linalg.norm(Ph)
    print(f"  residual of the genuine relation x^2+y^2-1 on held-out data: "
          f"{residt:.3e}   -> {'rejected' if residt > 1e-6 else 'ACCEPTED'}")
    rows.append(dict(env="spurious_variable", label="cross-trajectory check",
                     spurious_residual=float(resid),
                     genuine_residual=float(residt)))

    out = os.path.join(RESULTS, "negative_controls.csv")
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
