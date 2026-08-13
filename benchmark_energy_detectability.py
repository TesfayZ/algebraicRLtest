"""Is the energy direction actually in the numerical nullspace?

The paper's Table 3 asserts a raw degree-3 nullspace of dimension 15 on a
single-energy Acrobot trajectory: 14 trivial multiples of the unit-norm
constraints plus the energy direction. The 14 are exact algebra and hold at any
tolerance. The 15th is not: E - E_0 vanishes only up to the integrator's energy
drift, so whether it appears in the numerical nullspace depends on the
tolerance and on dt.

This is the quantitative form of the paper's own Remark on the tension between
the drift floor eta and the resolution of the discovery step. It is measured
here rather than assumed.

Writes Results/energy_detectability.csv.
"""

import os
import csv
import numpy as np
import sympy as sp

import envs
import deflation
import discover

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
os.makedirs(RESULTS, exist_ok=True)

VARS = envs.ACROBOT_VARS
sym = list(sp.symbols(" ".join(VARS), real=True))
G_TRUE = [sym[0]**2 + sym[1]**2 - 1, sym[2]**2 + sym[3]**2 - 1]

S0 = np.array([0.8, 0.4, 0.0, 0.0])
HORIZON = 40.0   # seconds of simulated time, held fixed across dt


def main():
    rows = []
    monos = deflation.monomials(sym, 3)
    print("=" * 78)
    print("Energy direction detectability in the degree-3 Acrobot nullspace")
    print("=" * 78)
    print(f"{'dt':>8} {'N':>7} {'relDrift':>11} {'s[-1]':>11} {'s[-2]':>11} "
          f"{'gap':>9}  {'raw dim @ tol':>28}")

    for dt in (0.2, 0.05, 0.01, 0.002):
        n = int(round(HORIZON / dt))
        traj = envs.acrobot_rollout(S0, dt, n)
        E = np.array([envs.acrobot_energy_coords(s) for s in traj])
        rel_drift = float(np.max(np.abs(E - E[0])) / abs(E[0]))

        data = envs.acrobot_obs(traj)
        Phi = discover._eval_monomials(data, sym, monos)
        # column-normalise so singular values are comparable across dt
        colnorm = np.linalg.norm(Phi, axis=0)
        colnorm[colnorm == 0] = 1.0
        Phin = Phi / colnorm
        s = np.linalg.svd(Phin, compute_uv=False)

        dims = {}
        for tol in (1e-10, 1e-8, 1e-6, 1e-4, 1e-3):
            dims[tol] = int(np.sum(s <= tol * s[0]))

        # s[-15] is the energy direction if the claim holds; s[-14] is the
        # smallest exactly-vanishing (trivial) direction.
        s14 = s[-14] / s[0]
        s15 = s[-15] / s[0]
        gap = s15 / max(s14, 1e-300)

        print(f"{dt:>8} {n:>7} {rel_drift:>11.2e} {s14:>11.2e} {s15:>11.2e} "
              f"{gap:>9.1f}  " + " ".join(f"{t:g}:{v}" for t, v in dims.items()))

        rows.append(dict(dt=dt, N=n, rel_energy_drift=rel_drift,
                         sigma_14=s14, sigma_15=s15, ratio=gap,
                         **{f"dim_tol_{t:g}": v for t, v in dims.items()}))

    print("\nInterpretation:")
    print("  sigma_14 = smallest singular value belonging to the 14 exactly")
    print("             vanishing trivial multiples (algebraic, tolerance free)")
    print("  sigma_15 = the energy direction, which vanishes only to the")
    print("             integrator's drift floor")
    print("  A tolerance strictly between them recovers dimension 15; one below")
    print("  sigma_15 recovers 14 and the energy invariant is invisible.")

    out = os.path.join(RESULTS, "energy_detectability.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
