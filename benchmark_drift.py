"""H4: simulator drift diagnostics, the de-risking experiment.

Backward error analysis predicts a sharp, falsifiable ordering: symplectic
integrators exhibit bounded invariant error over long horizons, whereas
non-symplectic methods drift secularly at a rate scaling as dt^p in the order p.
This experiment tests whether the discovered invariant reproduces that ordering,
and requires no policy training at all.

The quantity measured is the drift of the Acrobot energy, which the discovery
procedure recovers exactly at small dt (see benchmark_deflation.py).

Writes Results/drift_diagnostic.csv.
"""

import os
import csv
import numpy as np

import envs

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
os.makedirs(RESULTS, exist_ok=True)

S0 = np.array([0.8, 0.4, 0.0, 0.0])
HORIZON = 40.0

METHODS = [("symplectic_euler", 1, "symplectic"),
           ("verlet", 2, "symplectic"),
           ("rk4", 4, "non-symplectic"),
           ("euler", 1, "non-symplectic")]


def secular_slope(E, t):
    """Least-squares slope of the energy error against time; secular drift."""
    err = E - E[0]
    A = np.vstack([t, np.ones_like(t)]).T
    slope, _ = np.linalg.lstsq(A, err, rcond=None)[0]
    return float(slope)


def main():
    rows = []
    print("=" * 96)
    print("H4: invariant drift by integrator. Bounded (symplectic) vs secular "
          "(non-symplectic)")
    print("=" * 96)
    print(f"{'method':>18} {'class':>16} {'dt':>7} {'max|dE|/E0':>12} "
          f"{'secular slope':>15} {'bounded?':>10}")

    for method, order, cls in METHODS:
        for dt in (0.05, 0.01, 0.002):
            n = int(round(HORIZON / dt))
            traj = envs.acrobot_rollout_integrator(S0, dt, n, method)
            if not np.all(np.isfinite(traj)):
                print(f"{method:>18} {cls:>16} {dt:>7} {'diverged':>12}")
                rows.append(dict(method=method, order=order, integrator_class=cls,
                                 dt=dt, max_rel_drift=float("inf"),
                                 secular_slope=float("inf"), bounded=False))
                continue
            E = np.array([envs.acrobot_energy_coords(s) for s in traj])
            t = np.arange(len(E)) * dt
            rel = float(np.max(np.abs(E - E[0])) / abs(E[0]))
            slope = secular_slope(E, t) / abs(E[0])
            # bounded means the second half is no worse than the first half
            half = len(E) // 2
            d1 = np.max(np.abs(E[:half] - E[0]))
            d2 = np.max(np.abs(E[half:] - E[0]))
            bounded = bool(d2 < 3 * max(d1, 1e-300))
            print(f"{method:>18} {cls:>16} {dt:>7} {rel:>12.3e} "
                  f"{slope:>15.3e} {str(bounded):>10}")
            rows.append(dict(method=method, order=order, integrator_class=cls,
                             dt=dt, max_rel_drift=rel, secular_slope=slope,
                             first_half_drift=float(d1 / abs(E[0])),
                             second_half_drift=float(d2 / abs(E[0])),
                             bounded=bounded))
        print()

    # observed convergence order per method
    print("Observed order of the invariant error in dt:")
    for method, order, cls in METHODS:
        rs = [r for r in rows if r["method"] == method
              and np.isfinite(r["max_rel_drift"])]
        if len(rs) >= 2:
            a, b = rs[0], rs[-1]
            if b["max_rel_drift"] > 0:
                p = np.log(a["max_rel_drift"] / b["max_rel_drift"]) / \
                    np.log(a["dt"] / b["dt"])
                print(f"  {method:>18}: observed p = {p:5.2f}   (nominal {order})")

    # ------------------------------------------------------- horizon sweep ---
    # The bounded/secular distinction is a statement about long horizons. Over
    # a short one the error is dominated by the method's order p, not by
    # symplecticity, and RK4 beats both symplectic methods despite drifting
    # secularly. This sweep locates the horizon at which the distinction
    # actually becomes visible.
    print("\n" + "=" * 96)
    print("Horizon dependence: does max|dE| grow with T (secular) or "
          "saturate (bounded)?")
    print("=" * 96)
    dt = 0.01
    horizons = [40.0, 400.0, 4000.0]
    print(f"{'method':>18} " + " ".join(f"{'T='+str(int(T)):>13}" for T in horizons)
          + f" {'growth 40->4000':>17}")
    for method, order, cls in METHODS:
        vals = []
        for T in horizons:
            n = int(round(T / dt))
            traj = envs.acrobot_rollout_integrator(S0, dt, n, method)
            if not np.all(np.isfinite(traj)):
                vals.append(float("inf"))
                continue
            E = np.array([envs.acrobot_energy_coords(s) for s in traj])
            vals.append(float(np.max(np.abs(E - E[0])) / abs(E[0])))
        growth = vals[-1] / vals[0] if vals[0] > 0 and np.isfinite(vals[-1]) \
            else float("inf")
        print(f"{method:>18} " + " ".join(f"{v:>13.3e}" for v in vals)
              + f" {growth:>17.1f}x")
        rows.append(dict(method=method, integrator_class=cls,
                         experiment="horizon_sweep", dt=dt,
                         drift_T40=vals[0], drift_T400=vals[1],
                         drift_T4000=vals[2], growth_factor=growth))
    print("\nA bounded method's growth factor stays O(1) as T increases by 100x;")
    print("a secular one grows roughly linearly in T.")

    out = os.path.join(RESULTS, "drift_diagnostic.csv")
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
