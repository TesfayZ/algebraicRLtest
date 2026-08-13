"""How large must a parameter error be before the screen sees it, and why.

The version audit returns a null, and the positive controls show the pipeline
separates a dropped Coriolis term and a 30% mass error on the shipped class.
Between those two facts sits the question that decides what the null is worth:
a 1% error in `LINK_LENGTH_1` changes the trajectory by 1.6 rad/s, which stage 1
sees plainly, yet the screen does not flag the energy generator. Either the
screen is insensitive or the simulator is.

It is the simulator. Gymnasium ships Acrobot at dt = 0.2, and at that step size
RK4's own energy drift is a larger violation of conservation than a percent-level
parameter error is. The screen is measuring a real quantity against a floor the
integrator sets. This script demonstrates that by sweeping the fault magnitude at
several step sizes, holding the simulated horizon fixed so the comparison is
between step sizes and not between trajectory lengths.

The consequence is a statement about Gymnasium rather than about the method:
energy is not a usable invariant of Acrobot as shipped, and any diagnostic built
on it, algebraic or otherwise, inherits that floor.

Run:  ./venv/bin/python audit/sensitivity.py
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bugs                                                    # noqa: E402
import ideal_diff                                               # noqa: E402
from ideal_diff import screen                                  # noqa: E402
from run_audit import ENVS_DIR, HERE, RESULTS_DIR              # noqa: E402

BASE_TAG = "gymnasium-1.3.0"
HORIZON = 100.0          # seconds of simulated time, held fixed across dt
DTS = (0.2, 0.05, 0.01)  # 0.2 is what Gymnasium ships
RELS = (3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3)
PARAMS = ("LINK_MASS_2", "LINK_LENGTH_1")


def dump(name, mutation, n_steps, n_traj=8):
    out_dir = os.path.join(HERE, "data", "sensitivity")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{name}.npz")
    proc = subprocess.run(
        [os.path.join(ENVS_DIR, BASE_TAG, "bin", "python"),
         os.path.join(HERE, "dump_trajectories.py"),
         "--env-id", "Acrobot-v1", "--out", out, "--n-traj", str(n_traj),
         "--n-steps", str(n_steps), "--mutate", mutation],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"{name} dump failed:\n{proc.stderr}")
    z = np.load(out, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return [z[f"traj{i}"] for i in range(meta["n_traj"])]


def main():
    G, names, kinds = bugs.acrobot_reference()
    i_energy = names.index("energy")
    rows = []

    for dt in DTS:
        n_steps = int(round(HORIZON / dt))
        base = dump(f"base_dt{dt:g}", f"dt={dt!r}", n_steps)

        # The reference system's own energy drift is the floor everything else
        # is measured against, so report it before any fault is applied.
        from ideal_diff import generator_residual
        floor = generator_residual(G[i_energy], base, bugs.ACRO_SYMS,
                                   kind="conserved")
        print(f"\n=== dt = {dt:g} ({n_steps} steps, horizon {HORIZON:g}s) ===")
        print(f"    reference energy drift (the floor) = {floor:.3e}")

        for param in PARAMS:
            spec = 1.0  # every swept parameter has spec value 1.0 in Acrobot
            for rel in RELS:
                value = spec * (1.0 + rel)
                tag = f"{param}_{rel:g}_dt{dt:g}"
                trajs = dump(tag, f"dt={dt!r},{param}={value!r}", n_steps)
                d = screen(G, base, trajs, bugs.ACRO_SYMS,
                           names=names, kinds=kinds,
                           unit=ideal_diff.TRAJECTORY)
                v = d.verdicts[i_energy]
                rows.append(dict(dt=dt, n_steps=n_steps, param=param,
                                 rel_error=rel, drift_floor=floor,
                                 residual_ref=v.residual_ref,
                                 residual_test=v.residual_test, ratio=v.ratio,
                                 pvalue=v.pvalue, broken=v.broken,
                                 localised=";".join(d.localised) or "-"))
                print(f"    {param:<14} rel={rel:<7g} ratio={v.ratio:>9.3f} "
                      f"p={v.pvalue:>10.3e}  {'BROKEN' if v.broken else '-'}")

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "sensitivity.csv"), index=False)

    print("\n=== smallest relative error the screen localises ===")
    for (dt, param), grp in df.groupby(["dt", "param"]):
        hit = grp[grp["broken"]]["rel_error"]
        best = f"{hit.min():g}" if len(hit) else f">{max(RELS):g}"
        print(f"  dt={dt:<6g} {param:<14} {best}")


if __name__ == "__main__":
    main()
