"""Positive controls for the version audit.

The audit's headline is a null: eleven releases, bitwise identical dynamics. A
null is only worth reporting if the same pipeline, run unchanged, separates a
difference known to be present. These controls supply that evidence, and they
perturb the *shipped* Gymnasium class rather than a reimplementation of it, so
what is being validated is the instrument as actually applied.

The controls sit at different distances from the null. One is structural:
`book_or_nips=nips` selects the variant Gymnasium itself ships, which drops a
Coriolis term, so energy stops being conserved while both unit-norm identities
survive untouched. The rest are a sensitivity sweep on `LINK_LENGTH_1`, a
constant the dynamics actually read, taken down to a relative error of 1e-5.
The sweep answers the question a null invites, namely how small a change the
audit would have missed.

Gravity is not among them because Gymnasium hardcodes 9.8 inside `_dsdt`
rather than exposing it as an attribute, which is itself worth knowing: a
constant that cannot be reached from outside the class cannot be perturbed by
a configuration error either.

Run:  ./venv/bin/python audit/positive_control.py
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
from run_audit import ENVS_DIR, HERE, RESULTS_DIR, load, max_abs_diff  # noqa: E402

BASE_TAG = "gymnasium-1.3.0"
CONTROLS = [
    ("nips_coriolis", "book_or_nips=nips"),
    ("m2_1.3", "LINK_MASS_2=1.3"),
] + [(f"l1_rel_{r:g}", f"LINK_LENGTH_1={1.0 + r!r}") for r in
     (1e-2, 1e-3, 1e-4, 1e-5)]


def dump_control(name, mutation, n_traj=8, n_steps=500):
    out_dir = os.path.join(HERE, "data", "controls")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{name}.npz")
    python = os.path.join(ENVS_DIR, BASE_TAG, "bin", "python")
    proc = subprocess.run(
        [python, os.path.join(HERE, "dump_trajectories.py"),
         "--env-id", "Acrobot-v1", "--out", out, "--n-traj", str(n_traj),
         "--n-steps", str(n_steps), "--mutate", mutation],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"{name} dump failed:\n{proc.stderr}")
    z = np.load(out, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return [z[f"traj{i}"] for i in range(meta["n_traj"])]


def main():
    base = load("Acrobot-v1", BASE_TAG)
    if base is None:
        raise SystemExit("run audit/run_audit.py first to produce the baseline")
    G, names, kinds = bugs.acrobot_reference()

    rows = []
    print(f"baseline: {BASE_TAG} Acrobot-v1, unmutated\n")
    for name, mutation in CONTROLS:
        trajs = dump_control(name, mutation)
        diff = max_abs_diff(base["trajs"], trajs)
        d = screen(G, base["trajs"], trajs, bugs.ACRO_SYMS,
                   names=names, kinds=kinds,
                   unit=ideal_diff.TRAJECTORY)
        print(f"{name:>14}  ({mutation})")
        print(f"{'':>14}  stage 1 max|diff| = {diff:.3e}")
        print(f"{'':>14}  stage 2 localised = {d.localised or ['-']}")
        for v in d.verdicts:
            print(f"{'':>16}  {v.name:<14} ratio={v.ratio:>10.3e} "
                  f"p={v.pvalue:.3e} {'BROKEN' if v.broken else 'ok'}")
            rows.append(dict(control=name, mutation=mutation,
                             max_abs_diff=diff, generator=v.name,
                             residual_ref=v.residual_ref,
                             residual_test=v.residual_test, ratio=v.ratio,
                             pvalue=v.pvalue, broken=v.broken,
                             localised=";".join(d.localised) or "-"))
        print()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, "positive_controls.csv"), index=False)


if __name__ == "__main__":
    main()
