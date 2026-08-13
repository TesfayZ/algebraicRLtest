"""Do the reference generators actually hold on the shipped environments?

The audit's null says every release satisfies the reference set to the same
degree. That statement is vacuous if the degree is "not at all": two systems
that both violate a generator badly are still equal to each other, so the screen
would report nothing and mean nothing. This script closes that gap by reporting
each generator's absolute residual on the newest release of each family, which
is the precondition the null result rests on.

Run:  ./venv/bin/python audit/check_reference.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ideal_diff import generator_residual                      # noqa: E402
from run_audit import ORDERS, REFERENCES, RESULTS_DIR, load    # noqa: E402

#: Every environment in the matrix that carries an exact reference set, at the
#: newest release that ships it. InvertedDoublePendulum is included because it
#: is the environment whose release pairs separate the most, and the precondition
#: this script establishes is what makes that separation readable as amplified
#: rounding rather than as a violated constraint.
NEWEST = {
    "Acrobot-v1": "gymnasium-1.3.0",
    "Pendulum-v1": "gymnasium-1.3.0",
    "Reacher-v4": "mj-gymnasium-1.3.0-mujoco-3.10.0",
    "Reacher-v5": "mj-gymnasium-1.3.0-mujoco-3.10.0",
    "InvertedDoublePendulum-v4": "mj-gymnasium-1.3.0-mujoco-3.10.0",
    "InvertedDoublePendulum-v5": "mj-gymnasium-1.3.0-mujoco-3.10.0",
}


def main():
    rows = []
    for env_id, tag in NEWEST.items():
        assert tag in ORDERS[env_id], f"{tag} is not in the {env_id} release list"
        data = load(env_id, tag)
        if data is None:
            print(f"{env_id}: no dump for {tag}, run run_audit.py first")
            continue
        G, names, kinds, variables = REFERENCES[env_id]()
        print(f"\n=== {env_id} @ {tag} ===")
        for g, name, kind in zip(G, names, kinds):
            r = generator_residual(g, data["trajs"], variables, kind)
            rows.append(dict(env_id=env_id, tag=tag, generator=name,
                             kind=kind, residual=r))
            print(f"  {name:<16} {kind:<10} residual = {r:.3e}")

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "reference_residuals.csv"), index=False)


if __name__ == "__main__":
    main()
