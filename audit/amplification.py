"""How a last-bit difference between two shipped releases grows with the step.

`run_audit.py` classifies a release pair by its *first-step* separation rather
than by the largest separation over the rollout. This script is the measurement
behind that choice: it traces the separation step by step for a pair the
terminal threshold would have called a dynamics change, and for a pair on a
non-chaotic environment over the identical releases and horizon.

The distinction matters because the exact stage's guarantee, that unchanged code
on byte-identical input reproduces byte-identical doubles, is a statement about
exact arithmetic. In floating point it survives only until the dynamics amplify
the last place, which on a chaotic system takes a few hundred steps.

    ./venv/bin/python audit/amplification.py
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "Results")

# One chaotic environment and one that is not, over release pairs that exist for
# both, so the comparison isolates the dynamics rather than the binding.
CASES = [
    ("InvertedDoublePendulum-v4",
     "mj-gymnasium-0.29.1-mujoco-2.3.7", "mj-gymnasium-1.0.0-mujoco-3.1.6"),
    ("InvertedDoublePendulum-v4",
     "mj-gymnasium-1.2.3-mujoco-3.2.7", "mj-gymnasium-1.3.0-mujoco-3.10.0"),
    ("Reacher-v4",
     "mj-gymnasium-0.29.1-mujoco-2.3.7", "mj-gymnasium-1.0.0-mujoco-3.1.6"),
    ("Reacher-v4",
     "mj-gymnasium-1.2.3-mujoco-3.2.7", "mj-gymnasium-1.3.0-mujoco-3.10.0"),
]

PROBE_STEPS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 400, 499]

#: The window the growth rate is fitted over. Below `RATE_LO` the separation is
#: still quantised at a few units in the last place and its logarithm is a
#: staircase; above `RATE_HI` a chaotic pair is approaching the clip the
#: observation saturates at. Between them the growth is clean exponential, which
#: is the regime the paper's "one decade per N steps" describes.
RATE_LO, RATE_HI = 50, 400


def load(env_id, tag):
    path = os.path.join(DATA_DIR, env_id, f"{tag}.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return [z[f"traj{i}"] for i in range(meta["n_traj"])]


def trace(a, b):
    """Per-step separation, maximised over the paired trajectory set."""
    per_step = np.max([np.max(np.abs(x - y), axis=1) for x, y in zip(a, b)],
                      axis=0)
    return per_step


def growth(sep):
    """(steps per decade over the fitted window, total decades climbed).

    Both are quoted in the paper, so both are written to the CSV rather than
    only printed. A rate recomputed by a reader from two probe rows would agree
    with the text by construction; one that exists nowhere but on a terminal
    cannot be contradicted by anything, which is how a figure quoted from
    arithmetic survives a review pass.
    """
    lo, hi = RATE_LO, min(RATE_HI, len(sep) - 1)
    if not (sep[lo] > 0 and sep[hi] > 0 and sep[0] > 0):
        return float("nan"), float("nan")
    decades = np.log10(sep[hi] / sep[lo]) / (hi - lo)
    per_decade = (1.0 / decades) if decades > 0 else float("inf")
    return per_decade, float(np.log10(sep[hi] / sep[0]))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for env_id, older, newer in CASES:
        ta, tb = load(env_id, older), load(env_id, newer)
        if ta is None or tb is None:
            print(f"skip {env_id} {older} -> {newer}: dump missing")
            continue
        sep = trace(ta, tb)
        # Growth rate over the stretch where the separation is still resolvable
        # but no longer at the noise floor, reported in decades per step.
        per_decade, decades = growth(sep)
        print(f"\n=== {env_id} ===\n  {older}\n  {newer}")
        for s in PROBE_STEPS:
            if s < len(sep):
                print(f"    step {s:4d}   {sep[s]:.3e}")
                rows.append(dict(env_id=env_id, older=older, newer=newer,
                                 step=s, separation=float(sep[s]),
                                 steps_per_decade=per_decade,
                                 decades_climbed=decades))
        print(f"    growth: {per_decade:.1f} steps per decade between steps "
              f"{RATE_LO} and {min(RATE_HI, len(sep) - 1)}; "
              f"{decades:.1f} decades climbed from step 0")

    out = os.path.join(RESULTS_DIR, "amplification.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
