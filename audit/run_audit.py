"""Audit shipped simulator releases for algebraic drift.

Two stages, deliberately in this order.

Stage 1 is an exact comparison. Because `dump_trajectories.py` imposes the same
initial states and the same action sequence on every release, two releases whose
dynamics code is unchanged must produce bitwise identical trajectories. When
they do, no statistical question needs asking and a null result is a certainty
rather than a failure to reject, which is the one thing a two-sample test can
never give you.

Stage 2 runs the screen of Section 5.2 on every pair that stage 1 separates. Its
job is not detection, which stage 1 already settled, but localisation: naming
which reference generator the newer release stopped satisfying.

Run after ./audit/setup_envs.sh:
    ./venv/bin/python audit/run_audit.py            # dump, then analyse
    ./venv/bin/python audit/run_audit.py --analyse  # reuse existing dumps
"""
import argparse
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bugs                                                    # noqa: E402
from ideal_diff import TRAJECTORY, screen                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ENVS_DIR = os.path.join(HERE, "envs")
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "Results")

# Release order, oldest first. Adjacent pairs in this list are the version
# transitions a practitioner actually lived through.
#
# These two lists are the authoritative audited matrix: eleven classic-control
# releases and six MuJoCo binding configurations, seventeen in all, giving 350
# pairs over eleven environments. `setup_envs.sh` attempts one more row,
# gym 0.21.0, which does not install; do not read a release count off that
# script.
CLASSIC_ORDER = [
    "gym-0.23.1", "gym-0.25.2", "gym-0.26.2",
    "gymnasium-0.26.3", "gymnasium-0.27.1", "gymnasium-0.28.1",
    "gymnasium-0.29.1", "gymnasium-1.0.0", "gymnasium-1.1.1",
    "gymnasium-1.2.3", "gymnasium-1.3.0",
]

# The MuJoCo rows move two things independently. The first four advance
# gymnasium and the mujoco binding together, as a user upgrading would. Three
# rows carry gymnasium 1.3.0 and differ only in the binding, which is what
# attributes any difference to the physics library rather than to the wrapper;
# two of those three are additions for that purpose and the third is also the
# newest row of the upgrade path. Both READMEs say "three of which hold
# gymnasium at 1.3.0", and counting the rows is the way to read that.
MUJOCO_ORDER = [
    "mj-gymnasium-0.29.1-mujoco-2.3.7",
    "mj-gymnasium-1.0.0-mujoco-3.1.6",
    "mj-gymnasium-1.2.3-mujoco-3.2.7",
    "mj-gymnasium-1.3.0-mujoco-3.10.0",
    "mj-gymnasium-1.3.0-mujoco-3.1.6",
    "mj-gymnasium-1.3.0-mujoco-3.2.7",
]

# The v5 MuJoCo environments were introduced in gymnasium 1.0.0, so the 0.29.1
# row cannot supply them and their audit runs over a shorter list rather than
# pretending.
MUJOCO_ORDER_V5 = [t for t in MUJOCO_ORDER if not t.startswith("mj-gymnasium-0.")]


def pendulum_reference():
    """(generators, names, kinds) for Pendulum in its (cos, sin, w) observation.

    Gymnasium's Pendulum integrates thddot = 3g/(2l) sin(th) with theta measured
    from upright, which is the uniform-rod model. Multiplying by thdot and
    integrating gives the conserved quantity below; the constants are the
    shipped defaults g=10, l=1, so the coefficient of cos(theta) is 3g/(2l)=15.
    The quantity is only conserved up to the drift of the semi-implicit Euler
    step the environment uses, which is large, but drift is a property of the
    code and therefore identical between releases that did not change it.
    """
    c, s, w = sp.symbols("c s w", real=True)
    energy = sp.Rational(1, 2) * w**2 + 15 * c
    return ([c**2 + s**2 - 1, energy], ["unit-norm", "energy"],
            ["vanishing", "conserved"], (c, s, w))


def reacher_v4_reference():
    """The paper's Reacher set, in raw metres rather than nondimensionalised.

    Gymnasium reports the observation in metres, so the link lengths enter the
    forward-kinematics generators at their shipped values (0.1 and 0.11) instead
    of the scaled ones `bugs.REFERENCE_REACHER` uses.
    """
    G, names, kinds = bugs.reacher_reference(scale=1.0)
    return G, names, kinds, bugs.REACH_SYMS


def reacher_v5_reference():
    """v5 drops the identically-zero dz component, so its set loses `dz=0`.

    Everything else is unchanged and lives in the first ten coordinates, which
    v5 keeps in the same order.
    """
    G, names, kinds, syms = reacher_v4_reference()
    keep = [i for i, n in enumerate(names) if n != "dz=0"]
    return ([G[i] for i in keep], [names[i] for i in keep],
            [kinds[i] for i in keep], syms[:10])


def _idp_reference(n_obs):
    """The two unit-norm identities of InvertedDoublePendulum's observation.

    Both v4 and v5 report each hinge angle as a (sin, cos) pair, at observation
    indices (1, 3) for the first hinge and (2, 4) for the second, so the
    embedding identity holds on both; measured on shipped data it holds to
    2.2e-16. The trailing columns differ between versions and enter no
    generator: v4 is 11-wide and ends in three identically-zero constraint
    forces, v5 is 9-wide and ends in one. `screen` maps symbols to observation
    columns positionally, so the list has to be as wide as the observation
    actually is.
    """
    syms = sp.symbols(f"o0:{n_obs}", real=True)
    s1, s2, c1, c2 = syms[1], syms[2], syms[3], syms[4]
    return ([c1**2 + s1**2 - 1, c2**2 + s2**2 - 1],
            ["unit-norm-1", "unit-norm-2"],
            ["vanishing", "vanishing"], syms)


def no_reference():
    """No exact polynomial invariant in the shipped observation.

    These environments are audited by the exact stage alone. That stage is what
    delivers the null, since a release pair agreeing bitwise on byte-identical
    inputs has not touched the dynamics; the screen exists to localise a
    separation, and where there is none there is nothing to localise.
    """
    return [], [], [], ()


REFERENCES = {
    "Acrobot-v1": lambda: bugs.acrobot_reference() + (bugs.ACRO_SYMS,),
    "Pendulum-v1": pendulum_reference,
    "Reacher-v4": reacher_v4_reference,
    "Reacher-v5": reacher_v5_reference,
    "CartPole-v1": no_reference,
    "MountainCar-v0": no_reference,
    "MountainCarContinuous-v0": no_reference,
    "InvertedPendulum-v4": no_reference,
    "InvertedPendulum-v5": no_reference,
    "InvertedDoublePendulum-v4": lambda: _idp_reference(11),
    "InvertedDoublePendulum-v5": lambda: _idp_reference(9),
}

# Which release list each environment is audited over.
ORDERS = {
    "Acrobot-v1": CLASSIC_ORDER,
    "Pendulum-v1": CLASSIC_ORDER,
    "CartPole-v1": CLASSIC_ORDER,
    "MountainCar-v0": CLASSIC_ORDER,
    "MountainCarContinuous-v0": CLASSIC_ORDER,
    "Reacher-v4": MUJOCO_ORDER,
    "Reacher-v5": MUJOCO_ORDER_V5,
    "InvertedPendulum-v4": MUJOCO_ORDER,
    "InvertedPendulum-v5": MUJOCO_ORDER_V5,
    "InvertedDoublePendulum-v4": MUJOCO_ORDER,
    "InvertedDoublePendulum-v5": MUJOCO_ORDER_V5,
}


def dump_all(env_ids, n_traj, n_steps):
    """Run the dumper inside every built venv."""
    os.makedirs(DATA_DIR, exist_ok=True)
    script = os.path.join(HERE, "dump_trajectories.py")
    tags = sorted({t for e in env_ids for t in ORDERS[e]}, key=str)
    for tag in tags:
        python = os.path.join(ENVS_DIR, tag, "bin", "python")
        if not os.path.exists(python):
            print(f"  skip {tag}: venv not built")
            continue
        for env_id in [e for e in env_ids if tag in ORDERS[e]]:
            out_dir = os.path.join(DATA_DIR, env_id)
            os.makedirs(out_dir, exist_ok=True)
            out = os.path.join(out_dir, f"{tag}.npz")
            proc = subprocess.run(
                [python, script, "--env-id", env_id, "--out", out,
                 "--n-traj", str(n_traj), "--n-steps", str(n_steps)],
                capture_output=True, text=True)
            if proc.returncode != 0:
                tail = proc.stderr.strip().splitlines()[-1:] or ["(no stderr)"]
                print(f"  FAIL {tag} {env_id}: {tail[0]}")
            else:
                print(f"  ok   {tag} {env_id}")


def load(env_id, tag):
    path = os.path.join(DATA_DIR, env_id, f"{tag}.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    trajs = [z[f"traj{i}"] for i in range(meta["n_traj"])]
    return dict(meta=meta, trajs=trajs, initial_states=z["initial_states"])


def max_abs_diff(a, b):
    """Largest elementwise separation between two matched trajectory sets."""
    return max(float(np.max(np.abs(x - y))) for x, y in zip(a, b))


def first_step_diff(a, b):
    """Separation on the first step alone, before any dynamics can amplify it."""
    return max(float(np.max(np.abs(x[0] - y[0]))) for x, y in zip(a, b))


# Bitwise equality is the right test for pure-Python dynamics, where identical
# code on identical inputs must reproduce identical doubles. It is too strict
# once a compiled physics library is involved: recompiling MuJoCo can reorder
# floating-point operations and move the last bit without changing the model.
# Separations below this are reported as float noise rather than as a change,
# and the threshold is far under any physical scale in these environments.
FLOAT_NOISE = 1e-13


def classify(diff, first):
    """Verdict for one release pair, from its terminal and first-step separation.

    Thresholding the terminal separation alone is valid only while the dynamics
    are non-expansive over the audit horizon. On a chaotic system it is not: a
    difference of one unit in the last place at step 0 grows exponentially, so
    two releases running identical model files separate by O(1e-2) after a few
    hundred steps with nothing having changed. The first step is taken from
    imposed, byte-identical state, so a separation that is at machine epsilon
    there and large later is amplification and not a model change, and the two
    cases are distinguished rather than collapsed into one verdict.
    """
    if diff == 0.0:
        return "identical"
    if diff < FLOAT_NOISE:
        return "float-noise"
    return "float-noise-amplified" if first < FLOAT_NOISE else "DIFFERS"


def analyse(env_ids, alpha=0.01):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    meta_rows, pair_rows, gen_rows = [], [], []

    for env_id in env_ids:
        order = ORDERS[env_id]
        loaded = {t: load(env_id, t) for t in order}
        have = [t for t in order if loaded[t] is not None]
        if len(have) < 2:
            print(f"{env_id}: fewer than two releases dumped, skipping")
            continue

        G, names, kinds, variables = REFERENCES[env_id]()

        for tag in have:
            m = dict(loaded[tag]["meta"])
            m["tag"] = tag
            m["env_id"] = env_id
            meta_rows.append(m)

        print(f"\n=== {env_id}: {len(have)} releases ===")
        for a, b in itertools.combinations(have, 2):
            da, db = loaded[a], loaded[b]
            # The whole force of the exact stage is "unchanged code on
            # byte-identical input must reproduce byte-identical doubles", and
            # the input half of that has to be certified rather than assumed.
            # The initial states are drawn inside each release's own virtualenv
            # and NumPy is deliberately not pinned across rows, so nothing
            # guarantees a priori that two releases were handed the same
            # states: NEP 19's stream-compatibility promise covers RandomState
            # and explicitly not Generator, which is what the dumper uses.
            ic_ok = bool(np.array_equal(da["initial_states"],
                                        db["initial_states"]))
            if not ic_ok:
                print(f"  WARNING {env_id} {a} -> {b}: initial states differ "
                      f"between releases; this pair's separation is not "
                      f"evidence about the dynamics and is excluded")

            diff = max_abs_diff(da["trajs"], db["trajs"])
            first = first_step_diff(da["trajs"], db["trajs"])
            same_src = (da["meta"].get("_source_sha256")
                        == db["meta"].get("_source_sha256"))
            adjacent = have.index(b) == have.index(a) + 1
            verdict = classify(diff, first) if ic_ok else "incomparable"
            row = dict(env_id=env_id, older=a, newer=b, adjacent=adjacent,
                       initial_states_identical=ic_ok,
                       max_abs_diff=diff, first_step_diff=first,
                       stage1=verdict,
                       identical=(verdict != "DIFFERS"),
                       same_source_sha=same_src,
                       numpy_older=da["meta"]["numpy"],
                       numpy_newer=db["meta"]["numpy"])

            # Stage 2 only earns its keep where stage 1 found a real separation
            # and a reference set exists to localise against.
            if verdict == "DIFFERS" and G:
                d = screen(G, da["trajs"], db["trajs"], variables,
                           names=names, kinds=kinds, alpha=alpha,
                           unit=TRAJECTORY)
                row["localised"] = ";".join(d.localised) or "-"
                row["detected"] = d.detected
                for v in d.verdicts:
                    gen_rows.append(dict(
                        env_id=env_id, older=a, newer=b, generator=v.name,
                        residual_ref=v.residual_ref,
                        residual_test=v.residual_test,
                        ratio=v.ratio, pvalue=v.pvalue, broken=v.broken))
            else:
                row["localised"] = "-"
                row["detected"] = False
            pair_rows.append(row)

            if adjacent:
                detail = (f"{verdict} max={diff:.3e}" if diff else verdict)
                if verdict == "DIFFERS":
                    detail += f" -> {row['localised']}"
                print(f"  {a:>34} -> {b:<34} {detail}")

    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(os.path.join(RESULTS_DIR, "version_pairs.csv"), index=False)
    pd.DataFrame(meta_rows).to_csv(
        os.path.join(RESULTS_DIR, "version_metadata.csv"), index=False)
    # Written unconditionally, empty header included. Stage 2 runs only on a
    # pair `classify` separates, and on the shipped releases no pair is, so a
    # write guarded on `gen_rows` would never fire and whatever copy is already
    # on disk would survive, contradicting version_pairs.csv under a
    # reproducibility claim that every number comes from a script.
    gen_cols = ["env_id", "older", "newer", "generator", "residual_ref",
                "residual_test", "ratio", "pvalue", "broken"]
    pd.DataFrame(gen_rows, columns=gen_cols).to_csv(
        os.path.join(RESULTS_DIR, "version_generators.csv"), index=False)

    print("\n--- summary over all pairs, not only adjacent ---")
    for env_id, grp in pairs.groupby("env_id"):
        n_diff = int((~grp["identical"]).sum())
        n_amp = int((grp["stage1"] == "float-noise-amplified").sum())
        extra = f", {n_amp} amplified float noise" if n_amp else ""
        print(f"{env_id}: {len(grp)} pairs, {n_diff} separated by stage 1{extra}, "
              f"{int(grp['detected'].sum())} localised by the screen")
    print(f"TOTAL: {len(pairs)} release pairs over "
          f"{pairs['env_id'].nunique()} environments, "
          f"{int((~pairs['identical']).sum())} separated by stage 1, "
          f"{int((pairs['stage1'] == 'float-noise-amplified').sum())} "
          f"amplified float noise")
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse", action="store_true",
                    help="skip dumping and reuse audit/data")
    ap.add_argument("--env-id", action="append", dest="env_ids",
                    choices=sorted(REFERENCES))
    ap.add_argument("--n-traj", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=500)
    args = ap.parse_args()
    env_ids = args.env_ids or sorted(REFERENCES)

    if not args.analyse:
        print("=== dumping ===")
        dump_all(env_ids, args.n_traj, args.n_steps)
    analyse(env_ids)


if __name__ == "__main__":
    main()
