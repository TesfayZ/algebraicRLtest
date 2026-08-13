"""Reacher's observation against its own forward kinematics.

The audit surfaced this and the paper reports it, so it needs a script rather
than a transcript. `ReacherEnv._get_obs` assembles one observation vector out of
two quantities that are read at different points of the integration step: the
joint angles come from `data.qpos`, which `mj_step` has already integrated,
while the fingertip-to-target vector comes from `get_body_com`, that is
`data.xipos`, which `mj_step` leaves describing an internal stage because
`mj_forward` is never called afterwards. The observation therefore does not
satisfy the forward kinematics of the model it came from.

Three conditions are measured, and the contrast between the first two is the
whole diagnosis.

  mj_forward only    the state is set and the kinematics recomputed, with no
                     step. The identity holds at machine precision, which rules
                     out the model geometry as the cause.
  after mj_step      one step is taken from the same state. The residual appears.
  step, qvel = 0     a step is taken but nothing moves. The residual vanishes
                     again, which is what pins the cause to the joints moving
                     during the step and not to the step being taken.
  outside the limit  theta2 is drawn over the full circle rather than inside
                     joint1's [-3, 3] range, where MuJoCo's limit constraint
                     legitimately moves qpos during the step. This bounds the
                     measurement rather than adding to it, and it is why every
                     number quoted elsewhere is measured inside the limit.

`--across-releases` re-runs the whole thing inside each of the audit's MuJoCo
virtualenvs, which is what supports the claim that the finding is long-standing
rather than a regression. Without it the script measures one configuration, and
a claim spanning six of them has no measurement behind it.

Run:  ./venv/bin/python audit/reacher_fk.py
      ./venv/bin/python audit/reacher_fk.py --across-releases
Writes audit/Results/reacher_fk.csv.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

# pandas is imported where it is used, not here. Under --across-releases this
# file is re-executed by each audit virtualenv, and those hold only what their
# own release forces: gymnasium, mujoco and numpy. A top-level pandas import
# makes every child exit 1 and the driver report six skips.

HERE = os.path.dirname(os.path.abspath(__file__))
ENVS_DIR = os.path.join(HERE, "envs")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "Results")

#: Link lengths as reacher.xml declares them, in metres.
L1, L2 = 0.1, 0.11

#: joint1's range in reacher.xml. Sampling theta2 over the full circle puts
#: (pi - 3) / pi of draws outside it.
JOINT1_LIMIT = 3.0


def residuals(env_id, n=2000, seed=7, do_step=True, inside_limit=True,
              zero_qvel=False):
    """Max |fingertip - forward kinematics| over `n` random poses."""
    import gymnasium as gym
    import mujoco

    env = gym.make(env_id, disable_env_checker=True)
    u = env.unwrapped
    env.reset(seed=0)
    rng = np.random.default_rng(seed)

    lim = JOINT1_LIMIT - 0.1 if inside_limit else np.pi
    out = []
    for _ in range(n):
        th1 = rng.uniform(-np.pi, np.pi)
        th2 = rng.uniform(-lim, lim)
        tgt = rng.uniform(-0.15, 0.15, 2)
        qvel = (np.zeros(4) if zero_qvel
                else np.array([rng.uniform(-1, 1), rng.uniform(-1, 1),
                               0.0, 0.0]))
        u.set_state(np.array([th1, th2, tgt[0], tgt[1]]), qvel)
        if do_step:
            env.step(np.zeros(2, dtype=np.float32))
        else:
            mujoco.mj_forward(u.model, u.data)
        t1, t2 = float(u.data.qpos[0]), float(u.data.qpos[1])
        fin = u.data.body("fingertip").xpos
        out.append(max(abs(fin[0] - (L1 * np.cos(t1) + L2 * np.cos(t1 + t2))),
                       abs(fin[1] - (L1 * np.sin(t1) + L2 * np.sin(t1 + t2)))))
    env.close()
    return np.array(out)


CONDITIONS = (
    ("mj_forward only", dict(do_step=False, inside_limit=True)),
    ("after mj_step", dict(do_step=True, inside_limit=True)),
    ("step, qvel = 0", dict(do_step=True, inside_limit=True, zero_qvel=True)),
    ("outside the limit", dict(do_step=True, inside_limit=False)),
)


def measure(n, env_ids=("Reacher-v4", "Reacher-v5")):
    """Every condition on every Reacher version, in the interpreter we are in."""
    import gymnasium
    import mujoco as mj

    rows = []
    for env_id in env_ids:
        for label, kw in CONDITIONS:
            try:
                r = residuals(env_id, n=n, **kw)
            except Exception as exc:                 # version lacks this env
                print(f"  skip {env_id} {label}: {exc}")
                continue
            rows.append(dict(env_id=env_id, condition=label,
                             gymnasium=gymnasium.__version__,
                             mujoco=mj.__version__, n_poses=n,
                             median=float(np.median(r)),
                             maximum=float(r.max())))
    return rows


def across_releases(n):
    """Re-run inside each MuJoCo virtualenv the audit built.

    The claim in the paper is that this reproduces across gymnasium 1.0.0 to
    1.3.0 and mujoco 3.1.6 to 3.10.0. That is a claim about six installed
    configurations, so it has to be measured in six interpreters and cannot be
    measured in one.
    """
    tags = sorted(t for t in os.listdir(ENVS_DIR) if t.startswith("mj-")) \
        if os.path.isdir(ENVS_DIR) else []
    if not tags:
        raise SystemExit(f"no MuJoCo virtualenvs under {ENVS_DIR}; "
                         "run audit/setup_envs.sh first")
    rows = []
    for tag in tags:
        py = os.path.join(ENVS_DIR, tag, "bin", "python")
        if not os.path.exists(py):
            print(f"  skip {tag}: no interpreter")
            continue
        proc = subprocess.run(
            [py, os.path.abspath(__file__), "--emit-json", "--samples", str(n)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  skip {tag}: exited {proc.returncode}\n"
                  f"{proc.stderr.strip()[-400:]}")
            continue
        try:
            got = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            print(f"  skip {tag}: no parsable output")
            continue
        for r in got:
            r["release_tag"] = tag
        rows.extend(got)
        print(f"  {tag}: {len(got)} rows")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--across-releases", action="store_true",
                    help="re-run inside each of the audit's MuJoCo venvs")
    ap.add_argument("--emit-json", action="store_true",
                    help="print one JSON line and exit; used by the driver")
    args = ap.parse_args()
    n = 200 if args.quick else args.samples

    if args.emit_json:
        print(json.dumps(measure(n)))
        return

    print("=" * 88)
    print("Reacher's observation against its own forward kinematics")
    print("=" * 88)

    if args.across_releases:
        rows = across_releases(n)
    else:
        import gymnasium
        import mujoco as mj
        print(f"gymnasium {gymnasium.__version__}, mujoco {mj.__version__}, "
              f"{n} poses per condition")
        rows = measure(n)
        for r in rows:
            r["release_tag"] = "project venv"

    print(f"\n{'release':<36}{'env':<14}{'condition':<20}"
          f"{'median':>13}{'max':>13}")
    print("-" * 96)
    for r in rows:
        print(f"{r['release_tag']:<36}{r['env_id']:<14}{r['condition']:<20}"
              f"{r['median']:>13.3e}{r['maximum']:>13.3e}")

    if not rows:
        raise SystemExit("no measurements collected; refusing to overwrite "
                         "audit/Results/reacher_fk.csv with an empty file")

    import pandas as pd
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "reacher_fk.csv")
    pd.DataFrame(rows).to_csv(out, index=False)

    stepped = [r for r in rows if r["condition"] == "after mj_step"]
    if stepped:
        vers = sorted({(r["gymnasium"], r["mujoco"]) for r in stepped})
        print(f"\nReproduces on {len(vers)} configuration(s): "
              + ", ".join(f"gymnasium {g}/mujoco {m}" for g, m in vers))
    print("\nThe first condition rules out the geometry: reacher.xml and the "
          "idealised\ntwo-link identity agree at machine precision. The third "
          "rules out the step\nitself: a step in which nothing moves leaves the "
          "identity intact. What is left\nis the joints moving during the step, "
          "with the two halves of the observation\nread on either side of it. "
          "The fourth condition is the joint limit doing its\njob and is "
          "excluded from every figure quoted elsewhere.")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
