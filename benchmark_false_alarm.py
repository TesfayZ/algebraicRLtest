"""Calibration of the screen's false-alarm rate.

The screen reports a Bonferroni-corrected p-value per generator and declares a
constraint broken below alpha, which is a claim about a rate: over repeated
comparisons of two systems that do not differ, at most alpha of them should
report anything. This measures that rate.

How the null pairs are obtained. Both arms are produced by the same analytic
model at the same settings, so the physics is identical by construction and no
generator is violated in either. What differs is the draw: each arm gets its own
stream of initial conditions, and under measurement noise its own independent
observation-noise stream. That is exactly the situation the screen faces when
two logs of the same system are compared, and it is the null the alpha refers
to. Nothing here is a system under test; there is no fault to find.

Four settings are crossed, because each is a place the nominal rate can fail.

  environment   Acrobot, whose energy generator is a drift over a window, and
                Reacher, whose generators vanish pointwise on i.i.d. poses.
  noise         zero, where the reference residual sits at machine epsilon, and
                sigma = 1e-3, where it sits at a noise floor instead.
  floor         the absolute residual floor of `ideal_diff.RESIDUAL_FLOOR`, and
                no floor at all. A rank test has no scale, so with the floor
                removed two arms whose residuals differ systematically in the
                last significant figure separate completely and the test reports
                near-certainty about a difference of one part in 1e16.
  unit          contiguous blocks, and whole trajectories with an exact
                permutation null. Blocks cut from one trajectory share an
                initial condition, so they are not the independent replicates
                the block-level test counts them as, and the gap between the two
                rates is the size of that dependence.

Reported against the nominal alpha are the per-generator rate, the family-wise
rate over all generators of an environment, and a Wilson interval, which is the
interval that stays inside [0, 1] when the count is zero.

Writes Results/false_alarm.csv.
"""

import argparse
import csv
import os

import numpy as np

import bugs
import envs
import ideal_diff

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
os.makedirs(RESULTS, exist_ok=True)

ALPHA = 0.01
ACRO_KW = dict(dt=0.005, n_traj=8, n_steps=400)
REACH_KW = dict(N=4000, scale=envs.REACHER["l1"])


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial rate. Defined at k = 0 and k = n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def draw(system, seed, noise):
    """One healthy arm, with its own initial conditions and noise stream."""
    if system == "acrobot":
        return bugs.acrobot_trajectories(seed=seed, ic_seed=seed, noise=noise,
                                         **ACRO_KW)
    return bugs.reacher_samples(seed=seed, ic_seed=seed, noise=noise,
                                **REACH_KW)


def reference_set(system):
    if system == "acrobot":
        return bugs.acrobot_reference(), list(bugs.ACRO_SYMS)
    return bugs.reacher_reference(scale=envs.REACHER["l1"]), \
        list(bugs.REACH_SYMS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=200,
                    help="null comparisons per setting")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_pairs = 20 if args.quick else args.pairs
    noises = [0.0] if args.quick else [0.0, 1e-3]

    print("=" * 104)
    print(f"False-alarm calibration: {n_pairs} healthy-vs-healthy comparisons "
          f"per setting, nominal alpha = {ALPHA}")
    print("=" * 104)

    rows, summary = [], []
    for system in ("acrobot", "reacher"):
        (G, names, kinds), V = reference_set(system)
        for noise in noises:
            # Two fresh arms per pair, and no arm reused. Chaining
            # arms[i] against arms[i+1] off a list of n_pairs + 1 draws is
            # cheaper by half, but then every arm sits in two comparisons and
            # the trials are not the independent Bernoulli draws a Wilson
            # interval assumes. The rate is what this experiment exists to
            # report, with an interval, so the interval has to be valid by
            # construction rather than by a measurement that the dependence
            # happens to be small.
            arms = [draw(system, seed=1000 + i, noise=noise)
                    for i in range(2 * n_pairs)]
            for floor_name, floor in (("floor", ideal_diff.RESIDUAL_FLOOR),
                                      ("none", 0.0)):
                for unit in (ideal_diff.BLOCK, ideal_diff.TRAJECTORY):
                    fired = np.zeros(len(G), dtype=int)
                    family = 0
                    for i in range(n_pairs):
                        diff = ideal_diff.screen(
                            G, arms[2 * i], arms[2 * i + 1], V, names=names,
                            kinds=kinds, alpha=ALPHA, floor=floor, unit=unit)
                        for j, v in enumerate(diff.verdicts):
                            fired[j] += int(v.broken)
                        family += int(diff.detected)
                        rows.append(dict(
                            system=system, noise=noise, floor=floor,
                            unit=unit, pair=i, any_fired=diff.detected,
                            **{f"p_{n.replace(' ', '_')}": v.pvalue
                               for n, v in zip(names, diff.verdicts)}))
                    lo, hi = wilson(family, n_pairs)
                    summary.append(dict(
                        system=system, noise=noise, floor=floor, unit=unit,
                        n_pairs=n_pairs, family_false_alarms=family,
                        family_rate=family / n_pairs,
                        wilson_lo=lo, wilson_hi=hi,
                        **{f"rate_{n.replace(' ', '_')}": fired[j] / n_pairs
                           for j, n in enumerate(names)}))
                    per = "  ".join(f"{n}={fired[j] / n_pairs:.3f}"
                                    for j, n in enumerate(names))
                    print(f"{system:<9} sigma={noise:<7g} floor={floor_name:<6}"
                          f"{unit:<11} family-wise "
                          f"{family / n_pairs:.3f} "
                          f"[{lo:.3f}, {hi:.3f}]   {per}")

    print("\n" + "=" * 104)
    print("Read-off")
    print("=" * 104)
    for s in summary:
        verdict = ("within nominal" if s["wilson_hi"] <= ALPHA
                   else "above nominal" if s["family_rate"] > ALPHA
                   else "not resolved at this sample size")
        print(f"  {s['system']:<9} sigma={s['noise']:<7g} "
              f"floor={s['floor']:<8g} {s['unit']:<11} {verdict}")

    _write(os.path.join(RESULTS, "false_alarm.csv"), rows)
    _write(os.path.join(RESULTS, "false_alarm_summary.csv"), summary)


def _write(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
