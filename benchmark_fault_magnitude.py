"""How small a parameter error each method can still see, and at what alpha.

The catalogue fixes one magnitude per fault, which answers whether a method
detects a fault of that size and nothing about where its floor lies. Two sweeps
here supply the missing axis.

The first sweeps the size of the error. A single physical constant is moved by a
relative amount from 0.1% to 30%, everything else held at specification, and
each method is asked for a verdict at every step. The smallest error a method
still rejects at alpha is its detection floor on that parameter, and floors are
what a practitioner choosing a check actually needs.

The second sweeps the decision threshold. Every count in the main comparison is
scored at one alpha, which fixes one point of an operating curve and hides the
rest of it. Sweeping alpha over the whole catalogue and its healthy controls
gives the true- and false-positive rates as a curve, so a method that detects
more only by rejecting more often is visible as such.

How the arms are produced. Both come from the analytic model of `envs.py`, the
reference at specification and the test with one constant moved, from identical
initial conditions so that the difference between them is the constant and not
the draw. Trajectories are integrated at dt = 0.005 with RK4 over 400 steps,
which is the resolved regime; the shipped step size is a fault in its own right
and is not used as a reference anywhere.

Writes Results/fault_magnitude.csv and Results/roc.csv.
"""

import argparse
import csv
import os

import numpy as np

import baselines
import benchmark_bug_localisation as e1
import bugs
import envs
import ideal_diff

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
os.makedirs(RESULTS, exist_ok=True)

ALPHA = 0.01
MAGNITUDES = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]

#: (system, parameter, specification value, the generator it must break)
SWEPT = [
    ("acrobot", "m2", envs.ACROBOT["m2"], "energy"),
    ("acrobot", "g", envs.ACROBOT["g"], "energy"),
    ("acrobot", "lc2", envs.ACROBOT["lc2"], "energy"),
    ("reacher", "l2", envs.REACHER["l2"], "fk-x"),
]


def arms(system, param, value, noise, seed=1):
    """Reference at specification and test with one constant moved, paired."""
    if system == "acrobot":
        ref = bugs.acrobot_trajectories(seed=0, ic_seed=0, noise=noise,
                                        **e1.ACRO_KW)
        test = bugs.acrobot_trajectories(params={param: value}, seed=seed,
                                         ic_seed=0, noise=noise, **e1.ACRO_KW)
    else:
        ref = bugs.reacher_samples(seed=0, ic_seed=0, noise=noise,
                                   **e1.REACH_KW)
        test = bugs.reacher_samples(params={param: value}, seed=seed,
                                    ic_seed=0, noise=noise, **e1.REACH_KW)
    return ref, test


def reference_set(system):
    if system == "acrobot":
        return bugs.acrobot_reference(), list(bugs.ACRO_SYMS)
    return bugs.reacher_reference(scale=envs.REACHER["l1"]), \
        list(bugs.REACH_SYMS)


def magnitude_sweep(noise, magnitudes, rows):
    print("=" * 118)
    print(f"Detection floor: smallest relative parameter error still rejected "
          f"at alpha = {ALPHA}, sigma = {noise:g}")
    print("=" * 118)
    print(f"{'parameter':<14}{'rel. error':>11}{'value':>10}"
          f"{'MMD':>6}{'KS':>6}{'AUC p':>9}{'screen':>8}   "
          f"{'localised':<22}{'residual ratio':>16}")
    print("-" * 118)

    for system, param, spec, target in SWEPT:
        (G, names, kinds), V = reference_set(system)
        for rel in magnitudes:
            value = spec * (1.0 + rel)
            ref, test = arms(system, param, value, noise)
            diff = ideal_diff.screen(G, ref, test, V, names=names, kinds=kinds,
                                     alpha=ALPHA, unit=ideal_diff.TRAJECTORY)
            Xr, Xt = np.vstack(ref), np.vstack(test)
            _, mmd_p = baselines.mmd_permutation_test(Xr, Xt, seed=0)
            _, ks_p = baselines.ks_bonferroni(Xr, Xt, seed=0)
            auc_v, disc_p = baselines.discriminator_test(Xr, Xt, seed=0)
            hit = {v.name: v for v in diff.verdicts}
            ratio = hit[target].ratio if target in hit else float("nan")

            print(f"{system + '.' + param:<14}{rel:>11.3%}{value:>10.4f}"
                  f"{'Y' if mmd_p < ALPHA else '.':>6}"
                  f"{'Y' if ks_p < ALPHA else '.':>6}"
                  f"{disc_p:>9.1e}{'Y' if diff.detected else '.':>8}   "
                  f"{', '.join(diff.localised)[:21]:<22}{ratio:>16.3e}")

            rows.append(dict(
                system=system, parameter=param, spec_value=spec,
                relative_error=rel, value=value, noise=noise,
                mmd_p=mmd_p, mmd_detect=bool(mmd_p < ALPHA),
                ks_p=ks_p, ks_detect=bool(ks_p < ALPHA),
                discriminator_auc=auc_v, discriminator_p=disc_p,
                discriminator_detect=bool(disc_p < ALPHA),
                screen_detect=diff.detected,
                screen_localised="|".join(diff.localised),
                screen_exact=(diff.localised == [target]),
                target_generator=target, target_ratio=ratio,
                target_p=hit[target].pvalue if target in hit else float("nan")))

    print("\n  Floors, as the smallest swept error each method still rejects")
    for system, param, spec, target in SWEPT:
        sel = [r for r in rows
               if r["system"] == system and r["parameter"] == param
               and r["noise"] == noise]
        line = f"    {system + '.' + param:<14}"
        for label, key in (("MMD", "mmd_detect"), ("KS", "ks_detect"),
                           ("discriminator", "discriminator_detect"),
                           ("screen", "screen_detect")):
            got = [r["relative_error"] for r in sel if r[key]]
            line += (f"{label} {min(got):.1%}   " if got
                     else f"{label} none   ")
        print(line)


#: Smallest p-value each method can return, whatever the effect size.
#:
#: A permutation test cannot report below 1/(n_perm+1), and the screen's rank
#: test cannot report below the resolution of 32-against-32 under complete
#: separation. Sweeping alpha past a method's own floor stops measuring its
#: power and starts measuring its budget, so the floors are recorded next to
#: the curve and any row below one of them is marked rather than read.
P_FLOOR = {
    "mmd": 1.0 / (baselines.MMD_N_PERM + 1),
    "ks": 0.0,                     # asymptotic tail, no permutation floor
    "discriminator": 0.0,          # asymptotic Mann-Whitney tail
}


def _wilson(k, n, z=1.96):
    """Wilson interval for a binomial rate, the honest form of 0 out of 2."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - h) / d), float((c + h) / d)


def roc(noise, rows, unit=ideal_diff.TRAJECTORY):
    """True- and false-positive rates over the whole catalogue, alpha swept.

    The screen is scored at the trajectory unit by default. The block unit does
    not hold its nominal level, as the calibration run measures, so a curve
    drawn at that unit reports an operating point the instrument does not
    actually offer.
    """
    cat = bugs.catalogue()
    print("\n" + "=" * 118)
    print(f"Operating curves over the catalogue, sigma = {noise:g}, "
          f"screen unit = {unit}")
    print("=" * 118)

    records = []
    ref_cache = {}
    for bug in cat:
        key = (bug.system, tuple(sorted(bug.ref_kwargs.items())))
        if key not in ref_cache:
            ref_cache[key] = e1.reference_for(bug.system, noise=noise,
                                              overrides=bug.ref_kwargs)
        ref = ref_cache[key]
        test = e1.test_for(bug, regime=e1.PAIRED, noise=noise)
        (G, names, kinds), V = reference_set(bug.system)
        # alpha enters the screen only as the cut on the corrected p-values, so
        # the smallest corrected p over the generators is the whole curve.
        diff = ideal_diff.screen(G, ref, test, V, names=names, kinds=kinds,
                                 alpha=1.0, unit=unit)
        p_alg = min(v.pvalue for v in diff.verdicts)
        Xr, Xt = np.vstack(ref), np.vstack(test)
        _, mmd_p = baselines.mmd_permutation_test(Xr, Xt, seed=0)
        _, ks_p = baselines.ks_bonferroni(Xr, Xt, seed=0)
        auc_v, disc_p = baselines.discriminator_test(Xr, Xt, seed=0)
        records.append(dict(fault=bug.name, category=bug.category,
                            is_fault=bug.category != "control",
                            screen_p=p_alg, mmd_p=mmd_p, ks_p=ks_p,
                            discriminator_p=disc_p, discriminator_auc=auc_v))

    alphas = [1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 0.5, 1.0]
    faults = [r for r in records if r["is_fault"]]
    ctrl = [r for r in records if not r["is_fault"]]
    screen_floor = min(r["screen_p"] for r in records)
    floors = dict(P_FLOOR, screen=screen_floor)
    print(f"  p-value floors: screen {screen_floor:.2g}, "
          f"MMD {floors['mmd']:.2g} (permutation budget "
          f"{baselines.MMD_N_PERM}); a rate below a method's own floor is "
          f"arithmetic, not power.")
    print(f"  false-positive rates rest on {len(ctrl)} controls, so a measured "
          f"0.00 has Wilson upper bound "
          f"{_wilson(0, len(ctrl))[1]:.2f}.")
    print(f"{'alpha':>10}" + "".join(f"{k:>22}" for k in
                                     ("screen", "MMD", "KS", "discriminator")))
    print(f"{'':>10}" + "".join(f"{'TPR / FPR':>22}" for _ in range(4)))
    print("-" * 118)
    for a in alphas:
        cells = []
        row = dict(alpha=a, noise=noise, screen_unit=unit,
                   n_faults=len(faults), n_controls=len(ctrl))
        for label, key in (("screen", "screen_p"), ("mmd", "mmd_p"),
                           ("ks", "ks_p"), ("discriminator",
                                            "discriminator_p")):
            tpr = sum(1 for r in faults if r[key] < a) / max(1, len(faults))
            fpr = sum(1 for r in ctrl if r[key] < a) / max(1, len(ctrl))
            # Below the method's own floor no p-value can fall, so the rate
            # there is a property of the budget and is flagged as such.
            censored = a <= floors.get(label, 0.0)
            cells.append(f"{tpr:.2f} / {fpr:.2f}" + ("*" if censored else " "))
            row[f"{label}_tpr"] = tpr
            row[f"{label}_fpr"] = fpr
            row[f"{label}_fpr_lo"], row[f"{label}_fpr_hi"] = _wilson(
                sum(1 for r in ctrl if r[key] < a), len(ctrl))
            row[f"{label}_censored"] = censored
        rows.append(row)
        print(f"{a:>10.0e}" + "".join(f"{c:>22}" for c in cells))
    print("  * alpha at or below that method's p-value floor; the rate is "
          "arithmetic.")
    # The plateau each method reaches, and the alpha it needs to reach it, so
    # that three plateaus attained at three different alpha are not read as a
    # comparison at one operating point.
    print(f"\n  {'method':<16}{'peak TPR':>10}{'at alpha':>12}"
          f"{'FPR there':>12}{'FPR 95% upper':>16}")
    for label, key in (("screen", "screen_p"), ("mmd", "mmd_p"),
                       ("ks", "ks_p"), ("discriminator", "discriminator_p")):
        best = max(
            (r for r in rows if r["noise"] == noise and not r[f"{label}_censored"]),
            key=lambda r: (r[f"{label}_tpr"], -r["alpha"]), default=None)
        if best is None:
            continue
        print(f"  {label:<16}{best[f'{label}_tpr']:>10.2f}"
              f"{best['alpha']:>12.0e}{best[f'{label}_fpr']:>12.2f}"
              f"{best[f'{label}_fpr_hi']:>16.2f}")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    mags = [0.003, 0.03, 0.3] if args.quick else MAGNITUDES
    noises = [0.0] if args.quick else [0.0, 1e-3]

    mrows, rrows = [], []
    for noise in noises:
        magnitude_sweep(noise, mags, mrows)
    for noise in noises:
        roc(noise, rrows)

    _write(os.path.join(RESULTS, "fault_magnitude.csv"), mrows)
    _write(os.path.join(RESULTS, "roc.csv"), rrows)


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
