"""What the baselines can do once they are given the same features.

The screen localises a fault because it tests a named generator at a time. The
statistical baselines, handed the raw 6- or 11-dimensional observation, cannot
name anything, but that comparison confounds two differences: named features
against none, and an exact rank test against a kernel or classifier test. This
separates them.

Each method is run twice. On the raw observation, which is the setting of the
main comparison. And on the residual vector, whose j-th coordinate is the
scale-relative residual of the j-th reference generator over a block, so the
features are the named constraints themselves. A method scored on the residual
vector can localise: the coordinate whose null is rejected names the generator,
exactly as the screen's per-generator p-value does.

Where the faults come from. The catalogue in `bugs.py` fixes the fault, the
generators it must break, and where applicable the parameter value to recover,
before anything is measured. Each entry is applied to the analytic model, and
the reference arm is the same model with the fault removed, so ground truth for
localisation is the declared generator list and not a judgement made after
seeing the output.

What the ablation is for. If a distributional test on the residual vector
localises as well as the exact screen does, then the localisation is carried by
the reference generating set, and what exactness and canonicity buy is the
ability to obtain and compare that set at all rather than the localising step
itself. Reporting which of the two holds is the point.

Writes Results/baseline_residuals.csv.
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
N_BLOCKS = 32


def residual_features(G, trajs, V, kinds, n_blocks=N_BLOCKS):
    """(n_blocks, n_generators) matrix of per-block scale-relative residuals.

    `block_residuals` returns (relative, absolute); the baselines are handed the
    relative column, which is the same feature the screen's rank test reads.
    """
    cols = [ideal_diff.block_residuals(g, trajs, V, k, n_blocks)[0]
            for g, k in zip(G, kinds)]
    n = min(len(c) for c in cols)
    return np.stack([c[:n] for c in cols], axis=1)


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cat = bugs.catalogue()
    if args.quick:
        cat = ([b for b in cat if b.category == "parameter"][:3]
               + [b for b in cat if b.category == "control"])

    print("=" * 118)
    print("Baselines on raw observations and on the residual vector, "
          f"sigma={args.noise:g}")
    print("=" * 118)
    print(f"{'fault':<26}{'expected':<24}"
          f"{'KS-resid localised':<26}{'J':>5}"
          f"{'MMD':>6}{'AUC':>7}{'screen J':>10}")
    print("-" * 118)

    ref_cache, rows = {}, []
    for bug in cat:
        key = (bug.system, tuple(sorted(bug.ref_kwargs.items())))
        if key not in ref_cache:
            ref_cache[key] = e1.reference_for(bug.system, noise=args.noise,
                                              overrides=bug.ref_kwargs)
        ref = ref_cache[key]
        test = e1.test_for(bug, regime=e1.PAIRED, noise=args.noise)
        V = e1.ACRO_V if bug.system == "acrobot" else e1.REACH_V
        G, names, kinds = (bugs.acrobot_reference() if bug.system == "acrobot"
                           else bugs.reacher_reference(scale=envs.REACHER["l1"]))

        Fr = residual_features(G, ref, V, kinds)
        Ft = residual_features(G, test, V, kinds)

        # Localisation from a two-sided KS test per residual coordinate,
        # Bonferroni-corrected over the same family the screen corrects over.
        #
        # Argument order is (reference, test) here and in every other driver.
        # The subsample and permutation draws depend on it, so calling the same
        # baseline the other way round reseeds it and flips a verdict sitting
        # near alpha. Two drivers disagreeing on the order would move a
        # baseline's count by one against the row it is the contrast for.
        ks = baselines.ks_per_dimension(Fr, Ft)
        ks_local = [n for n, (p, _) in zip(names, ks) if p < ALPHA]

        _, mmd_p = baselines.mmd_permutation_test(Fr, Ft, n_sub=N_BLOCKS,
                                                  n_perm=2000, seed=0)
        auc_r, disc_p = baselines.discriminator_test(Fr, Ft, n_sub=N_BLOCKS,
                                                     epochs=200, seed=0)

        # The same three methods on the raw observation, for the contrast.
        Xr, Xt = np.vstack(ref), np.vstack(test)
        _, raw_mmd_p = baselines.mmd_permutation_test(Xr, Xt, seed=0)
        raw_ks_d, raw_ks_p = baselines.ks_bonferroni(Xr, Xt, seed=0)
        raw_auc, raw_disc_p = baselines.discriminator_test(Xr, Xt, seed=0)

        screen = ideal_diff.screen(G, ref, test, V, names=names, kinds=kinds,
                                   alpha=ALPHA, unit=ideal_diff.BLOCK)

        j_ks = jaccard(ks_local, bug.expect_broken)
        j_alg = jaccard(screen.localised, bug.expect_broken)
        print(f"{bug.name:<26}{'|'.join(bug.expect_broken)[:23]:<24}"
              f"{'|'.join(ks_local)[:25]:<26}{j_ks:>5.2f}"
              f"{'Y' if mmd_p < ALPHA else '.':>6}"
              f"{auc_r:>7.3f}{j_alg:>10.2f}")

        rows.append(dict(
            fault=bug.name, system=bug.system, category=bug.category,
            noise=args.noise, expected_broken="|".join(bug.expect_broken),
            ks_resid_localised="|".join(ks_local),
            ks_resid_jaccard=j_ks,
            ks_resid_exact=set(ks_local) == set(bug.expect_broken),
            mmd_resid_p=mmd_p, mmd_resid_detect=bool(mmd_p < ALPHA),
            disc_resid_auc=auc_r, disc_resid_p=disc_p,
            disc_resid_detect=bool(disc_p < ALPHA),
            raw_mmd_p=raw_mmd_p, raw_mmd_detect=bool(raw_mmd_p < ALPHA),
            raw_ks_stat=raw_ks_d, raw_ks_p=raw_ks_p,
            raw_ks_detect=bool(raw_ks_p < ALPHA),
            raw_disc_auc=raw_auc, raw_disc_p=raw_disc_p,
            raw_disc_detect=bool(raw_disc_p < ALPHA),
            screen_localised="|".join(screen.localised),
            screen_jaccard=j_alg,
            screen_exact=set(screen.localised) == set(bug.expect_broken),
            **{f"ks_resid_p_{n.replace(' ', '_')}": p
               for n, (p, _) in zip(names, ks)},
            **{f"screen_p_{v.name.replace(' ', '_')}": v.pvalue
               for v in screen.verdicts}))

    faults = [r for r in rows if r["category"] != "control"]
    ctrl = [r for r in rows if r["category"] == "control"]
    print("\n" + "=" * 118)
    print("Summary")
    print("=" * 118)
    for label, ex, det, fp in [
            ("KS on residuals", "ks_resid_exact", "ks_resid_localised",
             "ks_resid_localised"),
            ("algebraic screen", "screen_exact", "screen_localised",
             "screen_localised")]:
        n_ex = sum(1 for r in faults if r[ex])
        n_det = sum(1 for r in faults if r[det])
        n_fp = sum(1 for r in ctrl if r[fp])
        print(f"  {label:<20} detected {n_det}/{len(faults)}, "
              f"localised exactly {n_ex}/{len(faults)}, "
              f"{n_fp}/{len(ctrl)} false alarms on healthy controls")
    for label, key in [("MMD on residuals", "mmd_resid_detect"),
                       ("discriminator on residuals", "disc_resid_detect"),
                       ("MMD on raw states", "raw_mmd_detect"),
                       ("KS on raw states", "raw_ks_detect"),
                       ("discriminator on raw states", "raw_disc_detect")]:
        n_det = sum(1 for r in faults if r[key])
        n_fp = sum(1 for r in ctrl if r[key])
        print(f"  {label:<28} detected {n_det}/{len(faults)}, "
              f"{n_fp}/{len(ctrl)} false alarms   (localises nothing)")

    out = os.path.join(RESULTS, "baseline_residuals.csv")
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
