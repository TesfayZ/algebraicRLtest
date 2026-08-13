"""E1. Algebraic localisation and attribution of physics faults.

The experiment the paper turns on. Each fault in `bugs.catalogue()` is applied
to an environment whose exact invariants are known, and three questions are
asked of every method:

  detect      does the method say anything is wrong?
  localise    does it name the constraint that broke?
  attribute   does it recover the numerical value of the responsible parameter?

Statistical two-sample tests answer the first and are structurally incapable of
the other two, since their output is a scalar. The algebraic diagnostic answers
all three, and where it cannot it says which of the three verdicts applies
rather than guessing.

Writes Results/bug_localisation.csv and Results/bug_attribution.csv.
"""

import os
import csv
import time

import numpy as np
import sympy as sp

import bugs
import baselines
import envs
import ideal_diff
import recover

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
os.makedirs(RESULTS, exist_ok=True)

ACRO_V = list(bugs.ACRO_SYMS)
REACH_V = list(bugs.REACH_SYMS)

# The reference system is sampled at a resolved timestep. Remark 6 of the paper
# is why: at the shipped dt = 0.2 the integrator's own drift sits above the
# signal any conserved-quantity diagnostic would have to read, so a diagnostic
# run there measures the integrator and not the physics. `acrobot_dt_0.2` is in
# the catalogue precisely so that this shows up as a finding rather than as an
# unexamined choice of experimental setting.
ACRO_KW = dict(dt=0.005, n_traj=8, n_steps=400)
REACH_KW = dict(N=4000, scale=envs.REACHER["l1"])
ALPHA = 0.01                 # significance level for the two-sample tests

# Two regimes, because they ask different questions of the baselines.
#
# unpaired: reference and test are logged independently, which is the situation
#   anyone comparing two simulator releases or a model against its environment
#   is actually in. Nothing forces the two to visit the same states.
# paired: both start from identical initial conditions, so the only difference
#   is the fault. This is the regime the two-sample tests are entitled to, and
#   reporting it alongside the unpaired regime shows whether a baseline's
#   count depends on which one it was given.
UNPAIRED, PAIRED = "unpaired", "paired"

#: The healthy control of each system, used as the paired baseline's null.
_CONTROL = {}


def reference_for(system, noise=0.0, ic_seed=0, overrides=None):
    """The healthy arm a fault is screened against.

    `overrides` carries the settings a fault declares in `ref_kwargs`, which is
    how a fault in the sampling gets a reference on its own grid instead of one
    the residual can tell apart for reasons that have nothing to do with the
    fault.
    """
    base = ACRO_KW if system == "acrobot" else REACH_KW
    kw = dict(base)
    kw.update(overrides or {})
    if system == "acrobot":
        return bugs.acrobot_trajectories(seed=0, ic_seed=ic_seed, noise=noise,
                                         **kw)
    return bugs.reacher_samples(seed=0, ic_seed=ic_seed, noise=noise, **kw)


def test_for(bug, regime=UNPAIRED, seed=1, noise=0.0, overrides=None):
    base = ACRO_KW if bug.system == "acrobot" else REACH_KW
    kw = {k: v for k, v in base.items() if k not in bug.kwargs}
    kw.update({k: v for k, v in (overrides or {}).items()
               if k not in bug.kwargs})
    kw["noise"] = noise
    # Under pairing the initial states are drawn from the reference's stream;
    # the observation-noise stream stays independent, so a paired healthy
    # control under noise is two noisy views of the same physics.
    #
    # At sigma = 0 that leaves nothing to differ by, and a paired healthy
    # control is then the reference array compared with itself. That is
    # degenerate rather than wrong: it tests only that a method does not fire on
    # identical input, which is worth checking and is not a false-alarm rate.
    # `is_degenerate_control` marks those rows so the distinction is in the CSV
    # and not left for a reader to notice.
    kw["ic_seed"] = 0 if regime == PAIRED else seed
    return bug.trajectories(seed=seed, **kw)


def is_degenerate_control(bug, regime, noise):
    """Is this comparison a healthy arm against a bit-identical copy of itself?"""
    return bug.category == "control" and regime == PAIRED and noise == 0.0


def healthy_pair_for(bug, regime, noise):
    """A healthy (reference, test) pair on the same settings as `bug`'s.

    The paired baseline needs a null of its own, exactly as the screen compares
    test residuals against reference residuals. Built from the control of the
    same system so that the null is drawn under the settings the fault is
    screened at.

    `ref_kwargs` reaches both arms of the null. The baseline's statistic is a
    block mean over a window of trajectory, so it moves with the sampling grid
    on its own; a null drawn on the default grid while the statistic is computed
    on a decimated one compares two windows spanning different elapsed times,
    which is the confound `ref_kwargs` exists to remove.
    """
    healthy = _CONTROL[bug.system]
    return (reference_for(bug.system, noise=noise, overrides=bug.ref_kwargs),
            test_for(healthy, regime=regime, noise=noise,
                     overrides=bug.ref_kwargs))


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# --------------------------------------------------------------- level 2 ---

#: Difference lag for the conserved-quantity recovery. A conserved quantity has
#: zero difference at any lag while everything else grows with elapsed time, so
#: the lag buys separation against measurement noise at no cost in bias. Fifty
#: steps at dt = 0.005 is a quarter of a second, short against the secular drift
#: of RK4 at that step size and long against the noise.
LAG = 50
GAP = 0.1


def attribute(bug, test_trajs, lag=LAG, gap=GAP):
    """Run the discovery-and-fit half of the diagnostic for one fault."""
    if bug.system == "acrobot":
        G_known, _, _ = bugs.acrobot_reference()
        poly, s_min, s_next = recover.recover_conserved_quotient(
            test_trajs, ACRO_V, G_known[:2], degree=3, gap=gap, lag=lag)
        verdict, hyps = ideal_diff.diagnose(
            bugs.acrobot_energy_template(), bugs.acrobot_spec(), poly,
            ACRO_V, allow_offset=True)
        return verdict, hyps, s_min, s_next

    # Reacher's invariants are vanishing rather than conserved, and the fault
    # lives in the forward-kinematics identity, so the search is restricted to
    # the six variables that identity touches, chosen directly rather than
    # swept.
    scale = envs.REACHER["l1"]
    G_ref, names, _ = bugs.reacher_reference(scale=scale)
    sub = [bugs.RC1, bugs.RC2, bugs.RS1, bugs.RS2, bugs.RTX, bugs.RDX]
    cols = [REACH_V.index(v) for v in sub]
    data = test_trajs[0][:, cols]
    poly, s_min, s_next = recover.recover_vanishing_quotient(
        data, sub, [sub[0]**2 + sub[2]**2 - 1, sub[1]**2 + sub[3]**2 - 1],
        degree=2, gap=gap)
    verdict, hyps = ideal_diff.diagnose(
        bugs.reacher_fk_template("x"), bugs.reacher_spec(scale=scale), poly,
        sub, allow_offset=False)
    return verdict, hyps, s_min, s_next




# ------------------------------------------------------------------ main ---

def run_detection(cat, regime, noise, rows, run_baselines=True):
    """Detection and localisation for every fault, in one regime."""
    print(f"\n{'=' * 112}")
    print(f"Detection and localisation   regime={regime}   "
          f"observation noise sigma={noise:g}")
    print("=" * 112)
    print(f"{'fault':<26}{'category':<12}{'MMD':>6}{'KS':>6}{'AUC':>7}"
          f"{'pair':>6}{'alg':>6}   {'localised':<28}{'J':>5}{'max ratio':>11}")
    print("-" * 112)

    ref_cache = {}
    for bug in cat:
        key = (bug.system, tuple(sorted(bug.ref_kwargs.items())))
        if key not in ref_cache:
            ref_cache[key] = reference_for(bug.system, noise=noise,
                                           overrides=bug.ref_kwargs)
        ref = ref_cache[key]
        test = test_for(bug, regime=regime, noise=noise)
        V = ACRO_V if bug.system == "acrobot" else REACH_V
        G_ref, names, kinds = (
            bugs.acrobot_reference() if bug.system == "acrobot"
            else bugs.reacher_reference(scale=envs.REACHER["l1"]))

        t0 = time.time()
        diff = ideal_diff.screen(G_ref, ref, test, V, names=names, kinds=kinds,
                                 alpha=ALPHA, unit=ideal_diff.TRAJECTORY)
        t_alg = time.time() - t0
        # The same screen over blocks rather than trajectories, recorded
        # alongside. Blocks cut from one trajectory share an initial
        # condition, so the block-level p-value counts more independent
        # replicates than the design supplies and does not hold its nominal
        # level (Section~\ref{sec:limitations}); the trajectory-level test
        # above is the one the stated alpha is supported at and is what the
        # headline columns report.
        block_diff = ideal_diff.screen(G_ref, ref, test, V, names=names,
                                       kinds=kinds, alpha=ALPHA,
                                       unit=ideal_diff.BLOCK)

        if run_baselines:
            # Argument order is (reference, test) for every baseline and in
            # every driver. The subsample and permutation draws depend on it,
            # so calling one baseline (test, reference) elsewhere reseeds it and
            # moves verdicts that sit near alpha.
            Xr, Xt = np.vstack(ref), np.vstack(test)
            t0 = time.time()
            mmd_stat, mmd_p = baselines.mmd_permutation_test(Xr, Xt, seed=0)
            ks_d, ks_p = baselines.ks_bonferroni(Xr, Xt, seed=0)
            d_auc, d_p = baselines.discriminator_test(Xr, Xt, seed=0)
            if regime == PAIRED:
                r0, t0_ = healthy_pair_for(bug, regime, noise)
                pair_stat, pair_p, _ = baselines.paired_difference_test(
                    Xr, Xt, np.vstack(r0), np.vstack(t0_))
            else:
                pair_stat, pair_p = float("nan"), float("nan")
            t_stat = time.time() - t0
        else:
            mmd_stat = mmd_p = ks_d = ks_p = d_auc = d_p = float("nan")
            pair_stat = pair_p = float("nan")
            t_stat = 0.0

        mmd_hit = bool(mmd_p < ALPHA)
        ks_hit = bool(ks_p < ALPHA)
        auc_hit = bool(d_p < ALPHA)
        pair_hit = bool(pair_p < ALPHA)
        j = jaccard(diff.localised, bug.expect_broken)

        worst = max((v.ratio for v in diff.verdicts), default=float("nan"))
        print(f"{bug.name:<26}{bug.category:<12}"
              f"{'Y' if mmd_hit else '.':>6}{'Y' if ks_hit else '.':>6}"
              f"{d_auc:>7.3f}"
              f"{('Y' if pair_hit else '.') if regime == PAIRED else '-':>6}"
              f"{'Y' if diff.detected else '.':>6}   "
              f"{', '.join(diff.localised)[:27]:<28}{j:>5.2f}"
              f"{worst:>11.2e}")

        rows.append(dict(
            regime=regime, noise=noise,
            fault=bug.name, system=bug.system, category=bug.category,
            description=bug.description,
            expected_broken="|".join(bug.expect_broken),
            degenerate_control=is_degenerate_control(bug, regime, noise),
            mmd_stat=mmd_stat, mmd_p=mmd_p, mmd_detect=mmd_hit,
            ks_stat=ks_d, ks_p=ks_p, ks_detect=ks_hit,
            discriminator_auc=d_auc, discriminator_p=d_p,
            discriminator_detect=auc_hit,
            paired_ratio=pair_stat, paired_p=pair_p, paired_detect=pair_hit,
            algebraic_detect=diff.detected,
            algebraic_localised="|".join(diff.localised),
            localisation_jaccard=j,
            localisation_exact=set(diff.localised) == set(bug.expect_broken),
            block_detect=block_diff.detected,
            block_localised="|".join(block_diff.localised),
            block_localisation_exact=(set(block_diff.localised)
                                      == set(bug.expect_broken)),
            seconds_algebraic=t_alg, seconds_statistical=t_stat,
            **{f"resid_{v.name.replace(' ', '_')}": v.residual_test
               for v in diff.verdicts},
            **{f"p_{v.name.replace(' ', '_')}": v.pvalue
               for v in diff.verdicts},
            # Every p-value under complete separation is the rank test's
            # resolution floor and is therefore the same number for a 30% mass
            # error as for a 0.1% gravity error. The residual ratio is where the
            # size of the violation shows up.
            **{f"ratio_{v.name.replace(' ', '_')}": v.ratio
               for v in diff.verdicts},
            **{f"blockp_{v.name.replace(' ', '_')}": v.pvalue
               for v in block_diff.verdicts}))


def summarise_detection(rows, regime, noise):
    sel = [r for r in rows if r["regime"] == regime and r["noise"] == noise]
    faults = [r for r in sel if r["category"] != "control"]
    ctrl = [r for r in sel if r["category"] == "control"]
    degenerate = any(r["degenerate_control"] for r in ctrl)
    print(f"\n  summary, regime={regime}, sigma={noise:g}")
    methods = [("MMD permutation", "mmd_detect"),
               ("KS + Bonferroni", "ks_detect"),
               ("MLP discriminator", "discriminator_detect")]
    if regime == PAIRED:
        methods.append(("paired difference", "paired_detect"))
    methods.append(("algebraic screen", "algebraic_detect"))
    for label, key in methods:
        tp = sum(1 for r in faults if r[key])
        fp = sum(1 for r in ctrl if r[key])
        print(f"    {label:<20} {tp}/{len(faults)} faults detected, "
              f"{fp}/{len(ctrl)} {'identity checks passed' if degenerate else 'false alarms'}")
    if degenerate:
        print("    note: at sigma = 0 the paired controls are the reference "
              "arm against a\n          bit-identical copy of itself, so that "
              "column is an identity check and\n          not a false-alarm "
              "rate. The measured rate is in the calibration run.")
    exact = sum(1 for r in faults if r["localisation_exact"])
    mj = np.mean([r["localisation_jaccard"] for r in faults])
    print(f"    {'algebraic screen':<20} localised exactly {exact}/"
          f"{len(faults)}, mean Jaccard {mj:.2f}   (no baseline localises)")
    tp = sum(1 for r in faults if r["block_detect"])
    fp = sum(1 for r in ctrl if r["block_detect"])
    ex = sum(1 for r in faults if r["block_localisation_exact"])
    print(f"    {'  per block':<20} {tp}/{len(faults)} detected, "
          f"{fp}/{len(ctrl)} false alarms, localised exactly {ex}/{len(faults)}")


def run_attribution(cat, arows, noise=0.0, regime=UNPAIRED, verbose=True):
    if verbose:
        print(f"\n{'=' * 112}")
        print(f"Level 2: which parameter, and what value?   "
              f"regime={regime}   sigma={noise:g}")
        print("=" * 112)
        print(f"{'fault':<26}{'verdict':<36}{'top':<7}{'fitted':>10}"
              f"{'expected':>10}{'rel.err':>10}")
        print("-" * 112)

    for bug in cat:
        test = test_for(bug, regime=regime, noise=noise)
        t0 = time.time()
        verdict, hyps, s_min, s_next = attribute(bug, test)
        dt = time.time() - t0
        top = hyps[0] if hyps else None
        correct = bool(verdict == ideal_diff.PARAMETER_FAULT and top is not None
                       and top.param == bug.expect_param)
        rel_err = (abs(top.value - bug.expect_value) / abs(bug.expect_value)
                   if correct and bug.expect_value else float("nan"))
        if verbose:
            print(f"{bug.name:<26}{verdict:<36}"
                  f"{(top.param if top else '-'):<7}"
                  f"{(f'{top.value:.4f}' if top else '-'):>10}"
                  f"{(f'{bug.expect_value:.4f}' if bug.expect_value else '-'):>10}"
                  f"{(f'{rel_err:.2e}' if correct else '-'):>10}")

        arows.append(dict(
            noise=noise, regime=regime, lag=LAG, gap=GAP,
            fault=bug.name, system=bug.system, category=bug.category,
            attributable=bug.attributable,
            verdict=verdict, s_min=s_min, s_next=s_next,
            gap_ratio=(s_min / s_next if np.isfinite(s_next) and s_next > 0
                       else float("nan")),
            expected_param=bug.expect_param or "",
            expected_value=bug.expect_value if bug.expect_value else "",
            top_param=top.param if top else "",
            top_value=top.value if top else "",
            top_residual=top.residual if top else "",
            runner_up_residual=hyps[1].residual if len(hyps) > 1 else "",
            attribution_correct=correct, relative_error=rel_err,
            seconds=dt))


def summarise_attribution(arows, noise, regime=UNPAIRED):
    sel = [a for a in arows if a["noise"] == noise and a["regime"] == regime]
    par = [a for a in sel if a["category"] == "parameter"]
    ok = sum(1 for a in par if a["attribution_correct"])
    errs = [a["relative_error"] for a in par if a["attribution_correct"]]
    print(f"\n  summary, regime={regime}, sigma={noise:g}")
    print(f"    correct parameter named on {ok}/{len(par)} parameter faults"
          + (f", worst relative error {max(errs):.1e}" if errs else ""))
    # What matters for a non-parameter fault is that no parameter is *named*.
    # Both remaining verdicts do that, and distinguishing them here would be
    # scoring a distinction the user of the diagnostic does not act on.
    other = [a for a in sel if a["category"] in ("structural", "numerical")
             or (a["category"] == "observation" and a["attributable"])]
    refused = sum(1 for a in other
                  if a["verdict"] != ideal_diff.PARAMETER_FAULT)
    print(f"    named no parameter on {refused}/{len(other)} faults that no "
          f"parameter can explain")
    ctrl = [a for a in sel if a["category"] == "control"]
    clean = sum(1 for a in ctrl if a["verdict"] != ideal_diff.PARAMETER_FAULT)
    print(f"    reported no parameter fault on {clean}/{len(ctrl)} healthy "
          f"controls")


def attribution_noise_sweep(cat, srows):
    """Where does attribution stop working, and is one tolerance enough?

    Attribution reads coefficients off a numerical nullspace direction, so it
    inherits that direction's conditioning. The sweep is over both the
    measurement noise and the difference lag because those are the two
    quantities that set the spectral gap the recovery reads. Reporting the pair
    rather than a single ceiling is what makes the usable window visible, and
    the window is the honest form of the result.
    """
    par = [b for b in cat if b.category == "parameter"]
    sigmas = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    lags = [1, 10, 50, 100]

    print(f"\n{'=' * 112}")
    print("Attribution under measurement noise: correct parameter named and "
          f"within 5%, out of {len(par)} parameter faults")
    print("=" * 112)
    print(f"{'sigma':>10} | " + "".join(f"{'lag ' + str(l):>10}" for l in lags)
          + f"{'best':>8}")
    print("-" * 112)

    for sigma in sigmas:
        cells = []
        for lag in lags:
            n_ok = 0
            # The ceiling is set jointly by the spectral gap and the fit, so the
            # sweep records both for one reference fault rather than only the
            # count. Without them the paper's discussion of how much of the
            # ceiling is `gap` and how much is the fit rests on nothing written
            # down.
            probe_gap, probe_resid = float("nan"), float("nan")
            for bug in par:
                test = test_for(bug, regime=PAIRED, noise=sigma)
                # Reacher's invariants are kinematic and carry no lag; only the
                # conserved-quantity recovery on Acrobot is affected.
                verdict, hyps, s_min, s_next = attribute(bug, test, lag=lag)
                ok = (verdict == ideal_diff.PARAMETER_FAULT and hyps
                      and hyps[0].param == bug.expect_param
                      and abs(hyps[0].value - bug.expect_value)
                      <= 0.05 * abs(bug.expect_value))
                n_ok += int(bool(ok))
                if bug.expect_param == "m2":
                    if np.isfinite(s_next) and s_next > 0:
                        probe_gap = s_min / s_next
                    if hyps:
                        probe_resid = hyps[0].residual
            cells.append(n_ok)
            srows.append(dict(sigma=sigma, lag=lag, n_correct=n_ok,
                              n_faults=len(par), probe_fault="acrobot_m2_1.3",
                              probe_gap_ratio=probe_gap,
                              probe_fit_residual=probe_resid))
        print(f"{sigma:>10.0e} | " + "".join(f"{c:>10d}" for c in cells)
              + f"{max(cells):>8d}")

    print("\n  Attribution is a clean-data instrument. It reads coefficients "
          "off a\n  singular direction, so it inherits that direction's "
          "conditioning, and the\n  lag buys about a decade of noise tolerance "
          "by growing every non-conserved\n  direction while leaving the "
          "conserved one at zero. Level-1 screening, which\n  needs only a "
          "residual, degrades far more gracefully over the same range.")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the noise sweep")
    args = ap.parse_args()

    cat = bugs.catalogue()
    _CONTROL.update({b.system: b for b in cat if b.category == "control"})
    rows, arows = [], []

    print("=" * 112)
    print("E1. Algebraic localisation and attribution of injected physics "
          "faults")
    print("=" * 112)

    # Both regimes at zero noise, then the paired regime under measurement
    # noise, which is the setting closest to logged hardware data.
    settings = [(UNPAIRED, 0.0), (PAIRED, 0.0)]
    if not args.quick:
        settings.append((PAIRED, 1e-3))

    for regime, noise in settings:
        run_detection(cat, regime, noise, rows)
        summarise_detection(rows, regime, noise)

    # Attribution is run at every setting the detection table reports, so that
    # each of its cells is read from this CSV. Running it once and carrying the
    # count across regimes and noise levels would leave the cells that were not
    # measured indistinguishable from the one that was.
    for regime, noise in settings:
        run_attribution(cat, arows, noise=noise, regime=regime)
        summarise_attribution(arows, noise, regime)

    srows = []
    if not args.quick:
        attribution_noise_sweep(cat, srows)

    _write(os.path.join(RESULTS, "bug_localisation.csv"), rows)
    _write(os.path.join(RESULTS, "bug_attribution.csv"), arows)
    if srows:
        _write(os.path.join(RESULTS, "bug_attribution_noise.csv"), srows)


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