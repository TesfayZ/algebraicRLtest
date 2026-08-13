"""E3. Reward shaping from a discovered conserved quantity.

The invariant is not assumed here, it is discovered: `recover_conserved_quotient`
is run on finely integrated passive trajectories, and the polynomial that comes
back is what defines the potential. That detour matters, because Acrobot ships
with dt = 0.2 and at that step size the integrator's own drift sits above the
signal, so the energy is not recoverable from the environment as distributed
(Remark 6). Discovery is therefore run on a finely re-integrated copy and the
resulting potential is deployed on the shipped environment. That split is a real
constraint on the pipeline rather than a convenience, and it is reported as one.

Four conditions, chosen so that each rules out a specific alternative
explanation for any improvement:

  none        PPO on the environment reward. The baseline.
  discovered  the potential from the recovered energy.
  faulty      the potential from an energy recovered off a system with a known
              parameter fault. This is the link back to the diagnostic: if a
              wrong invariant shapes as well as a right one, the diagnostic
              buys nothing downstream.
  random      a degree-3 polynomial with no physical meaning, matched in scale.
  rnd         Random Network Distillation, a bonus that is not potential-based
              and carries no policy-invariance guarantee.

The faulty condition is gauged entirely within the faulty model. Recovery
returns a direction, so a scale and an offset have to be fixed before
|p(s) - E*| means anything, and the reference used to fix them is the faulty
model's own analytic energy; the target E* is the upright pose evaluated in the
faulty model as well. A deployer holding a wrong model has nothing else to
consult, so anything correct entering at that step would be a comparison the
setting does not offer. The faulty condition therefore carries the parameter
error in its coefficients, in its scale, in its offset and in its target.

The size of the parameter error is swept, because "wrong physics shapes fine"
generalises from one point only as far as the point where it stops being true,
and finding that point is more informative than confirming the one.

Returns reported are always unshaped environment returns.

Writes Results/shaping.csv.
"""

import os
import csv
import time

import numpy as np
import sympy as sp

import bugs
import envs
import ideal_diff
import recover
import shaping as sh

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
os.makedirs(RESULTS, exist_ok=True)

V = list(bugs.ACRO_SYMS)
G_REF, _, _ = bugs.acrobot_reference()
DEFAULT_RANDOM_DRAWS = 10

def energy_target(params=None):
    """Energy of the fully extended upright pose, the goal the potential aims at.

    Upright is theta1 = pi with the second link in line, so c1 = -1, c2 = 1 and
    both velocities are zero. Evaluating the potential energy there gives
    (m1 lc1 + m2 l1) g + m2 g lc2, which is what an engineer hand-designing
    energy pumping would also choose.

    The argument is the model the target is computed from, and that is the whole
    point of the function: a potential deployed from a faulty model must take
    its target from that model too, because the deployer does not have the
    correct one to consult.
    """
    p = dict(envs.ACROBOT)
    p.update(params or {})
    return ((p["m1"] * p["lc1"] + p["m2"] * p["l1"]) * p["g"]
            + p["m2"] * p["g"] * p["lc2"])


E_TARGET = energy_target()


def analytic_energy(params=None):
    """The energy polynomial of a given model, as the reference for its gauge."""
    p = dict(envs.ACROBOT)
    p.update(params or {})
    return sp.nsimplify(
        sp.expand(bugs.acrobot_energy_template().subs(bugs.acrobot_spec(p))),
        rational=True)


def discover_potential(params=None, seed=0):
    """Run the discovery pipeline and return (potential_poly, wall_seconds).

    Passive trajectories at a resolved timestep, exactly as Section 4 requires.
    """
    t0 = time.time()
    trajs = bugs.acrobot_trajectories(params=params, dt=0.005, n_traj=8,
                                      n_steps=400, seed=seed, ic_seed=0)
    poly, s_min, s_next = recover.recover_conserved_quotient(
        trajs, V, G_REF[:2], degree=3, gap=0.1, lag=50)
    return poly, time.time() - t0, s_min, s_next


def rescale_to_energy(poly, reference, variables, n=4000, seed=0):
    """Fix the gauge of a recovered conserved quantity against a reference.

    Recovery returns a direction, so the polynomial arrives with an arbitrary
    scale and offset. Neither changes what is conserved, but both change what
    |p(s) - E*| means, so the potential is meaningless until the gauge is
    pinned. We regress the recovered polynomial onto the reference energy over
    sampled states, which fixes exactly the two degrees of freedom that
    recovery cannot see.

    Note that the sampled cosines and sines are drawn from independent angles,
    so the points do not lie on c^2 + s^2 = 1. That is safe only because the
    recovered polynomial and the reference agree as coefficient vectors and not
    merely on the variety: `coefficient_error` measures 2.8e-8 between them. Two
    representatives of the same class modulo <c1^2+s1^2-1, c2^2+s2^2-1> agree on
    the variety and differ off it, so a gauge fitted off the variety would be
    biased if recovery ever returned a different representative. The assertion
    to check, if that changes, is the coefficient error and not the fit residual.
    """
    rng = np.random.default_rng(seed)
    obs = np.stack([
        np.cos(rng.uniform(-np.pi, np.pi, n)),
        np.sin(rng.uniform(-np.pi, np.pi, n)),
        np.cos(rng.uniform(-np.pi, np.pi, n)),
        np.sin(rng.uniform(-np.pi, np.pi, n)),
        rng.uniform(-4, 4, n), rng.uniform(-9, 9, n)], axis=-1)
    p = ideal_diff.eval_poly(poly, obs, variables)
    e = ideal_diff.eval_poly(reference, obs, variables)
    Aa = np.stack([p, np.ones_like(p)], axis=1)
    coef, *_ = np.linalg.lstsq(Aa, e, rcond=None)
    return sp.expand(float(coef[0]) * poly + float(coef[1]))


def coefficient_error(recovered, reference, variables):
    """Relative L2 distance between two polynomials' coefficient vectors."""
    import deflation
    basis = deflation.monomials(list(variables), 3, min_degree=0)
    a = deflation.coeff_vector(sp.expand(recovered), basis, list(variables))
    b = deflation.coeff_vector(sp.expand(reference), basis, list(variables))
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


_ONPOLICY = {}


def sample_states(n=20_000, seed=0):
    """States the gauge is fitted over: what a policy on Acrobot-v1 actually visits.

    A box of independently drawn angles and velocities is the obvious sample and
    the wrong one for this purpose. Acrobot-v1 admits velocities out to 4pi and
    9pi where such a box would have to be told the range, and a degree-3
    potential extrapolates as a cube, so two potentials matched in spread on the
    box separate off it. Rolling out a uniform-random policy on the shipped
    environment samples the region a run starts in, which is where a dense bonus
    either traps the policy or does not.

    Cached, so every gauge and every reported spread sees one sample.
    """
    key = (n, seed)
    if key not in _ONPOLICY:
        import gymnasium as gym
        env = gym.make("Acrobot-v1")
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        out = np.empty((n, 6))
        for i in range(n):
            out[i] = obs
            obs, _, term, trunc, _ = env.step(int(rng.integers(3)))
            if term or trunc:
                obs, _ = env.reset()
        env.close()
        _ONPOLICY[key] = out
    return _ONPOLICY[key]


def rescale_to_moments(poly, reference, variables, n=20_000, seed=0):
    """Match a polynomial's mean and spread to a reference's, over sampled states.

    This is the gauge for a polynomial that is *not* a representative of the
    reference's class, which is the random control's whole point.
    `rescale_to_energy` regresses one onto the other, which is right when the
    two agree as coefficient vectors and wrong here: for a polynomial
    uncorrelated with the energy the least-squares slope is driven by a
    correlation that does not exist, so it collapses towards zero and returns a
    near-constant potential. A near-constant potential has no shaping signal at
    all, so a control gauged that way cannot distinguish "an unphysical
    potential does not help" from "a potential two orders too small does not
    help", which is the distinction the control exists to make.

    Matching the first two moments instead fixes the same two degrees of
    freedom, scale and offset, and delivers what "matched in scale" claims: a
    potential whose spread over the visited states equals the energy's.
    """
    obs = sample_states(n=n, seed=seed)
    p = ideal_diff.eval_poly(poly, obs, variables)
    e = ideal_diff.eval_poly(reference, obs, variables)
    sd_p = float(np.std(p))
    if sd_p == 0.0:
        raise ValueError("degenerate random polynomial: zero spread")
    a = float(np.std(e)) / sd_p
    b = float(np.mean(e)) - a * float(np.mean(p))
    return sp.expand(a * poly + b)


def potential_spread(poly, variables, n=20_000, seed=0):
    """Standard deviation of a potential over the visited states.

    Reported per condition so that "matched in scale" is a number in the CSV
    and not an adjective in a docstring.
    """
    obs = sample_states(n=n, seed=seed)
    return float(np.std(ideal_diff.eval_poly(poly, obs, variables)))


def increment_spread(phi, gamma=0.99, n=20_000, seed=0):
    """Spread of the shaping term the agent actually receives, per step.

    Matching two potentials in spread over the state space does not match the
    per-step increment gamma*Phi(s') - Phi(s), which is what is added to a
    reward of -1 and therefore what decides whether the bonus dominates the
    task. Phi is the distance to a target, so the increment depends on how the
    potential varies between consecutive visited states and not only on how far
    it varies overall. Reported per condition beside the spread, because the
    random control's claim to be matched in scale rests on this number and not
    on the other one.
    """
    obs = sample_states(n=n, seed=seed)
    p = np.asarray(phi(obs), dtype=float)
    return float(np.std(gamma * p[1:] - p[:-1]))


def rescale_to_increment(poly, target, variables, reference_increment,
                         gamma=0.99):
    """Rescale a potential about `target` so its per-step increment matches.

    Two potentials matched in spread over the visited states are not matched in
    what the agent receives, and no gauge fixes both. The energy varies little
    between consecutive states because that is what being conserved means, while
    an arbitrary degree-3 polynomial of the same range varies about five times
    as much per step. Matching the spread therefore hands the random arm a
    larger per-step signal, and matching the increment hands it a smaller range.
    The two arms are run separately because the difference between them is the
    alternative explanation the control exists to exclude.

    Scaling about the target is exact rather than approximate: with
    `Phi = -|p - E*|`, sending `p` to `E* + alpha (p - E*)` sends `Phi` to
    `alpha Phi`, so the increment scales by `alpha` and the matching is a single
    division.
    """
    phi = sh.energy_potential(poly, variables, target)
    inc = increment_spread(phi, gamma=gamma)
    if inc == 0.0:
        raise ValueError("degenerate potential: zero per-step increment")
    alpha = reference_increment / inc
    return sp.expand(target + alpha * (poly - target))


def random_potential(variables, reference, seed=0):
    """A degree-3 polynomial with no physical meaning, matched in spread.

    The control that decides whether the shaping result says anything. If an
    arbitrary smooth potential of the same degree and magnitude accelerates
    learning as much as the discovered energy does, then the acceleration is an
    artefact of adding any dense signal to a sparse reward and the discovery
    step is doing no work. For that argument to run, "same magnitude" has to be
    true of the potential actually deployed, which is why the gauge here matches
    moments on the visited states and not a regression slope on a box.

    `seed` selects the draw. One draw supports a statement about one polynomial,
    so the experiment runs several and reports the spread across them.
    """
    import deflation
    rng = np.random.default_rng(seed + 4242)
    basis = deflation.monomials(list(variables), 3, min_degree=1)
    coefs = rng.normal(0, 1, len(basis))
    poly = sp.expand(sum(float(c) * m for c, m in zip(coefs, basis)))
    return rescale_to_moments(poly, reference, variables)


def mannwhitney(a, b):
    """Two-sided rank p-value between two conditions' per-seed scores.

    Five seeds against five is a small sample and the test says so: complete
    separation is worth p = 0.008, which is the most any comparison at this
    size can be worth, and anything short of complete separation is worth
    considerably less.
    """
    from scipy.stats import mannwhitneyu
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2 or (len(a) == len(b) and np.allclose(a, b)):
        return float("nan")
    try:
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def bootstrap_diff_ci(a, b, n_boot=10000, seed=0, alpha=0.05):
    """Percentile interval for the difference of two conditions' means."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(axis=1)
    db = b[rng.integers(0, len(b), (n_boot, len(b)))].mean(axis=1)
    d = da - db
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def env_fn():
    import gymnasium as gym
    return gym.make("Acrobot-v1")


def summarise(returns, steps, total, n_bins=15):
    """Learning curve on a common step grid, plus a sample-efficiency number."""
    grid = np.linspace(total / n_bins, total, n_bins)
    curve = []
    for g in grid:
        m = steps <= g
        curve.append(float(np.mean(returns[m][-30:])) if m.sum() else np.nan)
    return grid, np.array(curve)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=120_000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--param-sweep", action="store_true",
                    help="add faulty potentials at 10, 100 and 300 percent "
                         "mass error alongside the 30 percent one")
    ap.add_argument("--random-draws", type=int, default=DEFAULT_RANDOM_DRAWS,
                    help="how many random control potentials to draw; the "
                         "full paper run uses ten independent polynomials")
    args = ap.parse_args()
    total = 30_000 if args.quick else args.steps
    n_seeds = 2 if args.quick else args.seeds
    masses = [1.1, 1.3, 2.0, 4.0] if args.param_sweep else [1.3]
    n_random = 1 if args.quick else args.random_draws

    print("=" * 100)
    print("Potential-based shaping from a discovered conserved quantity")
    print("=" * 100)

    # ---- discovery, healthy and faulty
    good, t_good, sg, sn = discover_potential()
    if good is None:
        raise SystemExit("discovery failed on the healthy system; "
                         "nothing downstream is meaningful")
    good = rescale_to_energy(good, G_REF[2], V)
    err = coefficient_error(good, G_REF[2], V)
    print(f"  discovered the energy in {t_good:.1f}s, spectral gap "
          f"{sg / sn:.2e}")
    print(f"  agreement with the analytic energy: relative coefficient "
          f"error {err:.2e}")

    # Spread of each deployed potential over the sampled state space, so that
    # "matched in scale" is a measured number in the CSV. The random control is
    # only informative if its spread really does match the energy's, and the
    # gauge decides that: a regression onto the energy collapses for a
    # polynomial uncorrelated with it, giving a near-constant potential that is
    # no control at all, so the gauge here is by moments.
    spreads = {"discovered": potential_spread(good, V)}

    faulty_phis = []
    for m2 in masses:
        params = dict(m2=m2)
        poly, _, _, _ = discover_potential(params=params)
        if poly is None:
            print(f"  m2 = {m2:g}: no potential recovered")
            continue
        # Gauge and target both come from the faulty model, which is all a
        # deployer holding that model has.
        own = analytic_energy(params)
        poly = rescale_to_energy(poly, own, V)
        target = energy_target(params)
        rel = abs(m2 - envs.ACROBOT["m2"]) / envs.ACROBOT["m2"]
        name = "faulty" if len(masses) == 1 else f"faulty_m2_{m2:g}"
        print(f"  m2 = {m2:g} ({rel:.0%} error): potential recovered, "
              f"gauged against its own energy, target {target:.3f} "
              f"against {E_TARGET:.3f}")
        spreads[name] = potential_spread(poly, V)
        faulty_phis.append((name, sh.energy_potential(poly, V, target),
                            m2, rel, target,
                            coefficient_error(poly, G_REF[2], V)))

    phi_good = sh.energy_potential(good, V, E_TARGET)

    # Several draws, because one polynomial supports a claim about one
    # polynomial. Two gauges per draw, because no single gauge matches both the
    # range of the potential and the size of the increment the agent receives;
    # `rescale_to_increment` says why both arms are needed.
    inc_good = increment_spread(phi_good)
    random_phis = []
    for k in range(n_random):
        rand = random_potential(V, G_REF[2], seed=k)
        step = rescale_to_increment(rand, E_TARGET, V, inc_good)
        for poly, tag in ((rand, "random"), (step, "randstep")):
            name = tag if n_random == 1 else f"{tag}_{k}"
            spreads[name] = potential_spread(poly, V)
            random_phis.append((name,
                                sh.energy_potential(poly, V, E_TARGET)))

    conditions = [("none", None, False), ("discovered", phi_good, False)]
    conditions += [(n, p, False) for n, p, _, _, _, _ in faulty_phis]
    conditions += [(n, p, False) for n, p in random_phis]
    conditions += [("rnd", None, True)]

    # The spread of a potential over the visited states and the spread of the
    # increment the agent receives are different numbers, and the second is the
    # one a claim of matched scale has to make. Both are printed and both go to
    # the CSV.
    increments = {n: increment_spread(p)
                  for n, p in [("discovered", phi_good)]
                  + [(n, p) for n, p, _, _, _, _ in faulty_phis] + random_phis}
    print(f"\n  {'condition':<22}{'potential spread':>18}"
          f"{'per-step increment sd':>24}")
    for name in increments:
        print(f"  {name:<22}{spreads.get(name, float('nan')):>18.3f}"
              f"{increments[name]:>24.3f}")
    print("  Both are measured over states a random policy visits on "
          "Acrobot-v1, and\n  the environment reward is -1 per step.")

    rows, srows = [], []
    print(f"\n{'condition':<12}{'seed':>5}{'final return':>14}"
          f"{'AUC (mean ret)':>16}{'seconds':>9}")
    print("-" * 100)
    for name, phi, use_rnd in conditions:
        for seed in range(n_seeds):
            nov = sh.RND(6, seed=seed) if use_rnd else None
            t0 = time.time()
            rets, steps = sh.ppo(env_fn, phi=phi, total_steps=total,
                                 seed=seed, novelty=nov)
            dt = time.time() - t0
            grid, curve = summarise(rets, steps, total)
            final = float(np.mean(rets[-30:]))
            auc = float(np.nanmean(curve))
            print(f"{name:<12}{seed:>5}{final:>14.1f}{auc:>16.1f}{dt:>9.1f}")
            rows.append(dict(condition=name, seed=seed, total_steps=total,
                             final_return=final, auc_return=auc,
                             n_episodes=len(rets), seconds=dt,
                             **{f"curve_{int(g)}": c
                                for g, c in zip(grid, curve)}))

    print("\n" + "=" * 100)
    print("Summary: mean over seeds, unshaped environment return")
    print("=" * 100)
    print(f"{'condition':<16}{'final return':>16}{'AUC':>12}"
          f"{'vs none, p':>13}{'vs discovered, p':>19}")
    base = np.array([r["auc_return"] for r in rows
                     if r["condition"] == "none"])
    disc = np.array([r["auc_return"] for r in rows
                     if r["condition"] == "discovered"])
    for name, _, _ in conditions:
        sub = [r for r in rows if r["condition"] == name]
        f = np.array([r["final_return"] for r in sub])
        a = np.array([r["auc_return"] for r in sub])
        p_none = mannwhitney(a, base)
        p_disc = mannwhitney(a, disc)
        print(f"{name:<16}{f.mean():>10.1f} +- {f.std():<4.1f}{a.mean():>12.1f}"
              f"{p_none:>13.4f}{p_disc:>19.4f}")
        # The p column cannot carry an equivalence claim: five seeds against
        # five bottom out at 0.0079, so "not separable" is a statement about
        # the sample size. The interval on the difference from the discovered
        # potential is what an equivalence claim rests on, and it is recorded
        # per condition rather than for one comparison. Note also that no
        # correction is applied across the conditions in this table; with seven
        # of them 0.0079 x 7 = 0.055, so the effect sizes and intervals carry
        # the reading and the p column is a detection flag only.
        d_lo, d_hi = bootstrap_diff_ci(a, disc) if len(disc) else (
            float("nan"), float("nan"))
        n_lo, n_hi = bootstrap_diff_ci(a, base) if len(base) else (
            float("nan"), float("nan"))
        srows.append(dict(condition=name, n_seeds=len(sub),
                          final_return_mean=float(f.mean()),
                          final_return_sd=float(f.std()),
                          auc_mean=float(a.mean()), auc_sd=float(a.std()),
                          p_vs_none=p_none, p_vs_discovered=p_disc,
                          auc_diff_vs_discovered=float(a.mean() - disc.mean())
                          if len(disc) else float("nan"),
                          auc_diff_vs_discovered_lo=d_lo,
                          auc_diff_vs_discovered_hi=d_hi,
                          auc_diff_vs_none=float(a.mean() - base.mean())
                          if len(base) else float("nan"),
                          auc_diff_vs_none_lo=n_lo,
                          auc_diff_vs_none_hi=n_hi,
                          potential_spread=spreads.get(name, float("nan")),
                          increment_spread=increments.get(name,
                                                          float("nan"))))

    # A claim that two conditions are indistinguishable is not established by
    # failing to reject at five seeds. What it needs is an interval: if the
    # difference in AUC return could still be as large as the improvement
    # shaping is claimed to give, then nothing has been shown either way.
    rnd = np.array([r["auc_return"] for r in rows if r["condition"] == "rnd"])
    if len(rnd) and len(base):
        d = rnd.mean() - base.mean()
        lo, hi = bootstrap_diff_ci(rnd, base)
        gain = disc.mean() - base.mean() if len(disc) else float("nan")
        print(f"\n  RND against no shaping: difference in AUC return "
              f"{d:+.1f}, 95% interval [{lo:+.1f}, {hi:+.1f}]")
        print(f"  For comparison the discovered potential moves it by "
              f"{gain:+.1f}. An interval\n  wider than that gain does not "
              "support a claim that the two are equivalent.")

    print("\n  Policy invariance is the property to check, not to assume: "
          "shaping should\n  reach a given return sooner without changing the "
          "return it converges to.\n  The AUC column is where an improvement "
          "should appear and the final-return\n  column is where it should "
          "not.")

    # The equivalence the faulty sweep claims, stated as intervals. A claim
    # that a faulty potential shapes as well as the correct one is supported
    # only if the interval on that difference is narrow against the gain
    # shaping delivers in the first place.
    gain = disc.mean() - base.mean() if len(disc) and len(base) else float("nan")
    if np.isfinite(gain) and gain != 0:
        print(f"\n  Difference in AUC return from the discovered potential, "
              f"against the\n  {gain:+.1f} that potential gains over no shaping "
              f"at all. Percentile bootstrap\n  on five seeds, so the interval "
              f"is itself coarse.")
        print(f"    {'condition':<22}{'diff':>9}{'95% interval':>22}"
              f"{'width / gain':>15}")
        print("    " + "-" * 66)
        for s in srows:
            if s["condition"] in ("none", "discovered"):
                continue
            lo, hi = (s["auc_diff_vs_discovered_lo"],
                      s["auc_diff_vs_discovered_hi"])
            if not np.isfinite(lo):
                continue
            print(f"    {s['condition']:<22}"
                  f"{s['auc_diff_vs_discovered']:>+9.1f}"
                  f"{f'[{lo:+.1f}, {hi:+.1f}]':>22}"
                  f"{abs(hi - lo) / abs(gain):>14.0%}")

    if n_random > 1:
        print("\n" + "=" * 100)
        print("The random control across draws and across gauges")
        print("=" * 100)
        for tag, what in (("random", "matched in range"),
                          ("randstep", "matched in per-step increment")):
            names = [n for n, _ in random_phis if n.startswith(tag + "_")]
            aucs = [np.mean([r["auc_return"] for r in rows
                             if r["condition"] == n]) for n in names]
            finals = [np.mean([r["final_return"] for r in rows
                               if r["condition"] == n]) for n in names]
            print(f"  {tag:<10} ({what}), {len(names)} draws x {n_seeds} "
                  f"seeds: AUC {np.mean(aucs):.1f} +- {np.std(aucs):.1f} over "
                  f"draws, final {np.mean(finals):.1f} +- {np.std(finals):.1f}")
            print("             " + "  ".join(f"{n}={a:.1f}"
                                              for n, a in zip(names, aucs)))
            # The polynomial draw, not each of its PPO seeds, is the
            # independent unit for a claim about arbitrary random potentials.
            # Store that descriptive reduction separately from the per-draw
            # seed-level comparisons above; it deliberately has no pooled
            # p-value or seed bootstrap interval.
            srows.append(dict(
                condition=f"{tag}_across_draws", aggregation="draw_means",
                n_draws=len(names), n_seeds=n_seeds,
                final_return_mean=float(np.mean(finals)),
                final_return_sd=float(np.std(finals)),
                auc_mean=float(np.mean(aucs)), auc_sd=float(np.std(aucs)),
                auc_draw_min=float(np.min(aucs)), auc_draw_max=float(np.max(aucs)),
                final_draw_min=float(np.min(finals)),
                final_draw_max=float(np.max(finals))))

    if len(faulty_phis) > 1:
        print("\n" + "=" * 100)
        print("How wrong may the physics be before the potential stops "
              "helping?")
        print("=" * 100)
        print(f"{'mass error':>12}{'m2':>7}{'target':>10}"
              f"{'coeff. error':>14}{'AUC':>10}{'final':>10}{'vs none, p':>13}")
        for name, _, m2, rel, target, cerr in faulty_phis:
            sub = [r for r in rows if r["condition"] == name]
            a = np.array([r["auc_return"] for r in sub])
            f = np.array([r["final_return"] for r in sub])
            print(f"{rel:>12.0%}{m2:>7.2f}{target:>10.2f}{cerr:>14.2e}"
                  f"{a.mean():>10.1f}{f.mean():>10.1f}"
                  f"{mannwhitney(a, base):>13.4f}")

    out = os.path.join(RESULTS, "shaping.csv")
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

    out2 = os.path.join(RESULTS, "shaping_summary.csv")
    keys2 = []
    for r in srows:
        for k in r:
            if k not in keys2:
                keys2.append(k)
    with open(out2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys2)
        w.writeheader()
        w.writerows(srows)
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
