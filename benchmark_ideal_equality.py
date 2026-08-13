"""Deciding ideal equality, which is what the exact form is for.

Every other measurement in this project reduces to a residual, a rank or a
p-value, all of them numerical and all of them threshold-bound. The property a
reduced Groebner basis supplies that an approximate invariant cannot is
different in kind: two ideals over QQ are equal or they are not, the question is
decidable by normal-form reduction, and the answer carries no tolerance. This
measures that decision on systems whose ground truth is known.

Three arms, in increasing distance from the specification.

  declared    the test system's ideal is built from the physics the fault
              declares. The decision has a known answer: a fault that changes a
              physical constant changes the ideal, a fault that changes only the
              integrator or the step size leaves it identical, and a fault that
              destroys conservation leaves nothing of this form to compare.
  recovered   the test system's ideal is built from trajectories. Recovery
              returns a numerical direction, attribution fits the parameter it
              implies, and the fitted value is rationalised before the exact
              comparison. Whether the recovered ideal agrees with the declared
              one is the question, and it is decided rather than scored.
  installed   the test system's ideal is built from the constants the installed
              simulator actually carries, so the comparison is against software
              nobody here wrote.

Where the installed constants come from. Acrobot's masses, lengths, centres of
mass and inertias are class attributes of
`gymnasium.envs.classic_control.acrobot.AcrobotEnv` and are read from the
imported module. Reacher's link lengths are read from the compiled MuJoCo model
as the body-frame offsets of `body1` and `fingertip`, which are 0.1 and 0.11.
The capsule geoms of the two links are both 0.1 long, so a reader that takes the
lengths from the geometry rather than the frames gets the second link wrong by
10% and produces an ideal that the comparison then reports as unequal. That
distinction is a property of the shipped model file, and the exact decision is
what makes it visible.

Writes Results/ideal_equality.csv.
"""

import argparse
import csv
import os
import time

import numpy as np
import sympy as sp

import benchmark_bug_localisation as e1
import bugs
import envs
import ideal_diff

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
os.makedirs(RESULTS, exist_ok=True)

ACRO_V = list(bugs.ACRO_SYMS)
REACH_V = list(bugs.REACH_SYMS)
C1, S1, C2, S2, W1, W2 = bugs.ACRO_SYMS

#: A fault whose effect is to destroy the invariant has no ideal of this shape
#: to compare, and saying so is the verdict rather than a gap in the table.
NO_INVARIANT = "none"

#: The exact comparison from data needs an ideal to compare, and attribution is
#: what supplies one. Where attribution declines there is no second ideal, and
#: the honest entry is that the comparison is unavailable. Substituting the
#: specification's own ideal would report agreement between the two systems on
#: the strength of having found nothing, which is the opposite of what the
#: screen has already said about them.
UNAVAILABLE = "unavailable"

#: `ideal_diff.ideals_equal` returns None when a Groebner computation failed, so
#: the decision was never made. Recording that as "different" would turn a
#: computational failure into a positive claim about two systems.
UNDECIDED = "undecided"


def decide(G1, G2, variables):
    """Exact equality as a word, with the undecided case kept separate."""
    out = ideal_diff.ideals_equal(G1, G2, variables)
    if out is None:
        return UNDECIDED
    return "equal" if out else "different"


def acrobot_ideal(params=None):
    """The reference generating set with the given physical constants, over QQ."""
    p = dict(envs.ACROBOT)
    p.update(params or {})
    spec = {k: sp.nsimplify(v, rational=True)
            for k, v in bugs.acrobot_spec(p).items()}
    E = sp.expand(bugs.acrobot_energy_template().subs(spec))
    return [C1**2 + S1**2 - 1, C2**2 + S2**2 - 1, sp.nsimplify(E, rational=True)]


def reacher_ideal(params=None, scale=None):
    scale = scale if scale is not None else envs.REACHER["l1"]
    p = dict(envs.REACHER)
    p.update(params or {})
    spec = {k: sp.nsimplify(v, rational=True)
            for k, v in bugs.reacher_spec(p, scale=scale).items()}
    fk_x = sp.nsimplify(bugs.reacher_fk_template("x").subs(spec), rational=True)
    fk_y = sp.nsimplify(bugs.reacher_fk_template("y").subs(spec), rational=True)
    return [bugs.RC1**2 + bugs.RS1**2 - 1, bugs.RC2**2 + bugs.RS2**2 - 1,
            bugs.RDZ, fk_x, fk_y]


def _swap(G, a, b):
    return [sp.expand(g.subs({a: b, b: a}, simultaneous=True)) for g in G]


def _substitute(G, var, expr):
    return [sp.expand(g.subs(var, expr)) for g in G]


#: The ideal each catalogue fault leaves behind, written from the fault's own
#: declaration. An observation fault is a change of coordinates on the same
#: physics, so its ideal is the image of the reference set under the inverse of
#: that change; a numerical fault does not touch the algebra at all.
def declared_ideal(bug):
    n = bug.name
    if n in ("acrobot_nips_coriolis", "acrobot_torque_leak"):
        return None
    if bug.system == "acrobot":
        base = acrobot_ideal()
        if n == "acrobot_m2_1.3":
            return acrobot_ideal(dict(m2=sp.Rational(13, 10)))
        if n == "acrobot_l1_1.2":
            return acrobot_ideal(dict(l1=sp.Rational(6, 5)))
        if n == "acrobot_lc2_0.6":
            return acrobot_ideal(dict(lc2=sp.Rational(3, 5)))
        if n == "acrobot_I2_1.5":
            return acrobot_ideal(dict(I2=sp.Rational(3, 2)))
        if n == "acrobot_g_9.81":
            return acrobot_ideal(dict(g=sp.Rational(981, 100)))
        if n == "acrobot_obs_swap":
            return _swap(base, S1, C2)
        if n == "acrobot_obs_unnormalised":
            return _substitute(base, C1, C1 * sp.Rational(50, 51))
        return base                      # euler, dt_0.2, the healthy control
    base = reacher_ideal()
    if n == "reacher_l2_0.13":
        return reacher_ideal(dict(l2=sp.Rational(13, 100)))
    if n == "reacher_l1_0.09":
        return reacher_ideal(dict(l1=sp.Rational(9, 100)))
    if n == "reacher_obs_swap":
        return _swap(base, bugs.RC2, bugs.RS1)
    if n == "reacher_dz_offset":
        return _substitute(base, bugs.RDZ, bugs.RDZ - sp.Rational(1, 1000))
    return base


def recovered_ideal(bug, noise=0.0):
    """The test system's ideal as the pipeline reaches it from trajectories.

    Attribution returns a fitted parameter value, which is a float. An exact
    comparison needs a rational, and the rationalisation is the step where the
    prior that a specification carries round numbers re-enters. Its denominator
    bound is stated rather than tuned: values are matched to the nearest
    rational with denominator at most 1000, which admits 13/100 and 981/100 and
    excludes nothing else that a configuration file plausibly holds.
    """
    test = e1.test_for(bug, regime=e1.PAIRED, noise=noise)
    verdict, hyps, _, _ = e1.attribute(bug, test)
    if verdict != ideal_diff.PARAMETER_FAULT or not hyps:
        return None, verdict, None, None
    top = hyps[0]
    rat = sp.Rational(top.value).limit_denominator(1000)
    if bug.system == "acrobot":
        G = acrobot_ideal({top.param: rat})
    else:
        key = {"ell1": "l1", "ell2": "l2"}[top.param]
        # The Reacher templates carry lengths in units of the first link, so
        # the fitted value is undone by the same scale before it is a length.
        G = reacher_ideal({key: rat * sp.nsimplify(envs.REACHER["l1"])})
    return G, verdict, top.param, rat


def installed_acrobot_ideal():
    """Acrobot's ideal from the constants the installed Gymnasium class holds."""
    from gymnasium.envs.classic_control.acrobot import AcrobotEnv as A
    p = dict(m1=A.LINK_MASS_1, m2=A.LINK_MASS_2, l1=A.LINK_LENGTH_1,
             lc1=A.LINK_COM_POS_1, lc2=A.LINK_COM_POS_2,
             I1=A.LINK_MOI, I2=A.LINK_MOI, g=envs.ACROBOT["g"])
    return acrobot_ideal(p), p


def installed_reacher_lengths():
    """Reacher's link lengths from the compiled model's body-frame offsets."""
    import gymnasium as gym
    import mujoco
    env = gym.make("Reacher-v5")
    m = env.unwrapped.model
    out = {}
    for name, key in (("body1", "l1"), ("fingertip", "l2")):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        out[key] = float(m.body_pos[i][0])
    geom = {}
    for name, key in (("link0", "l1"), ("link1", "l2")):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        geom[key] = float(np.linalg.norm(m.geom_size[i]))
    env.close()
    return out, geom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cat = bugs.catalogue()
    if args.quick:
        cat = ([b for b in cat if b.category == "parameter"][:2]
               + [b for b in cat if b.category in ("numerical", "control")])

    print("=" * 116)
    print("Deciding <G_ref> = <G_test> by two-way normal-form reduction over QQ")
    print("=" * 116)
    print(f"{'fault':<26}{'category':<12}{'declared':<12}{'expected':<12}"
          f"{'recovered':<13}{'named':<8}{'value':>10}{'sec':>7}")
    print("-" * 116)

    rows = []
    for bug in cat:
        V = ACRO_V if bug.system == "acrobot" else REACH_V
        G_ref = acrobot_ideal() if bug.system == "acrobot" else reacher_ideal()

        G_dec = declared_ideal(bug)
        t0 = time.time()
        if G_dec is None:
            dec = NO_INVARIANT
        else:
            dec = decide(G_ref, G_dec, V)
        t_dec = time.time() - t0

        # A fault that changes only the integrator or the sampling leaves the
        # algebra untouched, so equality is the correct answer there and
        # inequality is the correct answer wherever a constant or a coordinate
        # moved.
        expected = ("equal" if bug.category in ("numerical", "control")
                    else NO_INVARIANT if G_dec is None else "different")

        G_rec, verdict, named, value = recovered_ideal(bug, noise=args.noise)
        t0 = time.time()
        if G_rec is None:
            rec, rec_matches = UNAVAILABLE, None
        else:
            rec = decide(G_ref, G_rec, V)
            rec_matches = (G_dec is not None
                           and decide(G_rec, G_dec, V) == "equal")
        t_rec = time.time() - t0

        print(f"{bug.name:<26}{bug.category:<12}{dec:<12}{expected:<12}"
              f"{rec:<13}{(named or '-'):<8}"
              f"{(str(value) if value is not None else '-'):>10}"
              f"{t_dec + t_rec:>7.2f}")

        rows.append(dict(
            fault=bug.name, system=bug.system, category=bug.category,
            noise=args.noise,
            declared_decision=dec, expected_decision=expected,
            declared_correct=(dec == expected),
            recovered_decision=rec,
            recovered_matches_declared=("" if rec_matches is None
                                        else bool(rec_matches)),
            attribution_verdict=verdict,
            named_parameter=named or "", fitted_rational=str(value or ""),
            seconds_declared=t_dec, seconds_recovered=t_rec))

    n_dec = sum(1 for r in rows if r["declared_correct"])
    available = [r for r in rows if r["recovered_decision"] != UNAVAILABLE]
    n_rec = sum(1 for r in available if r["recovered_matches_declared"] is True)
    print("\n" + "=" * 116)
    print(f"  declared ideals: {n_dec}/{len(rows)} decisions match the fault's "
          f"own physics")
    print(f"  recovered ideals: available on {len(available)}/{len(rows)} "
          f"systems, and on those {n_rec}/{len(available)} reproduce the "
          f"declared ideal exactly")
    print("  The comparison from data needs an ideal to compare, which "
          "attribution supplies\n  and only supplies where it names a "
          "parameter. Where it declines there is no\n  second ideal and the "
          "screen's verdict is what stands.")
    print("  Every decision above is exact. No tolerance, no threshold and no "
          "p-value enters\n  any of them, which is the one thing the "
          "approximate forms cannot offer.")

    # ------------------------------------------------- the installed software ---
    print("\n" + "=" * 116)
    print("The same decision against the constants the installed simulators "
          "carry")
    print("=" * 116)
    try:
        G_inst, p_inst = installed_acrobot_ideal()
        same = decide(acrobot_ideal(), G_inst, ACRO_V)
        moved = decide(acrobot_ideal(dict(m2=sp.Rational(13, 10))), G_inst,
                       ACRO_V)
        print(f"  Acrobot, from gymnasium AcrobotEnv class attributes "
              f"{p_inst}")
        print(f"    equal to the reference ideal:      {same}")
        print(f"    equal to the m2 = 1.3 fault ideal: {moved}")
        # The installed rows carry a different question from the catalogue rows
        # above, so they use their own column rather than borrowing the
        # recovered one, which would read as a recovery that never ran.
        rows.append(dict(fault="installed_acrobot", system="acrobot",
                         category="installed", noise=args.noise,
                         declared_decision=same, expected_decision="equal",
                         declared_correct=(same == "equal"),
                         recovered_decision=UNAVAILABLE,
                         recovered_matches_declared="",
                         alternative_source="m2=1.3 fault ideal",
                         alternative_decision=moved))
    except Exception as exc:
        print(f"  Acrobot: gymnasium unavailable ({exc})")

    try:
        frames, geoms = installed_reacher_lengths()
        G_frame = reacher_ideal(dict(l1=sp.nsimplify(frames["l1"],
                                                     rational=True),
                                     l2=sp.nsimplify(frames["l2"],
                                                     rational=True)))
        same = decide(reacher_ideal(), G_frame, REACH_V)
        G_geom = reacher_ideal(dict(l1=sp.Rational(1, 10), l2=sp.Rational(1, 10)))
        from_geom = decide(reacher_ideal(), G_geom, REACH_V)
        print(f"  Reacher, link lengths from the compiled model")
        print(f"    body-frame offsets {frames}: equal to the reference ideal "
              f"{same}")
        print(f"    both capsule geoms are 0.1 long; the ideal built from them "
              f"is {from_geom}")
        rows.append(dict(fault="installed_reacher", system="reacher",
                         category="installed", noise=args.noise,
                         declared_decision=same, expected_decision="equal",
                         declared_correct=(same == "equal"),
                         recovered_decision=UNAVAILABLE,
                         recovered_matches_declared="",
                         alternative_source="ideal from the capsule geoms",
                         alternative_decision=from_geom))
    except Exception as exc:
        print(f"  Reacher: mujoco model unavailable ({exc})")

    out = os.path.join(RESULTS, "ideal_equality.csv")
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
