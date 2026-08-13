"""Attribution when two constants are wrong at once.

The single-fault verdict is model selection over one-parameter explanations, and
the shape real debugging usually takes is one wrong entry in a configuration.
Two wrong entries is the case where that model is misspecified, and there are
three things the diagnostic can do about it. It can decline, which is correct.
It can name one parameter at a value that splits the difference between the two,
which is a wrong answer delivered with confidence and is the failure worth
knowing about. Or the two parameters can be jointly identifiable from the
recovered generator, in which case freeing both recovers both.

Pairs are chosen so that the two constants enter the energy differently. `m2`
multiplies every term, `g` only the potential ones, `I2` only the kinetic ones,
`lc2` both, and `l1` enters through the coupling term alone. A pair drawn from
one group is a harder case for identifiability than a pair drawn from two.

Each configuration is generated the same way as every other fault here: the
analytic model of `envs.py` with the two constants moved, integrated at
dt = 0.005 with RK4 over 400 steps from initial conditions shared with the
reference, and the invariant recovered from those trajectories alone. What is
scored is the single-fault verdict, the deviation of the joint fit from the two
true values, and the residual gap that decides identifiability.

Writes Results/multifault.csv.
"""

import argparse
import csv
import os

import numpy as np
import sympy as sp

import bugs
import envs
import ideal_diff
import recover

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
os.makedirs(RESULTS, exist_ok=True)

V = list(bugs.ACRO_SYMS)
LAG = 50
GAP = 0.1

#: Pairs of constants to move together, with the value each is moved to.
PAIRS = [
    (("m2", 1.3), ("g", 9.81)),
    (("m2", 1.3), ("l1", 1.2)),
    (("lc2", 0.6), ("I2", 1.5)),
    (("l1", 1.2), ("I2", 1.5)),
]

SYMBOL = {"m1": bugs.P_m1, "m2": bugs.P_m2, "l1": bugs.P_l1,
          "lc1": bugs.P_lc1, "lc2": bugs.P_lc2, "I1": bugs.P_I1,
          "I2": bugs.P_I2, "g": bugs.P_g}


def joint_fit(recovered, names, spec):
    """Free both parameters at once and fit them with the gauge.

    The single-fault ranking answers "which one", which has no answer when two
    are wrong. This answers "what are they", and its residual says whether the
    pair is identifiable from this generator at all: two constants that enter
    every coefficient in the same ratio are not separable however good the data.
    """
    params = [SYMBOL[n] for n in names]
    others = {q: v for q, v in spec.items() if q not in params}
    tmpl = sp.expand(bugs.acrobot_energy_template().subs(others))
    values, resid = ideal_diff.match_to_template(tmpl, params, recovered, V,
                                                 allow_offset=True)
    return values, resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    pairs = PAIRS[:2] if args.quick else PAIRS

    G_known, _, _ = bugs.acrobot_reference()
    spec = bugs.acrobot_spec()

    print("=" * 118)
    print("Two simultaneous parameter faults: what does single-fault "
          "attribution report?")
    print("=" * 118)
    print(f"{'faults':<22}{'verdict':<34}{'named':<7}{'fitted':>9}"
          f"{'true':>9}{'gap':>9}   {'joint fit':<28}")
    print("-" * 118)

    rows = []
    for (n1, v1), (n2, v2) in pairs:
        params = {n1: v1, n2: v2}
        trajs = bugs.acrobot_trajectories(params=params, dt=0.005, n_traj=8,
                                          n_steps=400, seed=1, ic_seed=0,
                                          noise=args.noise)
        poly, s_min, s_next = recover.recover_conserved_quotient(
            trajs, V, G_known[:2], degree=3, gap=GAP, lag=LAG)
        verdict, hyps = ideal_diff.diagnose(
            bugs.acrobot_energy_template(), spec, poly, V, allow_offset=True)

        top = hyps[0] if hyps else None
        gap = (hyps[1].residual / hyps[0].residual
               if len(hyps) > 1 and hyps[0].residual > 0 else float("inf"))
        # A named parameter that is neither of the two moved is the split-the-
        # difference failure; a named parameter that is one of the two is a
        # partial answer that still hides the other.
        named_is_moved = bool(top and top.param in params)

        jv, jr = (joint_fit(poly, [n1, n2], spec) if poly is not None
                  else (None, float("inf")))
        if jv is not None:
            got = {str(k): v for k, v in jv.items()}
            j1, j2 = got.get(n1, float("nan")), got.get(n2, float("nan"))
            err = max(abs(j1 - v1) / abs(v1), abs(j2 - v2) / abs(v2))
            jtxt = f"{n1}={j1:.4f} {n2}={j2:.4f}"
        else:
            j1 = j2 = err = float("nan")
            jtxt = "-"

        print(f"{n1 + '+' + n2:<22}{verdict:<34}"
              f"{(top.param if top else '-'):<7}"
              f"{(f'{top.value:.4f}' if top else '-'):>9}"
              f"{(f'{spec[SYMBOL[top.param]]:.4f}' if top else '-'):>9}"
              f"{gap:>9.2f}   {jtxt:<28}")

        rows.append(dict(
            fault_1=n1, value_1=v1, fault_2=n2, value_2=v2,
            noise=args.noise, recovered=poly is not None,
            s_min=s_min, s_next=s_next,
            verdict=verdict,
            named_parameter=top.param if top else "",
            named_value=top.value if top else "",
            named_is_one_of_the_two=named_is_moved,
            single_fault_gap=gap,
            joint_value_1=j1, joint_value_2=j2,
            joint_residual=jr, joint_worst_relative_error=err))

    named = [r for r in rows if r["verdict"] == ideal_diff.PARAMETER_FAULT]
    print("\n" + "=" * 118)
    print(f"  single-fault attribution named a parameter on {len(named)}/"
          f"{len(rows)} two-fault systems")
    if named:
        wrong = [r for r in named if not r["named_is_one_of_the_two"]]
        print(f"    of those, {len(wrong)} named a constant that was not one "
              f"of the two moved")
    # The count against a 5% bound and the worst error behind it. The count
    # alone reads the same whether the joint fit is exact or 4% out, so the
    # paper quotes the measured figure and this prints it.
    errs = [r["joint_worst_relative_error"] for r in rows
            if np.isfinite(r["joint_worst_relative_error"])]
    ok = [e for e in errs if e < 0.05]
    print(f"  freeing both parameters recovered both to within 5% on "
          f"{len(ok)}/{len(rows)} pairs")
    if errs:
        print(f"    worst relative error over those pairs: {max(errs):.2e}")

    out = os.path.join(RESULTS, "multifault.csv")
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
