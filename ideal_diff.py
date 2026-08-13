"""Algebraic diffing of two dynamical systems.

This is the operation the canonical form is actually needed for. A reduced
Groebner basis is the unique representative of an ideal under a fixed monomial
order, so two ideals are equal if and only if their reduced bases are, and
membership is decided by normal-form reduction alone. Neither statement holds
for an approximate invariant, a border basis, or a learned latent constraint,
which is the argument for paying the cost of exactness.

The module exposes the diagnostic at two levels, because they need different
things to work and fail in different ways.

Level 1, screening. Evaluate a *reference* generating set on trajectories drawn
from the system under test. A generator whose scale-relative residual jumps off
the noise floor names the physical constraint that broke. This needs no
discovery run on the test system, so it still reports when discovery would
abstain, and its output is a named constraint rather than a scalar.

Level 2, attribution. Re-discover on the test system, match each recovered
generator to a parametric template, and solve for the parameters the recovered
coefficients imply. This is what turns "the forward-kinematics constraint
broke" into "l2 is 0.13, not the 0.11 in the spec". It needs discovery to
succeed on the test system, which is a real precondition and is reported as
such rather than assumed.

Nothing here assumes the test system is a simulator. `G_ref` from an analytic
model against data from a learned world model is the same computation, and is
what `benchmark_worldmodel.py` runs.
"""

from dataclasses import dataclass, field

import numpy as np
import sympy as sp

import deflation


# --------------------------------------------------------------- evaluation ---

#: Cache of (monomial exponents, float coefficients) keyed by polynomial and
#: variable list. Every residual in this module is an evaluation of the same few
#: generators on many blocks of data, and the SymPy `Poly` construction, not the
#: arithmetic, dominates that. Decomposing once per generator rather than once
#: per block is what makes a few hundred screening runs affordable.
_DECOMP = {}


def _decompose(poly, variables):
    key = (sp.srepr(sp.sympify(poly)), tuple(str(v) for v in variables))
    hit = _DECOMP.get(key)
    if hit is None:
        p = sp.Poly(sp.expand(deflation.rebind(poly, variables)), *variables)
        hit = (list(p.monoms()), np.array([float(c) for c in p.coeffs()]))
        _DECOMP[key] = hit
    return hit


def eval_poly(poly, data, variables):
    """Evaluate `poly` on an (N, n) array whose columns follow `variables`."""
    monos, coeffs = _decompose(poly, variables)
    out = np.zeros(data.shape[0])
    for mono, coeff in zip(monos, coeffs):
        term = np.full(data.shape[0], float(coeff))
        for k, e in enumerate(mono):
            if e:
                term = term * data[:, k] ** e
        out += term
    return out


def term_scale(poly, data, variables):
    """RMS of sum_i |c_i m_i(x)|, the natural scale of `poly` on this data.

    Dividing the residual by this makes it dimensionless and comparable across
    generators of different degree and coefficient magnitude, which a raw
    residual is not. Without it a degree-3 generator on velocities of order 10
    looks "worse" than a unit-norm constraint purely through units.
    """
    monos, coeffs = _decompose(poly, variables)
    out = np.zeros(data.shape[0])
    for mono, coeff in zip(monos, coeffs):
        term = np.full(data.shape[0], abs(float(coeff)))
        for k, e in enumerate(mono):
            if e:
                term = term * np.abs(data[:, k]) ** e
        out += term
    return float(np.sqrt(np.mean(out ** 2)))


def relative_residual(poly, data, variables):
    """Scale-relative RMS residual of `poly` on `data`. Zero on the variety."""
    r = float(np.sqrt(np.mean(eval_poly(poly, data, variables) ** 2)))
    s = term_scale(poly, data, variables)
    return r / s if s > 0 else r


def relative_drift(poly, trajectories, variables):
    """Scale-relative variation of `poly` *along* each trajectory.

    The right residual for a conserved quantity, which does not vanish and so
    has no meaningful distance to zero. Energy on Acrobot is whatever the
    initial condition makes it; what a broken model changes is not its value
    but whether it stays put. We report the within-trajectory spread pooled
    across trajectories, divided by the same term scale as `relative_residual`
    so the two are on a common footing and one threshold serves both.
    """
    num, den = [], []
    for traj in trajectories:
        vals = eval_poly(poly, traj, variables)
        num.append(np.std(vals))
        den.append(term_scale(poly, traj, variables))
    d = float(np.mean(den))
    return float(np.mean(num)) / d if d > 0 else float(np.mean(num))


VANISHING = "vanishing"
CONSERVED = "conserved"


def absolute_residual(poly, data, variables):
    """RMS residual of `poly` on `data`, in the generator's own units."""
    return float(np.sqrt(np.mean(eval_poly(poly, data, variables) ** 2)))


def absolute_drift(poly, trajectories, variables):
    """Variation of `poly` along each trajectory, in the generator's own units."""
    return float(np.mean([np.std(eval_poly(poly, t, variables))
                          for t in trajectories]))


def generator_residual(poly, trajectories, variables, kind=VANISHING):
    """Dispatch to the residual appropriate to the generator's kind."""
    return generator_residual_pair(poly, trajectories, variables, kind)[0]


def generator_residual_pair(poly, trajectories, variables, kind=VANISHING):
    """(relative, absolute) residual, the two quantities the screen needs.

    The rank test runs on the relative residual, which is what makes generators
    of different degree and units comparable. `RESIDUAL_FLOOR` is an absolute
    floor set by the observation dtype, so it has to be applied to the absolute
    residual and not to that ratio.

    Applying it to the ratio, as this once did, makes it unreachable for any
    generator whose scale is carried by a single term: `term_scale` is then the
    RMS of that same term, so the ratio is identically 1.0 whenever the term is
    non-zero and 0.0 otherwise, and it carries no magnitude at all. Measured on
    Reacher's `dz` generator, a z-leak of 1e-30 screened as a broken physical
    constraint with the same p-value as a leak of 1e-3.
    """
    if kind == CONSERVED:
        return (relative_drift(poly, trajectories, variables),
                absolute_drift(poly, trajectories, variables))
    pooled = np.vstack(trajectories) if isinstance(trajectories, list) \
        else trajectories
    return (relative_residual(poly, pooled, variables),
            absolute_residual(poly, pooled, variables))


# ------------------------------------------------------- ideal-level queries ---

def in_ideal(q, G, variables):
    """Ideal membership by the normal-form criterion: q in <G> iff NF_G(q) = 0.

    Exact over QQ. Returns None, not False, when the reduction cannot be carried
    out. The distinction matters because the caller is deciding equality: a
    failed Groebner computation reported as non-membership would be a positive
    claim that two systems differ, drawn from having failed to compute, and the
    whole point of the exact decision is that it does not do that.
    """
    if not G:
        return sp.expand(q) == 0
    try:
        vs = list(variables)
        gb = sp.groebner([deflation.rebind(sp.expand(g), vs) for g in G],
                         *vs, order="grevlex", domain="QQ")
        return sp.expand(gb.reduce(deflation.rebind(sp.expand(q), vs))[1]) == 0
    except Exception:
        return None


def ideals_equal(G1, G2, variables):
    """Decide <G1> = <G2> by two-way normal-form reduction.

    Comparing reduced bases elementwise would also work and is the textbook
    statement, but two-way membership avoids recomputing a basis for whichever
    side already has one.

    Returns True, False, or None when any of the four membership queries could
    not be carried out, in which case the decision is unavailable rather than
    negative.
    """
    checks = ([in_ideal(g, G2, variables) for g in G1]
              + [in_ideal(g, G1, variables) for g in G2])
    if any(c is None for c in checks):
        return None
    return all(checks)


def reduced_basis(G, variables):
    """Reduced Groebner basis over QQ under grevlex, or `G` unchanged on failure."""
    if not G:
        return []
    try:
        vs = list(variables)
        gb = sp.groebner([deflation.rebind(sp.expand(g), vs) for g in G],
                         *vs, order="grevlex", domain="QQ")
        return list(gb.exprs)
    except Exception:
        return list(G)


# ------------------------------------------------------------ the diff report ---

@dataclass
class GeneratorVerdict:
    name: str
    poly: object
    residual_ref: float          # relative residual on reference data
    residual_test: float         # relative residual on the system under test
    ratio: float                 # residual_test / residual_ref, as effect size
    pvalue: float                # one-sided, Bonferroni-corrected
    broken: bool


@dataclass
class IdealDiff:
    verdicts: list = field(default_factory=list)
    detected: bool = False
    localised: list = field(default_factory=list)   # names of broken generators
    n_ref: int = 0

    def summary(self):
        if not self.detected:
            return "no algebraic difference detected"
        return "broken: " + ", ".join(self.localised)


def block_residuals(poly, trajectories, variables, kind=VANISHING,
                    n_blocks=32):
    """Per-block residuals, the replicates the screen's hypothesis test needs.

    A single pooled residual is one number, and one number cannot say whether
    the difference between two systems exceeds what resampling the same system
    would produce. Splitting into blocks buys replicates. Blocks are contiguous
    within a trajectory rather than randomly drawn, because a conserved
    quantity's drift is a property of a stretch of trajectory and shuffling
    across it would destroy exactly the quantity being measured.

    Returns (relative, absolute) arrays; see `generator_residual_pair`.
    """
    rel, absol = [], []
    per = max(1, n_blocks // max(1, len(trajectories)))
    for traj in trajectories:
        size = max(2, len(traj) // per)
        for start in range(0, len(traj) - size + 1, size):
            chunk = traj[start:start + size]
            r, a = generator_residual_pair(poly, [chunk], variables, kind)
            rel.append(r)
            absol.append(a)
    return np.array(rel), np.array(absol)


def trajectory_residuals(poly, trajectories, variables, kind=VANISHING,
                         n_units=8):
    """One residual per trajectory, the replicates that are actually independent.

    Blocks cut from the same trajectory share an initial condition, hence an
    energy level and a drift rate, so for a conserved generator they are far
    from independent and a rank test over them reports a p-value smaller than
    the data support. A trajectory is the unit the experiment randomises over,
    and one residual per trajectory is the conservative reading of the same
    measurement.

    Where the arm is a single array of independently drawn states, as Reacher's
    is, the rows carry no dependence to begin with and the array is split into
    `n_units` chunks so that the two arms are compared on the same footing.

    Returns (relative, absolute) arrays; see `generator_residual_pair`.
    """
    if len(trajectories) == 1:
        arr = trajectories[0]
        size = max(2, len(arr) // n_units)
        units = [arr[i:i + size] for i in range(0, len(arr) - size + 1, size)]
    else:
        units = list(trajectories)
    pairs = [generator_residual_pair(poly, [u], variables, kind)
             for u in units]
    return (np.array([p[0] for p in pairs]),
            np.array([p[1] for p in pairs]))


#: Absolute residual floor, below which two arms are treated as indistinguishable
#: and no test is run.
#:
#: This is a real parameter of the screen and not a numerical convenience, so it
#: is named here rather than left to a library default. The rank test has no
#: scale: given 32 blocks per arm whose residuals differ systematically in the
#: 17th significant figure, it separates them completely and returns p ~ 3e-12,
#: which would report a broken constraint on a system that is exact to machine
#: precision. Some absolute floor is therefore unavoidable, and the question is
#: only whether it is stated.
#:
#: The value is set by measurement rather than taste. Shipped classic-control
#: observations are float32, which floors the unit-norm identities at 1.7e-8
#: however exact the dynamics; shipped MuJoCo observations reach 1e-8 because
#: the two halves of the Reacher observation are read at different points of the
#: integration step. A residual below this floor is not evidence about physics
#: on any environment measured here.
#:
#: The cost is a stated blind spot: a real violation smaller than this is
#: invisible to the screen. The paper says so.
RESIDUAL_FLOOR = 1e-8


def _below_floor(a_abs, b_abs, floor):
    """Does every residual on both arms sit under the instrument floor?

    The floor is absolute and nothing else. A relative tolerance here would be a
    second, undeclared threshold: it would suppress any fault whose residual
    ratio is close enough to one, whatever the absolute size of the residuals,
    and that is a blind spot the paper does not claim. The rule is therefore
    exactly the one `RESIDUAL_FLOOR` documents, and it fires only when neither
    arm has risen off the floor at all.

    The arguments are the *absolute* residuals. Passing the scale-relative ones
    made the floor unreachable for any generator whose scale is carried by one
    term, since the ratio is then identically 1.0 or 0.0; see
    `generator_residual_pair`.
    """
    if floor <= 0:
        return False
    return (float(np.max(np.abs(a_abs))) <= floor
            and float(np.max(np.abs(b_abs))) <= floor)


def _mannwhitney_greater(a, b, a_abs=None, b_abs=None, floor=RESIDUAL_FLOOR):
    """One-sided p-value for "a is stochastically greater than b".

    Rank-based rather than a t-test: residuals are non-negative, heavy-tailed
    and nowhere near Gaussian, and a t-test on them reports significance driven
    by a single badly-conditioned block.

    `a`/`b` are the scale-relative residuals the test ranks; `a_abs`/`b_abs` are
    the absolute ones the floor is applied to. Arms that both sit under `floor`
    are reported as indistinguishable without running the test.
    """
    from scipy.stats import mannwhitneyu
    if len(a) < 3 or len(b) < 3:
        return 1.0
    if _below_floor(a if a_abs is None else a_abs,
                    b if b_abs is None else b_abs, floor):
        return 1.0
    try:
        # `method` is pinned rather than left to "auto". At 32 blocks per arm
        # auto selects the asymptotic normal approximation, and that choice sets
        # the smallest attainable p-value (3.26e-12 under complete separation,
        # against 1/C(64,32) = 5.5e-19 for the exact test), which the paper
        # quotes as the test's resolution limit. Leaving it to a library default
        # puts a reported number at the mercy of a scipy version bump.
        return float(mannwhitneyu(a, b, alternative="greater",
                                  method="asymptotic").pvalue)
    except ValueError:
        return 1.0


def _permutation_greater(a, b, a_abs=None, b_abs=None, floor=RESIDUAL_FLOOR,
                         max_exact=200000, n_perm=20000, seed=0):
    """One-sided permutation p-value on the difference of means.

    Used at the trajectory level, where the sample is small enough that the
    permutation distribution can be enumerated exactly. With eight units per arm
    there are 12870 splits, so the smallest attainable p-value is 7.8e-5, which
    is what complete separation between two arms of eight trajectories is worth
    and is still an order below the corrected alpha.

    `a_abs`/`b_abs` carry the absolute residuals the floor applies to; see
    `_below_floor`.
    """
    from math import comb
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return 1.0
    if _below_floor(a if a_abs is None else a_abs,
                    b if b_abs is None else b_abs, floor):
        return 1.0
    pooled = np.concatenate([a, b])
    N, n = len(pooled), len(a)
    # The difference of means is monotone in the sum over the first arm once
    # the pooled values are fixed, so the split can be scored by that sum alone.
    obs = float(a.sum())
    if comb(N, n) <= max_exact:
        idx = _splits(N, n)
        sums = pooled[idx].sum(axis=1)
        return float(np.mean(sums >= obs - 1e-12 * max(1.0, abs(obs))))
    rng = np.random.default_rng(seed)
    sums = np.array([pooled[rng.permutation(N)[:n]].sum()
                     for _ in range(n_perm)])
    return float((np.sum(sums >= obs - 1e-12 * max(1.0, abs(obs))) + 1)
                 / (n_perm + 1))


_SPLITS = {}


def _splits(N, n):
    """All (N choose n) index sets, cached; the exact permutation null's support."""
    key = (N, n)
    if key not in _SPLITS:
        from itertools import combinations
        _SPLITS[key] = np.array(list(combinations(range(N), n)), dtype=np.intp)
    return _SPLITS[key]


#: The unit of replication the screen's test is run over.
#:
#: `BLOCK` cuts each trajectory into contiguous chunks, which buys replicates
#: cheaply but supplies fewer independent ones than it counts: chunks of one
#: trajectory share an initial condition. `TRAJECTORY` uses one residual per
#: trajectory with an exact permutation null, which is the conservative reading.
#: Both are reported, since the difference between them is the size of the
#: dependence correction rather than a choice to be made silently.
BLOCK, TRAJECTORY = "block", "trajectory"


def screen(G_ref, ref_trajs, test_trajs, variables, names=None, kinds=None,
           alpha=0.01, n_blocks=32, floor=RESIDUAL_FLOOR, unit=TRAJECTORY):
    """Level 1. Which reference generators does the test system violate?

    `ref_trajs` and `test_trajs` are lists of (N_i, n) arrays. For each
    reference generator we compute its residual on independent trajectories by
    default and ask, one-sided, whether the test system's residuals are
    stochastically larger, correcting across generators by Bonferroni.  The
    ``BLOCK`` unit is retained only for the explicit sensitivity analysis:
    blocks from one trajectory are dependent, so it does not support a nominal
    p-value.

    A fixed ratio threshold was the obvious rule and it is the wrong one. Under
    measurement noise the reference residual rises off machine epsilon to the
    noise floor, so a fault has to clear a hundredfold *floor* rather than a
    hundredfold *signal*, and genuine faults stop being flagged at noise levels
    where they remain perfectly visible. Testing against the reference system's
    own block-to-block variability adapts to the floor instead of assuming it
    away, and it puts the screen on the same statistical footing as the
    two-sample baselines it is compared against.

    The output is still per generator, which is what the baselines cannot do:
    the same correction that yields a detection also names the constraint.
    """
    names = names or [f"g{i+1}" for i in range(len(G_ref))]
    kinds = kinds or [VANISHING] * len(G_ref)
    k = max(1, len(G_ref))
    diff = IdealDiff(n_ref=len(G_ref))
    for name, g, kind in zip(names, G_ref, kinds):
        if unit == TRAJECTORY:
            a, a_abs = trajectory_residuals(g, test_trajs, variables, kind)
            b, b_abs = trajectory_residuals(g, ref_trajs, variables, kind)
            p = min(1.0, _permutation_greater(a, b, a_abs, b_abs,
                                              floor=floor) * k)
        else:
            a, a_abs = block_residuals(g, test_trajs, variables, kind, n_blocks)
            b, b_abs = block_residuals(g, ref_trajs, variables, kind, n_blocks)
            p = min(1.0, _mannwhitney_greater(a, b, a_abs, b_abs,
                                              floor=floor) * k)
        r_ref, r_test = float(np.median(b)), float(np.median(a))
        ratio = r_test / r_ref if r_ref > 0 else np.inf
        broken = bool(p < alpha)
        diff.verdicts.append(GeneratorVerdict(name, g, r_ref, r_test, ratio,
                                              p, broken))
        if broken:
            diff.localised.append(name)
    diff.detected = len(diff.localised) > 0
    return diff




# ------------------------------------------------------------- attribution ---
#
# A recovered generator is not equal to its template, it is equal up to the
# gauge the discovery procedure cannot see. For a vanishing generator p and 2p
# cut out the same variety, so overall scale is free. For a conserved quantity
# E, E + c is conserved too, so an additive constant is free as well. Fitting
# the physical parameters therefore means fitting them jointly with that gauge,
# and pretending otherwise is what makes an anchor-normalised match fail on the
# very parameter it is supposed to find: every coefficient of the Acrobot energy
# contains m2, so no parameter-free monomial exists to anchor on.


def coefficient_vectors(template, recovered, variables, drop_constant=False):
    """Coefficient vectors of both polynomials over their union support.

    Returns (monomials, template_coeff_exprs, recovered_coeff_floats). The union
    rather than the intersection, so that a monomial present in one and absent
    from the other contributes a residual instead of being quietly ignored.
    """
    vs = list(variables)
    T = sp.Poly(sp.expand(deflation.rebind(template, vs)), *vs)
    Rp = sp.Poly(sp.expand(deflation.rebind(recovered, vs)), *vs)
    t_coeffs = dict(zip(T.monoms(), T.coeffs()))
    r_coeffs = dict(zip(Rp.monoms(), Rp.coeffs()))

    zero = tuple([0] * len(vs))
    support = set(t_coeffs) | set(r_coeffs)
    if drop_constant:
        support.discard(zero)
    monos = sorted(support)
    t_vec = [sp.sympify(t_coeffs.get(m, 0)) for m in monos]
    r_vec = np.array([float(r_coeffs.get(m, 0)) for m in monos])
    return monos, t_vec, r_vec


def match_to_template(template, params, recovered, variables,
                      allow_offset=False, n_restarts=8, seed=0):
    """Fit `params` so that the template matches `recovered` up to gauge.

    Minimises ||c_T(theta) - alpha * c_R|| / ||c_T(theta)|| over theta and the
    scale alpha, dropping the constant monomial first when `allow_offset` is
    set, which is the right treatment for a conserved quantity. Normalising by
    the template's own norm makes the residual dimensionless, so residuals from
    different single-fault hypotheses are comparable and the ranking in
    `attribute_single_fault` means something.

    Returns (values, residual), or (None, inf) if the fit fails outright.
    """
    from scipy.optimize import least_squares

    ps = list(params)
    monos, t_vec, r_vec = coefficient_vectors(template, recovered, variables,
                                              drop_constant=allow_offset)
    if not monos or np.linalg.norm(r_vec) == 0:
        return None, np.inf

    f_t = sp.lambdify(ps, t_vec, "numpy")

    def residual(x):
        theta, alpha = x[:-1], x[-1]
        tv = np.array([float(v) for v in f_t(*theta)], dtype=float)
        nrm = np.linalg.norm(tv)
        if nrm == 0:
            return np.ones_like(tv) * 1e3
        return (tv - alpha * r_vec) / nrm

    # Seed alpha from the norm ratio at the spec parameters, which is the right
    # order of magnitude even when the parameters are badly wrong.
    rng = np.random.default_rng(seed)
    best, best_cost = None, np.inf
    for i in range(n_restarts):
        theta0 = np.ones(len(ps)) if i == 0 else rng.uniform(0.2, 2.0, len(ps))
        try:
            tv0 = np.array([float(v) for v in f_t(*theta0)], dtype=float)
            a0 = float(np.dot(tv0, r_vec) / max(np.dot(r_vec, r_vec), 1e-30))
            out = least_squares(residual, np.append(theta0, a0))
        except Exception:
            continue
        if out.cost < best_cost:
            best, best_cost = out.x, out.cost
    if best is None:
        return None, np.inf
    values = {p: float(v) for p, v in zip(ps, best[:-1])}
    return values, float(np.linalg.norm(residual(best)))


@dataclass
class FaultHypothesis:
    param: str
    value: float
    spec: float
    residual: float

    def deviation(self):
        return abs(self.value - self.spec) / max(abs(self.spec), 1e-12)


def attribute_single_fault(template, spec, recovered, variables,
                           allow_offset=False):
    """Level 2. Which single parameter explains the recovered coefficients?

    `spec` maps each parameter symbol of `template` to the value the environment
    specification claims. For each parameter in turn we free that one, hold the
    rest at spec, and fit; the hypothesis with the smallest residual in
    coefficient space is the diagnosis. This is model selection over
    single-fault explanations, which is the shape real debugging takes: some
    entry of the configuration is wrong and the question is which one.

    A well-posed diagnosis has a clear winner whose residual is near zero while
    every rival sits orders of magnitude above. A flat ranking means the
    parameters are not separately identifiable from this generator alone, and
    `identifiable` is what callers should check before believing the argmin.
    """
    out = []
    for p in spec:
        # Compared by value, not by identity. Sympy's symbol cache usually makes
        # the two agree, but it is a cache and not a guarantee, and a miss here
        # would silently free two parameters instead of one.
        others = {q: v for q, v in spec.items() if q != p}
        tmpl_p = sp.expand(sp.sympify(template).subs(others))
        values, resid = match_to_template(tmpl_p, [p], recovered, variables,
                                          allow_offset=allow_offset)
        if values is None or p not in values:
            continue
        out.append(FaultHypothesis(str(p), float(values[p]), float(spec[p]),
                                   float(resid)))
    out.sort(key=lambda h: h.residual)
    return out


#: the three verdicts attribution can return, and they are not a ranking
PARAMETER_FAULT = "parameter fault"
CONSISTENT_WITH_SPEC = "consistent with spec"
NOT_A_PARAMETER_FAULT = "not explicable as a parameter fault"


def diagnose(template, spec, recovered, variables, allow_offset=False,
             gap=10.0, abs_tol=1e-3, min_deviation=1e-3):
    """Turn a recovered generator into one of three verdicts.

    `recovered` is None when the discovery step left no unique direction, which
    happens exactly when the system stopped having the invariant at all. That
    is the correct verdict for a structural fault: energy is not conserved
    under a dropped Coriolis term for *any* setting of the masses, so no
    parameter assignment explains it and the diagnostic says so instead of
    returning the closest fit.

    Otherwise a fault is diagnosed only when one parameter both fits far better
    than its rivals and lands away from its specified value. The second half
    matters more than it looks: on a healthy system every single-fault
    hypothesis fits well, each at its own spec value, so a rule that reported
    the argmin would name a fault on a correct simulator every time.

    The two ways of failing that test are opposite in meaning and must not
    share a verdict. If the winner fits but is not separated from its rival,
    the data is consistent with the specification and the parameters are not
    identifiable from it. If the winner does not fit at all, the recovered
    generator *contradicts* every single-parameter hypothesis, including the
    specification itself, which is the same situation a dropped Coriolis term
    produces and is the opposite of agreement. Routing both to
    `CONSISTENT_WITH_SPEC` reported a 30% second-link mass error as a clean
    bill of health at sigma = 1e-7, with the correct parameter sitting first in
    the returned list at 1.3056 and beating its rival elevenfold, purely
    because its residual missed `abs_tol` by a factor of 1.4.

    Returns (verdict, hypotheses).
    """
    if recovered is None:
        return NOT_A_PARAMETER_FAULT, []
    hyps = attribute_single_fault(template, spec, recovered, variables,
                                  allow_offset=allow_offset)
    if not hyps:
        return NOT_A_PARAMETER_FAULT, []
    top = hyps[0]
    if not fits(hyps, abs_tol=abs_tol):
        return NOT_A_PARAMETER_FAULT, hyps
    if not separated(hyps, gap=gap):
        return CONSISTENT_WITH_SPEC, hyps
    if top.deviation() < min_deviation:
        return CONSISTENT_WITH_SPEC, hyps
    return PARAMETER_FAULT, hyps


def fits(hypotheses, abs_tol=1e-3):
    """Does the best single-fault hypothesis actually explain the generator?

    Rules out the case where every hypothesis is bad and one is merely least
    bad. Failing this is evidence *against* the specification, not for it.
    """
    return bool(hypotheses) and hypotheses[0].residual <= abs_tol


def separated(hypotheses, gap=10.0):
    """Does the best hypothesis beat its nearest rival by `gap`?

    Rules out the case where two parameters enter the generator in the same way
    and are genuinely indistinguishable from it. Failing this is
    non-identifiability, which is reported rather than resolved by tie-break.
    """
    if not hypotheses:
        return False
    if len(hypotheses) == 1:
        return True
    best, second = hypotheses[0], hypotheses[1]
    if best.residual <= 0:
        return second.residual > 0
    return second.residual / best.residual > gap


def identifiable(hypotheses, gap=10.0, abs_tol=1e-3):
    """Both conditions at once. Retained for callers that want the conjunction.

    `diagnose` deliberately does not use this: it needs to tell the two
    conditions apart, because they carry opposite verdicts.
    """
    return fits(hypotheses, abs_tol=abs_tol) and separated(hypotheses, gap=gap)