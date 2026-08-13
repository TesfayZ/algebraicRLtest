"""Verify every analytic/symbolic claim the paper makes in Appendix A.

Each check prints PASS/FAIL and the measured quantity, so a claim that has
drifted from the text is visible rather than silently wrong. Nonzero exit on
any failure. Writes Results/verified_properties.csv.

Run:  ./venv/bin/python verify_properties.py
"""

import os
import sys
import itertools
from math import comb
import numpy as np
import sympy as sp

import envs

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
os.makedirs(RESULTS, exist_ok=True)

rows = []
failures = []


def check(name, ok, measured, claimed):
    status = "PASS" if ok else "FAIL"
    if not ok:
        failures.append(name)
    print(f"[{status}] {name}\n        measured={measured}\n        claimed ={claimed}")
    rows.append(dict(check=name, status=status, measured=str(measured),
                     claimed=str(claimed)))


def check_close(name, measured, claimed, rel=0.05):
    """Assert a measured quantity against the number the paper prints.

    An order-of-magnitude bound passes whether or not the paper's figure is the
    one the code produces, so a value that drifts inside the bound survives
    unnoticed. Every quantity checked this way is deterministic under a fixed
    seed, which makes a relative comparison the right one: `rel` is a tolerance
    on the printed precision, not on the measurement.
    """
    measured, claimed = float(measured), float(claimed)
    if claimed == 0.0:
        ok = measured == 0.0
        err = 0.0 if ok else float("inf")
    else:
        err = abs(measured / claimed - 1.0)
        ok = err <= rel
    check(name, ok, f"{measured:.4g} ({err:.1%} from the printed value)",
          f"{claimed:.4g}")


# ============================================================ A.1 Acrobot ===
print("\n=== A.1  Acrobot energy as a polynomial in the observation ===")

rng = np.random.default_rng(0)
N = 2000
th = rng.uniform(-np.pi, np.pi, size=(N, 2))
om = rng.uniform(-4 * np.pi, 4 * np.pi, size=(N, 2))
states = np.concatenate([th, om], axis=1)

E_coord = np.array([envs.acrobot_energy_coords(s) for s in states])
E_obs = envs.acrobot_energy_obs(envs.acrobot_obs(states))
max_disc = float(np.max(np.abs(E_coord - E_obs)))
check_close("A.1 energy(obs polynomial) == energy(coords)", max_disc, 1.14e-13)

# The exact rational coefficients, derived symbolically rather than asserted.
c1, s1, c2, s2, w1, w2 = sp.symbols("c1 s1 c2 s2 w1 w2", real=True)
P = envs.ACROBOT
m1, m2, l1, lc1, lc2, I1, I2, g = (sp.Rational(str(P[k])) for k in
                                   ["m1", "m2", "l1", "lc1", "lc2", "I1", "I2", "g"])
M11 = m1 * lc1**2 + m2 * (l1**2 + lc2**2) + I1 + I2 + 2 * m2 * l1 * lc2 * c2
M12 = m2 * lc2**2 + I2 + m2 * l1 * lc2 * c2
M22 = m2 * lc2**2 + I2
E_sym = sp.expand(
    sp.Rational(1, 2) * M11 * w1**2 + M12 * w1 * w2 + sp.Rational(1, 2) * M22 * w2**2
    - (m1 * lc1 + m2 * l1) * g * c1 - m2 * g * lc2 * (c1 * c2 - s1 * s2))
E_poly = sp.Poly(E_sym, c1, s1, c2, s2, w1, w2)
denoms = sorted({sp.Rational(co).q for co in E_poly.coeffs()})
maxden = max(denoms)
print(f"        symbolic energy: {E_sym}")
check("A.1 max coefficient denominator <= Qmax=16", maxden <= 16,
      f"denominators {denoms}, max {maxden}", "max 10, denominators {4,2,8,10}")
# The paper prints the denominator set and its maximum, so assert both. The
# inequality above passes for any max <= 16 and cannot see either.
check("A.1 coefficient denominator set", denoms == [2, 4, 8, 10],
      f"{denoms}", "{2,4,8,10}")
check_close("A.1 max coefficient denominator", maxden, 10)
check("A.1 energy total degree", E_poly.total_degree() == 3,
      E_poly.total_degree(), 3)
# The paper states this count as "exactly the eight terms printed there", so the
# predicate has to be the comparison and not a constant.
check("A.1 number of monomials in E", len(E_poly.coeffs()) == 8,
      len(E_poly.coeffs()), "8 terms in eq. (acrobotE)")

# ============================================== A.2 Passive drift under RK4 ===
print("\n=== A.2  Passive energy drift under RK4 ===")

s0 = np.array([0.1, -0.2, 0.3, -0.15])
E0 = envs.acrobot_energy_coords(s0)
drift = {}
for dt in [0.2, 0.05, 0.01]:
    n = int(round(40.0 / dt))
    traj = envs.acrobot_rollout(s0, dt, n, torque=0.0)
    E = np.array([envs.acrobot_energy_coords(s) for s in traj])
    drift[dt] = float(np.max(np.abs(E - E0)) / abs(E0))
    print(f"        dt={dt:<5} steps={n:<5} rel drift={drift[dt]:.3e}")

order = np.log(drift[0.2] / drift[0.05]) / np.log(0.2 / 0.05)
check("A.2 drift ordering monotone in dt",
      drift[0.2] > drift[0.05] > drift[0.01],
      {k: f"{v:.2e}" for k, v in drift.items()},
      "6.51e-4 > 6.86e-7 > 2.20e-10")
for _dt, _claim in ((0.2, 6.51e-4), (0.05, 6.86e-7), (0.01, 2.20e-10)):
    check_close(f"A.2 relative energy drift at dt={_dt}", drift[_dt], _claim)
check_close("A.2 observed convergence order", order, 4.95)

# ============================================ A.3 Power balance under torque ===
print("\n=== A.3  Power balance under torque ===")

bal = {}
for dt in [0.05, 0.01, 0.002]:
    traj = envs.acrobot_rollout(s0, dt, 200, torque=1.0)
    E = np.array([envs.acrobot_energy_coords(s) for s in traj])
    w2_ = traj[:-1, 3]
    resid = np.abs(E[1:] - E[:-1] - dt * 1.0 * w2_)
    bal[dt] = float(np.mean(resid))
    print(f"        dt={dt:<6} mean residual={bal[dt]:.3e}  "
          f"resid/dt^2={bal[dt]/dt**2:.3f}")

ratios = [bal[dt] / dt**2 for dt in [0.05, 0.01, 0.002]]
spread = max(ratios) / min(ratios)
check("A.3 balance-law residual scales as dt^2", spread < 4.0,
      f"resid/dt^2 = {[f'{r:.2f}' for r in ratios]}, spread {spread:.2f}x",
      "0.99 / 1.06 / 1.53, i.e. O(dt^2)")
for _dt, _claim in ((0.05, 2.48e-3), (0.01, 1.06e-4), (0.002, 6.12e-6)):
    check_close(f"A.3 mean balance residual at dt={_dt}", bal[_dt], _claim)
for _r, _claim in zip(ratios, (0.99, 1.06, 1.53)):
    check_close(f"A.3 residual/dt^2 = {_claim}", _r, _claim)

# ================================================= A.4 The twisted cubic ===
print("\n=== A.4  Twisted cubic: reduced GB is not a minimal generating set ===")

x, y, z = sp.symbols("x y z")
gb = sp.groebner([y - x**2, z - x**3], x, y, z, order="grevlex")
gb_list = list(gb.exprs)
print(f"        reduced GB (grevlex): {gb_list}")
check("A.4 reduced GB has 3 elements for a 2-generator ideal",
      len(gb_list) == 3, len(gb_list), 3)

# The third element must be an algebraic consequence of the first two.
gb2 = sp.groebner([y - x**2, z - x**3], x, y, z, order="grevlex")
third_reduces = all(sp.simplify(gb2.reduce(e)[1]) == 0 for e in gb_list)
check("A.4 all GB elements lie in the 2-generator ideal", third_reduces,
      third_reduces, True)

# ============================================== A.5 Deflation on Acrobot ===
print("\n=== A.5  Deflation on Acrobot, degree <= 3 ===")

import deflation  # noqa: E402  (local module)

acro_vars = sp.symbols("c1 s1 c2 s2 w1 w2", real=True)
monos3 = deflation.monomials(acro_vars, 3)
check("A.5 dictionary size M = C(9,3)", len(monos3) == 84, len(monos3), 84)

g1 = acro_vars[0]**2 + acro_vars[1]**2 - 1
g2 = acro_vars[2]**2 + acro_vars[3]**2 - 1

# Trivial multiples of g1, g2 living in degree <= 3.
mult = []
for g in (g1, g2):
    for m in deflation.monomials(acro_vars, 1):
        mult.append(sp.expand(m * g))
rank_mult = deflation.poly_rank(mult, acro_vars, 3)
check("A.5 trivial multiples of g1,g2 in degree<=3", rank_mult == 14,
      rank_mult, 14)

E_acro = E_sym  # degree-3 energy polynomial, in the same variable order
ideal_deg3 = mult + [E_acro - sp.Symbol("E0")]
rank_all = deflation.poly_rank(
    [sp.expand(p) for p in mult] + [sp.expand(E_acro)], acro_vars, 3,
    allow_constant=True)
check("A.5 raw nullspace dimension (14 multiples + energy)",
      rank_all == 15, rank_all, 15)

gb_acro = sp.groebner([g1, g2], *acro_vars, order="grevlex", domain="QQ")
deflated = [sp.expand(gb_acro.reduce(p)[1]) for p in mult]
n_killed = sum(1 for p in deflated if sp.simplify(p) == 0)
check("A.5 normal-form reduction kills all 14 trivial multiples",
      n_killed == 14, n_killed, 14)

E_nf = sp.expand(gb_acro.reduce(sp.expand(E_acro))[1])
check("A.5 energy direction survives deflation", sp.simplify(E_nf) != 0,
      "nonzero" if sp.simplify(E_nf) != 0 else "zero", "nonzero")

deflated_rank = deflation.poly_rank(
    [p for p in deflated if sp.simplify(p) != 0] + [E_nf], acro_vars, 3,
    allow_constant=True)
check("A.5 deflated nullspace dimension", deflated_rank == 1,
      deflated_rank, 1)

# ---- orthogonal projection, for the comparison in Table lin-vs-gb ----
print("\n=== A.5b  Orthogonal projection vs normal-form reduction, n=4 ===")
pv = sp.symbols("x1 y1 x2 y2", real=True)
g = pv[0]**2 + pv[1]**2 - 1
proj_rank, nf_rank, raw_rank = deflation.projection_vs_normalform(g, pv, 3)
check("A.5b raw degree<=3 rank of <g>", raw_rank == 5, raw_rank,
      "5 multiples m*g with deg m <= 1")
check("A.5b orthogonal projection removes 1 direction",
      raw_rank - proj_rank == 1, raw_rank - proj_rank, 1)
check("A.5b normal form removes all 5", raw_rank - nf_rank == raw_rank,
      raw_rank - nf_rank, raw_rank)

# ================================== A.6 Reacher forward kinematics + units ===
print("\n=== A.6  Reacher forward kinematics and the units failure ===")

rng = np.random.default_rng(1)
M = 5000
th1 = rng.uniform(-np.pi, np.pi, M)
th2 = rng.uniform(-np.pi, np.pi, M)
tx = rng.uniform(-0.2, 0.2, M)
ty = rng.uniform(-0.2, 0.2, M)
w1 = rng.uniform(-10, 10, M)
w2 = rng.uniform(-10, 10, M)
obs = envs.reacher_obs_from_state(th1, th2, tx, ty, w1, w2)

L1, L2 = envs.REACHER["l1"], envs.REACHER["l2"]
C1, C2, S1, S2 = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
TX, TY, DX, DY, DZ = obs[:, 4], obs[:, 5], obs[:, 8], obs[:, 9], obs[:, 10]

r1 = DX + TX - L1 * C1 - L2 * (C1 * C2 - S1 * S2)
r2 = DY + TY - L1 * S1 - L2 * (S1 * C2 + C1 * S2)
maxr = float(max(np.max(np.abs(r1)), np.max(np.abs(r2))))
check_close("A.6 FK identity residual over 5000 poses", maxr, 7.29e-17)
check("A.6 unit-norm residual", float(np.max(np.abs(C1**2 + S1**2 - 1))) < 1e-15,
      f"{float(np.max(np.abs(C1**2 + S1**2 - 1))):.3e}", "0 exactly")
check("A.6 dz identically zero", float(np.max(np.abs(DZ))) == 0.0,
      float(np.max(np.abs(DZ))), 0.0)

den_m = [sp.Rational(str(L1)).q, sp.Rational(str(L2)).q]
check("A.6 in metres, l2 denominator exceeds Qmax=16", max(den_m) > 16,
      f"denominators {den_m}", "10 and 100; 100 > 16")
# The paper prints both denominators, not just that the larger clears 16.
check("A.6 in metres, the denominators are 10 and 100",
      sorted(den_m) == [10, 100], f"{sorted(den_m)}", "[10, 100]")
den_nd = [sp.Rational(str(L1 / L1)).q, sp.nsimplify(L2 / L1, rational=True).q]
check("A.6 after rescaling by l1, both admissible", max(den_nd) <= 16,
      f"denominators {den_nd}", "1 and 10")
check("A.6 after rescaling by l1, the denominators are 1 and 10",
      sorted(den_nd) == [1, 10], f"{sorted(den_nd)}", "[1, 10]")

# ============================ A.8 The parametric templates used by level 2 ===
print("\n=== A.8  Symbolic templates agree with the numeric models ===")
import bugs                      # noqa: E402
import ideal_diff                # noqa: E402

# The energy exists twice in this codebase: once numerically in envs.py and once
# symbolically in bugs.py so that attribution can differentiate it with respect
# to physical parameters. A duplicated definition is a place for the two to
# drift apart, so the equality is asserted rather than assumed.
E_sym = sp.expand(bugs.acrobot_energy_template().subs(bugs.acrobot_spec()))
_rng = np.random.default_rng(11)
_th = _rng.uniform(-np.pi, np.pi, (3000, 2))
_w = _rng.uniform(-5, 5, (3000, 2))
_obs = envs.acrobot_obs(np.concatenate([_th, _w], axis=1))
_num = envs.acrobot_energy_obs(_obs)
_tmpl = ideal_diff.eval_poly(E_sym, _obs, list(bugs.ACRO_SYMS))
_err = float(np.max(np.abs(_num - _tmpl)))
check_close("A.8 energy template == envs.acrobot_energy_obs", _err, 2.84e-14)

# Same for Reacher's forward-kinematics template against the analytic
# observation builder.
_scale = envs.REACHER["l1"]
_fk = sp.expand(bugs.reacher_fk_template("x").subs(
    bugs.reacher_spec(scale=_scale)))
_robs = bugs.reacher_samples(N=3000, seed=5, scale=_scale)[0]
_fkr = float(np.max(np.abs(ideal_diff.eval_poly(_fk, _robs,
                                                list(bugs.REACH_SYMS)))))
check_close("A.8 Reacher FK template vanishes on generated poses", _fkr,
            8.88e-16)

# The refusal behaviour of the three-verdict rule is a claim in Section 6.3 and
# is cheap to assert directly: a healthy system must not be reported as faulty.
_G, _, _ = bugs.acrobot_reference()
_healthy = bugs.acrobot_trajectories(dt=0.005, n_traj=6, n_steps=300, seed=2)
import recover                   # noqa: E402
_poly, _, _ = recover.recover_conserved_quotient(_healthy,
                                                 list(bugs.ACRO_SYMS),
                                                 _G[:2], degree=3, lag=50)
_verdict, _ = ideal_diff.diagnose(bugs.acrobot_energy_template(),
                                  bugs.acrobot_spec(), _poly,
                                  list(bugs.ACRO_SYMS), allow_offset=True)
check("A.8 healthy system is not reported as a parameter fault",
      _verdict != ideal_diff.PARAMETER_FAULT, _verdict,
      "anything but 'parameter fault'")

# ===== A.8b Functional independence and the sufficiency diagnostic ===
# Conserved quantities form a subalgebra, so E conserved implies E^2 conserved
# and no Groebner basis removes the redundancy. The claim is that Jacobian rank
# does, and that the same rank answers the sufficiency question.
print("\n=== A.8b  Functional independence by Jacobian rank ===")
import independence                # noqa: E402

_ind_states = np.vstack(bugs.acrobot_trajectories(dt=0.005, n_traj=8,
                                                  n_steps=400, seed=0))[:40]
_ACRO = list(bugs.ACRO_SYMS)
_E = _G[2]
_keep3, _rank3 = independence.functionally_independent_subset(
    [_G[0], _G[1], _E], _ACRO, _ind_states)
check("A.8b the three reference generators are functionally independent",
      _keep3 == [0, 1, 2] and _rank3 == 3,
      f"kept {_keep3}, rank {_rank3}", "all three kept, rank 3")
_keepE2, _rankE2 = independence.functionally_independent_subset(
    [_G[0], _G[1], _E, sp.expand(_E**2)], _ACRO, _ind_states)
check("A.8b E^2 is discarded given E", _keepE2 == [0, 1, 2] and _rankE2 == 3,
      f"kept {_keepE2}, rank {_rankE2}", "E^2 dropped, rank stays 3")
# An affine reparameterisation of E is the same invariant and must also go.
_keepAff, _ = independence.functionally_independent_subset(
    [_G[0], _G[1], _E, sp.expand(3 * _E + 7)], _ACRO, _ind_states)
check("A.8b an affine image of E is discarded given E", _keepAff == [0, 1, 2],
      f"kept {_keepAff}", "3E+7 dropped")
# The certificate is order-dependent, which is exactly why it is not a
# canonical form. Asserting that keeps the paper's disclaimer honest.
_keepRev, _ = independence.functionally_independent_subset(
    [sp.expand(_E**2), _E], _ACRO, _ind_states)
check("A.8b the subset is order-dependent, so it is not canonical",
      _keepRev == [0], f"kept {_keepRev} from [E^2, E]",
      "E^2 kept when presented first")
check("A.8b generic Jacobian rank of the reference set",
      independence.jacobian_rank([_G[0], _G[1], _E], _ACRO, _ind_states) == 3,
      independence.jacobian_rank([_G[0], _G[1], _E], _ACRO, _ind_states), 3)

# Sufficiency, with d_traj supplied analytically: Acrobot's admissible set is
# 4-dimensional (two angles, two velocities) inside R^6, and at fixed energy the
# trajectory manifold is 3-dimensional. local_dimension is not used here; its
# docstring records why.
_suff, _rk, _tgt, _n = independence.sufficiency([_G[0], _G[1], _E], _ACRO,
                                               _ind_states, d_traj=3)
check("A.8b zero residual is locally sufficient at fixed energy",
      _suff == independence.LOCALLY_SUFFICIENT,
      f"{_suff}: rank J = {_rk}, n - d_traj = {_tgt}, n = {_n}",
      "locally sufficient, rank 3 = 6 - 3")
# Dropping the energy under-determines the dynamics, which is the other branch.
_suff2, _rk2, _tgt2, _ = independence.sufficiency([_G[0], _G[1]], _ACRO,
                                                  _ind_states, d_traj=3)
check("A.8b the unit-norm identities alone are not sufficient",
      _suff2 == independence.UNDER_DETERMINED,
      f"{_suff2}: rank J = {_rk2}, n - d_traj = {_tgt2}",
      "under-determined, rank 2 < 3")

# ==================== A.9 Quotient recovery needs deflation, not projection ===
print("\n=== A.9  Without the quotient projection there is nothing to find ===")
import deflation as _defl        # noqa: E402

_trajs = bugs.acrobot_trajectories(dt=0.005, n_traj=8, n_steps=400, seed=0)
_Phi, _monos = recover.difference_matrix(_trajs, list(bugs.ACRO_SYMS), 3,
                                         lag=50)
_cn = np.linalg.norm(_Phi, axis=0)
_cn[_cn == 0] = 1.0
_vraw = np.linalg.svd(_Phi / _cn, full_matrices=False)[2][-1] / _cn
_praw = recover._to_poly(_vraw, _monos)
_pq, _smin, _snext = recover.recover_conserved_quotient(
    _trajs, list(bugs.ACRO_SYMS), _G[:2], degree=3, gap=0.1, lag=50)

_BASIS = _defl.monomials(list(bugs.ACRO_SYMS), 3, min_degree=1)
_lookup = {sp.Poly(m, *bugs.ACRO_SYMS).monoms()[0]: i
           for i, m in enumerate(_BASIS)}


def _vec(poly):
    """Coefficient vector modulo constants, which are gauge for a conserved p."""
    p = sp.Poly(sp.expand(poly), *bugs.ACRO_SYMS)
    out = np.zeros(len(_BASIS))
    for mono, c in zip(p.monoms(), p.coeffs()):
        if mono in _lookup:
            out[_lookup[mono]] = float(c)
    return out


def _align(a, b):
    va, vb = _vec(a), _vec(b)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    return abs(float(va @ vb / (na * nb))) if na and nb else 0.0


_mult = [sp.expand(m * g) for g in _G[:2]
         for m in _defl.monomials(list(bugs.ACRO_SYMS), 1)]
_A = np.array([_vec(p) for p in _mult])
_rank = int(np.sum(np.linalg.svd(_A, compute_uv=False) > 1e-10))
_Q = np.linalg.svd(_A, full_matrices=False)[2][:_rank]
_vn = _vec(_praw) / np.linalg.norm(_vec(_praw))
_frac = float(np.linalg.norm(_Q @ _vn))

check("A.9 trivial filtered piece has dimension 14", _rank == 14,
      _rank, 14)
check("A.9 undeflated least-varying direction is a trivial multiple",
      _frac > 0.999 and _align(_praw, _G[2]) < 1e-3,
      f"fraction in component {_frac:.4f}, "
      f"alignment with energy {_align(_praw, _G[2]):.4f}",
      "fraction 1.0, alignment 0.0")
# The fraction is printed to four decimals, so compare it as a value; the
# boolean above is an order-of-magnitude bound that a drift to 0.9991 passes.
check_close("A.9 undeflated fraction inside the trivial piece", _frac, 1.0)
# The alignment is deliberately NOT compared as a value. It measures 2.6e-07,
# which rounds to the printed 0.0000, but the undeflated near-null space is
# 14-dimensional and numerically degenerate, so which direction comes out of it
# is a tie-break of the extraction rather than a property of the system. The
# bound above is the right guard, and the dimension below is the real claim.
check("A.9 undeflated direction is not identified (degenerate subspace)",
      _rank == 14, f"near-null dimension {_rank}",
      "14, so the direction is a tie-break not a property of the system")
check("A.9 deflated least-varying direction is the energy",
      _pq is not None and _align(_pq, _G[2]) > 0.999,
      f"alignment with energy {_align(_pq, _G[2]):.4f}" if _pq is not None
      else "no direction recovered", "alignment 1.0")
check_close("A.9 deflated alignment with the energy",
            _align(_pq, _G[2]) if _pq is not None else 0.0, 1.0)

# ================================================= A.10 the E4 audit ===
#
# The audit of Section 9.5 runs against per-release virtualenvs that this
# script cannot assume exist, so these checks read the CSVs the audit wrote
# rather than re-running it. When the audit has not been run they report SKIP
# and do not fail the suite: a claim that cannot be checked here is different
# from a claim that is wrong, and conflating them would make the exit code
# useless. Rebuild them with ./audit/setup_envs.sh followed by the four audit
# scripts.
print("\n=== A.10  Auditing shipped simulator releases (E4) ===")

import csv  # noqa: E402

AUDIT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "audit", "Results")


def _read(name):
    path = os.path.join(AUDIT, name)
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def skip(name, why):
    print(f"[SKIP] {name}\n        {why}")
    rows.append(dict(check=name, status="SKIP", measured=why, claimed="-"))


_pairs = _read("version_pairs.csv")
if _pairs is None:
    skip("A.10 audit over shipped releases", "audit/Results absent; run the audit")
else:
    # The null: no pair separated by more than float noise, and the paper
    # quotes the pair count, so a changed matrix must change the text.
    _differs = [r for r in _pairs if r["stage1"] == "DIFFERS"]
    check("A.10 no shipped release pair changed the dynamics",
          len(_differs) == 0,
          f"{len(_differs)} of {len(_pairs)} pairs differ", "0 differ")
    check("A.10 number of release pairs audited", len(_pairs) == 350,
          len(_pairs), "350 pairs (Sec. 9.5)")
    _envs = {r["env_id"] for r in _pairs}
    check("A.10 number of environments audited", len(_envs) == 11,
          f"{len(_envs)} environments", "11 environments")

    # The separation a pair is judged on is its *first step*, taken from an
    # imposed byte-identical state. Thresholding the horizon instead is only
    # valid while the dynamics are non-expansive: on the one chaotic
    # environment in the matrix a last-place difference grows to O(1) within an
    # episode, which is why the two quantities are checked separately.
    _worst_first = max(float(r["first_step_diff"]) for r in _pairs)
    check("A.10 largest first-step separation is float noise",
          _worst_first < 1e-13, f"{_worst_first:.3e}", "< 1e-13")
    check_close("A.10 largest first-step separation", _worst_first, 2.22e-16)
    _amp = [r for r in _pairs if r["stage1"] == "float-noise-amplified"]
    check("A.10 chaotic amplification is confined to the chaotic environment",
          len(_amp) == 13
          and {r["env_id"] for r in _amp} == {"InvertedDoublePendulum-v4",
                                              "InvertedDoublePendulum-v5"},
          f"{len(_amp)} pairs over "
          f"{sorted({r['env_id'] for r in _amp})}",
          "13 pairs, InvertedDoublePendulum v4 and v5 only")
    _worst_stable = max(float(r["max_abs_diff"]) for r in _pairs
                        if r["stage1"] != "float-noise-amplified")
    check_close("A.10 largest separation off the chaotic environment",
                _worst_stable, 9.1e-15)

_amp_trace = _read("amplification.csv")
if _amp_trace is None:
    skip("A.10 chaotic amplification trace",
         "audit/Results absent; run audit/amplification.py")
else:
    # The paper quotes this trace step by step, so a changed dump has to change
    # the text. The Reacher rows are the control: same release pair, same
    # horizon, no growth, which is what attributes the growth to the dynamics.
    def _row(env, step, older="mj-gymnasium-0.29.1-mujoco-2.3.7"):
        r = [x for x in _amp_trace
             if x["env_id"] == env and int(x["step"]) == step
             and x["older"] == older]
        return r[0] if r else None

    def _sep(env, step, older="mj-gymnasium-0.29.1-mujoco-2.3.7"):
        r = _row(env, step, older)
        return float(r["separation"]) if r else None

    _idp0 = _sep("InvertedDoublePendulum-v4", 0)
    _idp400 = _sep("InvertedDoublePendulum-v4", 400)
    check_close("A.10 chaotic pair starts at one unit in the last place",
                _idp0 if _idp0 is not None else float("nan"), 2.22e-16)
    check_close("A.10 chaotic pair separation at step 400",
                _idp400 if _idp400 is not None else float("nan"), 6.43e-2)

    # The growth rate and the number of decades climbed are both quoted in the
    # text and in the figure caption, so both are read from the CSV here. A rate
    # that exists only on a terminal cannot be contradicted by anything, which
    # is how a figure quoted from arithmetic survives a review pass.
    _r1 = _row("InvertedDoublePendulum-v4", 400)
    _r2 = _row("InvertedDoublePendulum-v4", 400,
               older="mj-gymnasium-1.2.3-mujoco-3.2.7")
    if _r1 is None or _r2 is None or "steps_per_decade" not in (_r1 or {}):
        skip("A.10 chaotic growth rate",
             "amplification.csv predates the rate columns; rerun "
             "audit/amplification.py")
    else:
        check_close("A.10 growth rate on the first chaotic pair",
                    float(_r1["steps_per_decade"]), 32.4)
        check_close("A.10 growth rate on the second chaotic pair",
                    float(_r2["steps_per_decade"]), 41.6)
        # The caption says "fourteen decades", so the predicate is the rounding
        # and not an inequality that 12 or 16 would also satisfy.
        _dec = [float(_r1["decades_climbed"]), float(_r2["decades_climbed"])]
        check("A.10 the chaotic pairs climb fourteen decades",
              all(round(d) == 14 for d in _dec),
              f"{_dec[0]:.1f} and {_dec[1]:.1f} decades", "fourteen each")
    _rea0, _rea400 = _sep("Reacher-v4", 0), _sep("Reacher-v4", 400)
    check("A.10 non-chaotic control does not amplify over the same pair",
          _rea0 is not None and _rea400 is not None
          and _rea400 < 1e-13 and abs(_rea400 - _rea0) < 1e-13,
          f"step 0 {_rea0:.3e}, step 400 {_rea400:.3e}"
          if _rea0 is not None else "absent",
          "flat at 5.6e-17")
    # The bound above passes for anything up to four orders larger than the
    # figure the paper prints, so compare the value the paper quotes.
    check_close("A.10 non-chaotic control separation at step 400",
                _rea400 if _rea400 is not None else float("nan"), 5.55e-17)

_ctrl = _read("positive_controls.csv")
if _ctrl is None:
    skip("A.10 positive controls", "audit/Results absent; run positive_control.py")
else:
    # A null is only meaningful if the same pipeline also separates known
    # faults, so these two checks establish that it does.
    for _name, _claim in (("nips_coriolis", "4.21"), ("m2_1.3", "11.56")):
        _r = [r for r in _ctrl if r["control"] == _name
              and r["generator"] == "energy"]
        if not _r:
            skip(f"A.10 control {_name}", "row absent from positive_controls.csv")
            continue
        _ratio, _broken = float(_r[0]["ratio"]), _r[0]["broken"] == "True"
        check(f"A.10 control {_name} localises to energy alone",
              _broken and _r[0]["localised"] == "energy",
              f"localised {_r[0]['localised']!r}", "localised 'energy'")
        check_close(f"A.10 control {_name} energy residual ratio", _ratio,
                    float(_claim))
    _clean = [r for r in _ctrl if r["generator"].startswith("unit-norm")]
    check("A.10 controls leave both unit-norm identities clean",
          all(r["broken"] == "False" for r in _clean),
          f"{sum(r['broken'] == 'True' for r in _clean)} of {len(_clean)} broken",
          "0 broken")

_sens = _read("sensitivity.csv")
if _sens is None:
    skip("A.10 drift floor versus step size", "audit/Results absent; run sensitivity.py")
else:
    # Table 10 claims the shipped step size blinds the screen below a few
    # percent, and that a finer step recovers the bottom of the sweep.
    _floor = {float(r["dt"]): float(r["drift_floor"]) for r in _sens}
    for _dt, _claim in ((0.2, 3.55e-3), (0.05, 4.19e-6), (0.01, 1.52e-8)):
        check_close(f"A.10 reference energy drift at dt={_dt}", _floor[_dt],
                    _claim)
    check("A.10 drift floor falls by >100x from dt=0.2 to dt=0.05",
          _floor[0.2] / _floor[0.05] > 100,
          f"{_floor[0.2] / _floor[0.05]:.1f}x", ">100x")
    _at = lambda dt, p: [float(r["rel_error"]) for r in _sens
                         if float(r["dt"]) == dt and r["param"] == p
                         and r["broken"] == "True"]
    _shipped = _at(0.2, "LINK_MASS_2")
    check("A.10 at shipped dt the screen misses a 1% mass error",
          min(_shipped) > 0.01 if _shipped else False,
          f"smallest localised {min(_shipped):g}" if _shipped else "none localised",
          "3e-2")
    _fine = _at(0.05, "LINK_MASS_2")
    check("A.10 at dt=0.05 the screen reaches the bottom of the sweep",
          bool(_fine) and min(_fine) <= 1e-3,
          f"smallest localised {min(_fine):g}" if _fine else "none localised",
          "<=1e-3")

_ref = _read("reference_residuals.csv")
if _ref is None:
    skip("A.10 observation dtype floors", "audit/Results absent; run check_reference.py")
else:
    _r = {(x["env_id"], x["generator"]): float(x["residual"]) for x in _ref}
    # float32 observations put a floor on the unit-norm identities that no
    # amount of exactness in the dynamics can remove; float64 ones do not.
    check_close("A.10 float32 classic-control observations floor unit-norm",
                _r[("Acrobot-v1", "unit-norm th1")], 1.70e-8)
    check_close("A.10 float64 MuJoCo observations reach machine precision",
                _r[("Reacher-v4", "unit-norm th1")], 3.57e-17)
    # The Reacher finding: fk sits far above the unit-norm floor in the same
    # float64 observation, which is what made it worth running down.
    check("A.10 Reacher fk residual sits far above its unit-norm floor",
          _r[("Reacher-v4", "fk-y")] > 1e4 * _r[("Reacher-v4", "unit-norm th1")],
          f"fk-y {_r[('Reacher-v4', 'fk-y')]:.3e} against "
          f"unit-norm {_r[('Reacher-v4', 'unit-norm th1')]:.3e}",
          "nine orders apart")
    check_close("A.10 Reacher fk-x residual", _r[("Reacher-v4", "fk-x")],
                1.1e-8)
    check_close("A.10 Reacher fk-y residual", _r[("Reacher-v4", "fk-y")],
                1.58e-7)

_fk = _read("reacher_fk.csv")
if _fk is None:
    skip("A.10 Reacher forward-kinematics reproducer",
         "audit/Results absent; run audit/reacher_fk.py")
else:
    # Keyed by configuration as well as by condition. The file holds one block
    # per installed (gymnasium, mujoco) pair, so a two-part key silently
    # compares whichever block happens to be written last, and the mj_forward
    # rows differ between blocks by a third, which is six times the tolerance
    # here. The configuration the paper quotes is named.
    _CFG = ("1.3.0", "3.10.0")
    _f = {(x["env_id"], x["condition"]): x for x in _fk
          if (x["gymnasium"], x["mujoco"]) == _CFG}
    check("A.10 the quoted Reacher configuration is present",
          len(_f) >= 8, f"{len(_f)} rows at gymnasium/mujoco {_CFG}",
          "8 rows, four conditions on each of v4 and v5")
    # The contrast between the first two rows is the diagnosis: the geometry is
    # exact, and the residual appears only once a step has run.
    check_close("A.10 Reacher fk holds after mj_forward",
                _f[("Reacher-v4", "mj_forward only")]["median"], 2.08e-17)
    check_close("A.10 Reacher fk median residual after mj_step",
                _f[("Reacher-v4", "after mj_step")]["median"], 6.20e-9)
    check_close("A.10 Reacher fk maximum residual after mj_step",
                _f[("Reacher-v4", "after mj_step")]["maximum"], 2.27e-8)
    check_close("A.10 Reacher fk residual outside joint1's limit",
                _f[("Reacher-v4", "outside the limit")]["maximum"], 2.46e-4)
    check("A.10 the Reacher finding reproduces on v4 and v5",
          all(_f[("Reacher-v4", c)]["maximum"]
              == _f[("Reacher-v5", c)]["maximum"]
              for c in ("mj_forward only", "after mj_step")),
          "v4 and v5 agree", "identical on both versions")

    _cfg = sorted({(x["gymnasium"], x["mujoco"]) for x in _fk})
    check("A.10 the Reacher finding spans six installed configurations",
          len(_cfg) >= 6, f"{len(_cfg)} configurations",
          "at least 6 (run reacher_fk.py --across-releases)")
    check_close("A.10 a step in which nothing moves leaves fk intact",
                _f[("Reacher-v4", "step, qvel = 0")]["median"], 2.78e-17)

# ======================= A.11 Defects a results table would not catch ===
#
# Each of these asserts a condition no results table can see, because the
# reported experiments do not visit the case. A defect in one of them changes a
# published number without changing anything the tables display, so it has to
# fail here or not at all.
print("\n=== A.11  Defects that a results table would not catch ===")

_AV = list(bugs.ACRO_SYMS)
_RV = list(bugs.REACH_SYMS)

# The verdict branch. `diagnose` must not report a generator that no
# single-parameter hypothesis explains as agreeing with the specification.
_TM, _SP = bugs.acrobot_energy_template(), bugs.acrobot_spec()
_E13 = sp.expand(_TM.subs({**_SP, bugs.P_m2: 1.3}))
_v_clean, _h_clean = ideal_diff.diagnose(_TM, _SP, _E13, _AV,
                                         allow_offset=True)
check("A.11 an exact parameter fault is named",
      _v_clean == ideal_diff.PARAMETER_FAULT and _h_clean[0].param == "m2",
      f"{_v_clean}, {_h_clean[0].param}", "parameter fault, m2")

_perturbed = sp.expand(_E13 + 0.02 * bugs.S1 * bugs.C2)
_v_bad, _h_bad = ideal_diff.diagnose(_TM, _SP, _perturbed, _AV,
                                     allow_offset=True)
check("A.11 a generator no hypothesis fits is not consistent with spec",
      _v_bad == ideal_diff.NOT_A_PARAMETER_FAULT, _v_bad,
      "not explicable as a parameter fault")
check("A.11 the winner there is separated, so the fit and not the gap decided",
      ideal_diff.separated(_h_bad) and not ideal_diff.fits(_h_bad),
      f"separated={ideal_diff.separated(_h_bad)}, "
      f"fits={ideal_diff.fits(_h_bad)}",
      "separated but not fitting")

# The residual floor is absolute, so a leak far below it must not be flagged
# whatever the scale-relative residual does. dz=0 is the single-term generator
# whose relative residual carries no magnitude at all.
_G_r, _n_r, _k_r = bugs.reacher_reference(scale=envs.REACHER["l1"])
_ref_r = bugs.reacher_samples(N=2000, seed=0, ic_seed=0,
                              scale=envs.REACHER["l1"])
_flagged = {}
for _delta in (1e-3, 1e-30):
    _test_r = bugs.reacher_samples(
        N=2000, seed=1, ic_seed=0, scale=envs.REACHER["l1"],
        obs_bug=bugs.offset_column(10, _delta))
    _d_r = ideal_diff.screen(_G_r, _ref_r, _test_r, _RV, names=_n_r,
                             kinds=_k_r, alpha=0.01)
    _flagged[_delta] = bool(next(v.broken for v in _d_r.verdicts
                                 if v.name == "dz=0"))
check("A.11 a dz leak above the residual floor is flagged", _flagged[1e-3],
      str(_flagged[1e-3]), "True")
check("A.11 a dz leak 22 orders below the floor is not",
      not _flagged[1e-30], str(_flagged[1e-30]), "False")

# A wide design matrix's nullspace includes the directions that are in it for
# dimensional reasons alone.
_Phi_wide = np.random.default_rng(0).normal(size=(3, 10))
_ns_wide, _ = _defl.numerical_nullspace(_Phi_wide)
check("A.11 numerical_nullspace finds all 7 directions when M > N",
      _ns_wide.shape[0] == 7, f"{_ns_wide.shape[0]} directions", "7")
check("A.11 and they really are in the nullspace",
      float(np.max(np.abs(_Phi_wide @ _ns_wide.T))) < 1e-12,
      f"{float(np.max(np.abs(_Phi_wide @ _ns_wide.T))):.1e}", "< 1e-12")

# E3's random control is only informative if it is genuinely matched in scale.
import benchmark_shaping as _bs    # noqa: E402
import baselines as _bl            # noqa: E402
import ideal_diff as _id           # noqa: E402
import inspect as _inspect         # noqa: E402
import make_figures as _fig        # noqa: E402
import re as _re                   # noqa: E402
_E_ref = bugs.acrobot_reference()[0][2]
_rand = _bs.random_potential(_AV, _E_ref)
_sd_e = _bs.potential_spread(_E_ref, _AV)
_sd_r = _bs.potential_spread(_rand, _AV)
check_close("A.11 the random control's spread matches the energy's", _sd_r,
            _sd_e, rel=0.01)

# Statistical contracts governing the reported experiments.
_mmd_src = _inspect.getsource(_bl._median_heuristic_gamma)
check("A.11 MMD bandwidth is selected from an unlabelled pooled sample",
      "rng.choice(len(Z)" in _mmd_src and "Z[:200]" not in _mmd_src,
      "pooled rng-selected rows" if "rng.choice(len(Z)" in _mmd_src else
      "labelled prefix", "pooled, permutation-invariant selection")
check("A.11 screening defaults to the trajectory-level exact null",
      _inspect.signature(_id.screen).parameters["unit"].default == _id.TRAJECTORY,
      _inspect.signature(_id.screen).parameters["unit"].default,
      _id.TRAJECTORY)
check("A.11 E3's full random-control design uses ten polynomial draws",
      _bs.DEFAULT_RANDOM_DRAWS == 10, str(_bs.DEFAULT_RANDOM_DRAWS), "10")
_curve_rows = [
    {"condition": "random_0", "curve_1": "0", "curve_2": "0"},
    {"condition": "random_0", "curve_1": "0", "curve_2": "0"},
    {"condition": "random_1", "curve_1": "10", "curve_2": "10"},
    {"condition": "random_1", "curve_1": "10", "curve_2": "10"},
]
_, _curve_mean, _curve_se = _fig.aggregate_curves(_curve_rows)["random"]
check("A.11 E3 figure aggregates random curves at the draw unit",
      np.allclose(_curve_mean, [5.0, 5.0]) and np.allclose(_curve_se, [5.0, 5.0]),
      f"mean={_curve_mean.tolist()}, se={_curve_se.tolist()}",
      "mean=[5, 5], se=[5, 5] over two draw means")

_shape_path = os.path.join(RESULTS, "shaping.csv")
_shape_summary_path = os.path.join(RESULTS, "shaping_summary.csv")
if os.environ.get("CCR_QUICK") == "1":
    skip("A.11 E3 random-control result layout",
         "quick driver run: paper-scale ten-draw artifact intentionally skipped")
elif not (os.path.exists(_shape_path) and os.path.exists(_shape_summary_path)):
    skip("A.11 E3 random-control result layout",
         "E3 results absent; run benchmark_shaping.py")
else:
    with open(_shape_path, newline="") as _f:
        _shape_rows = list(csv.DictReader(_f))
    with open(_shape_summary_path, newline="") as _f:
        _shape_summary = list(csv.DictReader(_f))
    _names = {r["condition"] for r in _shape_rows}
    _expected_random = {f"{tag}_{i}" for tag in ("random", "randstep")
                        for i in range(_bs.DEFAULT_RANDOM_DRAWS)}
    _aggregate = {r["condition"]: r for r in _shape_summary}
    _layout_ok = (_expected_random <= _names
                  and all(_aggregate.get(f"{tag}_across_draws", {}).get("n_draws")
                          == str(_bs.DEFAULT_RANDOM_DRAWS)
                          for tag in ("random", "randstep")))
    check("A.11 E3 results retain ten draws and draw-level summaries", _layout_ok,
          f"{len(_names & _expected_random)}/20 conditions, "
          f"aggregates={sorted(k for k in _aggregate if k.endswith('_across_draws'))}",
          "20 random conditions and two 10-draw summaries")
    if _layout_ok:
        with open(os.path.join(os.path.dirname(RESULTS), "doc",
                               "algebraicRLtest.tex")) as _f:
            _tex = _f.read()
        _table_ok = True
        _table_values = []
        for _tag, _label in (("random", "matched range"),
                             ("randstep", "matched step")):
            _draw_rows = [_aggregate[f"{_tag}_across_draws"]]
            _mean = float(_draw_rows[0]["auc_mean"])
            _lo = float(_draw_rows[0]["auc_draw_min"])
            _hi = float(_draw_rows[0]["auc_draw_max"])
            _needle = (rf"rand\. {_label}.*?\${_mean:.1f}\$ "
                       rf"\$\[{_lo:.1f}, {_hi:.1f}\]\$")
            _table_values.append(f"{_tag}: {_mean:.1f} [{_lo:.1f}, {_hi:.1f}]")
            _table_ok = _table_ok and bool(_re.search(_needle, _tex, _re.DOTALL))
        check("A.11 E3 table matches ten-draw random-control summaries", _table_ok,
              "; ".join(_table_values),
              "draw mean and range transcribed in tab:shaping")

# E3's gauge. The paper quotes the recovered energy's agreement with the
# analytic form and the five targets the potentials aim at, and none of them is
# a column of shaping.csv: the shaping run records returns, and these come out
# of the discovery step that precedes it. Recomputing them here is a second of
# arithmetic and puts every E3 number the text prints behind a check. Nothing
# below trains a policy.
_pot, _, _, _ = _bs.discover_potential()
check("A.11 the shaping potential is recovered at all", _pot is not None,
      "recovered" if _pot is not None else "no decisive spectral gap",
      "a direction with a decisive gap")
if _pot is not None:
    _gauged = _bs.rescale_to_energy(_pot, _bs.analytic_energy(), _AV)
    check_close("A.11 recovered energy against the analytic form",
                _bs.coefficient_error(_gauged, _bs.analytic_energy(), _AV),
                2.8e-8)
check_close("A.11 the correct potential's target E*", _bs.energy_target(), 19.60)
for _rel, _claim in ((0.1, 21.07), (0.3, 24.01), (1.0, 34.30), (3.0, 63.70)):
    check_close(f"A.11 target E* at a {_rel:.0%} mass error",
                _bs.energy_target(dict(m2=1.0 + _rel)), _claim)

# ===================== A.12 The operating-characteristic table ===
#
# Table tab:roc transcribes 48 cells out of one CSV, which is 48 chances to
# mistype one. The whole table is compared here instead of a representative row,
# and the censoring marks are derived from each method's own p-value floor
# rather than trusted, since a mark on the wrong row would turn a measurement
# into arithmetic or the reverse.
print("\n=== A.12  Operating characteristics over alpha ===")

_roc_path = os.path.join(RESULTS, "roc.csv")
if not os.path.exists(_roc_path):
    skip("A.12 operating-characteristic table",
         "Results/roc.csv absent; run benchmark_fault_magnitude.py")
else:
    with open(_roc_path, newline="") as _f:
        _roc = list(csv.DictReader(_f))
    _bykey = {(r["noise"], float(r["alpha"])): r for r in _roc}

    # (noise, alpha) -> (mmd, ks, discriminator, screen), as printed.
    _TAB = {
        ("0.0", 1e-12):  (0.00, 0.13, 0.53, 0.00),
        ("0.0", 1e-6):   (0.00, 0.20, 0.53, 0.00),
        ("0.0", 1e-4):   (0.33, 0.20, 0.53, 0.00),
        ("0.0", 1e-3):   (0.33, 0.20, 0.53, 1.00),
        ("0.0", 1e-2):   (0.40, 0.40, 0.53, 1.00),
        ("0.0", 0.05):   (0.53, 0.40, 0.53, 1.00),
        ("0.001", 1e-12): (0.00, 0.20, 0.53, 0.00),
        ("0.001", 1e-6):  (0.00, 0.27, 0.53, 0.00),
        ("0.001", 1e-4):  (0.33, 0.27, 0.53, 0.00),
        ("0.001", 1e-3):  (0.33, 0.27, 0.53, 0.80),
        ("0.001", 1e-2):  (0.40, 0.47, 0.53, 0.80),
        ("0.001", 0.05):  (0.53, 0.47, 0.53, 0.80),
    }
    _cols = ["mmd", "ks", "discriminator", "screen"]
    _bad = []
    for _key, _printed in _TAB.items():
        _r = _bykey.get(_key)
        if _r is None:
            _bad.append(f"{_key}: no row")
            continue
        for _c, _p in zip(_cols, _printed):
            _m = float(_r[f"{_c}_tpr"])
            if abs(_m - _p) > 0.005:
                _bad.append(f"{_key} {_c}: {_m:.3f} against printed {_p:.2f}")
    check("A.12 every cell of tab:roc matches the CSV", not _bad,
          "all 48 cells match" if not _bad else "; ".join(_bad[:4]),
          "48 rates, to the two decimals printed")

    # The censored cells are the ones below the method's own floor, and each
    # floor is arithmetic: the exact permutation null over 8 units per arm,
    # Bonferroni corrected over Acrobot's three generators, and 1/(B+1) for MMD.
    import baselines as _bl                                   # noqa: E402
    _screen_floor = 3.0 / comb(16, 8)
    _mmd_floor = 1.0 / (_bl.MMD_N_PERM + 1)
    check_close("A.12 the screen's exact-null p-value floor", _screen_floor,
                2.33e-4)
    check_close("A.12 MMD's permutation p-value floor", _mmd_floor, 1.0e-4)
    _mismatch = [f"{k} {c}" for k in _TAB for c, fl in
                 (("screen", _screen_floor), ("mmd", _mmd_floor))
                 if k in _bykey
                 and (_bykey[k][f"{c}_censored"].lower() == "true") != (k[1] < fl)]
    check("A.12 the censoring marks follow each method's own floor",
          not _mismatch, "all marks agree" if not _mismatch
          else "; ".join(_mismatch[:4]), "censored exactly below the floor")

    # The false-positive column the caption claims for the whole table.
    _fpr = [float(_bykey[k][f"{c}_fpr"]) for k in _TAB if k in _bykey
            for c in _cols]
    check("A.12 no method false-alarms anywhere in the table",
          _fpr and max(_fpr) == 0.0, f"max FPR {max(_fpr):g}" if _fpr else "no rows",
          "0.00 over the two controls")
    _wil = max(float(_bykey[k]["screen_fpr_hi"]) for k in _TAB if k in _bykey)
    check_close("A.12 Wilson upper bound on a zero rate over two controls",
                _wil, 0.66)

# ================================ A.13 What exactness is needed for (E1e) ===
#
# The ablation separates three claims the paper now makes, and each is a count
# rather than a bound, so every one goes through `check` or `check_close`
# against the printed figure. The interesting direction is the last: the paper
# says the screen tolerates coefficient error and the equality decision does
# not, and a check that only confirmed the first half would let the second
# drift.
print("\n=== A.13  An approximate reference set in place of the exact one ===")

_ap_path = os.path.join(RESULTS, "approximate_reference_summary.csv")
_rec_path = os.path.join(RESULTS, "approximate_recovery.csv")
_eq_path = os.path.join(RESULTS, "approximate_equality.csv")
if not all(os.path.exists(p) for p in (_ap_path, _rec_path, _eq_path)):
    skip("A.13 exactness ablation",
         "Results/approximate_*.csv absent; run "
         "benchmark_approximate_reference.py")
else:
    with open(_ap_path, newline="") as _f:
        _ap = list(csv.DictReader(_f))
    with open(_rec_path, newline="") as _f:
        _rec = list(csv.DictReader(_f))
    with open(_eq_path, newline="") as _f:
        _eq = list(csv.DictReader(_f))

    _pert = [r for r in _ap if r["arm"] == "perturbed"]
    _intact = [r for r in _pert if float(r["coeff_deviation"]) <= 1e-4]
    _n_faults = {int(r["n_faults"]) for r in _pert}
    check("A.13 the ablation screens the same fifteen faults as E1",
          _n_faults == {15}, sorted(_n_faults), "15")
    check("A.13 perturbing every coefficient by 1e-4 loses at most one localisation",
          _intact and all(int(r["n_localised_exact"]) >= 14 for r in _intact),
          f"{min(int(r['n_localised_exact']) for r in _intact)}/15 worst of "
          f"{len(_intact)} runs", "at least 14/15 in every run")
    _worst_ok = max(float(r["coeff_deviation"]) for r in _intact)
    check_close("A.13 largest coefficient deviation the screen survives",
                _worst_ok, 9.83e-5)

    # The knee, quoted as a range over the five perturbation draws because one
    # draw is one sign pattern and the paper prints the spread.
    def _band(dev):
        _sel = [int(r["n_localised_exact"]) for r in _pert
                if abs(float(r["coeff_deviation"]) / dev - 1.0) < 0.05]
        return (min(_sel), max(_sel)) if _sel else (None, None)

    check("A.13 at 1e-3 the screen starts losing faults", _band(9.83e-4)
          == (13, 14), _band(9.83e-4), "(13, 14)")
    check("A.13 at 1e-1 it has lost most of them", _band(9.96e-2) == (3, 4),
          _band(9.96e-2), "(3, 4)")

    # An approximate set loses faults; it does not manufacture them. That is
    # the half of the result a localisation count alone would hide.
    _fa = sum(int(r["n_false_alarms"]) for r in _pert)
    _ctrl = sum(int(r["n_controls"]) for r in _pert)
    check("A.13 no perturbation magnitude raises a false alarm", _fa == 0,
          f"{_fa}/{_ctrl}", "0 over 184 healthy controls")
    check("A.13 the perturbation sweep covers 184 controls", _ctrl == 184,
          _ctrl, "184")

    _clean = [float(r["coeff_deviation"]) for r in _rec
              if r["arm"] == "recovered" and float(r["noise"]) == 0.0
              and r["recovered"] == "True"]
    check_close("A.13 a numerically recovered set on clean data",
                max(_clean), 2.44e-8)

    _by = {(r["arm"], f"{float(r['noise']):.0e}"): r for r in _ap
           if r["arm"] in ("exact", "recovered", "forced")}
    for _arm, _sig, _want in (("exact", "0e+00", 15), ("recovered", "0e+00", 15),
                              ("exact", "1e-06", 14), ("recovered", "1e-06", 14),
                              ("exact", "1e-04", 13), ("recovered", "1e-04", 3),
                              ("forced", "1e-04", 11)):
        _r = _by[(_arm, _sig)]
        check(f"A.13 {_arm} set at sigma={_sig} localises {_want}/15",
              int(_r["n_localised_exact"]) == _want,
              f"{_r['n_localised_exact']}/15", f"{_want}/15")

    # The mechanism, which the counts alone do not pin down: at sigma = 1e-4
    # the recovery abstains on the energy, and forcing it through returns a
    # direction with no relation to the one it is named for.
    check("A.13 at sigma=1e-4 the recovery drops the energy generator",
          "acrobot:energy" in _by[("recovered", "1e-04")]["dropped"],
          _by[("recovered", "1e-04")]["dropped"] or "nothing dropped",
          "acrobot:energy")
    _forced_E = [float(r["coeff_deviation"]) for r in _rec
                 if r["arm"] == "forced" and float(r["noise"]) == 1e-4
                 and r["generator"] == "energy"]
    check("A.13 the forced energy at sigma=1e-4 is not the energy",
          bool(_forced_E) and _forced_E[0] > 0.5,
          f"coefficient deviation {_forced_E[0]:.2f}" if _forced_E else "absent",
          "order 1, i.e. a different direction")

    # And the decision that does need exactness.
    _pos = [r for r in _eq if float(r["eps"]) > 0]
    _diff = [r for r in _pos if r["equal"] == "False"]
    check("A.13 the equality decision separates every perturbed set",
          len(_diff) == len(_pos) and len(_pos) == 90,
          f"{len(_diff)}/{len(_pos)} declared different", "90/90")
    check_close("A.13 the smallest perturbation it still separates",
                min(float(r["eps"]) for r in _diff), 1e-12)
    check("A.13 and it holds on the unperturbed set",
          all(r["equal"] == "True" for r in _eq if float(r["eps"]) == 0),
          [r["equal"] for r in _eq if float(r["eps"]) == 0], "True for both systems")

# ============================================================== summary ===
out = os.path.join(RESULTS, "verified_properties.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["check", "status", "measured", "claimed"])
    w.writeheader()
    w.writerows(rows)

print(f"\n{'='*70}")
n_skip = sum(r["status"] == "SKIP" for r in rows)
n_run = len(rows) - n_skip
print(f"{n_run - len(failures)}/{n_run} checks passed"
      + (f", {n_skip} skipped" if n_skip else "") + f". Wrote {out}")
if failures:
    print("FAILURES:")
    for f_ in failures:
        print(f"  - {f_}")
    sys.exit(1)
print("All claimed properties verified.")