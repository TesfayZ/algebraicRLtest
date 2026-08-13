"""Functional independence and the sufficiency diagnostic, both by Jacobian rank.

Two things in this file, and they are the same computation used twice.

*Functional independence* is the irredundancy notion the conserved side needs.
Conserved quantities form a subalgebra rather than an ideal, so `E` conserved
implies `E^2` conserved, and no reduced Groebner basis removes that redundancy:
`E^2` is not a member of the ideal `<E>` in any sense the vanishing machinery
can exploit, because neither vanishes. What does separate them is the Jacobian.
At a regular point the rank of `J = [grad p_1; ...; grad p_k]` equals the number
of functionally independent functions among the `p_i`, and `grad(E^2) = 2E grad E`
is parallel to `grad E`, so admitting `E^2` after `E` does not raise the rank.

A maximal functionally independent subset is not a canonical form. It is not
unique, it depends on the order candidates are examined and on the points
sampled, and it is not invariant under a change of generators. It is an
irredundancy certificate and nothing more.

*Sufficiency.* A zero constraint residual certifies membership of the constraint
variety, not dynamical correctness. Which of the two holds for a given system is
decidable locally from the same rank: by the regular-value form of the implicit
function theorem the level set of `G` through a regular point `s` has dimension
`n - rank J_G(s)`, so comparing that against an independently estimated local
dimension `d_traj` of the trajectory manifold says whether the level set locally
coincides with the trajectory manifold. Equality means the residual is locally
sufficient as well as necessary; a level set of larger dimension means `G`
under-determines the dynamics and only necessity holds.

This is local and sampling-dependent, never a global proof. The implicit
function theorem certifies equivalence only where the rank is actually checked,
and unsampled regions may have lower rank. The verdict is evidence, not a
guarantee, and both entry points return the numbers they rest on so a caller
cannot read a verdict without also seeing them.
"""

import numpy as np
import sympy as sp

import deflation

# Verdicts of `sufficiency`.
LOCALLY_SUFFICIENT = "locally sufficient"
UNDER_DETERMINED = "under-determined, necessity only"
OVER_DETERMINED = "over-determined for the sampled manifold"


def _lambdified_gradients(polys, variables):
    """Numeric gradient rows for each polynomial, as a list of callables.

    `rebind` is applied because a polynomial that came back from the support
    search carries assumption-free symbols while the algebraic side uses
    `real=True`, and sympy treats those as distinct.
    """
    vs = list(variables)
    out = []
    for p in polys:
        expr = deflation.rebind(sp.expand(p), vs)
        grad = [sp.diff(expr, v) for v in vs]
        out.append(sp.lambdify(vs, grad, "numpy"))
    return out


def jacobian_at(polys, variables, point, _grads=None):
    """Jacobian of `polys` at one point, shape (k, n)."""
    grads = _grads if _grads is not None else _lambdified_gradients(polys,
                                                                   variables)
    rows = []
    for g in grads:
        row = g(*[float(x) for x in point])
        rows.append([float(np.asarray(c).reshape(()) ) for c in row])
    return np.asarray(rows, dtype=float)


def _rank(J, tol_rel=1e-8):
    if J.size == 0:
        return 0
    s = np.linalg.svd(J, compute_uv=False)
    if s.size == 0 or s[0] == 0.0:
        return 0
    return int(np.sum(s > tol_rel * s[0]))


def jacobian_rank(polys, variables, points, tol_rel=1e-8):
    """Generic rank of the Jacobian over sampled points.

    The maximum, not the mean: rank drops on special sets, so the generic value
    is what the implicit function theorem argument needs, and a point that
    happens to be singular should not lower it.
    """
    if not polys:
        return 0
    grads = _lambdified_gradients(polys, variables)
    best = 0
    for pt in np.asarray(points, dtype=float):
        best = max(best, _rank(jacobian_at(polys, variables, pt, _grads=grads),
                               tol_rel=tol_rel))
    return best


def functionally_independent_subset(polys, variables, points, tol_rel=1e-8):
    """Greedy maximal functionally independent subset.

    Returns `(indices, rank)`, where `indices` selects the kept polynomials in
    the order given and `rank` is the generic Jacobian rank of the whole input,
    which is also the size of the returned subset. A polynomial is kept when
    adding it raises the rank, so `E^2` presented after `E` is dropped.

    The order dependence is real and is the reason this is a certificate rather
    than a canonical form: presented as `[E^2, E]` the routine keeps `E^2`.
    """
    if not polys:
        return [], 0
    grads = _lambdified_gradients(polys, variables)
    pts = np.asarray(points, dtype=float)
    kept = []
    rank_kept = 0
    for i in range(len(polys)):
        trial = kept + [i]
        best = 0
        for pt in pts:
            J = np.asarray([jacobian_at([polys[j]], variables, pt,
                                        _grads=[grads[j]])[0] for j in trial])
            best = max(best, _rank(J, tol_rel=tol_rel))
        if best > rank_kept:
            kept = trial
            rank_kept = best
    return kept, rank_kept


def local_dimension(points, n_neighbours=40, tol_rel=1e-3, max_centres=200,
                    seed=0):
    """Local PCA estimate of the intrinsic dimension of a sampled manifold.

    For each of a few centres, take its nearest neighbours, centre them and
    count the singular values above `tol_rel` of the largest. The median over
    centres is returned, because a single centre near a fold overestimates.

    This estimator is not reliable enough to carry a sufficiency verdict, and
    the reported numbers do not use it. Measured on passive Acrobot over
    `tol_rel` in [1e-4, 1e-1] and `n_neighbours` in {10, 20, 40, 80}, a single
    trajectory (analytically a 1-dimensional curve) is estimated at 2 to 6 and
    never at 1, and an 8-trajectory ensemble (analytically 4-dimensional) at 1
    to 4. There is no plateau to read the answer off: a local neighbourhood's
    singular spectrum decays smoothly, because curvature over a finite window
    contributes directions that no threshold separates from genuine ones.

    Supply `d_traj` analytically to `sufficiency` where the dynamics are known,
    which on every system in this repository they are. This function is kept
    because the diagnostic is stated for the case where they are not, and a
    caller who reaches for it should see the instability documented rather than
    discover it.
    """
    X = np.asarray(points, dtype=float)
    if X.shape[0] < n_neighbours + 1:
        n_neighbours = max(2, X.shape[0] - 1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=min(max_centres, X.shape[0]),
                     replace=False)
    dims = []
    for i in idx:
        d = np.linalg.norm(X - X[i], axis=1)
        nb = X[np.argsort(d)[:n_neighbours + 1]]
        nb = nb - nb.mean(axis=0)
        s = np.linalg.svd(nb, compute_uv=False)
        if s.size and s[0] > 0:
            dims.append(int(np.sum(s > tol_rel * s[0])))
    return int(np.median(dims)) if dims else 0


def sufficiency(polys, variables, points, d_traj, tol_rel=1e-8):
    """Is a zero residual on `polys` locally sufficient, or only necessary?

    Returns `(verdict, rank_J, n - d_traj, n)`. The caller is handed the rank
    and the target so that the verdict is never readable on its own.

    `d_traj` is the local dimension of the trajectory manifold, estimated by
    `local_dimension` or supplied analytically. Note that it is a property of
    the *sampling*: an ensemble spanning many energy levels has a larger
    trajectory manifold than one at fixed energy, and the same generating set
    can therefore be locally sufficient for the second and under-determining
    for the first. That is not a defect of the diagnostic, it is the question
    being asked.
    """
    n = len(list(variables))
    rank = jacobian_rank(polys, variables, points, tol_rel=tol_rel)
    target = n - int(d_traj)
    if rank == target:
        verdict = LOCALLY_SUFFICIENT
    elif rank < target:
        verdict = UNDER_DETERMINED
    else:
        verdict = OVER_DETERMINED
    return verdict, rank, target, n
