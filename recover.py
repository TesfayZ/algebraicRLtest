"""Recovering a generator from data, in the form attribution needs.

Two things differ from the discovery pipeline of the companion method
\\citep{sr-gb-csnp2026}, and both are forced by what the diagnostic is for.

The dictionary is the difference dictionary of Section 3.2, since the quantity
being recovered on Acrobot is conserved rather than vanishing, and it is
evaluated across trajectories at different energies so that a per-trajectory
constant cannot pass itself off as an invariant.

More importantly, snap-rounding is bypassed. The rationality prior is a prior on
the system being *correct*: a link length of 0.11 m is a round number because
somebody chose it, and a mass of 1.0 kg likewise. A system with a fault has no
such guarantee, and generically its invariant has coefficients that are not
small rationals at all. Snapping them would either fail outright or, worse,
round a faulty coefficient back onto the healthy value and hide the bug. The
diagnostic therefore works with the raw numerical nullspace direction and lets
the template fit in `ideal_diff` supply the structure that snap-rounding would
otherwise have supplied.
"""

import warnings

import numpy as np
import sympy as sp

import deflation
import discover


def difference_matrix(trajectories, variables, degree, min_degree=1, lag=1):
    """Design matrix of {m(s_{t+lag}) - m(s_t)} over each trajectory.

    The constant monomial is excluded by `min_degree=1`: its difference column
    is identically zero, so including it would contribute a nullspace direction
    that is an artefact of the dictionary rather than a property of the system.

    `lag` is the one knob here that buys noise robustness for free. A conserved
    quantity has zero difference at every lag, so lengthening the lag leaves the
    signal being searched for exactly where it was, while every direction that
    is *not* conserved grows roughly linearly in the elapsed time. Measurement
    noise, meanwhile, is unchanged. The separation the recovery depends on is a
    ratio between those two, so it improves in direct proportion to the lag,
    until secular integrator drift over the longer window starts to matter.
    """
    monos = deflation.monomials(list(variables), degree, min_degree=min_degree)
    blocks = []
    for traj in trajectories:
        # Clamp rather than skip: a trajectory shorter than the requested lag
        # still carries information, and dropping it silently would make a
        # coarse-timestep system look like it had no data rather than like the
        # coarse-timestep system it is.
        k = max(1, min(lag, len(traj) - 1))
        if len(traj) <= k:
            continue
        Phi = discover._eval_monomials(traj, list(variables), monos)
        blocks.append(Phi[k:] - Phi[:-k])
    if not blocks:
        return np.zeros((0, len(monos))), monos
    return np.vstack(blocks), monos


def vanishing_matrix(data, variables, degree):
    monos = deflation.monomials(list(variables), degree, min_degree=0)
    return discover._eval_monomials(data, list(variables), monos), monos


def nullspace(Phi, tol_rel=1e-6):
    """Nullspace in *monomial* coordinates, via the column-normalised SVD.

    Column normalisation is what makes one tolerance meaningful across a
    dictionary whose columns span many orders of magnitude. Its cost is that the
    right singular vectors come back in normalised coordinates and have to be
    divided by the column norms before anything algebraic touches them; skipping
    that step leaves the nullspace dimension unchanged under deflation and reads
    as deflation doing nothing.

    `full_matrices=True` matters when the dictionary is wider than the sample:
    there the design matrix has a guaranteed nullspace of dimension at least
    M - N that no singular value reports, because the thin SVD returns only
    min(N, M) right singular vectors. `discover.nullspace_of` and
    `deflation.numerical_nullspace` both handle this; a third nullspace routine
    that silently drops directions is the wrong thing to leave lying around.
    """
    colnorm = np.linalg.norm(Phi, axis=0)
    colnorm[colnorm == 0] = 1.0
    M = Phi.shape[1]
    U, s, Vt = np.linalg.svd(Phi / colnorm, full_matrices=True)
    n_null = int(np.sum(s <= tol_rel * s[0])) if s.size else 0
    n_null += M - len(s)                 # directions beyond the thin SVD
    if n_null == 0:
        return np.zeros((0, M)), s
    V = Vt[M - n_null:] / colnorm
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    Q, _ = np.linalg.qr(V.T)
    return Q.T[:n_null], s


def deflate_modulo_constants(V, monos_nc, G_known, variables, degree,
                             tol=1e-8):
    """Normal-form deflation of a difference-dictionary nullspace.

    Reduction modulo G does not respect the exclusion of the constant monomial:
    on Acrobot NF(c1^2) is 1 - s1^2, which has a constant term even though its
    argument does not. So the reduction is carried out in the full degree-<=d
    space and the constant coordinate is discarded afterwards. That discard is
    not a numerical convenience, it is the quotient by constants in which
    conserved quantities actually live, E and E + c being the same invariant.
    """
    vs = list(variables)
    monos_full = deflation.monomials(vs, degree, min_degree=0)
    if monos_full[1:] != list(monos_nc):
        raise ValueError("constant-free monomial list is not the tail of the "
                         "full one; the two orderings have diverged")
    V_full = np.hstack([np.zeros((V.shape[0], 1)), V])
    T = discover.normal_form_matrix(monos_full, G_known, vs)
    W = (V_full @ T.T)[:, 1:]
    if W.size == 0:
        return np.zeros((0, V.shape[1]))
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    keep = int(np.sum(s > tol * max(1.0, s[0])))
    return Vt[:keep]


def recover_conserved(trajectories, variables, G_known, degree=3,
                      tol_rel=1e-6):
    """The surviving conserved direction after deflating what is already known.

    Returns (poly, n_raw, n_deflated). `poly` is None unless deflation leaves
    exactly one direction, which is the only case in which the answer is
    unambiguous. Leaving two or more is reported rather than resolved: picking
    one by any tie-break would be the step that turns a diagnostic into a guess.
    """
    vs = list(variables)
    Phi, monos = difference_matrix(trajectories, vs, degree)
    V, _ = nullspace(Phi, tol_rel)
    n_raw = V.shape[0]
    if n_raw == 0:
        return None, 0, 0
    if G_known:
        V = deflate_modulo_constants(V, monos, G_known, vs, degree)
    n_def = V.shape[0]
    if n_def != 1:
        return None, n_raw, n_def
    return _to_poly(V[0], monos), n_raw, n_def


def recover_vanishing(data, variables, G_known, degree=2, tol_rel=1e-8):
    """The surviving vanishing direction after deflation, same contract."""
    vs = list(variables)
    Phi, monos = vanishing_matrix(data, vs, degree)
    V, _ = nullspace(Phi, tol_rel)
    n_raw = V.shape[0]
    if n_raw == 0:
        return None, 0, 0
    if G_known:
        V, _ = discover.deflate_nullspace(V, monos, G_known, vs)
    n_def = V.shape[0]
    if n_def != 1:
        return None, n_raw, n_def
    return _to_poly(V[0], monos), n_raw, n_def


# ------------------------------------------- the tolerance-free formulation ---
#
# Everything above decides a nullspace *dimension* against a tolerance, then
# deflates what it found. That order is fragile under measurement noise, and not
# marginally so: with noisy observations c1^2 + s1^2 - 1 no longer vanishes, so
# the fourteen trivial multiples are no longer in the nullspace at all, and the
# tolerance that would admit the energy admits a great deal else first.
#
# Reversing the order removes the problem and the hyperparameter together.
# Project the dictionary onto the quotient by the ideal already known, and then
# ask for the least-varying direction *in that quotient*. No tolerance is
# needed, because no dimension is being decided; the answer is a single smallest
# singular vector, and the spectral gap behind it says whether to believe it.
#
# This also settles what deflation is for. Without it the least-varying
# direction of the difference dictionary is one of the trivial multiples, which
# are conserved for algebraic reasons and say nothing about the physics. The
# quotient is the only space in which the question has the intended meaning.


def quotient_basis(monos_nc, G_known, variables, degree):
    """Orthonormal basis of (degree-<=d polynomials) / <G_known>, mod constants.

    Rows are coefficient vectors against `monos_nc`, the constant-free monomial
    list. Built from the image of the normal-form map, with the constant
    direction projected out afterwards for the reason `deflate_modulo_constants`
    gives: reduction introduces constants even when its argument has none.
    """
    vs = list(variables)
    monos_full = deflation.monomials(vs, degree, min_degree=0)
    if monos_full[1:] != list(monos_nc):
        raise ValueError("constant-free monomial list is not the tail of the "
                         "full one")
    if not G_known:
        return np.eye(len(monos_nc))
    T = discover.normal_form_matrix(monos_full, G_known, vs)
    B = image_basis(T)[:, 1:]            # drop the constant coordinate
    if B.size == 0:
        return np.zeros((0, len(monos_nc)))
    s2, Vt2 = np.linalg.svd(B, full_matrices=False)[1:]
    keep = int(np.sum(s2 > 1e-10 * max(1.0, s2[0])))
    return Vt2[:keep]


def image_basis(T, tol=1e-10):
    """Orthonormal basis of the image of the normal-form map, as rows.

    The quotient P_d / <G> is represented by the span of the standard
    monomials, which is the *image* of NF_G. Both recovery paths must use this
    same complement, or the representative each returns differs from the other
    by an ideal element, and the template fit downstream is asked to match a
    fixed representative against a moving one.

    The row space of T, `ker(T)^perp`, is a complement of the same dimension and
    is not the same subspace: on the Reacher problem the smallest principal
    cosine between the two is 0.5. The vanishing path used it and the conserved
    path used the image, which happened to be inert there only because the
    forward-kinematics generator lies in both.
    """
    U, s, _ = np.linalg.svd(T, full_matrices=False)
    rank = int(np.sum(s > tol * max(1.0, s[0])))
    return U[:, :rank].T


def _orthonormal_rows(A, tol=1e-12, rank_expected=None):
    """Orthonormal basis of the row space of `A`, as rows.

    Needed wherever a quotient basis is carried into column-normalised
    coordinates. Rescaling an orthonormal basis column by column does not leave
    it orthonormal, and a merely row-normalised basis is not one either: the
    singular values of `Phi @ B.T` are then the singular values of a composition
    with a non-orthogonal map, not those of `Phi` restricted to the subspace, so
    the acceptance ratio sigma_min/sigma_next stops being a property of the
    subspace and acquires a dependence on the units of the state variables.

    `rank_expected` closes the second half of that same problem. The callers
    build `A` as an exactly-constructed quotient basis rescaled column by
    column, so its true rank is known before the decomposition and the relative
    threshold below is the only thing that can disagree with it. That threshold
    is relative to `s[0]`, and the spread of `A`'s singular values is set by the
    dynamic range of the column norms, hence by the units of the state
    variables: at a column condition number near 1/tol the threshold silently
    discards genuine quotient directions, the acceptance ratio moves by orders,
    and recovery degrades to the weaker refusal with no warning. Passing the
    known rank makes the decomposition a change of basis rather than a fresh
    rank decision, which is what the surrounding path claims to be.
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    rank = int(np.sum(s > tol * max(1.0, s[0])))
    if rank_expected is not None and rank != rank_expected:
        warnings.warn(
            f"quotient basis lost rank under column rescaling: numerical rank "
            f"{rank} against the exactly-constructed {rank_expected}, at column "
            f"condition number {s[0] / s[-1]:.3e}. Truncating to "
            f"{rank_expected}; the recovered direction and the acceptance ratio "
            f"are both suspect at this conditioning. Nondimensionalise the "
            f"state variables.", RuntimeWarning, stacklevel=2)
        rank = rank_expected
    return Vt[:rank]


def least_varying_direction(Phi, B):
    """Smallest right singular direction of `Phi` restricted to the span of `B`.

    Returns (coeff_vector_in_monomial_coordinates, s_min, s_next). The caller
    decides whether to believe it from the gap between the two singular values:
    a genuine invariant leaves a direction along which the design matrix is
    orders of magnitude smaller than along any other, and a system that has
    stopped having the invariant leaves no such direction.
    """
    if B.shape[0] < 2:
        return None, np.inf, np.inf
    Phi_q = Phi @ B.T
    U, s, Vt = np.linalg.svd(Phi_q, full_matrices=False)
    v = Vt[-1] @ B                       # back to monomial coordinates
    return v, float(s[-1]), float(s[-2])


def recover_conserved_quotient(trajectories, variables, G_known, degree=3,
                               gap=0.1, lag=1):
    """Conserved quantity as the least-varying direction in the quotient.

    Returns (poly, s_min, s_next). `poly` is None when the spectral gap is not
    decisive. For a system that conserves nothing there is always a smallest
    singular direction, and returning it regardless would manufacture an
    invariant out of the least badly behaved combination of monomials.
    """
    vs = list(variables)
    Phi, monos = difference_matrix(trajectories, vs, degree, lag=lag)
    if Phi.shape[0] < 2:
        return None, np.inf, np.inf
    # Column-normalise so one gap threshold means the same thing across a
    # dictionary whose columns span many orders of magnitude.
    colnorm = np.linalg.norm(Phi, axis=0)
    colnorm[colnorm == 0] = 1.0
    B = quotient_basis(monos, G_known, vs, degree)
    if B.shape[0] < 2:
        return None, np.inf, np.inf
    # Express the basis in normalised coordinates. `B` is exactly constructed,
    # so its rank is known and the rescaling must not change it.
    Bn = _orthonormal_rows(B * colnorm, rank_expected=B.shape[0])
    v, s_min, s_next = least_varying_direction(Phi / colnorm, Bn)
    if v is None or not (s_min < gap * s_next):
        return None, s_min, s_next
    return _to_poly(v / colnorm, monos), s_min, s_next


def recover_vanishing_quotient(data, variables, G_known, degree=2, gap=0.1):
    """The vanishing counterpart, on a plain (not difference) dictionary.

    The constant coordinate is kept here, unlike in `quotient_basis`: a
    vanishing generator carries a genuine constant term, where for a conserved
    quantity the constant is gauge and is projected out.
    """
    vs = list(variables)
    Phi, monos_full = vanishing_matrix(data, vs, degree)
    colnorm = np.linalg.norm(Phi, axis=0)
    colnorm[colnorm == 0] = 1.0
    if G_known:
        T = discover.normal_form_matrix(monos_full, G_known, vs)
        B = image_basis(T)
    else:
        B = np.eye(len(monos_full))
    if B.shape[0] < 2:
        return None, np.inf, np.inf
    Bn = _orthonormal_rows(B * colnorm, rank_expected=B.shape[0])
    v, s_min, s_next = least_varying_direction(Phi / colnorm, Bn)
    if v is None or not (s_min < gap * s_next):
        return None, s_min, s_next
    return _to_poly(v / colnorm, monos_full), s_min, s_next


def _to_poly(vec, monos, rel_tol=1e-8):
    """Nullspace vector to a sympy polynomial, dropping numerical dust.

    The threshold is relative to the largest coefficient. An absolute one would
    delete genuinely small coefficients on a badly scaled dictionary, which on
    the energy would silently remove the gravity terms.
    """
    scale = np.max(np.abs(vec))
    expr = sp.Integer(0)
    for c, m in zip(vec, monos):
        if abs(c) > rel_tol * scale:
            expr += sp.Float(float(c) / scale) * m
    return sp.expand(expr)