"""Local nullspace and deflation machinery shared by the diagnostic.

The normal-form matrix, Groebner deflation, and the orthogonal-projection
deflation it is compared against, all self-contained over numpy/scipy/sympy.
`recover.py` imports `_eval_monomials`, `normal_form_matrix`, `nullspace_of`
and `deflate_nullspace` from here; nothing in this module or its callers
reaches outside the project.

Groebner deflation. The degree-d nullspace is reduced modulo the running
reduced Groebner basis, which removes the entire degree-d graded component of
the ideal already recovered rather than the single direction that orthogonal
projection removes. `project_deflate` is the orthogonal-projection rule the
paper compares it against (Table 5, `sec:deflation`).

Implementation note on deflation. Reducing each floating-point nullspace vector
one at a time is numerically poor: the reduced set is not orthonormal and
near-zero survivors are hard to threshold. Instead we build the matrix of the
linear map NF_G restricted to the degree-<=d monomial space, which has exact
rational entries, apply it to the nullspace basis, and re-orthonormalise by SVD.
That is mathematically the same projection and much better conditioned.
"""

import numpy as np
import sympy as sp

import deflation


# ------------------------------------------------- normal form as a matrix ---

def normal_form_matrix(monos, G, variables):
    """Matrix T of the linear map NF_G on span(monos).

    NF_G(sum_i c_i m_i) = sum_j (T c)_j m_j. Exact rational arithmetic, then
    cast to float once at the end.
    """
    if not G:
        return np.eye(len(monos))
    gb = sp.groebner(list(G), *variables, order="grevlex", domain="QQ")
    index = {}
    for i, m in enumerate(monos):
        index[sp.Poly(m, *variables).monoms()[0]] = i
    T = sp.zeros(len(monos), len(monos))
    for i, m in enumerate(monos):
        nf = sp.expand(gb.reduce(m)[1])
        if nf == 0:
            continue
        p = sp.Poly(nf, *variables)
        for mono, c in zip(p.monoms(), p.coeffs()):
            if mono in index:
                T[index[mono], i] = c
            else:
                raise ValueError(
                    f"NF of {m} left the degree-bounded space: {nf}")
    return np.array(T.evalf(), dtype=float)


def deflate_nullspace(V, monos, G, variables, tol=1e-8):
    """Apply NF_G to each row of V, re-orthonormalise, drop the null directions.

    Returns (V_deflated, n_removed).
    """
    if V.shape[0] == 0:
        return V, 0
    T = normal_form_matrix(monos, G, variables)
    W = V @ T.T
    if W.size == 0:
        return np.zeros((0, V.shape[1])), V.shape[0]
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    keep = s > tol * max(1.0, s[0])
    n_keep = int(np.sum(keep))
    return Vt[:n_keep], V.shape[0] - n_keep


def project_deflate(V, g_vec, tol=1e-8):
    """The baseline rule: orthogonal projection of a single direction."""
    if V.shape[0] == 0:
        return V, 0
    g = np.asarray(g_vec, float)
    g = g / np.linalg.norm(g)
    W = V - np.outer(V @ g, g)
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    keep = s > tol * max(1.0, s[0])
    n_keep = int(np.sum(keep))
    return Vt[:n_keep], V.shape[0] - n_keep


# ------------------------------------------------------------ nullspace ---

def nullspace_of(Phi, tol_rel=1e-8):
    """Numerical nullspace of the design matrix, with a relative threshold.

    `full_matrices=True` matters when M > N. There the design matrix has a
    guaranteed nullspace of dimension at least M - N that no singular value
    reports, because the thin SVD returns only min(N, M) right singular
    vectors, and a nullspace routine that silently drops directions is the
    wrong thing to leave lying around.
    """
    U, s, Vt = np.linalg.svd(Phi, full_matrices=True)
    thresh = tol_rel * s[0] if s.size else 0.0
    n_null = int(np.sum(s <= thresh))
    M = Phi.shape[1]
    n_null += M - len(s)                 # directions beyond the thin SVD
    if n_null == 0:
        return np.zeros((0, M)), s
    return Vt[M - n_null:], s


def _eval_monomials(sub_data, sub_vars, monos):
    Phi = np.ones((sub_data.shape[0], len(monos)))
    for j, m in enumerate(monos):
        exps = sp.Poly(m, *sub_vars).monoms()[0]
        col = np.ones(sub_data.shape[0])
        for k, e in enumerate(exps):
            if e:
                col = col * sub_data[:, k] ** e
        Phi[:, j] = col
    return Phi