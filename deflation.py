"""Groebner deflation and arity-graded invariant discovery.

This is the methodological contribution of the paper: replacing the orthogonal
projection deflation of the underlying discovery method with normal-form
reduction modulo the running reduced Groebner basis.

The two rules differ in what they remove from the degree-d nullspace:

  orthogonal projection   V <- (I - g g^T / ||g||^2) V
        removes span(g), a single direction.

  Groebner deflation      V <- { NF_G(v) : v in V } \\ {0}
        removes the entire degree-d graded component of <G>, of dimension
        C(n + d - deg g, d - deg g) for a single generator g.

The two coincide only at d = deg g.
"""

from math import comb
from itertools import combinations, combinations_with_replacement

import numpy as np
import sympy as sp


# ------------------------------------------------------------- monomials ---

def monomials(variables, degree, min_degree=0):
    """All monomials of total degree in [min_degree, degree], grevlex-ish order."""
    out = []
    for d in range(min_degree, degree + 1):
        if d == 0:
            out.append(sp.Integer(1))
            continue
        for combo in combinations_with_replacement(variables, d):
            m = sp.Integer(1)
            for v in combo:
                m *= v
            out.append(m)
    return out


def rebind(expr, target_vars):
    """Rebind an expression's symbols onto `target_vars`, matched by name.

    sr_gb builds its monomial library with assumption-free symbols, while the
    algebraic side of this codebase uses real=True. Sympy treats those as
    distinct symbols, so a polynomial written in one set is seen as having
    symbolic *coefficients* when viewed over the other, which surfaces much
    later as "Cannot convert expression to float". Rebinding by name at the
    boundary is the fix.
    """
    mapping = {}
    by_name = {str(v): v for v in target_vars}
    for s in expr.free_symbols:
        if str(s) in by_name:
            mapping[s] = by_name[str(s)]
    return expr.subs(mapping) if mapping else expr


def monomial_arity(mono, variables):
    """Number of distinct variables appearing in a monomial."""
    return len([v for v in variables if mono.has(v)])


def coeff_vector(poly, basis, variables):
    """Coordinates of `poly` in the monomial `basis`. Raises if poly leaves span."""
    p = sp.Poly(sp.expand(poly), *variables)
    lookup = {}
    for i, m in enumerate(basis):
        mp = sp.Poly(m, *variables)
        lookup[mp.monoms()[0]] = i
    vec = np.zeros(len(basis))
    for mono, c in zip(p.monoms(), p.coeffs()):
        if mono not in lookup:
            raise ValueError(f"monomial {mono} of {poly} outside the basis")
        vec[lookup[mono]] = float(c)
    return vec


def poly_rank(polys, variables, degree, allow_constant=True, tol=1e-10):
    """Rank of the span of `polys` inside the degree-<=`degree` monomial space."""
    if not polys:
        return 0
    basis = monomials(variables, degree, min_degree=0 if allow_constant else 1)
    rows = [coeff_vector(p, basis, variables) for p in polys]
    A = np.array(rows)
    if A.size == 0:
        return 0
    s = np.linalg.svd(A, compute_uv=False)
    return int(np.sum(s > tol * max(1.0, s[0])))


# ------------------------------------------------- the two deflation rules ---

def orthogonal_projection_deflate(V, g_vec):
    """The rule of the underlying method: project out one direction."""
    g = np.asarray(g_vec, dtype=float)
    g = g / np.linalg.norm(g)
    return V - np.outer(V @ g, g)


def groebner_deflate(V_polys, G, variables):
    """Normal-form reduction of each nullspace vector modulo the basis G.

    Returns the surviving (nonzero) normal forms.
    """
    if not G:
        return list(V_polys)
    gb = sp.groebner(list(G), *variables, order="grevlex", domain="QQ")
    out = []
    for v in V_polys:
        nf = sp.expand(gb.reduce(sp.expand(v))[1])
        if sp.simplify(nf) != 0:
            out.append(nf)
    return out


def projection_vs_normalform(g, variables, degree):
    """Deflate <g> out of the degree-<=`degree` space both ways; return ranks.

    Returns (rank after orthogonal projection, rank after normal form, raw rank)
    where "raw" is the dimension of the degree-<=degree graded piece of <g>.
    """
    deg_g = sp.Poly(g, *variables).total_degree()
    mult = [sp.expand(m * g) for m in monomials(variables, degree - deg_g)]
    raw_rank = poly_rank(mult, variables, degree)

    basis = monomials(variables, degree)
    V = np.array([coeff_vector(p, basis, variables) for p in mult])
    g_vec = coeff_vector(g, basis, variables)
    Vp = orthogonal_projection_deflate(V, g_vec)
    s = np.linalg.svd(Vp, compute_uv=False)
    proj_rank = int(np.sum(s > 1e-10 * max(1.0, s[0])))

    nf = groebner_deflate(mult, [g], variables)
    nf_rank = poly_rank(nf, variables, degree) if nf else 0
    return proj_rank, nf_rank, raw_rank


def graded_component_dim(n, deg_g, d):
    """dim of the degree-<=d graded piece of a principal ideal <g>, deg g = deg_g."""
    if d < deg_g:
        return 0
    return comb(n + d - deg_g, d - deg_g)


# --------------------------------------------------- numerical null space ---

def design_matrix(data, variables, degree, arity_subset=None):
    """Evaluate the degree-<=`degree` monomial dictionary on `data` (N x n)."""
    if arity_subset is None:
        cols = list(range(len(variables)))
        vs = list(variables)
    else:
        cols = list(arity_subset)
        vs = [variables[i] for i in cols]
    monos = monomials(vs, degree)
    sub = data[:, cols]
    Phi = np.ones((data.shape[0], len(monos)))
    for j, m in enumerate(monos):
        p = sp.Poly(m, *vs)
        exps = p.monoms()[0]
        col = np.ones(data.shape[0])
        for k, e in enumerate(exps):
            if e:
                col = col * sub[:, k] ** e
        Phi[:, j] = col
    return Phi, monos, vs


def numerical_nullspace(Phi, tol=None, max_dim=None):
    """Right singular vectors of Phi below the numerical rank threshold.

    `full_matrices=True` is required and not a preference. With
    `full_matrices=False` the decomposition returns only min(N, M) right
    singular vectors, so `Vt[len(s):]` is empty and the M > N case silently
    loses the M - N directions that are in the nullspace for dimensional
    reasons alone: on a 3x10 design matrix with a 7-dimensional nullspace it
    returned nothing at all.
    """
    U, s, Vt = np.linalg.svd(Phi, full_matrices=True)
    N, M = Phi.shape
    if tol is None:
        tol = max(N, M) * np.finfo(float).eps * s[0]
    idx = np.where(s <= tol)[0]
    ns = Vt[len(s) - len(idx):len(s)] if len(idx) else np.zeros((0, M))
    # The trailing rows of Vt beyond the singular values are the guaranteed
    # nullspace directions when the dictionary is wider than the sample.
    if M > len(s):
        ns = np.vstack([Vt[len(s):], ns]) if ns.size else Vt[len(s):]
    if max_dim is not None and ns.shape[0] > max_dim:
        ns = ns[-max_dim:]
    return ns, s


def vectors_to_polys(V, monos):
    """Turn nullspace row vectors into sympy polynomials over the monomial list."""
    out = []
    for row in V:
        expr = sp.Integer(0)
        for c, m in zip(row, monos):
            if abs(c) > 1e-12:
                expr += sp.Float(c) * m
        out.append(sp.expand(expr))
    return out
