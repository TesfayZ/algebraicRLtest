"""Sequential orthogonal projection deflates at most one direction in total.

The paper states that orthogonal projection removes only span(g), a single
direction, where normal-form reduction removes the whole graded component. The
measured deflation ablation shows something stronger: projecting a *second*
generator out after the first removes nothing at all, so the total is one
direction however many generators have been recovered.

The reason is elementary. After projecting g1 out of a subspace S, the result
is S' = S \\cap g1^perp. A second generator g2 with <g1, g2> != 0 does not lie
in S', and v |-> v - <v, g2> g2 is injective on any subspace not containing a
vector parallel to g2, so the rank does not drop.

This test confirms it symbolically and numerically, and confirms that the
statement fails exactly when g2 IS orthogonal to g1, which is the case the
paper's own wording implicitly assumes.
"""

import numpy as np
import sympy as sp

import deflation
import discover


def rank_of(V, tol=1e-8):
    if V.shape[0] == 0:
        return 0
    s = np.linalg.svd(V, compute_uv=False)
    return int(np.sum(s > tol * max(1.0, s[0])))


def run(g_list, variables, degree, label):
    monos = deflation.monomials(variables, degree)
    mult = []
    for g in g_list:
        dg = sp.Poly(g, *variables).total_degree()
        mult += [sp.expand(m * g)
                 for m in deflation.monomials(variables, degree - dg)]
    V = np.array([deflation.coeff_vector(p, monos, variables) for p in mult])
    Q, _ = np.linalg.qr(V.T)
    V = Q.T[:rank_of(V)]
    raw = V.shape[0]

    Vp = V.copy()
    per_step = []
    for g in g_list:
        gv = deflation.coeff_vector(g, monos, variables)
        before = Vp.shape[0]
        Vp, _ = discover.project_deflate(Vp, gv)
        per_step.append(before - Vp.shape[0])

    Vg, rm = discover.deflate_nullspace(V, monos, g_list, variables)

    print(f"\n{label}")
    print(f"  raw graded component        : {raw}")
    print(f"  projection, removed per step: {per_step}  (total {sum(per_step)})")
    print(f"  projection, surviving       : {Vp.shape[0]}")
    print(f"  normal form, removed        : {rm}")
    print(f"  normal form, surviving      : {Vg.shape[0]}")
    return per_step, raw, rm


# --- Case 1: the Acrobot generators. Both contain the constant -1, so they are
#     not orthogonal as coefficient vectors.
v6 = list(sp.symbols("c1 s1 c2 s2 w1 w2", real=True))
g1 = v6[0]**2 + v6[1]**2 - 1
g2 = v6[2]**2 + v6[3]**2 - 1
inner = float(np.dot(
    deflation.coeff_vector(g1, deflation.monomials(v6, 3), v6),
    deflation.coeff_vector(g2, deflation.monomials(v6, 3), v6)))
print(f"<g1, g2> as coefficient vectors = {inner}  (nonzero: they share the "
      f"constant term)")
steps, raw, rm = run([g1, g2], v6, 3, "Acrobot g1, g2 at degree 3")
assert steps == [1, 0], f"expected [1, 0], got {steps}"
assert rm == raw, f"normal form should remove all {raw}, removed {rm}"

# --- Case 2: orthogonal generators. Here each projection does remove one.
v4 = list(sp.symbols("x y z w", real=True))
h1 = v4[0] * v4[1]
h2 = v4[2] * v4[3]
inner2 = float(np.dot(
    deflation.coeff_vector(h1, deflation.monomials(v4, 3), v4),
    deflation.coeff_vector(h2, deflation.monomials(v4, 3), v4)))
print(f"\n<h1, h2> = {inner2}  (orthogonal)")
steps2, raw2, rm2 = run([h1, h2], v4, 3, "Orthogonal generators xy, zw at degree 3")
assert steps2 == [1, 1], f"expected [1, 1], got {steps2}"

print("\nBoth cases behave as predicted.")
print("Projection removes one direction per generator only when the generators")
print("are mutually orthogonal as coefficient vectors, and one direction in")
print("total otherwise. Normal-form reduction removes the entire component in")
print("both cases.")
