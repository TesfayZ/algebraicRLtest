"""Deflation ablation on real Acrobot trajectory data.

Tests the paper's headline claim on numerically sampled trajectories rather
than on the exact ideal: at degree 3 the raw nullspace is dominated by trivial
multiples of the two unit-norm constraints, and Groebner deflation removes
exactly those while orthogonal projection removes one direction per generator.

The experiment sweeps both the integrator step dt and the nullspace tolerance,
because the two interact. The trivial multiples vanish to machine precision at
every dt, being exact algebra. The energy direction vanishes only to the
integrator's drift floor, so admitting it requires a tolerance above that floor,
and a tolerance that loose also admits spurious directions. Deflation is what
makes the loose tolerance usable: it removes the 14 algebraically trivial
directions first, so what survives at a loose tolerance is a short list that can
actually be searched.

Writes Results/deflation_ablation.csv and Results/deflation_pooled.csv.
"""

import os
import csv
import numpy as np
import sympy as sp

import envs
import deflation
import discover

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
os.makedirs(RESULTS, exist_ok=True)

VARS = envs.ACROBOT_VARS
sym = list(sp.symbols(" ".join(VARS), real=True))
c1, s1, c2, s2, w1, w2 = sym
G_TRUE = [c1**2 + s1**2 - 1, c2**2 + s2**2 - 1]

E_TRUE = sp.expand(
    sp.Rational(7, 4) * w1**2 + sp.Rational(1, 2) * c2 * w1**2
    + sp.Rational(5, 4) * w1 * w2 + sp.Rational(1, 2) * c2 * w1 * w2
    + sp.Rational(5, 8) * w2**2
    - sp.Rational(147, 10) * c1 - sp.Rational(49, 10) * c1 * c2
    + sp.Rational(49, 10) * s1 * s2)

S0 = np.array([0.8, 0.4, 0.0, 0.0])
HORIZON = 40.0


def nullspace_at(Phi, tol_rel):
    """Nullspace basis expressed in the *monomial* basis.

    The SVD is taken on the column-normalised matrix, which is what makes the
    singular-value threshold meaningful across dictionaries of mixed scale, but
    its right singular vectors are coordinates against the normalised columns.
    They have to be divided back by the column norms before anything algebraic
    (normal-form reduction, comparison against a known polynomial) touches them.
    """
    colnorm = np.linalg.norm(Phi, axis=0)
    colnorm[colnorm == 0] = 1.0
    Phin = Phi / colnorm
    U, s, Vt = np.linalg.svd(Phin, full_matrices=False)
    n_null = int(np.sum(s <= tol_rel * s[0]))
    if n_null == 0:
        return np.zeros((0, Phi.shape[1])), s
    V = Vt[len(s) - n_null:] / colnorm          # back to monomial coordinates
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    # re-orthonormalise: the unscaling destroys orthogonality
    Q, _ = np.linalg.qr(V.T)
    return Q.T[:n_null], s


def energy_alignment(V, monos):
    """How much of span{E, 1} is captured by V (0..1).

    The recovered conserved quantity is E - E_0, not E: the trajectory's energy
    level enters as a constant term. Comparing against E alone caps the
    achievable alignment at |E| / |E - E_0|, which on this trajectory is 0.807,
    so the target subspace has to include the constant monomial.
    """
    if V.shape[0] == 0:
        return 0.0
    e = deflation.coeff_vector(E_TRUE, monos, sym)
    one = deflation.coeff_vector(sp.Integer(1), monos, sym)
    B = np.stack([e, one])
    Q, _ = np.linalg.qr(B.T)          # orthonormal basis of span{E, 1}
    Q = Q.T
    # fraction of the E direction (within span{E,1}) recovered by V
    e_in = Q @ e
    e_in = e_in / np.linalg.norm(e_in)
    e_full = Q.T @ e_in
    proj = V.T @ (V @ e_full)
    return float(np.linalg.norm(proj))


def energy_residual(V, monos):
    """Residual of V's span outside span{E, 1}; 0 means V is exactly the energy."""
    if V.shape[0] != 1:
        return float("nan")
    e = deflation.coeff_vector(E_TRUE, monos, sym)
    one = deflation.coeff_vector(sp.Integer(1), monos, sym)
    Q, _ = np.linalg.qr(np.stack([e, one]).T)
    v = V[0] / np.linalg.norm(V[0])
    return float(np.linalg.norm(v - Q @ (Q.T @ v)))


def main():
    monos = deflation.monomials(sym, 3)
    M = len(monos)
    rows = []

    print("=" * 100)
    print("Deflation on a single-energy Acrobot trajectory, degree 3, "
          f"n=6, M={M}")
    print("=" * 100)
    print(f"{'dt':>7} {'tol':>8} {'raw':>5} {'proj':>6} {'GB':>4} "
          f"{'|P_V E|':>9}  {'|P_GB E|':>9} {'resid':>10}  note")

    for dt in (0.2, 0.05, 0.01, 0.002):
        n = int(round(HORIZON / dt))
        traj = envs.acrobot_rollout(S0, dt, n)
        E = np.array([envs.acrobot_energy_coords(s) for s in traj])
        rel_drift = float(np.max(np.abs(E - E[0])) / abs(E[0]))
        data = envs.acrobot_obs(traj)
        Phi = discover._eval_monomials(data, sym, monos)

        for tol in (1e-10, 1e-6, 1e-4, 1e-3):
            V, svals = nullspace_at(Phi, tol)
            raw = V.shape[0]
            if raw == 0:
                continue

            Vg, rm_gb = discover.deflate_nullspace(V, monos, G_TRUE, sym)
            Vp = V.copy()
            rm_proj = 0
            for g in G_TRUE:
                gv = deflation.coeff_vector(g, monos, sym)
                Vp, r = discover.project_deflate(Vp, gv)
                rm_proj += r

            align_raw = energy_alignment(V, monos)
            align_gb = energy_alignment(Vg, monos)
            resid = energy_residual(Vg, monos)
            if Vg.shape[0] == 1 and resid < 1e-6:
                note = "energy isolated exactly"
            elif Vg.shape[0] == 0:
                note = "energy below tolerance, invisible"
            elif align_gb > 0.99:
                note = f"energy present among {Vg.shape[0]} directions"
            else:
                note = "spurious directions admitted"

            print(f"{dt:>7} {tol:>8.0e} {raw:>5} {Vp.shape[0]:>6} "
                  f"{Vg.shape[0]:>4} {align_raw:>9.4f}  {align_gb:>9.4f} "
                  f"{resid:>10.2e}  {note}")

            rows.append(dict(
                dt=dt, rel_energy_drift=rel_drift, tol=tol, M=M, N=n + 1,
                raw_nullspace=raw,
                after_projection=Vp.shape[0], removed_projection=rm_proj,
                after_groebner=Vg.shape[0], removed_groebner=rm_gb,
                energy_alignment_raw=align_raw,
                energy_alignment_after_groebner=align_gb,
                energy_residual=resid,
                note=note))
        print()

    out = os.path.join(RESULTS, "deflation_ablation.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")

    # ---------------- pooled across energies: the Section 3.2 claim ----------
    print("\n" + "=" * 100)
    print("Pooled across 40 trajectories at different energies "
          "(Section 3.2: the energy invariant should vanish from the ideal)")
    print("=" * 100)
    rng = np.random.default_rng(0)
    chunks = []
    for _ in range(40):
        s0 = np.concatenate([rng.uniform(-np.pi, np.pi, 2),
                             rng.uniform(-1.5, 1.5, 2)])
        chunks.append(envs.acrobot_obs(envs.acrobot_rollout(s0, 0.01, 200)))
    pooled = np.concatenate(chunks, axis=0)
    Phi = discover._eval_monomials(pooled, sym, monos)
    prows = []
    for tol in (1e-10, 1e-6, 1e-4, 1e-3):
        V, _ = nullspace_at(Phi, tol)
        Vg, _ = discover.deflate_nullspace(V, monos, G_TRUE, sym) \
            if V.shape[0] else (V, 0)
        a = energy_alignment(V, monos)
        print(f"  tol={tol:.0e}  raw={V.shape[0]:>3}  after GB deflation="
              f"{Vg.shape[0]:>3}   |P_V E|={a:.4f}")
        prows.append(dict(tol=tol, N=pooled.shape[0], raw_nullspace=V.shape[0],
                          after_groebner=Vg.shape[0], energy_alignment=a))
    out2 = os.path.join(RESULTS, "deflation_pooled.csv")
    with open(out2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prows[0].keys()))
        w.writeheader()
        w.writerows(prows)
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
