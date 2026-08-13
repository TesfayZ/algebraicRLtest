"""A catalogue of physics faults, and the symbolic templates that diagnose them.

Every entry is a fault a real simulator or robot description has plausibly
carried: a mistyped length in an XML file, a mass ratio copied from the wrong
datasheet, a dropped term in a hand-derived equation of motion, an integrator
swapped for speed, a mis-ordered observation vector. Each is applied to a
reference environment whose exact invariants are known analytically, so what
*should* break is known before anything is measured.

The point of fixing a catalogue rather than perturbing at random is that the
diagnosis has a ground truth at two levels. A fault has a known *locus*, the
generator it must break, and where it is a parameter fault it also has a known
*value*, the number the diagnostic ought to recover. Detection alone is a weak
claim, and any two-sample test can make it; the catalogue is built so that
localisation and attribution can be scored too.

The categories are:

  parameter    a physical constant differs from the specification. The ideal
               changes but stays an ideal of the same shape, so the recovered
               coefficients should name the offending constant.
  structural   a term is missing from the equations of motion. The system stops
               conserving what it should; no parameter assignment explains it.
  numerical    the integrator or step size differs. The exact invariants are
               unchanged as algebra, and only their drift moves.
  observation  the state is packed into the observation wrongly. The constraint
               that breaks is a bookkeeping identity, not a physical law.

`nips_coriolis` is not hypothetical. It is a variant Gymnasium ships, and it
does not conserve energy, which is exactly the property this diagnostic exists
to check.
"""

import numpy as np
import sympy as sp

import envs

# --------------------------------------------------------- symbolic templates ---

ACRO_SYMS = sp.symbols("c1 s1 c2 s2 w1 w2", real=True)
C1, S1, C2, S2, W1, W2 = ACRO_SYMS

REACH_SYMS = sp.symbols("c1 c2 s1 s2 tx ty w1 w2 dx dy dz", real=True)
RC1, RC2, RS1, RS2, RTX, RTY, RW1, RW2, RDX, RDY, RDZ = REACH_SYMS

# Physical parameters carried symbolically, so a recovered coefficient can be
# solved back to the constant that produced it.
P_m1, P_m2, P_l1, P_lc1, P_lc2, P_I1, P_I2, P_g = sp.symbols(
    "m1 m2 l1 lc1 lc2 I1 I2 g", positive=True)
ACRO_PARAMS = (P_m1, P_m2, P_l1, P_lc1, P_lc2, P_I1, P_I2, P_g)

P_L1, P_L2 = sp.symbols("ell1 ell2", positive=True)
REACH_PARAMS = (P_L1, P_L2)


def acrobot_energy_template():
    """Total mechanical energy in the observation, with symbolic parameters.

    The same expression `envs.acrobot_energy_obs` evaluates numerically. Keeping
    one symbolic and one numeric copy is a duplication, so `verify_properties`
    checks that substituting the spec values into this reproduces the other to
    machine precision.
    """
    M11_const = P_m1 * P_lc1**2 + P_m2 * (P_l1**2 + P_lc2**2) + P_I1 + P_I2
    M11_c2 = 2 * P_m2 * P_l1 * P_lc2
    M12_const = P_m2 * P_lc2**2 + P_I2
    M12_c2 = P_m2 * P_l1 * P_lc2
    M22 = P_m2 * P_lc2**2 + P_I2

    T = (sp.Rational(1, 2) * (M11_const + M11_c2 * C2) * W1**2
         + (M12_const + M12_c2 * C2) * W1 * W2
         + sp.Rational(1, 2) * M22 * W2**2)
    V = (-(P_m1 * P_lc1 + P_m2 * P_l1) * P_g * C1
         - P_m2 * P_g * P_lc2 * (C1 * C2 - S1 * S2))
    return sp.expand(T + V)


def acrobot_spec(params=None):
    """Spec values of the symbolic parameters, as a dict keyed by symbol."""
    p = params or envs.ACROBOT
    return {P_m1: p["m1"], P_m2: p["m2"], P_l1: p["l1"], P_lc1: p["lc1"],
            P_lc2: p["lc2"], P_I1: p["I1"], P_I2: p["I2"], P_g: p["g"]}


def reacher_fk_template(axis="x"):
    """Forward-kinematics identity with the link lengths carried symbolically."""
    if axis == "x":
        return sp.expand(RDX + RTX - P_L1 * RC1
                         - P_L2 * (RC1 * RC2 - RS1 * RS2))
    return sp.expand(RDY + RTY - P_L1 * RS1 - P_L2 * (RS1 * RC2 + RC1 * RS2))


def reacher_spec(params=None, scale=1.0):
    p = params or envs.REACHER
    return {P_L1: p["l1"] / scale, P_L2: p["l2"] / scale}


# ----------------------------------------------- reference generating sets ---

def acrobot_reference():
    """(generators, names, kinds) for Acrobot, in the 6-d observation."""
    E = sp.expand(acrobot_energy_template().subs(acrobot_spec()))
    E = sp.nsimplify(E, rational=True)
    return ([C1**2 + S1**2 - 1, C2**2 + S2**2 - 1, E],
            ["unit-norm th1", "unit-norm th2", "energy"],
            ["vanishing", "vanishing", "conserved"])


def reacher_reference(scale=1.0):
    spec = reacher_spec(scale=scale)
    fk_x = sp.nsimplify(reacher_fk_template("x").subs(spec), rational=True)
    fk_y = sp.nsimplify(reacher_fk_template("y").subs(spec), rational=True)
    return ([RC1**2 + RS1**2 - 1, RC2**2 + RS2**2 - 1, RDZ, fk_x, fk_y],
            ["unit-norm th1", "unit-norm th2", "dz=0", "fk-x", "fk-y"],
            ["vanishing"] * 5)


# ------------------------------------------------------- data generation ---

def acrobot_trajectories(params=None, dt=0.01, n_traj=8, n_steps=500, seed=0,
                         integrator="rk4", obs_bug=None, torque=0.0,
                         ic_seed=None, noise=0.0, decimate=1):
    """List of (n_steps+1, 6) observation arrays from independent initial states.

    Several initial energies on purpose. A single trajectory would let any
    per-trajectory constant masquerade as a conserved quantity, which is the
    failure mode Section 3.2 of the paper describes.

    `ic_seed` separates the initial conditions from everything else. Passing the
    same `ic_seed` to a reference and a faulty system pairs them: the two start
    from identical states and any difference downstream is the fault rather than
    the draw. That is the regime the two-sample baselines are entitled to, and
    it is reported alongside the unpaired one instead of only the setting that
    flatters the method being proposed.

    `decimate` keeps every k-th observation. It exists so that a fault in the
    step size can be compared against a reference on the *same* sampling grid:
    integrating finely and then keeping every 40th sample gives a reference
    whose blocks span the same elapsed time and hold the same number of points
    as an arm run at dt = 0.2, which a residual computed by `np.std` is
    otherwise sensitive to on both counts.
    """
    p = dict(envs.ACROBOT)
    p.update(params or {})
    ic_rng = np.random.default_rng(seed if ic_seed is None else ic_seed)
    noise_rng = np.random.default_rng(seed + 90210)
    out = []
    for _ in range(n_traj):
        s0 = np.array([ic_rng.uniform(-1.0, 1.0), ic_rng.uniform(-1.0, 1.0),
                       ic_rng.uniform(-0.5, 0.5), ic_rng.uniform(-0.5, 0.5)])
        traj = envs.acrobot_rollout_integrator(s0, dt, n_steps, integrator,
                                               torque=torque, p=p)
        obs = envs.acrobot_obs(traj)
        if decimate > 1:
            obs = obs[::decimate]
        if obs_bug is not None:
            obs = obs_bug(obs)
        if noise > 0:
            obs = obs + noise_rng.normal(0.0, noise, obs.shape)
        out.append(obs)
    return out


def reacher_samples(params=None, N=4000, seed=0, scale=1.0, obs_bug=None,
                    ic_seed=None, noise=0.0):
    """A single (N, 11) block. Reacher's invariants are kinematic, not dynamic,
    so independent poses carry the same information as trajectories and cost
    less."""
    p = dict(envs.REACHER)
    p.update(params or {})
    rng = np.random.default_rng(seed if ic_seed is None else ic_seed)
    noise_rng = np.random.default_rng(seed + 90210)
    th1 = rng.uniform(-np.pi, np.pi, N)
    th2 = rng.uniform(-np.pi, np.pi, N)
    tx = rng.uniform(-0.2, 0.2, N)
    ty = rng.uniform(-0.2, 0.2, N)
    w1 = rng.uniform(-10, 10, N)
    w2 = rng.uniform(-10, 10, N)
    obs = envs.reacher_obs_from_state(th1, th2, tx, ty, w1, w2, p=p).copy()
    for j in (4, 5, 8, 9, 10):          # tx, ty, dx, dy, dz are lengths
        obs[:, j] /= scale
    if obs_bug is not None:
        obs = obs_bug(obs)
    if noise > 0:
        obs = obs + noise_rng.normal(0.0, noise, obs.shape)
    return [obs]


# --------------------------------------------------------- observation bugs ---

def swap_columns(i, j):
    def f(obs):
        out = obs.copy()
        out[:, [i, j]] = out[:, [j, i]]
        return out
    return f


def scale_column(i, factor):
    def f(obs):
        out = obs.copy()
        out[:, i] = out[:, i] * factor
        return out
    return f


def offset_column(i, delta):
    def f(obs):
        out = obs.copy()
        out[:, i] = out[:, i] + delta
        return out
    return f


# ------------------------------------------------------------- the catalogue ---

#: generators that carry no free parameter, so no fitted value can be read off
#: them and level-2 attribution does not apply. Scoring a refusal here would be
#: scoring the diagnostic on a question it was never asked.
UNPARAMETERISED = {"unit-norm th1", "unit-norm th2", "dz=0"}


class Bug:
    """One injected fault, with the reference arm it must be compared against.

    `ref_kwargs` is how a fault that changes the *sampling* rather than the
    physics declares a matched reference. A residual is computed over a window
    of a trajectory, so an arm sampled at a different step size differs from the
    reference in the number of points per block and the elapsed time each block
    spans, both of which move the residual on their own. Where that applies the
    fault carries the reference settings that equalise them, and the comparison
    isolates the step size.
    """

    def __init__(self, name, system, category, description, expect_broken,
                 expect_param=None, expect_value=None, ref_kwargs=None,
                 **kwargs):
        self.name = name
        self.system = system              # "acrobot" | "reacher"
        self.category = category
        self.description = description
        self.expect_broken = expect_broken     # generator names that must break
        self.expect_param = expect_param       # parameter the fault lives in
        self.expect_value = expect_value       # value it was set to
        self.ref_kwargs = ref_kwargs or {}     # reference-arm overrides
        self.kwargs = kwargs

    @property
    def attributable(self):
        """Does any broken generator carry parameters to attribute a fault to?"""
        return bool(set(self.expect_broken) - UNPARAMETERISED)

    def trajectories(self, seed=0, **overrides):
        kw = dict(self.kwargs)
        kw.update(overrides)
        if self.system == "acrobot":
            return acrobot_trajectories(seed=seed, **kw)
        return reacher_samples(seed=seed, **kw)


REFERENCE_ACROBOT = dict(params=None, dt=0.01, integrator="rk4", obs_bug=None)
REFERENCE_REACHER = dict(params=None, scale=envs.REACHER["l1"], obs_bug=None)


def catalogue(scale=None):
    """The full fault list. `scale` is Reacher's nondimensionalising length."""
    scale = scale if scale is not None else envs.REACHER["l1"]
    R = dict(scale=scale)

    return [
        # ---- parameter faults: the ideal moves, and the coefficients say how
        Bug("acrobot_m2_1.3", "acrobot", "parameter",
            "second link mass 1.3 instead of 1.0",
            ["energy"], expect_param="m2", expect_value=1.3,
            params=dict(m2=1.3)),
        Bug("acrobot_l1_1.2", "acrobot", "parameter",
            "first link length 1.2 instead of 1.0",
            ["energy"], expect_param="l1", expect_value=1.2,
            params=dict(l1=1.2)),
        Bug("acrobot_lc2_0.6", "acrobot", "parameter",
            "second centre of mass at 0.6 instead of 0.5",
            ["energy"], expect_param="lc2", expect_value=0.6,
            params=dict(lc2=0.6)),
        Bug("acrobot_I2_1.5", "acrobot", "parameter",
            "second link inertia 1.5 instead of 1.0",
            ["energy"], expect_param="I2", expect_value=1.5,
            params=dict(I2=1.5)),
        Bug("acrobot_g_9.81", "acrobot", "parameter",
            "gravity 9.81 instead of 9.8, a plausible unit-of-care difference",
            ["energy"], expect_param="g", expect_value=9.81,
            params=dict(g=9.81)),
        Bug("reacher_l2_0.13", "reacher", "parameter",
            "second link length 0.13 instead of 0.11 in the XML",
            ["fk-x", "fk-y"], expect_param="ell2", expect_value=0.13 / 0.1,
            params=dict(l2=0.13), **R),
        Bug("reacher_l1_0.09", "reacher", "parameter",
            "first link length 0.09 instead of 0.10 in the XML",
            ["fk-x", "fk-y"], expect_param="ell1", expect_value=0.9,
            params=dict(l1=0.09), **R),

        # ---- structural faults: no parameter assignment can explain them
        Bug("acrobot_nips_coriolis", "acrobot", "structural",
            "the nips variant, which drops a Coriolis term and stops "
            "conserving energy",
            ["energy"], params=dict(variant="nips")),
        Bug("acrobot_torque_leak", "acrobot", "structural",
            "a constant unmodelled torque of 0.05 on the second joint",
            ["energy"], torque=0.05),

        # ---- numerical faults: the algebra is untouched, the drift is not
        Bug("acrobot_euler", "acrobot", "numerical",
            "explicit Euler substituted for RK4",
            ["energy"], integrator="euler"),
        # Sampled on a common 0.2 s grid over a common 10 s window: the
        # reference is integrated at dt = 0.005 and decimated by 40, so both
        # arms carry 51 points per trajectory and every block spans the same
        # elapsed time. What is left between them is the step size.
        Bug("acrobot_dt_0.2", "acrobot", "numerical",
            "the shipped timestep of 0.2 instead of a resolved 0.005",
            ["energy"], dt=0.2, n_steps=50,
            ref_kwargs=dict(dt=0.005, n_steps=2000, decimate=40)),

        # ---- observation faults: bookkeeping, not physics
        # Scrambling the observation breaks the energy too, since the energy is
        # a polynomial *in the observation*: permute two of its arguments and
        # the quantity it computes is no longer the one that is conserved.
        Bug("acrobot_obs_swap", "acrobot", "observation",
            "sin(th1) and cos(th2) transposed in the observation vector",
            ["unit-norm th1", "unit-norm th2", "energy"],
            obs_bug=swap_columns(1, 2)),
        Bug("acrobot_obs_unnormalised", "acrobot", "observation",
            "the first angle's cosine scaled by 1.02",
            ["unit-norm th1", "energy"], obs_bug=scale_column(0, 1.02)),
        Bug("reacher_dz_offset", "reacher", "observation",
            "a 1e-3 z offset leaks into the fingertip-to-target vector",
            ["dz=0"], obs_bug=offset_column(10, 1e-3), **R),
        Bug("reacher_obs_swap", "reacher", "observation",
            "cos(th2) and sin(th1) transposed in the observation vector",
            ["unit-norm th1", "unit-norm th2", "fk-x", "fk-y"],
            obs_bug=swap_columns(1, 2), **R),

        # ---- the negative control: no fault at all
        Bug("none_acrobot", "acrobot", "control",
            "reference dynamics, reseeded; nothing should fire", []),
        Bug("none_reacher", "reacher", "control",
            "reference kinematics, reseeded; nothing should fire", [], **R),
    ]