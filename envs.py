"""Environment models and ground-truth invariants for the Tier A/B/C suite.

Everything here is analytic: the dynamics are reimplemented so that trajectories
can be generated at arbitrary timestep and with a chosen integrator, which the
Gymnasium envs do not expose. Where a Gymnasium env exists we cross-check
against it (see `verify_properties.py`).

Acrobot follows the Sutton-Barto book model as implemented in
gymnasium/envs/classic_control/acrobot.py, with the standard parameters
m1=m2=1, l1=l2=1, lc1=lc2=0.5, I1=I2=1, g=9.8, dt=0.2, torque in {-1,0,1}.
"""

import numpy as np

# ---------------------------------------------------------------- Acrobot ---

ACROBOT = dict(m1=1.0, m2=1.0, l1=1.0, l2=1.0, lc1=0.5, lc2=0.5,
               I1=1.0, I2=1.0, g=9.8, dt=0.2, variant="book",
               b1=0.0, b2=0.0)

MAX_VEL_1 = 4 * np.pi
MAX_VEL_2 = 9 * np.pi


def acrobot_dsdt(s_augmented, p=ACROBOT):
    """Book-model acrobot derivative. State (th1, th2, dth1, dth2), last entry torque."""
    m1, m2 = p["m1"], p["m2"]
    l1, lc1, lc2 = p["l1"], p["lc1"], p["lc2"]
    I1, I2, g = p["I1"], p["I2"], p["g"]
    a = s_augmented[-1]
    s = s_augmented[:-1]
    theta1, theta2, dtheta1, dtheta2 = s
    d1 = (m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * np.cos(theta2))
          + I1 + I2)
    d2 = m2 * (lc2**2 + l1 * lc2 * np.cos(theta2)) + I2
    phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.0)
    phi1 = (-m2 * l1 * lc2 * dtheta2**2 * np.sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2)
            + phi2)
    # The Gymnasium default is book_or_nips="book", which carries the
    # -m2*l1*lc2*dtheta1^2*sin(theta2) Coriolis term in the second equation.
    # The "nips" variant drops it and is NOT energy conserving; using it makes
    # passive energy drift plateau at ~1e-4 regardless of dt, which looks like
    # an integrator failure but is a model inconsistency. bugs.py uses the
    # switch deliberately, as a structural fault with a known ground truth.
    coriolis = 0.0 if p.get("variant", "book") == "nips" else \
        m2 * l1 * lc2 * dtheta1**2 * np.sin(theta2)
    # Viscous joint damping enters the two equations of motion as generalised
    # forces -b_i * dtheta_i. The first joint carries no actuator, so its
    # damping joins phi1 on the left; the second joint's subtracts from the
    # commanded torque. With b1 = b2 = 0 this is the passive book model and
    # energy is conserved exactly; with either nonzero the conserved quantity is
    # replaced by the power balance of Section 3.4.
    b1, b2 = p.get("b1", 0.0), p.get("b2", 0.0)
    phi1 = phi1 + b1 * dtheta1
    a = a - b2 * dtheta2
    ddtheta2 = ((a + d2 / d1 * phi1 - coriolis - phi2)
                / (m2 * lc2**2 + I2 - d2**2 / d1))
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return np.array([dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0])


def acrobot_energy_coords(state, p=ACROBOT):
    """Total mechanical energy T+V in (th1, th2, dth1, dth2) coordinates.

    Potential zero is the hanging-down configuration of the pivot; heights are
    measured with y = -l*cos(theta) as in the book model, where theta is
    measured from the downward vertical.
    """
    m1, m2 = p["m1"], p["m2"]
    l1, lc1, lc2 = p["l1"], p["lc1"], p["lc2"]
    I1, I2, g = p["I1"], p["I2"], p["g"]
    th1, th2, d1, d2 = state[0], state[1], state[2], state[3]

    # Kinetic energy of the two-link chain.
    M11 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * np.cos(th2)) + I1 + I2
    M12 = m2 * (lc2**2 + l1 * lc2 * np.cos(th2)) + I2
    M22 = m2 * lc2**2 + I2
    T = 0.5 * (M11 * d1**2 + 2 * M12 * d1 * d2 + M22 * d2**2)

    # Potential energy, angles from the downward vertical.
    V = (-m1 * g * lc1 * np.cos(th1)
         - m2 * g * (l1 * np.cos(th1) + lc2 * np.cos(th1 + th2)))
    return T + V


def acrobot_energy_obs(obs, p=ACROBOT):
    """The same energy written as a polynomial in the 6-d Gymnasium observation.

    obs = (c1, s1, c2, s2, w1, w2) with ci=cos(thi), si=sin(thi).
    Derived by substituting cos(th1+th2) = c1c2 - s1s2 into the coordinate form.
    """
    c1, s1, c2, s2, w1, w2 = (obs[..., i] for i in range(6))
    m1, m2 = p["m1"], p["m2"]
    l1, lc1, lc2 = p["l1"], p["lc1"], p["lc2"]
    I1, I2, g = p["I1"], p["I2"], p["g"]

    M11_const = m1 * lc1**2 + m2 * (l1**2 + lc2**2) + I1 + I2
    M11_c2 = 2 * m2 * l1 * lc2
    M12_const = m2 * lc2**2 + I2
    M12_c2 = m2 * l1 * lc2
    M22 = m2 * lc2**2 + I2

    T = (0.5 * (M11_const + M11_c2 * c2) * w1**2
         + (M12_const + M12_c2 * c2) * w1 * w2
         + 0.5 * M22 * w2**2)
    V = (-(m1 * lc1 + m2 * l1) * g * c1
         - m2 * g * lc2 * (c1 * c2 - s1 * s2))
    return T + V


def acrobot_obs(state):
    th1, th2, d1, d2 = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
    return np.stack([np.cos(th1), np.sin(th1), np.cos(th2), np.sin(th2), d1, d2],
                    axis=-1)


# ------------------------------------------------------------- integrators ---

def rk4(deriv, y0, t, *args):
    """Fixed-step RK4 over the time grid `t`, matching gymnasium.utils rk4."""
    yout = np.zeros((len(t), len(y0)), dtype=float)
    yout[0] = y0
    for i in range(len(t) - 1):
        this, dt = t[i], t[i + 1] - t[i]
        y0 = yout[i]
        k1 = np.asarray(deriv(y0, this, *args))
        k2 = np.asarray(deriv(y0 + dt / 2.0 * k1, this + dt / 2.0, *args))
        k3 = np.asarray(deriv(y0 + dt / 2.0 * k2, this + dt / 2.0, *args))
        k4 = np.asarray(deriv(y0 + dt * k3, this + dt, *args))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout


def acrobot_rollout(s0, dt, n_steps, torque=0.0, p=ACROBOT):
    """Passive (torque=0) or constant-torque rollout, RK4, no velocity clipping.

    Velocity clipping is deliberately omitted: Gymnasium clips omega to
    +-4pi/+-9pi, which is a non-physical projection that destroys energy
    conservation exactly. Trajectories here stay inside the clip range.
    """
    def deriv(y, t):
        return acrobot_dsdt(np.append(y, torque), p)[:4]

    traj = np.zeros((n_steps + 1, 4))
    traj[0] = s0
    for i in range(n_steps):
        out = rk4(deriv, traj[i], [0.0, dt])
        traj[i + 1] = out[-1]
    return traj


def acrobot_rollout_integrator(s0, dt, n_steps, method, torque=0.0, p=ACROBOT):
    """Roll out with a named integrator, for the H4 drift diagnostic.

    Symplectic Euler and velocity Verlet are applied in the (q, qdot) splitting;
    for a mass matrix depending on q this is semi-implicit rather than exactly
    symplectic, which is the standard practical variant and is what the
    backward-error ordering in the paper refers to.
    """
    def accel(q, qd, tq):
        y = np.array([q[0], q[1], qd[0], qd[1], tq])
        d = acrobot_dsdt(y, p)
        return d[2:4]

    traj = np.zeros((n_steps + 1, 4))
    traj[0] = s0
    for i in range(n_steps):
        q = traj[i][:2].copy()
        qd = traj[i][2:].copy()
        if method == "euler":
            a = accel(q, qd, torque)
            traj[i + 1] = np.concatenate([q + dt * qd, qd + dt * a])
        elif method == "symplectic_euler":
            a = accel(q, qd, torque)
            qd_new = qd + dt * a
            traj[i + 1] = np.concatenate([q + dt * qd_new, qd_new])
        elif method == "verlet":
            a = accel(q, qd, torque)
            q_new = q + dt * qd + 0.5 * dt**2 * a
            a_new = accel(q_new, qd + dt * a, torque)
            qd_new = qd + 0.5 * dt * (a + a_new)
            traj[i + 1] = np.concatenate([q_new, qd_new])
        elif method == "rk4":
            def deriv(y, t):
                return acrobot_dsdt(np.append(y, torque), p)[:4]
            traj[i + 1] = rk4(deriv, traj[i], [0.0, dt])[-1]
        else:
            raise ValueError(method)
    return traj


# --------------------------------------------------------------- Reacher ---

REACHER = dict(l1=0.1, l2=0.11)


def reacher_obs_from_state(th1, th2, tx, ty, w1, w2, p=REACHER):
    """Gymnasium Reacher-v4 11-d observation, built analytically.

    (cos th1, cos th2, sin th1, sin th2, target_x, target_y, w1, w2, dx, dy, dz)
    where (dx, dy, dz) = fingertip - target.

    v5 drops the identically-zero `dz` and is 10-wide, keeping the first ten
    entries in this order. Nothing here targets v5; the release audit builds its
    v5 reference set separately.
    """
    l1, l2 = p["l1"], p["l2"]
    fx = l1 * np.cos(th1) + l2 * np.cos(th1 + th2)
    fy = l1 * np.sin(th1) + l2 * np.sin(th1 + th2)
    return np.stack([np.cos(th1), np.cos(th2), np.sin(th1), np.sin(th2),
                     tx, ty, w1, w2, fx - tx, fy - ty, np.zeros_like(fx)],
                    axis=-1)


REACHER_VARS = ["c1", "c2", "s1", "s2", "tx", "ty", "w1", "w2", "dx", "dy", "dz"]
ACROBOT_VARS = ["c1", "s1", "c2", "s2", "w1", "w2"]
