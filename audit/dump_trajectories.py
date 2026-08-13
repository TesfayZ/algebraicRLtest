"""Dump paired trajectories from a shipped simulator release.

Runs *inside* a per-version virtualenv, so it may import nothing beyond numpy
and whichever of `gym` / `gymnasium` that venv holds. Everything algebraic
happens later, in the project venv, against the .npz this writes.

The design point is pairing. Initial states are set directly rather than drawn
from `reset()`, and the action sequence is fixed, so two releases receive
byte-identical inputs. Any difference in the output is then the code and not
the draw, which is the confound that makes a two-sample test useless for this
question (Section 6.2 of the paper). It also means an *exact* comparison is
meaningful: if two releases agree bitwise, no code change touched the dynamics,
and a null result from the screen is a certainty rather than a failure to
reject.
"""
import argparse
import hashlib
import json
import sys

import numpy as np


def load_gym():
    """Return (module, name, version). Gymnasium is preferred when both exist."""
    try:
        import gymnasium as g
        return g, "gymnasium", g.__version__
    except ImportError:
        pass
    import gym as g
    return g, "gym", g.__version__


def make(mod, env_id):
    """`make` across the API break, without triggering deprecation shims.

    gym >= 0.26 wraps everything in OrderEnforcing + PassiveEnvChecker, and the
    checker rejects the direct state assignment below. `disable_env_checker`
    does not exist before 0.24, hence the fallback.
    """
    try:
        return mod.make(env_id, disable_env_checker=True)
    except TypeError:
        return mod.make(env_id)


def reset(env, seed):
    """reset() returns obs before gym 0.26 and (obs, info) after."""
    try:
        out = env.reset(seed=seed)
    except TypeError:
        env.seed(seed)
        out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def step(env, action):
    """step() returns 4 items before gym 0.26 and 5 after."""
    out = env.step(action)
    if len(out) == 5:
        obs, rew, term, trunc, info = out
        return obs, rew, bool(term or trunc)
    obs, rew, done, info = out
    return obs, rew, bool(done)


SPECS = {
    # Acrobot's observation is already the paper's Acrobot coordinate system,
    # (c1, s1, c2, s2, w1, w2), so no lifting is needed. Action 1 is zero
    # torque: the passive system is the one whose energy is conserved, and the
    # actuated case would need the balance law of Section 3.3 instead.
    "Acrobot-v1": dict(
        n_state=4,
        action=1,
        ic_low=np.array([-1.0, -1.0, -0.5, -0.5]),
        ic_high=np.array([1.0, 1.0, 0.5, 0.5]),
        attrs=["dt", "LINK_LENGTH_1", "LINK_LENGTH_2", "LINK_MASS_1",
               "LINK_MASS_2", "LINK_COM_POS_1", "LINK_COM_POS_2",
               "LINK_MOI", "MAX_VEL_1", "MAX_VEL_2", "book_or_nips",
               "torque_noise_max", "AVAIL_TORQUE"],
    ),
    # Pendulum observation is (cos, sin, w). Zero torque again.
    "Pendulum-v1": dict(
        n_state=2,
        action=np.array([0.0], dtype=np.float32),
        ic_low=np.array([-np.pi, -1.0]),
        ic_high=np.array([np.pi, 1.0]),
        attrs=["dt", "g", "m", "l", "max_speed", "max_torque"],
    ),
    # The three environments below carry no exact polynomial invariant in the
    # shipped observation: CartPole's action set has no zero element, so the
    # cart is driven every step, and both MountainCar variants sit in a
    # cosine potential that is transcendental in the position coordinate. They
    # are audited by the exact stage alone, which needs no reference set, and
    # they are here because that stage is what produces the null: a release
    # pair that agrees bitwise has not touched the dynamics whether or not an
    # invariant exists to screen against.
    "CartPole-v1": dict(
        n_state=4,
        action=0,
        ic_low=np.array([-0.05, -0.05, -0.05, -0.05]),
        ic_high=np.array([0.05, 0.05, 0.05, 0.05]),
        attrs=["tau", "gravity", "masscart", "masspole", "total_mass",
               "length", "polemass_length", "force_mag",
               "kinematics_integrator", "theta_threshold_radians",
               "x_threshold"],
    ),
    "MountainCar-v0": dict(
        n_state=2,
        action=1,                       # 1 is coast; 0 and 2 push
        ic_low=np.array([-0.6, -0.01]),
        ic_high=np.array([-0.4, 0.01]),
        attrs=["min_position", "max_position", "max_speed", "goal_position",
               "force", "gravity"],
    ),
    "MountainCarContinuous-v0": dict(
        n_state=2,
        action=np.array([0.0], dtype=np.float32),
        ic_low=np.array([-0.6, -0.01]),
        ic_high=np.array([-0.4, 0.01]),
        attrs=["min_position", "max_position", "max_speed", "goal_position",
               "power"],
    ),
}

# Reacher is the better audit target of the two families. Its invariants are
# forward kinematics and two unit-norm identities, all purely kinematic, so
# they carry no dependence on the integrator and there is no drift floor for a
# fault to hide behind the way there is for Acrobot's energy at dt = 0.2.
#
# The v4 observation is already the paper's Reacher coordinate system,
# (c1, c2, s1, s2, tx, ty, w1, w2, dx, dy, dz). v5 is 10-dimensional because it
# drops the identically-zero dz component, which is a documented breaking change
# and therefore a useful thing for the audit to rediscover on its own.
#
# State is (qpos, qvel) = ((th1, th2, target_x, target_y), (w1, w2, 0, 0)); the
# target is static, so its velocity entries stay zero. Targets are drawn inside
# the radius-0.2 disc the environment itself samples from.
_REACHER = dict(
    n_state=6,
    nq=4, nv=2,
    action=np.zeros(2, dtype=np.float32),
    ic_low=np.array([-np.pi, -np.pi, -0.15, -0.15, -1.0, -1.0]),
    ic_high=np.array([np.pi, np.pi, 0.15, 0.15, 1.0, 1.0]),
    attrs=["dt", "frame_skip", "model_path"],
    setter="mujoco",
)
SPECS["Reacher-v4"] = dict(_REACHER)
SPECS["Reacher-v5"] = dict(_REACHER)

# InvertedPendulum is (slider x, hinge theta) with both velocities free, and
# carries no exact polynomial invariant in its four-dimensional observation:
# the cart is unactuated here but the pole's potential is a cosine of an angle
# the observation reports raw. Exact stage only, as for the classic-control
# additions above.
SPECS["InvertedPendulum-v4"] = dict(
    n_state=4, nq=2, nv=2,
    action=np.zeros(1, dtype=np.float32),
    ic_low=np.array([-0.05, -0.05, -0.05, -0.05]),
    ic_high=np.array([0.05, 0.05, 0.05, 0.05]),
    attrs=["dt", "frame_skip", "model_path"],
    setter="mujoco",
)
SPECS["InvertedPendulum-v5"] = dict(SPECS["InvertedPendulum-v4"])

# InvertedDoublePendulum reports both hinge angles as (sin, cos) pairs, so it
# carries two exact unit-norm identities of the same kind Acrobot and Reacher
# do, and is the one addition that exercises the screen rather than only the
# exact stage. State is (x, th1, th2) with the three matching velocities.
SPECS["InvertedDoublePendulum-v4"] = dict(
    n_state=6, nq=3, nv=3,
    action=np.zeros(1, dtype=np.float32),
    ic_low=np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1]),
    ic_high=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    attrs=["dt", "frame_skip", "model_path"],
    setter="mujoco",
)
SPECS["InvertedDoublePendulum-v5"] = dict(SPECS["InvertedDoublePendulum-v4"])


def set_state(env, spec, s0):
    """Impose an initial state, whichever state representation the family uses.

    For the MuJoCo family the first `nq` entries of `s0` are the free
    generalised coordinates and the next `nv` are their velocities; anything
    the model carries beyond that (Reacher's static target velocity, for
    instance) is zeroed rather than left at whatever `reset` produced.
    """
    if spec.get("setter") == "mujoco":
        u = env.unwrapped
        nq, nv = spec["nq"], spec["nv"]
        qpos = np.zeros(u.model.nq, dtype=np.float64)
        qvel = np.zeros(u.model.nv, dtype=np.float64)
        qpos[:nq] = s0[:nq]
        qvel[:nv] = s0[nq:nq + nv]
        u.set_state(qpos, qvel)
    else:
        env.unwrapped.state = np.array(s0, dtype=np.float64)


def env_metadata(env, attrs):
    """Constants and source fingerprint of the unwrapped environment class.

    The hash of the defining source file separates "this release changed the
    dynamics" from "this release changed something else in the package", which
    is the first question triage asks.
    """
    u = env.unwrapped
    meta = {}
    for a in attrs:
        if hasattr(u, a):
            v = getattr(u, a)
            meta[a] = v.tolist() if isinstance(v, np.ndarray) else v
    try:
        import inspect
        src = inspect.getsource(type(u))
        meta["_source_sha256"] = hashlib.sha256(src.encode()).hexdigest()
        meta["_source_lines"] = len(src.splitlines())
    except Exception as exc:                                  # pragma: no cover
        meta["_source_sha256"] = f"unavailable: {exc!r}"
    return meta


def mutate(env, spec_str):
    """Apply `attr=value` overrides to the unwrapped environment.

    This exists for positive controls. An audit that reports "identical" for
    every pair is worthless unless the same pipeline is shown to separate a
    difference that is known to be there, so the controls perturb the *shipped*
    class rather than a reimplementation of it. `--mutate book_or_nips=nips`
    selects the variant that drops a Coriolis term and stops conserving energy,
    which is a real alternative Gymnasium ships rather than an injected bug.
    """
    for item in spec_str.split(","):
        if not item.strip():
            continue
        attr, _, raw = item.partition("=")
        attr = attr.strip()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw.strip()
        if not hasattr(env.unwrapped, attr):
            raise SystemExit(f"no attribute {attr!r} on {type(env.unwrapped).__name__}")
        setattr(env.unwrapped, attr, value)


def rollout(env, spec, s0, n_steps):
    """One trajectory from an imposed initial state under a fixed action."""
    reset(env, seed=0)
    set_state(env, spec, s0)
    obs = []
    for _ in range(n_steps):
        o, _, _ = step(env, spec["action"])
        obs.append(np.asarray(o, dtype=np.float64))
    return np.array(obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", required=True, choices=sorted(SPECS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-traj", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=500)
    ap.add_argument("--ic-seed", type=int, default=12345)
    ap.add_argument("--mutate", default="",
                    help="positive control, e.g. 'book_or_nips=nips' or 'g=9.81'")
    args = ap.parse_args()

    mod, mod_name, mod_version = load_gym()
    spec = SPECS[args.env_id]

    # Initial states come from a fixed RNG that lives here, not in the
    # environment, so every release under test starts from the same states
    # regardless of how its own seeding was reorganised.
    rng = np.random.default_rng(args.ic_seed)
    s0s = rng.uniform(spec["ic_low"], spec["ic_high"],
                      size=(args.n_traj, spec["n_state"]))

    env = make(mod, args.env_id)
    if args.mutate:
        mutate(env, args.mutate)
    meta = env_metadata(env, spec["attrs"])
    trajs = [rollout(env, spec, s0, args.n_steps) for s0 in s0s]
    env.close()

    meta.update(mutation=args.mutate, package=mod_name, version=mod_version,
                env_id=args.env_id,
                python=sys.version.split()[0], numpy=np.__version__,
                n_traj=args.n_traj, n_steps=args.n_steps, ic_seed=args.ic_seed)
    np.savez_compressed(args.out, initial_states=s0s,
                        meta=json.dumps(meta),
                        **{f"traj{i}": t for i, t in enumerate(trajs)})
    print(json.dumps({k: v for k, v in meta.items()
                      if not k.startswith("_")}, sort_keys=True))


if __name__ == "__main__":
    main()
