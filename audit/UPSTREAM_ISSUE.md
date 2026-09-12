# Draft issue for Farama-Foundation/Gymnasium

Filed as Farama-Foundation/Gymnasium#1690: https://github.com/Farama-Foundation/Gymnasium/issues/1690.
Reproducer below runs against a stock install and needs nothing from this
repository.

---

**Title:** Reacher observation mixes post-step `qpos` with pre-integration
`xipos`, so the forward-kinematics identity fails by ~1e-8

**Body:**

### Summary

`ReacherEnv._get_obs` builds a single observation vector out of two quantities
that are not read at the same point of the integration step. The joint angles
come from `data.qpos`, which `mj_step` has already integrated, while the
fingertip-to-target vector comes from `get_body_com(...)`, that is `data.xipos`,
which still holds the kinematics of an internal integrator stage because
`mj_forward` is not called after `mj_step`.

The consequence is that the observation does not satisfy its own forward
kinematics. Writing `l1 = 0.1` and `l2 = 0.11` for the link lengths in
`reacher.xml`, the identity

```
obs[8] + obs[4] == l1*cos(theta1) + l2*cos(theta1 + theta2)
obs[9] + obs[5] == l1*sin(theta1) + l2*sin(theta1 + theta2)
```

should hold exactly, and instead fails by about `6e-9` in the median and `2e-8`
at worst whenever the joints are moving.

This is small enough not to affect policy learning. It matters for anything
that treats the observation as a consistent kinematic state, such as model
verification, system identification, or tests that assert physical invariants.

### Where it comes from

`gymnasium/envs/mujoco/mujoco_env.py`:

```python
def _step_mujoco_simulation(self, ctrl, n_frames):
    self.data.ctrl[:] = ctrl
    mujoco.mj_step(self.model, self.data, nstep=n_frames)
    mujoco.mj_rnePostConstraint(self.model, self.data)
```

`mj_step` leaves `data.xpos` and `data.xipos` describing the state before the
final integration; nothing recomputes them afterwards. `set_state` does call
`mj_forward`, which is why the identity holds exactly on a freshly set state and
only breaks once a step has run.

`gymnasium/envs/mujoco/reacher_v4.py` (v5 is the same in the relevant respect):

```python
def _get_obs(self):
    theta = self.data.qpos.flat[:2]
    return np.concatenate([
        np.cos(theta),                 # post-step qpos
        np.sin(theta),                 # post-step qpos
        self.data.qpos.flat[2:],       # post-step qpos
        self.data.qvel.flat[:2],       # post-step qvel
        self.get_body_com("fingertip") - self.get_body_com("target"),  # stale xipos
    ])
```

### Reproducer

```python
import mujoco, numpy as np, gymnasium as g

L1, L2 = 0.1, 0.11
env = g.make("Reacher-v4", disable_env_checker=True)
u = env.unwrapped
env.reset(seed=0)
rng = np.random.default_rng(7)

for label, do_step in (("mj_forward only", False), ("after mj_step", True)):
    errs = []
    for _ in range(2000):
        th1 = rng.uniform(-np.pi, np.pi)
        th2 = rng.uniform(-2.9, 2.9)          # inside joint1's [-3, 3] limit
        tgt = rng.uniform(-0.15, 0.15, 2)
        qvel = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), 0.0, 0.0])
        u.set_state(np.array([th1, th2, tgt[0], tgt[1]]), qvel)
        if do_step:
            env.step(np.zeros(2, dtype=np.float32))
        else:
            mujoco.mj_forward(u.model, u.data)
        t1, t2 = u.data.qpos[0], u.data.qpos[1]
        fin = u.data.body("fingertip").xpos
        errs.append(max(
            abs(fin[0] - (L1*np.cos(t1) + L2*np.cos(t1 + t2))),
            abs(fin[1] - (L1*np.sin(t1) + L2*np.sin(t1 + t2)))))
    errs = np.array(errs)
    print(f"{label:>16}  median={np.median(errs):.3e}  max={errs.max():.3e}")
```

Output on `gymnasium==1.3.0`, `mujoco==3.10.0`, Python 3.10:

```
 mj_forward only  median=2.082e-17  max=8.327e-17
   after mj_step  median=6.199e-09  max=2.274e-08
```

Note that `theta2` must be sampled inside `joint1`'s `[-3, 3]` range. Outside
it the limit constraint legitimately moves `qpos` during the step and the
residual reaches `2e-4`, which is the constraint working correctly rather than
this issue.

### Scope

Reproduces identically, to the digit, on six installed configurations:
`gymnasium` 0.29.1 with `mujoco` 2.3.7, 1.0.0 with 3.1.6, 1.2.3 with 3.2.7, and
1.3.0 with each of 3.1.6, 3.2.7 and 3.10.0. Both `Reacher-v4` and `Reacher-v5`
are affected wherever both ship. The behaviour has been stable across releases,
so this is long-standing rather than a regression.

Setting `qvel = 0` before the step returns the residual to `2.8e-17`, which
locates the cause: with nothing moving, the pre-integration kinematics and the
post-step `qpos` describe the same configuration, so the identity holds again.

The same pattern would affect any environment whose observation mixes
`qpos`-derived and body-position-derived components. We have only checked
Reacher.

### Suggested fix

Call `mujoco.mj_forward(self.model, self.data)` after `mj_step` in
`_step_mujoco_simulation`, so that derived kinematic quantities correspond to
the integrated state. That costs one extra forward pass per step. A cheaper
alternative for Reacher alone is to compute the fingertip position from `qpos`
rather than from `get_body_com`.

Either changes observation values at the `1e-8` level, so it would want a
version bump if observation reproducibility is a concern.
