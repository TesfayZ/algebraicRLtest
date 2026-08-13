# Auditing shipped simulator releases

The fault catalogue in the paper is injected by us, which is the standard and
fair objection to any mutation-based evaluation: the faults were written by the
person whose method catches them. This directory answers that objection by
pointing the same instrument, unchanged, at software nobody here wrote, namely
the actual release history of `gym`, `gymnasium` and the `mujoco` bindings.

## Design

Every release gets its own virtualenv, all pinned to CPython 3.10 so the
interpreter is not a confound. `dump_trajectories.py` runs *inside* each one and
imposes the same initial states and the same action sequence on every release,
so releases receive byte-identical inputs. That pairing is what makes an exact
comparison meaningful, and it is the regime a two-sample test cannot exploit:
with inputs held fixed, unchanged dynamics code must reproduce unchanged doubles.

The analysis runs in two stages, in this order.

1. **Exact comparison.** Two releases whose dynamics are unchanged agree
   bitwise. When they do, no statistical question needs asking and the null is a
   certainty rather than a failure to reject. Separations below `1e-13` are
   reported as float noise, because recompiling a physics library can reorder
   floating-point operations without changing the model. A pair is judged on its
   **first-step** separation, not its terminal one; see "Chaos" below.
2. **Screening.** The paper's level-1 screen runs on every pair stage 1
   separates, where a reference generating set exists. Its job is not detection,
   which stage 1 already settled, but localisation.

## The matrix

11 environments, 350 release pairs.

| family | releases | environments | pairs |
|--------|----------|--------------|-------|
| classic control | 11, `gym` 0.23.1 → `gymnasium` 1.3.0 | Acrobot-v1, Pendulum-v1, CartPole-v1, MountainCar-v0, MountainCarContinuous-v0 | 5 × 55 = 275 |
| MuJoCo | 6 configs, `mujoco` 2.3.7 → 3.10.0 | Reacher, InvertedPendulum, InvertedDoublePendulum, each v4 and v5 | 75 |

Three of the MuJoCo configurations hold `gymnasium` at 1.3.0 and move only the
binding, so a difference could be attributed to the physics library rather than
the wrapper. The v5 environments were introduced in `gymnasium` 1.0.0, so they
run over the five releases that ship them rather than pretending to six.

Counting v4 and v5 separately, as the audit does, six of the eleven carry an
exact polynomial reference set the screen can localise against: Acrobot (two
unit-norm identities and energy), Pendulum (unit-norm and energy), Reacher-v4
and Reacher-v5 (forward kinematics), and InvertedDoublePendulum-v4 and v5 (two
unit-norm identities each, holding on shipped data to `2.2e-16`). The remaining
five carry none: CartPole's action set has no zero element so the cart is driven
every step, both MountainCar variants sit in a cosine potential transcendental
in the position coordinate, and InvertedPendulum reports its hinge angle raw
rather than as a `(sin, cos)` pair. Those five are audited by the exact stage
alone, which is the stage that produces the null.

`gym==0.21.0` is absent from the matrix: it ships an `opencv-python (>=3.)`
specifier that modern resolvers reject, and 0.23.1 covers the same pre-0.26 API
era.

## Scripts

| script | what it does |
|--------|--------------|
| `setup_envs.sh` | builds one venv per release, with the pins each release forces |
| `dump_trajectories.py` | runs inside a venv; paired rollouts to `.npz`, plus constants and a source hash |
| `run_audit.py` | the two-stage audit over every release pair |
| `amplification.py` | per-step separation growth, chaotic environment against a non-chaotic control |
| `positive_control.py` | proves the pipeline separates known faults in the *shipped* class |
| `sensitivity.py` | how large a parameter error must be before the screen sees it, versus `dt` |
| `check_reference.py` | the precondition: do the reference generators hold on shipped data at all |
| `reacher_fk.py` | isolates the Reacher forward-kinematics residual; `--across-releases` re-runs it inside each MuJoCo venv |

```bash
./audit/setup_envs.sh
./venv/bin/python audit/run_audit.py
./venv/bin/python audit/amplification.py
./venv/bin/python audit/positive_control.py
./venv/bin/python audit/sensitivity.py
./venv/bin/python audit/check_reference.py
./venv/bin/python audit/reacher_fk.py --across-releases
```

Results land in `audit/Results/`.

## What the audit found

**No algebraic drift, across every pair tested.** All 350 pairs, 311 bitwise
identical and 39 separated only by float noise, show no change of dynamics. All
275 classic-control pairs are bitwise identical with a maximum separation of
exactly zero. Among the 75 MuJoCo pairs the largest separation off the chaotic
environment is `9.1e-15`, consistent with a recompiled physics library
reordering floating-point operations.

The null here is a certainty and not a failure to reject, which is the point of
putting the exact comparison first: with inputs held byte-identical, agreement
to the last bit is a proof that the arithmetic did not change, whereas a
two-sample test that fails to separate leaves the question open.

**The instrument is not blind.** Run against the shipped `AcrobotEnv` class,
the same pipeline separates and correctly localises:

| control | stage 1 max abs diff | localised | energy ratio | p |
|---------|---------------------|-----------|--------------|---|
| `book_or_nips=nips` (a variant Gymnasium ships, drops a Coriolis term) | 1.00e+01 | `energy` | 4.21 | 1.7e-06 |
| `LINK_MASS_2=1.3` | 9.67e+00 | `energy` | 11.56 | 9.8e-12 |

Both unit-norm identities stay clean in both cases, which is the localisation
claim doing its work: the screen names the constraint that broke and not the
ones that did not.

### Chaos: an exact comparison has to be read at the first step

Thirteen pairs, all on InvertedDoublePendulum, separate by `4.5e-2` or more at
the end of a 500-step rollout, up to the `1e+1` at which the observation
saturates its own clip. On the terminal separation alone those would be the
largest dynamics changes in the audit. They are not changes at all.

`amplification.py` traces one pair step by step:

| step | InvertedDoublePendulum-v4 | Reacher-v4 (control) |
|------|---------------------------|----------------------|
| 0    | 2.220e-16 | 5.551e-17 |
| 20   | 3.908e-14 | 5.551e-17 |
| 50   | 1.014e-12 | 5.551e-17 |
| 100  | 4.836e-09 | 5.551e-17 |
| 200  | 1.420e-05 | 5.551e-17 |
| 400  | 6.433e-02 | 5.551e-17 |

Both members of the pair receive the same imposed initial state and differ on
the first step by one unit in the last place. The growth is smooth exponential,
one decade per 32 steps on this pair and per 42 on the other, which is the
system's Lyapunov exponent acting on a rounding difference. Reacher, over the
identical release pairs and horizon, is flat.

So the exact stage's guarantee is narrower than it first appears. "Unchanged
code on identical input reproduces identical doubles" holds at every step in
exact arithmetic, but in floating point it holds only until the dynamics amplify
the last place, and on a chaotic system that is a few hundred steps. `classify`
therefore judges a pair by its first-step separation. Thresholding the horizon
instead would report a physics change in a benchmark whose physics did not
change, and loosening the threshold to compensate would give up the certainty
that motivated putting an exact comparison first. **The remedy is a shorter
comparison window, not a larger tolerance.**

### Gymnasium's shipped `dt` is what limits the screen, not the screen

A 1% error in `LINK_LENGTH_1` moves the trajectory by 1.6 rad/s, which stage 1
sees easily, yet the screen does not flag it. At the shipped `dt = 0.2` RK4's
own energy drift is a larger violation of conservation than a percent-level
parameter error is. Reducing `dt` at fixed horizon moves the floor and the
detection threshold together:

| `dt` | reference energy drift | smallest relative error localised |
|------|------------------------|-----------------------------------|
| 0.2 (shipped) | 3.55e-03 | 3e-2 (`LINK_MASS_2`), 1e-1 (`LINK_LENGTH_1`) |
| 0.05 | 4.19e-06 | ≤1e-3 (both, the bottom of the sweep) |
| 0.01 | 1.52e-08 | ≤1e-3 (both, the bottom of the sweep) |

So energy is not a usable invariant of Acrobot as shipped, and any diagnostic
built on it inherits that floor regardless of how it is implemented.

### Classic-control observations are float32

`Acrobot-v1` and `Pendulum-v1` return float32 observations, which puts a floor
of about `1.7e-08` on the unit-norm identities no matter how exact the
underlying dynamics are. MuJoCo environments return float64 and reach `3.6e-17`
on the same identities. Since level-2 attribution is measured to degrade above a
relative noise of `1e-7`, the observation dtype alone places attribution on
shipped classic-control environments at the edge of its ceiling. The fix is to
read `env.unwrapped.state` rather than the observation.

### Reacher's observation vector is not internally self-consistent

The audit flagged the forward-kinematics generators sitting at `1.1e-08`
(`fk-x`) and `1.6e-07` (`fk-y`) in a float64 pipeline, roughly nine orders above
the machine precision the unit-norm identities reach in the same observation.
The cause is not the model geometry.

Evaluated after `mj_forward`, the identity is exact:

| condition | median `|fk|` | max `|fk|` |
|-----------|---------------|------------|
| `mj_forward`, any velocity | 2.1e-17 | 8.3e-17 |
| `mj_step`, `qvel = 0` (nothing moves) | 2.8e-17 | 1.1e-16 |
| `mj_step`, `qvel ~ U(-1,1)` | 6.2e-09 | 2.3e-08 |

All three rows are produced by `reacher_fk.py` and are in
`Results/reacher_fk.csv`. The middle row is the one that pins the cause: a step
in which nothing moves leaves the identity intact, so it is the joints moving
during the step and not the step itself. Run
`./venv/bin/python audit/reacher_fk.py --across-releases` to reproduce all three
inside each MuJoCo virtualenv; the residual is identical to three significant
figures on all six configurations, from `gymnasium` 0.29.1 with `mujoco` 2.3.7
through `gymnasium` 1.3.0 with `mujoco` 3.10.0.

So the geometry in `reacher.xml` and the idealised two-link identity agree to
machine precision, and the residual appears only when the joints actually move
during a step. `MujocoEnv._step_mujoco_simulation` calls `mj_step` followed by
`mj_rnePostConstraint`, and never calls `mj_forward` afterwards, so `data.xpos`
and `data.xipos` still describe an internal integrator stage rather than the
final integrated `qpos`. `ReacherEnv._get_obs` then builds one observation out
of both:

```python
np.cos(theta), np.sin(theta),        # from data.qpos, the integrated state
self.data.qpos.flat[2:],             # from data.qpos
self.data.qvel.flat[:2],             # from data.qvel
self.get_body_com("fingertip") - self.get_body_com("target"),  # from data.xipos, stale
```

The two halves of the observation are therefore read at different points of the
integration step. At `6e-09` this is far below anything that matters for policy
learning, and it is not a physics error. It matters here because it puts a floor
of about `1e-08` relative on any exact diagnostic applied to shipped MuJoCo
observations, which is one order inside the `1e-07` ceiling measured for level-2
attribution. Reading `env.unwrapped.data.qpos` instead of the observation
removes it entirely.

One sampling detail bounds this measurement: `joint1` is limited to `[-3, 3]`
while a naive sweep draws `theta2` from `[-pi, pi]`. The 4.5% of samples outside
the limit trigger MuJoCo's limit constraint, which moves `qpos` during the step
and inflates the maximum residual to `2e-04`. That is the constraint doing its
job, not a defect, and every number in the table above is measured inside the
limit.

The same pattern would appear in any MuJoCo environment whose observation mixes
`qpos`-derived and body-position-derived components. Whether it does has not
been tested beyond Reacher.
