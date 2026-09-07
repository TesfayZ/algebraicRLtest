# Diagnosing Faults in Reinforcement Learning Simulators and World Models with Canonical Polynomial Invariants

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22650767.svg)](https://doi.org/10.5281/zenodo.22650767)

The paper is archived on Zenodo at [10.5281/zenodo.22650767](https://doi.org/10.5281/zenodo.22650767). Latex references here refer to the document in `doc/*.tex`

Localising and attributing physics faults with canonical polynomial invariants.

**The thesis.** Exactness and canonicity buy one thing approximate invariants do
not: decidable equality. Every downstream use that needs only a differentiable
residual is served about as well by an approximate invariant. The uses that
genuinely require the canonical form are *diagnostic*: deciding whether two
systems have the same algebra, and localising where they differ. So this is a
debugging instrument, not a regulariser.

The instrument has two levels:

- **Screening** evaluates a reference generating set on the system under test
  and reports, per generator with a Bonferroni-corrected p-value, *which
  physical constraint broke*. No discovery run on the faulty system is needed.
- **Attribution** recovers the invariant from the faulty system and fits it
  against a parametric template, returning *the value of the parameter
  responsible*: "`l2` is 0.13, not the 0.11 in the spec."

## Install

Runs locally; there is no notebook and no build step.

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

`torch` is used only by the E2 world model and the E3 PPO implementation, and it
gets its own line because the CPU wheel lives on a separate index; everything
algebraic needs only numpy, scipy and sympy. `matplotlib` is used only by
`make_figures.py`. `requirements.txt` covers the project venv alone: the release
audit builds one virtualenv per shipped release on CPython 3.10 from the pinned
table in `audit/setup_envs.sh`, which is the point of that experiment.

The pinned interpreter is Python 3.14 (SymPy 1.14, NumPy 2.5), which is what the
paper's reproducibility appendix states.

## Running the experiments

```bash
./venv/bin/python run_all_experiments.py            # everything
./venv/bin/python run_all_experiments.py --quick    # reduced sweeps
./venv/bin/python run_all_experiments.py --only E1  # one experiment
```

Every script is independently runnable and writes its own CSV into `Results/`.
The `SCRIPTS` dict inside `run_all_experiments.py` is the authoritative
inventory; trust it over this table if the two disagree. The driver runs the
verification appendix after its result-producing experiments, then renders
figures last. Where a script's own
default is smaller than the setting the paper reports, the driver passes the
difference through `EXTRA_ARGS`, which is why C1 runs at `--pairs 500` rather
than at its 200-pair default.

The full E3 run uses `--param-sweep --random-draws 10`. Its random-potential
controls have five PPO seeds within each of ten independently drawn
polynomials; summaries and figure bands aggregate the seeds within a polynomial
first. `--quick` uses one draw for smoke testing and must not be used for paper
figures or random-control claims.

### Claim to script

The paper names results, not filenames. This table is where the mapping lives.

| id   | script | what it establishes | where in the paper |
|------|--------|--------------------|--------------------|
| A    | `verify_properties.py` | 137 analytic/symbolic checks, each compared against the figure the paper prints to 5% relative; nonzero exit if any claim has drifted from the text. The A.11 block asserts conditions no results table can display, so that a defect there fails here instead of quietly changing a number | "Verified Environment and Algebraic Properties" |
| P    | `test_projection_saturates.py` | sequential orthogonal projection removes one direction *in total*, not one per generator | "Deflation by normal-form reduction"; "Deflation on Acrobot" |
| D1   | `benchmark_deflation.py` | deflation ablation on Acrobot over Δt × tolerance | the deflation table, `tab:deflation` |
| D1b  | `benchmark_energy_detectability.py` | the energy direction sits at the integrator's drift floor | Remark "the tension between tolerating drift and resolving it" |
| D4   | `benchmark_negative_controls.py` | Hopper/HalfCheetah return nothing; spurious variable rejected | "Negative controls", in the experimental protocols appendix |
| D5   | `benchmark_drift.py` | integrator order recovered from data | "Passive drift under RK4" |
| B    | `benchmark_balance_law.py` | Tier B: the power balance of an actuated damped Acrobot, and the damping coefficients recovered from it | "Tier B: actuated and dissipative" |
| C1   | `benchmark_false_alarm.py` | the screen's false-alarm rate over hundreds of healthy-vs-healthy pairs, with and without the residual floor, per block and per trajectory | the screening section, and the residual-floor discussion |
| **E1** | `benchmark_bug_localisation.py` | **the headline**: detection, exact localisation and attribution against three statistical baselines | "Localising and attributing physics faults", `tab:e1`, `tab:attrnoise` |
| E1b  | `benchmark_baseline_residuals.py` | the same baselines handed the per-generator residual vector, which is what separates *named features* from *exact algebra* | the localisation discussion |
| E1c  | `benchmark_fault_magnitude.py` | the smallest parameter error each method still detects, and the operating characteristics over twelve decades of alpha | the detection-floor discussion, `tab:floor` and `tab:roc` |
| E1d  | `benchmark_multifault.py` | what single-fault attribution reports when two constants are wrong at once | "Limitations" |
| E1e  | `benchmark_approximate_reference.py` | the exactness ablation: an approximate reference set substituted for the exact one, both perturbed and numerically recovered, screened and then asked for the equality decision | "What Canonicity Buys, and What It Does Not", `tab:approx` |
| E2   | `benchmark_worldmodel.py` | consistency regularisation cuts the algebraic residual but barely moves rollout fidelity; one-step MSE stays the better predictor | "Does exact consistency improve a world model?" |
| E3   | `benchmark_shaping.py` | discovered-energy shaping, a faulty-potential mass sweep, and two random-potential controls over ten independent polynomial draws; random-control aggregates are reported at the draw unit | "Shaping, and the control that decides what it means", `tab:shaping` |
| E5   | `benchmark_ideal_equality.py` | the decidable-equality claim exercised: `<G_ref> = <G_test>` decided against declared, recovered and installed systems | "What Canonicity Buys" |
| **E4** | `audit/` (seven scripts) | **the instrument on software nobody here wrote**: 350 shipped release pairs over 11 environments, no dynamics change; three findings about the benchmarks | "Auditing shipped simulator releases", `tab:auditdt` |
| E4b  | `audit/reacher_fk.py` | the Reacher forward-kinematics inconsistency, isolated: exact after `mj_forward`, 6.2e-9 median after `mj_step` | "The audit found something we did not put there" |
| F    | `make_figures.py` | the paper's three figures, rendered from the CSVs above and never from a fresh run | `fig:shaping`, `fig:amplification`, `fig:calibration` |

`benchmark_drift.py` (D5) sweeps integrator families and horizons more widely
than the paper reports; only its RK4 drift figures reach the text.

`make_figures.py` (F) runs last and reads only committed CSVs, so it cannot
disagree with an experiment: if a number moves, rerun the owning script and then
rerun F. It writes vector PDFs into `doc/figures/`. Each figure carries something its table cannot: the E3
curves show *when* in training the faulty potentials track the correct one, the
amplification trace shows a growth *rate* over fifteen decades, and the
calibration forest plot replaces a table of interval endpoints with the
comparison the endpoints were there to support.

E2 and E3 both report results that cut against the hypotheses they were written
to test. That is deliberate and it is in the paper.

## The release audit

`audit/` lives in its own directory because it needs one virtualenv per shipped
release rather than the project venv. See [audit/README.md](audit/README.md) for
the design and the full results.

```bash
./audit/setup_envs.sh                            # one venv per release, needs uv
./venv/bin/python audit/run_audit.py             # the two-stage audit
./venv/bin/python audit/amplification.py         # separation growth, step by step
./venv/bin/python audit/positive_control.py      # known faults, shipped class
./venv/bin/python audit/sensitivity.py           # detection floor versus dt
./venv/bin/python audit/check_reference.py       # do the generators hold at all
./venv/bin/python audit/reacher_fk.py            # the Reacher FK inconsistency
```

**Where the releases come from.** `setup_envs.sh` installs each shipped release
from PyPI at a pinned version into its own virtualenv under `audit/envs/`, all
on CPython 3.10 so the interpreter is not a confound.

The authoritative **audited** matrix is `CLASSIC_ORDER` and `MUJOCO_ORDER` in
`run_audit.py`: eleven classic-control releases from `gym` 0.23.1 to `gymnasium`
1.3.0, and six MuJoCo binding configurations from `mujoco` 2.3.7 to 3.10.0,
three of which carry `gymnasium` 1.3.0 and differ only in the binding.
Seventeen releases, 350 pairs, 11 environments. Do not read a release count off
`setup_envs.sh`: its `ROWS` array is what the script *attempts* and holds one
further row, `gym` 0.21.0, which pins an `opencv-python` specifier modern
resolvers reject. That row fails and leaves an empty `audit/envs/gym-0.21.0`
behind; `gym` 0.23.1 covers the same pre-0.26 API era.

NumPy is pinned per row rather than globally, because releases
before `gym` 0.26 use `np.bool8` and fail on NumPy >= 1.24; `dump_trajectories.py`
writes the resolved NumPy version into every `.npz` so a NumPy change can be told
apart from a dynamics change. Because NumPy moves across rows and NEP 19's
stream guarantee covers `RandomState` and not `Generator`, `run_audit.py`
compares the recorded initial states of every pair and refuses to classify one
whose states differ. All 350 pairs pass that check, so "identical input" is
certified rather than assumed. Environments are built through the public `make`
entry point at the identifiers each release ships, no source file is patched, and
nothing is read from a private attribute.

The short version: across 350 release pairs over 11 environments, `gym` 0.23.1
through `gymnasium` 1.3.0 and `mujoco` 2.3.7 through 3.10.0 are unchanged in
their dynamics; the same pipeline localises the Coriolis term the `nips` variant
drops. Three properties of the benchmarks came out of it.

1. Gymnasium's shipped `dt = 0.2` puts Acrobot's RK4 energy drift above a 1%
   parameter error, so energy is not a checkable invariant as distributed.
2. Reacher's observation fails its own forward kinematics by `6e-9`, because
   `mj_forward` is not called after `mj_step` and the two halves of the
   observation are read at different points of the integration step. Written up
   for upstream in [audit/UPSTREAM_ISSUE.md](audit/UPSTREAM_ISSUE.md); not
   submitted.
3. On InvertedDoublePendulum, the one chaotic environment in the matrix, two
   releases running identical model files separate by up to `1e+1` over 500
   steps from a first-step difference of `2.2e-16`. That is Lyapunov growth of a
   rounding difference, not a model change, so the audit classifies a pair by
   its **first-step** separation. Reacher, over the same release pairs and
   horizon, does not separate at all.

## Modules

- `envs.py`: Acrobot and Reacher reimplemented analytically, so trajectories can
  be generated at arbitrary Δt and integrator, which Gymnasium does not expose.
  **Uses the `book` Acrobot variant**; the `nips` variant drops a Coriolis term
  and does not conserve energy, and `bugs.py` uses that switch deliberately as a
  structural fault with known ground truth.
- `deflation.py`: monomial machinery, the two deflation rules, graded-component
  dimensions, and `rebind` (see the traps below).
- `discover.py`: the normal-form matrix, Groebner deflation, the
  orthogonal-projection deflation it is compared against, and the nullspace
  routines. Self-contained; `recover.py` is its only caller.
- `ideal_diff.py`: the diagnostic core. Ideal equality and membership by
  normal-form reduction; `screen` (level 1); `attribute_single_fault` and
  `diagnose` (level 2, returning one of three verdicts).
- `recover.py`: recovering a generator from data. Contains two paths, one
  deciding a nullspace dimension against a tolerance and one projecting onto the
  quotient with no tolerance at all; the comparison between them is a result in
  the paper, so both stay.
- `independence.py`: functional independence and the sufficiency diagnostic,
  both by Jacobian rank. A Gröbner basis cannot drop `E^2` given `E`, since
  neither vanishes and conserved quantities form a subalgebra; the rank can. The
  same rank says whether a zero residual is locally sufficient for the dynamics
  or only necessary.
- `bugs.py`: the fault catalogue, parametric templates, and reference
  generating sets.
- `baselines.py`: MMD permutation test, KS+Bonferroni, MLP discriminator.
- `worldmodel.py` / `shaping.py`: the E2 dynamics model and the E3 PPO
  implementation, both self-contained.

## Four places where the natural implementation is wrong

Each of these has an implementation that looks correct, runs without complaint,
and returns something that reads as a finding. They are recorded because the
code alone does not show why it is written the way it is.

**Orthonormality of the quotient basis.** The quotient basis is built in
monomial coordinates, and the design matrix is column-normalised, so the basis
has to be carried into the normalised coordinates before the restriction
`Phi @ B.T` is taken. Rescaling an orthonormal basis column by column does not
leave it orthonormal, and normalising its rows afterwards does not either. Get
this wrong and the singular values of the restriction are those of a composition
with a non-orthogonal map rather than those of `Phi` restricted to the subspace:
the accepted gap ratio then depends on the units of the state variables, moving
by an order of magnitude when a single velocity coordinate is scaled by 100. It
also silently weakens the level-2 verdicts, routing two faults to *consistent
with specification* that the orthonormalised version correctly refuses as *not
explicable as a parameter fault*. `recover._orthonormal_rows` is the fix.

**Coordinate frames.** The SVD is taken on the *column-normalised* design
matrix, so its right singular vectors are coordinates against normalised
columns. They must be divided back by the column norms before any algebraic
operation. Skip that and the nullspace dimension is unchanged by deflation,
which reads as "Gröbner deflation does nothing".

**Anchor-normalised template matching.** Every coefficient of the Acrobot energy
contains `m2`, so there is no monomial whose template coefficient is
parameter-free to normalise against, and an anchor-based implementation silently
skips the true hypothesis while a wrong one wins. A recovered generator is only
defined up to scale (and, for a conserved quantity, an additive constant), so
the gauge has to be *fitted jointly with the parameters*, which is what
`match_to_template` does.

**Tolerance-based nullspace under noise.** Deciding a nullspace dimension
against a tolerance and *then* deflating fails once observations are noisy:
`c1^2+s1^2-1` stops vanishing, so the 14 trivial multiples leave the nullspace
entirely and the tolerance that admits the energy admits much else first.
Reversing the order removes the failure and the hyperparameter together.

## Paper

Numbers in the paper are pasted from script output and never hand-edited. The
verified-properties appendix is referred to by name and not by letter here,
because its letter moves whenever an appendix is added or dropped.