"""Run the experiment suite and report each script's status.

Each script is independently runnable and writes its CSV to ``Results/``.

    python run_all_experiments.py             # everything
    python run_all_experiments.py --quick     # skip the slowest sweeps
    python run_all_experiments.py --only D1
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, "venv", "bin", "python")
if not os.path.exists(PY):
    PY = sys.executable

# ID -> (script, description, slow)
SCRIPTS = {
    "A":  ("verify_properties.py",
           "Appendix A: all analytic and symbolic environment properties", False),
    "P":  ("test_projection_saturates.py",
           "Sequential orthogonal projection saturates at one direction", False),
    "D1": ("benchmark_deflation.py",
           "Deflation ablation on Acrobot, dt x tolerance sweep", False),
    "D1b": ("benchmark_energy_detectability.py",
            "Energy direction vs integrator drift floor", False),
    "D4": ("benchmark_negative_controls.py",
           "Tier C negative controls and spurious-variable stress test", True),
    "D5": ("benchmark_drift.py",
           "Integrator drift diagnostic, order and horizon sweeps", True),
    "B":  ("benchmark_balance_law.py",
           "Tier B: power balance and damping recovery, actuated and damped",
           True),
    "C1": ("benchmark_false_alarm.py",
           "False-alarm rate of the screen over healthy-vs-healthy pairs", True),
    "E1": ("benchmark_bug_localisation.py",
           "Fault localisation and attribution vs statistical baselines", True),
    "E1b": ("benchmark_baseline_residuals.py",
            "The baselines given the same per-generator residual features",
            True),
    "E1c": ("benchmark_fault_magnitude.py",
            "Detection floor against fault magnitude, and operating curves",
            True),
    "E1d": ("benchmark_multifault.py",
            "Attribution when two constants are wrong at once", False),
    "E1e": ("benchmark_approximate_reference.py",
            "An approximate reference set in place of the exact one", True),
    "E2": ("benchmark_worldmodel.py",
           "Algebraic unit tests for learned world models", True),
    "E3": ("benchmark_shaping.py",
           "Shaping from a discovered conserved quantity, PPO on Acrobot", True),
    "E5": ("benchmark_ideal_equality.py",
           "Deciding ideal equality against declared, recovered and installed "
           "systems", True),
    # The full release audit is run separately because it needs one environment
    # per simulator release.
    "E4b": ("audit/reacher_fk.py",
            "Reacher's observation against its own forward kinematics", False),
    # Read the completed result files; skip in quick mode.
    "F":  ("make_figures.py",
           "Render the paper's figures from the committed CSVs", True),
}

# Verification consumes experiment outputs; figure generation consumes all CSVs.
ORDER = ["P", "D1", "D1b", "D4", "D5", "B", "C1",
         "E1", "E1b", "E1c", "E1d", "E1e", "E2", "E3", "E5", "E4b", "A", "F"]

#: scripts that understand --quick themselves, rather than being skipped
QUICK_FLAG = {"B", "C1", "E1", "E1b", "E1c", "E1e", "E2", "E3", "E5"}

#: Full-run settings required by selected experiments.
EXTRA_ARGS = {"C1": ["--pairs", "500"],
              "E3": ["--param-sweep", "--random-draws", "10"],
              "E4b": ["--across-releases"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip slow scripts")
    ap.add_argument("--only", nargs="*", help="run only these ids")
    args = ap.parse_args()

    ids = args.only if args.only else ORDER
    results = []
    for key in ids:
        if key not in SCRIPTS:
            print(f"unknown id {key}; known: {', '.join(ORDER)}")
            continue
        script, desc, slow = SCRIPTS[key]
        cmd = [PY, os.path.join(HERE, script)]
        if not args.quick:
            cmd += EXTRA_ARGS.get(key, [])
        if args.quick and slow:
            # Run scripts that implement a reduced sweep; skip the others.
            if key in QUICK_FLAG:
                cmd.append("--quick")
            else:
                print(f"[skip] {key:4s} {desc}")
                results.append((key, "SKIPPED", 0.0))
                continue
        print(f"\n{'='*78}\n[{key}] {desc}\n{'='*78}")
        t0 = time.time()
        child_env = os.environ.copy()
        if args.quick and key == "A":
            # Use checks compatible with the reduced E3 output.
            child_env["CCR_QUICK"] = "1"
        rc = subprocess.call(cmd, cwd=HERE, env=child_env)
        dt = time.time() - t0
        results.append((key, "OK" if rc == 0 else f"FAILED(rc={rc})", dt))

    print(f"\n{'='*78}\nSummary\n{'='*78}")
    for key, status, dt in results:
        print(f"  {key:5s} {SCRIPTS[key][0]:38s} {status:14s} {dt:7.1f}s")
    bad = [k for k, s, _ in results if s.startswith("FAILED")]
    if bad:
        print(f"\nFAILED: {', '.join(bad)}")
        sys.exit(1)
    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()