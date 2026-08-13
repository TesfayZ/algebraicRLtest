#!/usr/bin/env bash
# Build one isolated virtualenv per shipped simulator release.
#
# Every venv is pinned to the same CPython (3.10, the widest window that spans
# gym 0.21 through gymnasium 1.3) so the interpreter is not a confound. numpy
# cannot be held fixed the same way, because releases before gym 0.26 use
# np.bool8 and die on numpy >= 1.24; the pin per row records what each release
# actually forces, and dump_trajectories.py writes the resolved numpy version
# into every .npz so triage can separate a dynamics change from a numpy change.
#
#   ./audit/setup_envs.sh          # build all
#   ./audit/setup_envs.sh gym      # build only rows whose tag matches
set -uo pipefail

cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p envs
FILTER="${1:-}"

# What this array is, and what it is not. ROWS is the list of releases this
# script *attempts*; it is not the audited matrix. The first row is expected to
# fail: gym 0.21.0 pins an opencv-python specifier modern resolvers reject, and
# what it leaves behind is an empty envs/gym-0.21.0 directory. It is kept so the
# attempt is on the record rather than looking like an oversight, and gym 0.23.1
# covers the same pre-0.26 API era.
#
# The authority for what is audited is CLASSIC_ORDER and MUJOCO_ORDER in
# run_audit.py, which is what produces the 350 pairs over 11 environments.
#
# tag                spec                  extra pins
ROWS=(
  "gym-0.21.0|gym==0.21.0|numpy<1.24 setuptools==65.5.0 wheel==0.38.4"
  "gym-0.23.1|gym==0.23.1|numpy<1.24"
  "gym-0.25.2|gym==0.25.2|numpy<1.24"
  "gym-0.26.2|gym==0.26.2|numpy<2"
  "gymnasium-0.26.3|gymnasium==0.26.3|numpy<2"
  "gymnasium-0.27.1|gymnasium==0.27.1|numpy<2"
  "gymnasium-0.28.1|gymnasium==0.28.1|numpy<2"
  "gymnasium-0.29.1|gymnasium==0.29.1|numpy<2"
  "gymnasium-1.0.0|gymnasium==1.0.0|numpy<2"
  "gymnasium-1.1.1|gymnasium==1.1.1|numpy<2"
  "gymnasium-1.2.3|gymnasium==1.2.3|numpy<2"
  "gymnasium-1.3.0|gymnasium==1.3.0|numpy<2"
  # MuJoCo rows. Reacher's invariants are kinematic, so unlike Acrobot's energy
  # they carry no dependence on the integrator and no drift floor to hide
  # behind. The last two rows hold gymnasium fixed and move only the mujoco
  # binding, which is what separates "the wrapper changed" from "the physics
  # changed".
  "mj-gymnasium-0.29.1-mujoco-2.3.7|gymnasium==0.29.1 mujoco==2.3.7|numpy<2 imageio packaging"
  "mj-gymnasium-1.0.0-mujoco-3.1.6|gymnasium==1.0.0 mujoco==3.1.6|numpy<2 imageio packaging"
  "mj-gymnasium-1.2.3-mujoco-3.2.7|gymnasium==1.2.3 mujoco==3.2.7|numpy<2 imageio packaging"
  "mj-gymnasium-1.3.0-mujoco-3.10.0|gymnasium==1.3.0 mujoco==3.10.0|numpy<2 imageio packaging"
  "mj-gymnasium-1.3.0-mujoco-3.1.6|gymnasium==1.3.0 mujoco==3.1.6|numpy<2 imageio packaging"
  "mj-gymnasium-1.3.0-mujoco-3.2.7|gymnasium==1.3.0 mujoco==3.2.7|numpy<2 imageio packaging"
)

ok=(); failed=()
for row in "${ROWS[@]}"; do
  IFS='|' read -r tag spec pins <<< "$row"
  [[ -n "$FILTER" && "$tag" != *"$FILTER"* ]] && continue

  venv="envs/$tag"
  if [[ -x "$venv/bin/python" ]]; then
    echo "== $tag: already built"; ok+=("$tag"); continue
  fi

  echo "== $tag: building"
  uv venv --python 3.10 "$venv" >/dev/null 2>&1 || { failed+=("$tag (venv)"); continue; }

  # gym 0.21 ships metadata modern resolvers reject, so it needs the old
  # setuptools in place first and the build run without isolation.
  if [[ "$tag" == "gym-0.21.0" ]]; then
    uv pip install --python "$venv/bin/python" setuptools==65.5.0 wheel==0.38.4 >/dev/null 2>&1
    if ! uv pip install --python "$venv/bin/python" --no-build-isolation \
         "$spec" numpy'<1.24' pygame 2>&1 | tail -2; then
      failed+=("$tag (install)"); continue
    fi
  else
    if ! uv pip install --python "$venv/bin/python" $spec $pins pygame 2>&1 | tail -2; then
      failed+=("$tag (install)"); continue
    fi
  fi

  # A venv that installs but cannot construct the environment is worse than one
  # that fails loudly, so prove it works now rather than during the audit.
  if "$venv/bin/python" -c "
import sys
try:
    import gymnasium as g
except ImportError:
    import gym as g
e = g.make('Acrobot-v1')
print('   ok', g.__version__)
" 2>&1 | tail -2; then
    ok+=("$tag")
  else
    failed+=("$tag (smoke)")
  fi
done

echo
echo "built:  ${#ok[@]}  ${ok[*]:-}"
echo "failed: ${#failed[@]} ${failed[*]:-}"
