#!/usr/bin/env bash
# Run LeWM baseline evals and refresh Stage-1 JSON artifacts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LE_WM="${ROOT}/le-wm"

export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

cd "${LE_WM}"

run_eval() {
  local config="$1"
  shift
  echo "=== eval ${config} $* ==="
  python eval.py --config-name="${config}" "$@"
}

case "${1:-all}" in
  tworoom)
    shift || true
    for seed in "${@:-42 0 1 2}"; do
      run_eval tworoom.yaml policy=quentinll/lewm-tworooms "seed=${seed}"
    done
    ;;
  pusht)
    shift || true
    for seed in "${@:-0 1 2}"; do
      run_eval pusht.yaml policy=quentinll/lewm-pusht "seed=${seed}"
    done
    ;;
  tworoom_hard)
    shift || true
    for seed in "${@:-42 0 1 2}"; do
      run_eval tworoom_hard.yaml policy=quentinll/lewm-tworooms "seed=${seed}"
    done
    ;;
  tworoom_short_cem)
    shift || true
    for seed in "${@:-42 0 1 2}"; do
      run_eval tworoom_hard.yaml policy=quentinll/lewm-tworooms solver=cem_short "seed=${seed}"
    done
    ;;
  collect)
    python "${ROOT}/scripts/collect_baseline_results.py"
    ;;
  all)
    run_eval tworoom.yaml policy=quentinll/lewm-tworooms seed=42
    run_eval pusht.yaml policy=quentinll/lewm-pusht seed=0
    python "${ROOT}/scripts/collect_baseline_results.py"
    ;;
  *)
    echo "Usage: $0 {tworoom|pusht|tworoom_hard|tworoom_short_cem|collect|all} [seeds...]"
    exit 1
    ;;
esac

if [[ "${1:-}" != "collect" ]]; then
  python "${ROOT}/scripts/collect_baseline_results.py"
fi
