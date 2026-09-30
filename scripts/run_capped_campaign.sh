#!/usr/bin/env bash
set -euo pipefail
# Separate, bounded capped-utility study. The original campaign remains unchanged.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-smoke}"
TAG="${2:-capped_${MODE}_v1}"
if [[ ! "$TAG" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo 'Tag must contain only letters, numbers, underscores and hyphens.'; exit 2
fi
case "$MODE" in
    smoke) SCENARIOS=(uniform_high); N=8; REPS=2; TRAIN=12; EVAL=6; EVERY=2; THRESHOLD=0; MU=1; BOOT=20 ;;
    pilot) SCENARIOS=(uniform_high patchy_high split_high_low); N=64; REPS=10; TRAIN=5000; EVAL=1000; EVERY=50; THRESHOLD=0.25; MU=0.1; BOOT=2000 ;;
    *) echo 'Usage: bash scripts/run_capped_campaign.sh {smoke|pilot} [tag]'; exit 2 ;;
esac
if [[ -n "$(git status --porcelain)" ]]; then
    echo 'Commit source changes before generating study provenance.'; exit 1
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
COMMON=(--reward-mode capped_harvest --seed 20261001 --workers "${WORKERS:-2}"
    --scenarios "${SCENARIOS[@]}" --populations "$N" --replicates "$REPS"
    --training-steps "$TRAIN" --evaluation-steps "$EVAL" --record-every "$EVERY"
    --record-network-every "$EVERY" --rewire-every "$EVERY" --rewire-threshold "$THRESHOLD" --rewire-theta 0
    --rewire-mu "$MU" --network-eval frozen)
if [[ "${RESUME:-0}" == 1 ]]; then COMMON+=(--resume-conditions); fi
ROOT=results/q_learning_baseline/experiments
"$PYTHON_BIN" -m cognitive_tools.experiment "${COMMON[@]}" --run-name "${TAG}_b0" --social-mode none --rewiring none
"$PYTHON_BIN" -m cognitive_tools.experiment "${COMMON[@]}" --run-name "${TAG}_s1" --social-mode fixed --rewiring none
"$PYTHON_BIN" -m cognitive_tools.experiment "${COMMON[@]}" --run-name "${TAG}_r1" --social-mode fixed --rewiring prediction_error
"$PYTHON_BIN" -m cognitive_tools.experiment "${COMMON[@]}" --run-name "${TAG}_r0" --social-mode fixed --rewiring random_matched \
    --matched-rewire-schedule "$ROOT/${TAG}_r1/data/rewiring_schedule.csv"
"$PYTHON_BIN" -m cognitive_tools.analysis --run "$ROOT/${TAG}_b0" --run "$ROOT/${TAG}_s1" --run "$ROOT/${TAG}_r1" --run "$ROOT/${TAG}_r0" \
    --analysis-name "${TAG}_analysis" --profile focused --bootstrap-reps "$BOOT"
