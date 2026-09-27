#!/usr/bin/env bash
set -euo pipefail

# Core social-learning campaign runner.
#
# Usage:
#   bash scripts/run_social_campaign.sh pilot pilot_v1
#   CONFIRM_FULL=YES bash scripts/run_social_campaign.sh full confirmatory_v1
#
# Optional environment variables:
#   PYTHON_BIN=python
#   RESUME=1                 Skip already-complete run directories.
#   FULL_REPLICATES=100      Confirmatory replicate count (freeze before use).
#   BOOTSTRAP_REPS=2000      Override analysis bootstrap resamples.
#
# Primary scientific evaluation is always --network-eval frozen.
# Adaptive-network evaluation is a separate robustness campaign and is not
# mixed into the matched-R0 confirmatory analysis.

REPO_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-}"
TAG="${2:-}"

if [[ "${MODE}" != "pilot" && "${MODE}" != "full" ]]; then
    echo "Usage:"
    echo "  bash scripts/run_social_campaign.sh pilot <tag>"
    echo "  CONFIRM_FULL=YES bash scripts/run_social_campaign.sh full <tag>"
    exit 2
fi

if [[ -z "${TAG}" ]]; then
    TAG="social_${MODE}_v1"
fi

if [[ "${MODE}" == "full" && "${CONFIRM_FULL:-NO}" != "YES" ]]; then
    echo "Full confirmatory execution requires CONFIRM_FULL=YES."
    echo "Freeze the final replicate count and parameter grid before running."
    exit 2
fi

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to start from a dirty Git worktree."
    git status --short
    echo
    echo "Commit/stash source changes first. Generated results should be ignored."
    exit 1
fi

GIT_SHA="$(git rev-parse HEAD)"
echo "Campaign mode: ${MODE}"
echo "Campaign tag:  ${TAG}"
echo "Git SHA:       ${GIT_SHA}"

SCENARIOS=(
    uniform_high
    patchy_high
    split_high_low
)

THETAS=(
    0.00
    0.25
    1.00
)

if [[ "${MODE}" == "pilot" ]]; then
    POPULATIONS=(32 64)
    REPLICATES=10
    MUS=(0.10)
    DEFAULT_BOOTSTRAP_REPS=2000
else
    POPULATIONS=(64)
    REPLICATES="${FULL_REPLICATES:-100}"
    MUS=(0.05 0.10 0.20)
    DEFAULT_BOOTSTRAP_REPS=5000
fi

BOOTSTRAP_REPS="${BOOTSTRAP_REPS:-${DEFAULT_BOOTSTRAP_REPS}}"

EXPERIMENT_ROOT="results/q_learning_baseline/experiments"
ANALYSIS_ROOT="results/q_learning_baseline/social_analysis"
ANALYSIS_NAME="${TAG}_analysis"

COMMON=(
    --scenarios "${SCENARIOS[@]}"
    --populations "${POPULATIONS[@]}"
    --replicates "${REPLICATES}"
    --training-steps 5000
    --evaluation-steps 1000
    --record-every 50
    --record-network-every 50
    --seed 42
    --coupling 0.10
    --low-harvest 0.002
    --high-harvest 0.020
    --metabolism 0.002
    --initial-energy 1.0
    --energy-capacity 1.0
    --initial-resource-fraction 0.50
    --alpha 0.10
    --gamma 0.95
    --epsilon 0.20
    --epsilon-min 0.02
    --epsilon-decay 0.9995
    --social-k 4
    --ba-m 2
    --rewire-every 50
    --rewire-threshold 0.25
    --forecast-alpha 0.50
    --network-eval frozen
)

echo
echo "Preflight: compile"
"${PYTHON_BIN}" -m py_compile \
    Cognitive_tools/ecology.py \
    Cognitive_tools/env.py \
    Cognitive_tools/model.py \
    Cognitive_tools/qlearning.py \
    Cognitive_tools/social.py \
    baseline_validation_experiment.py \
    social_experiment_analysis.py

echo
echo "Preflight: tests"
"${PYTHON_BIN}" -m pytest -q

RUN_PATHS=()

run_name_path() {
    printf '%s/%s' "${EXPERIMENT_ROOT}" "$1"
}

record_run() {
    RUN_PATHS+=("$(run_name_path "$1")")
}

run_experiment() {
    local name="$1"
    shift

    local path
    path="$(run_name_path "${name}")"

    if [[ -e "${path}" ]]; then
        if [[ "${RESUME:-0}" == "1" \
              && -f "${path}/config.json" \
              && -f "${path}/data/evaluation_summary.csv" ]]; then
            echo
            echo "RESUME: keeping completed run ${name}"
            record_run "${name}"
            return
        fi

        echo "Run directory already exists: ${path}"
        echo "Remove/rename it, or use RESUME=1 for a verified completed run."
        exit 1
    fi

    echo
    echo "RUN: ${name}"
    "${PYTHON_BIN}" baseline_validation_experiment.py \
        --run-name "${name}" \
        "${COMMON[@]}" \
        "$@"

    record_run "${name}"
}

theta_treatment() {
    case "$1" in
        0|0.0|0.00) printf 'r1' ;;
        0.25|.25)    printf 'r2' ;;
        1|1.0|1.00)  printf 'r3' ;;
        *)
            echo "Unsupported core theta: $1" >&2
            exit 1
            ;;
    esac
}

parameter_tag() {
    # 0.05 -> 005, 0.10 -> 010, 0.20 -> 020
    printf '%s' "$1" | tr -d '.'
}

# ---------------------------------------------------------------------------
# Fixed controls
# ---------------------------------------------------------------------------

run_experiment "${TAG}_b0" \
    --social-mode none \
    --rewiring none

run_experiment "${TAG}_s1" \
    --social-mode fixed \
    --social-network random_k \
    --rewiring none

run_experiment "${TAG}_s2" \
    --social-mode fixed \
    --social-network ba \
    --rewiring none

# ---------------------------------------------------------------------------
# Adaptive treatments and their exact event-count-matched R0 controls
# ---------------------------------------------------------------------------

for mu in "${MUS[@]}"; do
    mu_tag="$(parameter_tag "${mu}")"

    for theta in "${THETAS[@]}"; do
        treatment="$(theta_treatment "${theta}")"

        adaptive_name="${TAG}_${treatment}_mu${mu_tag}"
        r0_name="${TAG}_r0_${treatment}_mu${mu_tag}"

        run_experiment "${adaptive_name}" \
            --social-mode fixed \
            --social-network random_k \
            --rewiring prediction_error \
            --rewire-theta "${theta}" \
            --rewire-mu "${mu}"

        adaptive_schedule="$(
            run_name_path "${adaptive_name}"
        )/data/rewiring_schedule.csv"

        if [[ ! -f "${adaptive_schedule}" ]]; then
            echo "Missing adaptive rewiring schedule: ${adaptive_schedule}"
            exit 1
        fi

        run_experiment "${r0_name}" \
            --social-mode fixed \
            --social-network random_k \
            --rewiring random_matched \
            --rewire-theta "${theta}" \
            --rewire-mu "${mu}" \
            --matched-rewire-schedule "${adaptive_schedule}"
    done
done

# ---------------------------------------------------------------------------
# Cross-treatment analysis
# ---------------------------------------------------------------------------

analysis_args=(
    "${PYTHON_BIN}"
    social_experiment_analysis.py
    --analysis-name "${ANALYSIS_NAME}"
    --bootstrap-reps "${BOOTSTRAP_REPS}"
)

for path in "${RUN_PATHS[@]}"; do
    analysis_args+=(--run "${path}")
done

echo
echo "ANALYZE: ${ANALYSIS_NAME}"
"${analysis_args[@]}"

manifest="${ANALYSIS_ROOT}/${ANALYSIS_NAME}/analysis_manifest.json"

"${PYTHON_BIN}" - "${manifest}" "${GIT_SHA}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
expected_sha = sys.argv[2]

with manifest_path.open() as file:
    manifest = json.load(file)

warnings = list(manifest.get("warnings", []))
if warnings:
    raise SystemExit(
        "Analysis completed with warnings:\n- " + "\n- ".join(warnings)
    )

inputs = manifest.get("inputs", [])
if not inputs:
    raise SystemExit("Analysis manifest has no input runs.")

dirty = [
    row["run_id"]
    for row in inputs
    if row.get("git_worktree_dirty") is True
]
if dirty:
    raise SystemExit(
        "Input runs recorded a dirty worktree: " + ", ".join(dirty)
    )

shas = {
    row.get("git_commit_sha")
    for row in inputs
    if row.get("git_commit_sha")
}
if shas != {expected_sha}:
    raise SystemExit(
        f"Input runs do not all use campaign SHA {expected_sha}: {sorted(shas)}"
    )

unpaired_r0 = [
    row["run_id"]
    for row in inputs
    if row.get("treatment") == "R0"
    and not row.get("paired_adaptive_run_id")
]
if unpaired_r0:
    raise SystemExit(
        "Unpaired matched-R0 runs: " + ", ".join(unpaired_r0)
    )

print("Campaign manifest validation passed.")
print(f"Inputs: {len(inputs)}")
print(f"Git SHA: {expected_sha}")
PY

echo
echo "Campaign complete."
echo "Experiments: ${EXPERIMENT_ROOT}/${TAG}_*"
echo "Analysis:    ${ANALYSIS_ROOT}/${ANALYSIS_NAME}"
