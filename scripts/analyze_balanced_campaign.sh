#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TAG="${1:-balanced_full_v1}"
if [[ ! "$TAG" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo 'Tag must use letters, digits, underscores or hyphens' >&2
    exit 2
fi
if [[ $# -gt 2 || ( $# -eq 2 && "$2" != '--allow-presentation-only-commit-drift' ) ]]; then
    echo 'Usage: bash scripts/analyze_balanced_campaign.sh [tag] [--allow-presentation-only-commit-drift]' >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/cognitive-mpl}"
CAMPAIGN="results/q_learning_baseline/campaigns/$TAG"
ROOT="results/q_learning_baseline/experiments"
BOOT="$("$PYTHON_BIN" - "$CAMPAIGN/freeze.json" "$TAG" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
tag = sys.argv[2]
if not path.is_file():
    raise SystemExit(f"Missing frozen campaign metadata: {path}")
freeze = json.loads(path.read_text())
if freeze.get("tag") != tag or freeze.get("mode") not in {"pilot", "full"}:
    raise SystemExit(f"Unexpected frozen campaign metadata: {path}")
print(int(freeze["bootstrap_resamples"]))
PY
)"

RUNS=(--run "$ROOT/${TAG}_b0")
for PROFILE in equal random normal_centered low_propensity_majority high_propensity_majority; do
    for DYNAMICS in fixed adaptive_bounded; do
        RUNS+=(--run "$ROOT/${TAG}_${PROFILE}_${DYNAMICS}")
    done
done
for ((i=1; i<${#RUNS[@]}; i+=2)); do
    if [[ ! -f "${RUNS[i]}/complete.json" ]]; then
        echo "Run is not complete: ${RUNS[i]}" >&2
        exit 1
    fi
done

EXTRA=()
if [[ $# -eq 2 ]]; then EXTRA+=("$2"); fi
"$PYTHON_BIN" -m cognitive_tools.analysis --profile visibility "${RUNS[@]}" \
    --analysis-name "${TAG}_analysis" --bootstrap-reps "$BOOT" "${EXTRA[@]}"
"$PYTHON_BIN" -m scripts.plot_visibility_story \
    --analysis-dir "results/q_learning_baseline/social_analysis/${TAG}_analysis"
