#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODE="${1:-smoke}"
TAG="${2:-visibility_${MODE}_v1}"
if [[ ! "$TAG" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo 'Tag must use letters, digits, underscores or hyphens'; exit 2
fi
case "$MODE" in
    smoke) SCENARIOS=(uniform_high); N=8; REPS=2; TRAIN=60; EVAL=6; EVERY=10; BOOT=100; SEED=20261011 ;;
    pilot) SCENARIOS=(uniform_high patchy_high split_high_low); N=64; REPS=10; TRAIN=5000; EVAL=1000; EVERY=50; BOOT=2000; SEED=20261012 ;;
    full) SCENARIOS=(uniform_high patchy_high split_high_low); N=64; REPS=100; TRAIN=5000; EVAL=1000; EVERY=50; BOOT=5000; SEED=20261013
        if [[ "${CONFIRM_FULL:-NO}" != YES ]]; then
            echo 'Set CONFIRM_FULL=YES for the 3,300-condition campaign'; exit 2
        fi ;;
    *) echo 'Usage: bash scripts/run_visibility_campaign.sh smoke|pilot|full [tag]'; exit 2 ;;
esac
if [[ -n "$(git status --porcelain)" ]]; then
    echo 'Commit source and protocol changes before running to preserve provenance'; exit 1
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
SPEC=configs/visibility/visibility_profiles_v1.json
ROOT=results/q_learning_baseline/experiments
CAMPAIGN=results/q_learning_baseline/campaigns/$TAG
mkdir -p "$CAMPAIGN"
"$PYTHON_BIN" - "$CAMPAIGN/freeze.json" "$SPEC" "$MODE" "$TAG" "$SEED" "$N" "$REPS" "$TRAIN" "$EVAL" "$BOOT" <<'PYFREEZE'
import hashlib,json,subprocess,sys
from pathlib import Path
output,spec,mode,tag,seed,n,reps,train,evaluation,bootstrap=sys.argv[1:]
metadata=dict(study_protocol='visibility_bounded_search_v1', mode=mode, tag=tag,
              git_commit_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              visibility_profile_spec_sha256=hashlib.sha256(Path(spec).read_bytes()).hexdigest(),
              seed=int(seed), population=int(n), replicates=int(reps),
              training_steps=int(train), evaluation_steps=int(evaluation),
              bootstrap_resamples=int(bootstrap), treatments=11,
              ecologies=['uniform_high'] if mode=='smoke' else
                        ['uniform_high','patchy_high','split_high_low'])
path=Path(output)
if path.exists() and json.loads(path.read_text()) != metadata:
    raise SystemExit('Frozen campaign metadata differs; use a new tag')
path.write_text(json.dumps(metadata,indent=2))
PYFREEZE
COMMON=(--study-protocol visibility_bounded_search_v1 --reward-mode capped_harvest
    --visibility-spec "$SPEC" --attention-k 4 --social-k 4 --social-network random_k
    --rewire-theta 0.25 --rewire-mu 0.10 --rewire-threshold 0.25 --rewire-every 50
    --network-eval frozen --seed "$SEED" --workers "${WORKERS:-2}"
    --scenarios "${SCENARIOS[@]}" --populations "$N" --replicates "$REPS"
    --training-steps "$TRAIN" --evaluation-steps "$EVAL"
    --record-every "$EVERY" --record-network-every "$EVERY")
if [[ "${RESUME:-0}" == 1 ]]; then COMMON+=(--resume-conditions); fi
RUNS=(--run "$ROOT/${TAG}_b0")
"$PYTHON_BIN" -m cognitive_tools.experiment "${COMMON[@]}" \
    --run-name "${TAG}_b0" --social-mode none --network-dynamics none --rewiring none
for PROFILE in equal random normal_centered low_propensity_majority high_propensity_majority; do
    for DYNAMICS in fixed adaptive_bounded; do
        REWIRING=none
        if [[ "$DYNAMICS" == adaptive_bounded ]]; then REWIRING=prediction_error; fi
        RUN_NAME="${TAG}_${PROFILE}_${DYNAMICS}"
        "$PYTHON_BIN" -m cognitive_tools.experiment "${COMMON[@]}" \
            --run-name "$RUN_NAME" --social-mode fixed --visibility-profile "$PROFILE" \
            --network-dynamics "$DYNAMICS" --rewiring "$REWIRING"
        RUNS+=(--run "$ROOT/$RUN_NAME")
    done
done
"$PYTHON_BIN" -m cognitive_tools.analysis --profile visibility "${RUNS[@]}" \
    --analysis-name "${TAG}_analysis" --bootstrap-reps "$BOOT"
