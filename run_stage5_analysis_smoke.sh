#!/usr/bin/env bash
set -euo pipefail

COMMON=(
  --scenarios uniform_high
  --populations 8
  --replicates 2
  --training-steps 200
  --evaluation-steps 50
  --record-every 20
  --record-network-every 50
  --rewire-every 50
  --network-eval frozen
)

python baseline_validation_experiment.py \
  --run-name stage5_smoke_b0 \
  --social-mode none \
  --rewiring none \
  "${COMMON[@]}"

python baseline_validation_experiment.py \
  --run-name stage5_smoke_s1 \
  --social-mode fixed \
  --social-network random_k \
  --social-k 4 \
  --rewiring none \
  "${COMMON[@]}"

python baseline_validation_experiment.py \
  --run-name stage5_smoke_s2 \
  --social-mode fixed \
  --social-network ba \
  --ba-m 2 \
  --rewiring none \
  "${COMMON[@]}"

for THETA_LABEL in "0:r1" "0.25:r2" "1:r3"; do
  THETA="${THETA_LABEL%%:*}"
  LABEL="${THETA_LABEL##*:}"

  ADAPTIVE_RUN="stage5_smoke_${LABEL}"
  R0_RUN="stage5_smoke_r0_${LABEL}"

  python baseline_validation_experiment.py \
    --run-name "${ADAPTIVE_RUN}" \
    --social-mode fixed \
    --social-network random_k \
    --social-k 4 \
    --rewiring prediction_error \
    --rewire-theta "${THETA}" \
    --rewire-mu 0.10 \
    --rewire-threshold 0.25 \
    --forecast-alpha 0.50 \
    "${COMMON[@]}"

  python baseline_validation_experiment.py \
    --run-name "${R0_RUN}" \
    --social-mode fixed \
    --social-network random_k \
    --social-k 4 \
    --rewiring random_matched \
    --rewire-theta "${THETA}" \
    --matched-rewire-schedule \
      "results/q_learning_baseline/experiments/${ADAPTIVE_RUN}/data/rewiring_schedule.csv" \
    "${COMMON[@]}"
done

python social_experiment_analysis.py \
  --analysis-name stage5_core_smoke \
  --bootstrap-reps 200 \
  --run results/q_learning_baseline/experiments/stage5_smoke_b0 \
  --run results/q_learning_baseline/experiments/stage5_smoke_s1 \
  --run results/q_learning_baseline/experiments/stage5_smoke_s2 \
  --run results/q_learning_baseline/experiments/stage5_smoke_r1 \
  --run results/q_learning_baseline/experiments/stage5_smoke_r0_r1 \
  --run results/q_learning_baseline/experiments/stage5_smoke_r2 \
  --run results/q_learning_baseline/experiments/stage5_smoke_r0_r2 \
  --run results/q_learning_baseline/experiments/stage5_smoke_r3 \
  --run results/q_learning_baseline/experiments/stage5_smoke_r0_r3

printf '\nStage 5 smoke analysis complete.\n'
printf 'Inspect: results/q_learning_baseline/social_analysis/stage5_core_smoke\n'
