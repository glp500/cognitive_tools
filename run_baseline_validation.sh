#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="baseline_validation_v1"

python baseline_validation_experiment.py \
    --run-name "${RUN_NAME}" \
    --populations 8 16 32 \
    --replicates 20 \
    --training-steps 5000 \
    --evaluation-steps 1000 \
    --record-every 50 \
    --seed 42 \
    --coupling 0.10 \
    --low-harvest 0.002 \
    --high-harvest 0.020 \
    --metabolism 0.002 \
    --initial-energy 1.0 \
    --energy-capacity 1.0 \
    --initial-resource-fraction 0.50 \
    --alpha 0.10 \
    --gamma 0.95 \
    --epsilon 0.20 \
    --epsilon-min 0.02 \
    --epsilon-decay 0.9995

python baseline_validation_figures.py \
    --run-name "${RUN_NAME}"

echo
echo "Complete."
echo "Results: results/q_learning_baseline/experiments/${RUN_NAME}"
