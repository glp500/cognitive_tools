#!/usr/bin/env bash
set -euo pipefail

# Targeted robustness checks. These use four representative environments,
# N=16, and 10 replicates to keep the sensitivity suite tractable.
SCENARIOS=(
    uniform_high
    centralized_low
    patchy_high_central_low
    split_high_low
)

COMMON=(
    --scenarios "${SCENARIOS[@]}"
    --populations 16
    --replicates 10
    --evaluation-steps 1000
    --record-every 50
    --seed 42
    --coupling 0.10
    --low-harvest 0.002
    --metabolism 0.002
    --initial-energy 1.0
    --energy-capacity 1.0
    --initial-resource-fraction 0.50
    --alpha 0.10
    --epsilon 0.20
    --epsilon-min 0.02
    --epsilon-decay 0.9995
)

# Discount-factor sensitivity.
for GAMMA in 0.90 0.95 0.99; do
    NAME="sensitivity_gamma_${GAMMA/./}"

    python baseline_validation_experiment.py \
        --run-name "${NAME}" \
        "${COMMON[@]}" \
        --training-steps 5000 \
        --high-harvest 0.020 \
        --gamma "${GAMMA}"

    python baseline_validation_figures.py --run-name "${NAME}"
done

# High-extraction intensity sensitivity.
for HIGH in 0.010 0.020 0.040; do
    NAME="sensitivity_highharvest_${HIGH/./}"

    python baseline_validation_experiment.py \
        --run-name "${NAME}" \
        "${COMMON[@]}" \
        --training-steps 5000 \
        --high-harvest "${HIGH}" \
        --gamma 0.95

    python baseline_validation_figures.py --run-name "${NAME}"
done

# Training-duration sensitivity.
for STEPS in 2500 5000 10000; do
    NAME="sensitivity_training_${STEPS}"

    python baseline_validation_experiment.py \
        --run-name "${NAME}" \
        "${COMMON[@]}" \
        --training-steps "${STEPS}" \
        --high-harvest 0.020 \
        --gamma 0.95

    python baseline_validation_figures.py --run-name "${NAME}"
done
