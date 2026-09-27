# Stage 3 — decompose network evaluation

Apply these files at repository root:

```text
M  baseline_validation_experiment.py
M  tests/test_experiment_lifecycle.py
M  README.md
M  PROVENANCE.md
M  SOCIAL_EXPERIMENT.md
D  baseline_validation_experiment_STAGE2_CHANGES.md
```

No changes are required in:

```text
Cognitive_tools/social.py
Cognitive_tools/model.py
Cognitive_tools/env.py
Cognitive_tools/ecology.py
Cognitive_tools/qlearning.py
baseline_validation_figures.py
```

The existing baseline figure script remains compatible because the historical
`fresh_reset` evaluation label is retained for the primary fresh-ecology +
terminal-network condition. New evaluation rows use additional explicit
network fields and new mode names only for reset/adaptive topology conditions.

## Evaluation semantics

Default:

```bash
--network-eval frozen
```

Social runs then contain:

```text
continuation
    trained ecology
    terminal training network
    frozen network

fresh_reset
    fresh ecology
    terminal training network carried forward
    frozen network
    network_start=terminal

fresh_reset_network
    fresh ecology
    exact initial training network restored
    frozen network
    network_start=initial
```

`fresh_reset` keeps its old name only for backward compatibility with
`baseline_validation_figures.py`.

Optional prediction-error robustness:

```bash
--network-eval adaptive
```

This keeps the frozen evaluations and additionally runs:

```text
fresh_adaptive_network
    trained Q-table, frozen
    fresh ecology
    terminal training network
    terminal training forecast carried
    network continues adapting
```

The terminal graph and forecasts are copied before adaptive evaluation.
Evaluation rewiring uses a separate deterministic RNG stream.

`random_matched` R0 intentionally does not support adaptive evaluation yet,
because no matched evaluation-phase event-count schedule exists. Use the
frozen decomposition for confirmatory R0-vs-R1/R2/R3 comparisons.

## Validation

Run:

```bash
python -m py_compile \
    Cognitive_tools/social.py \
    baseline_validation_experiment.py \
    baseline_validation_figures.py

pytest -q
```

### S1 lifecycle smoke

```bash
python baseline_validation_experiment.py \
    --run-name stage3_s1_smoke \
    --social-mode fixed \
    --social-network random_k \
    --social-k 4 \
    --rewiring none \
    --network-eval frozen \
    --scenarios uniform_high patchy_high \
    --populations 8 \
    --replicates 1 \
    --training-steps 300 \
    --evaluation-steps 100 \
    --record-every 20 \
    --record-network-every 20
```

Check the available Q-learning evaluation modes:

```bash
python - <<'PY'
import csv
from pathlib import Path

path = Path(
    "results/q_learning_baseline/experiments/"
    "stage3_s1_smoke/data/evaluation_summary.csv"
)

with path.open() as f:
    rows = list(csv.DictReader(f))

modes = sorted({
    row["evaluation_mode"]
    for row in rows
    if row["strategy"] == "q_learning"
})

print(modes)
assert "continuation" in modes
assert "fresh_reset" in modes
assert "fresh_reset_network" in modes
PY
```

For S1 the terminal graph equals the initial graph, so the two fresh Q-policy
evaluations should be mechanically identical apart from evaluation metadata.

### R2 frozen decomposition smoke

```bash
python baseline_validation_experiment.py \
    --run-name stage3_r2_frozen_smoke \
    --social-mode fixed \
    --social-network random_k \
    --social-k 4 \
    --rewiring prediction_error \
    --rewire-theta 0.25 \
    --rewire-mu 0.10 \
    --rewire-every 50 \
    --rewire-threshold 0.25 \
    --forecast-alpha 0.50 \
    --network-eval frozen \
    --scenarios uniform_high patchy_high \
    --populations 8 \
    --replicates 1 \
    --training-steps 500 \
    --evaluation-steps 100 \
    --record-every 20 \
    --record-network-every 50
```

### R2 adaptive robustness smoke

```bash
python baseline_validation_experiment.py \
    --run-name stage3_r2_adaptive_smoke \
    --social-mode fixed \
    --social-network random_k \
    --social-k 4 \
    --rewiring prediction_error \
    --rewire-theta 0.25 \
    --rewire-mu 0.10 \
    --rewire-every 50 \
    --rewire-threshold 0.25 \
    --forecast-alpha 0.50 \
    --network-eval adaptive \
    --scenarios uniform_high patchy_high \
    --populations 8 \
    --replicates 1 \
    --training-steps 500 \
    --evaluation-steps 100 \
    --record-every 20 \
    --record-network-every 50
```

The Q-learning rows should additionally contain:

```text
fresh_adaptive_network
```

and the evaluation output includes:

```text
network_start
network_adaptive
evaluation_total_rewires
```

while `evaluation_timeseries.csv` additionally includes:

```text
evaluation_rewires_step
evaluation_rewires_cumulative
```

## Commit

After the tests and smoke runs pass:

```bash
git rm baseline_validation_experiment_STAGE2_CHANGES.md

git add \
    baseline_validation_experiment.py \
    tests/test_experiment_lifecycle.py \
    README.md \
    PROVENANCE.md \
    SOCIAL_EXPERIMENT.md

git commit -m "feat: decompose network evaluation"
```
