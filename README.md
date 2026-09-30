# cognitive_tools

`cognitive_tools` studies how ecological conditions and restricted social
observation generate population misperception and collective organization in a
renewable common-pool resource system.

## Current research design

The current primary experiment is the [Stage-5 visibility and bounded-search
study](docs/specs/visibility-bounded-search-study.md). It asks when ecology and
restricted social observation distort population perception (RQ1), and how
these conditions shape visibility concentration and extraction (RQ2). The
hypotheses test ecological differences, initial visibility profiles, and fixed
versus adaptive bounded observation. Resource persistence, reserve welfare and
wealth inequality are secondary consequences. Agents learn with the validated
[capped-harvest commons utility](docs/reviews/capped-harvest-validation-results-2026-09-30.md).

Each observer attends four sources. Five profiles manipulate initial visibility:
equal, random, normal centered, low-propensity majority and high-propensity
majority. Adaptive bounded search is the sole changing-network treatment. The
33-condition matrix uses three ecologies × (five profiles × two dynamics + B0).

The earlier B0/S1/S2/R0/R1/R2/R3 design remains available for reproducing the
[Stage-4 study](docs/ideas/ecology-perception-organization.md). Its campaign
script and focused analysis retain their historical meaning.

### Run the new experiment

From a clean committed checkout with the [installation](#installation) complete:

```bash
# Fast end-to-end test: all 11 treatments, one ecology, 2 replicates.
PYTHON_BIN=python WORKERS=2 bash scripts/run_visibility_campaign.sh smoke visibility_smoke_v1

# Smaller scientific campaign: all 33 conditions, 10 replicates each.
PYTHON_BIN=python WORKERS=8 bash scripts/run_visibility_campaign.sh pilot visibility_pilot_v1

# Frozen full campaign: 33 conditions × 100 replicates = 3,300 runs.
CONFIRM_FULL=YES PYTHON_BIN=python WORKERS=14 bash scripts/run_visibility_campaign.sh full visibility_full_v1
```

Use `RESUME=1` with the same command and tag to recover completed conditions
under the same code/specification. Results are written to
`results/q_learning_baseline/experiments/<tag>_*` and the five main figures to
`results/q_learning_baseline/social_analysis/<tag>_analysis/figures/`.
The campaign script runs the matching analysis automatically. Review
`docs/specs/visibility-bounded-search-study.md` for measurement and inference
limits before interpreting pilot or full results.

## Code layout

```text
cognitive_tools/
    ecology.py       Resource dynamics and landscape primitives
    scenarios.py     Supported ecological scenarios
    model.py         Stationary agents, extraction, wealth, and welfare
    env.py           PettingZoo environment
    qlearning.py     Independent tabular learners
    social.py        Observation networks, forecasting, and rewiring
    experiment.py    Canonical experiment CLI, lifecycle, and provenance
    analysis.py      Canonical analysis CLI and optional diagnostics
    focused_analysis.py  Frozen study summaries, comparisons, and four figures
scripts/
    run_social_campaign.sh
tests/
docs/
```

The experiment module separates configuration, environment construction,
training, evaluation, measurement, and serialization into named functions.
The analysis reads completed runs without rerunning the model or modifying its
inputs. There are no historical experiment implementations in the active tree.

## Installation

Use Python 3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

`pyproject.toml` is the dependency and tooling configuration. Direct scientific
dependencies are pinned to the versions used for the cleanup baseline; run
metadata records the actual runtime versions. No Conda-specific setup is needed.
For an exact archived environment, retain its complete package inventory too.

## Minimal experiment

This small B0 run checks the full lifecycle, including fixed-policy controls:

```bash
python -m cognitive_tools.experiment \
    --run-name example_b0 \
    --scenarios uniform_high --populations 8 --replicates 2 \
    --training-steps 20 --evaluation-steps 10 --record-every 5
```

Directories are created automatically. The defaults select ecology-only
learning. This short run is a software check, not evidence about treatment
effects. Use a distinct run name to avoid overwriting another run.

Inspect all options with:

```bash
python -m cognitive_tools.experiment --help
python -m cognitive_tools.analysis --help
```

The installed `cognitive-experiment` and `cognitive-analysis` commands expose
the same entry points.

## One campaign workflow

Start from a clean, committed worktree:

```bash
bash scripts/run_social_campaign.sh smoke smoke_v1
```

This runs B0, S1, S2, R1–R3, their three matched R0 controls, and the analysis.
It uses tiny runs, forces enough rewiring opportunities to exercise pairing,
and validates the resulting manifest. It also compiles the package, runs tests,
checks schedules, and verifies that every input records the campaign commit.

For research runs, the same script accepts `pilot` or `full`:

```bash
bash scripts/run_social_campaign.sh pilot pilot_v1
CONFIRM_FULL=YES bash scripts/run_social_campaign.sh full confirmatory_v1
```

Campaigns use all available logical CPUs by default (`WORKERS=0`). Set
`WORKERS=14`, for example, to cap concurrency. The standalone CLI defaults to
one worker and accepts `--workers 0` or a positive count. Conditions keep their
original seeds and are assembled in canonical order regardless of completion
order. Each completed condition is saved atomically under its run's `conditions/`.

To resume after interruption, repeat the campaign command with `RESUME=1` and
the same tag, worker setting, environment, and clean code revision. Completed
conditions are reused; an interrupted condition restarts. Use a new tag for
the parallel campaign rather than reusing output from an older revision.

The full campaign is fixed at 27 conditions × 100 replicates, N=64, mu=0.10,
and base seed 20260928 (distinct from pilot seed 42).
The protocol and campaign settings are documented in
[docs/experiment.md](docs/experiment.md#campaigns).

## Analysis

Analyze one or more compatible runs:

```bash
python -m cognitive_tools.analysis \
    --analysis-name example_analysis \
    --run results/q_learning_baseline/experiments/example_b0
```

Add another `--run` for each treatment. Include each adaptive source run with
its matched R0. The default focused profile produces replicate summaries, planned paired
contrasts, and four main figures. Use `--profile diagnostics` for the retained
network-memory, policy, and parameter diagnostics. See the
[input contract and interpretation constraints](docs/analysis.md).

## Tests and style

```bash
ruff format --check .
ruff check .
pytest -q
python -m compileall -q cognitive_tools
```

The tests cover ecological equations, Q updates, network construction,
rewiring, temporal ordering, evaluation lifecycle, measurement, and pairing.
Scenario regressions cover every supported landscape on square and rectangular
grids. GitHub Actions runs formatting, lint, and tests on Python 3.12.

## Outputs and reproducibility

Generated files stay under ignored `results/`:

- Experiments: `results/q_learning_baseline/experiments/<run-name>/`
- Analysis: `results/q_learning_baseline/social_analysis/<analysis-name>/`

These path names are retained to preserve the current run and analysis contract.
Each experiment records configuration, seeds, Git SHA, worktree state, runtime
versions, and tabular measurements. Analysis records input hashes and pairings.
Archive whole paired run directories, including configuration and schedules.
Do not commit generated tables, plots, or animations.

The pre-cleanup source and historical campaign definitions are preserved by the
`pre-cleanup-scientific-snapshot` tag. The cleanup also verified a separate local
results archive; its location and checksum are in
[the cleanup record](docs/cleanup.md). No Git history was rewritten.

## Scientific documentation

- [Experiment](docs/experiment.md): design, mechanisms, outcomes, and campaigns.
- [Analysis](docs/analysis.md): input contract, pairing, intervals, and outputs.
- [Population payoff validation](docs/payoff-validation.md): separate Schelling experiment and usage.
- [Payoff validation results](docs/reviews/payoff-validation-results-2026-09-29.md): pilot and held-out findings.
- [Incentive redesign review](docs/reviews/incentive-redesign-review-2026-09-29.md): cost comparison, development evidence, and proposed revision.
- [Social-dilemma revision specification](docs/specs/social-dilemma-revision.md): implemented opt-in capped-harvest variant; held-out gate passed.
- [Capped-harvest study](docs/capped-harvest-study.md): separate learning recipe and utility accounting.
- [Capped validation results](docs/reviews/capped-harvest-validation-results-2026-09-30.md): held-out evidence and claim limits.
- [Provenance](docs/provenance.md): scientific sources and implementation lineage.
- [Bibliography](docs/references.bib): source citation metadata.
