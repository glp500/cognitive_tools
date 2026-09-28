# cognitive_tools

`cognitive_tools` is a simulation framework for studying adaptive social learning
in renewable common-pool resource systems. Agents learn extraction policies from
ecological and social information. Some treatments allow agents to change who
they observe when their local predictions are inaccurate. The experiments test
whether this adaptive rewiring changes sustainability, welfare, and inequality
beyond the effects of network structure or random turnover.

## Aim and research questions

Study how adaptive social information networks interact with individual
reinforcement learning and ecological feedback in a renewable common-pool
resource system.

**Primary question:** How does prediction-error-driven rewiring of who agents
observe affect resource sustainability, welfare, and inequality, relative to
fixed networks and matched random turnover?

**Secondary question:** How do these effects vary across ecological conditions
and the local or global scope of rewiring?

Agents stay in place, extract a renewable resource, and learn independently.
Social links carry observations of previous actions; they do not transfer
resources, rewards, or Q values. Sustainability, welfare, and inequality are
measured outcomes, not rewards optimized directly by the agents.

## Treatments

| ID | Information network | Rewiring |
|---|---|---|
| B0 | No social information | None |
| S1 | Random directed network with fixed attention capacity | None |
| S2 | Fixed symmetric preferential-attachment network | None |
| R0 | Same initial network as its paired adaptive run | Random observers, matched event counts and search scope |
| R1 | Random directed network | Prediction error; local search (`theta=0`) |
| R2 | Random directed network | Prediction error; mixed search (`theta=0.25`) |
| R3 | Random directed network | Prediction error; global search (`theta=1`) |

R0 must follow its adaptive run because it reads that run's rewiring schedule.
There is a separate matched R0 for each adaptive treatment and parameter setting.
The [experiment specification](docs/experiment.md) defines the mechanisms,
causal contrasts, measurement semantics, and evaluation conditions.

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
    analysis.py      Canonical analysis CLI, tables, and figures
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

Freeze the parameter grid and replicate count before a confirmatory campaign.
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
its matched R0. Analysis retains replicate distributions, bootstrap intervals,
paired effects, network-memory contrasts, and parameter summaries. See the
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
- [Provenance](docs/provenance.md): scientific sources and implementation lineage.
- [Bibliography](docs/references.bib): source citation metadata.
