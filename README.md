# cognitive_tools

`cognitive_tools` studies how ecological conditions and restricted social
observation generate population misperception and collective organization in a
renewable common-pool resource system.

## Current research design

The prospective primary experiment is the [capacity-matched visibility
study](docs/specs/capacity-matched-visibility-study.md). It compares uniform,
dispersed and segregated resource landscapes with the same total carrying
capacity. The dispersed and segregated maps contain identical local capacity
values, so their direct comparison isolates spatial arrangement. Attention
profiles and fixed/adaptive observation follow the historical
[Stage-5 protocol](docs/specs/visibility-bounded-search-study.md). The previous
[capped-harvest commons validation](docs/reviews/capped-harvest-validation-results-2026-09-30.md)
applies only to its tested landscapes. The new maps independently passed the
[balanced-landscape payoff gate](docs/reviews/balanced-capacity-payoff-gate-2026-10-01.md)
for both summed and discounted returns.

The revised prospective hypotheses ask whether inequality in actual observer
counts is associated with local-view error over training (H1), whether
segregated resources increase that error relative to capacity-matched dispersed
resources (H2), and whether adaptive observation reduces it relative to a
paired fixed network (H3). The analysis saves within-run checkpoint and
time-adjusted H1 associations, a secondary run-average association, and
replicate-averaged H2/H3 contrasts. It also compares
collective extraction, visibility inequality, resource persistence, reserve
welfare, and wealth inequality for RQ2. Directional H2/H3 predictions are
pilot-informed. The independent 100-replicate full study has now completed:
the primary H1 association ran opposite its prediction, H2 remained
unresolved, and H3 showed a small reduction in local-view error. See the
[full-run recovery audit](docs/reviews/balanced-full-run-recovery-2026-10-01.md)
for estimates, health checks, and its Git-provenance limitation.

Each observer attends four sources. Five profiles manipulate initial visibility:
equal, random, normal centered, low-propensity majority and high-propensity
majority. Adaptive bounded search is the sole changing-network treatment. The
33-condition matrix uses three new ecologies × (five profiles × two dynamics + B0).

The earlier B0/S1/S2/R0/R1/R2/R3 design remains available for reproducing the
[Stage-4 study](docs/ideas/ecology-perception-organization.md). Its campaign
script and focused analysis retain their historical meaning.

### Run the capacity-matched experiment

From a clean committed checkout with the [installation](#installation) complete:

```bash
# Mechanical test: three ecologies × 11 treatments × 2 short replicates.
PYTHON_BIN=python WORKERS=2 bash scripts/run_balanced_campaign.sh smoke balanced_smoke_v1

# Small figure test: N=64, all 33 cells, 2 short replicates; renders story candidates.
PYTHON_BIN=python WORKERS=8 bash scripts/run_balanced_campaign.sh figure_smoke balanced_figures_v1

# Smaller scientific campaign: all 33 conditions, 10 replicates each.
PYTHON_BIN=python WORKERS=8 bash scripts/run_balanced_campaign.sh pilot balanced_pilot_v1

# Reproduce the held-out commons-dilemma gate before a confirmatory full run.
python -m cognitive_tools.payoff --output results/payoff_validation/balanced_gate_v1/run \
    --purpose validation --workers 14 --population 64 --replicates 100 \
    --replicate-start 4000 --seed 20261014 --focal-count 4 --assignments 1 \
    --compositions 0 63 --horizons 1000 --gamma 0.95 \
    --reward-mode capped_harvest \
    --scenarios balanced_uniform balanced_dispersed balanced_segregated
python -m cognitive_tools.payoff_analysis \
    --run results/payoff_validation/balanced_gate_v1/run \
    --output results/payoff_validation/balanced_gate_v1/analysis --resamples 5000
python -m scripts.validate_balanced_gate results/payoff_validation/balanced_gate_v1/analysis

# Frozen full campaign: 33 conditions × 100 replicates = 3,300 runs.
CONFIRM_FULL=YES BALANCED_PAYOFF_GATE=results/payoff_validation/balanced_gate_v1/analysis \
    PYTHON_BIN=python WORKERS=14 bash scripts/run_balanced_campaign.sh full balanced_full_v1
```

Use `RESUME=1` with the same command and tag to recover completed conditions
under the same clean code revision and specification. The runner now checks
the frozen Git revision and clean worktree between treatments, so a code or
documentation edit during a campaign stops the next treatment before its
provenance changes. Results are written to
`results/q_learning_baseline/experiments/<tag>_*` and the five main figures to
`results/q_learning_baseline/social_analysis/<tag>_analysis/figures/`.
The campaign script first writes `results/q_learning_baseline/campaigns/<tag>/landscape_validation.csv`,
then runs the matching analysis. Its `figure_smoke`, `pilot` and `full` modes also
render five story candidates under `<tag>_analysis/story_candidates/`. Use a new
tag for each new campaign. Review the [new study specification](docs/specs/capacity-matched-visibility-study.md)
before interpreting results. The older `run_visibility_campaign.sh` remains
available for reproducing the earlier Stage-5 design.

If all condition runs completed but the final combined analysis stopped, run
the analysis helper rather than restarting 3,300 simulations:

```bash
PYTHON_BIN=python bash scripts/analyze_balanced_campaign.sh balanced_full_v1 \
    --allow-presentation-only-commit-drift
```

The helper requires all 11 `complete.json` markers, uses the frozen bootstrap
count, runs the visibility analysis, and renders the five story figures. The
optional flag is for the recorded `balanced_full_v1` revision drift: analysis
keeps the original input commit IDs and accepts them only if Git verifies that
all intervening changes are documentation or presentation code. Scientific
source changes still fail. Input runs that recorded a dirty worktree remain
explicit warnings in `analysis_manifest.json`; the flag does not certify their
uncommitted contents. Omit the flag for a campaign recorded at one revision.
The [full-run recovery audit](docs/reviews/balanced-full-run-recovery-2026-10-01.md)
records the affected commits, completed-run count, full-study estimates, and
provenance limitation. The recovered analysis and five story figures are under
`results/q_learning_baseline/social_analysis/balanced_full_v1_analysis/`.

### Re-render the visual story

After a completed balanced pilot or full analysis, presentation-only figure changes can be applied without rerunning simulations or bootstrap estimates:

```bash
MPLCONFIGDIR=/tmp/cognitive-mpl python -m scripts.plot_visibility_main \
    --analysis-dir results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis
MPLCONFIGDIR=/tmp/cognitive-mpl python -m scripts.plot_visibility_story \
    --analysis-dir results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis
```

For the full campaign, replace the analysis directory with `balanced_full_v1_analysis` after `analysis_health.json` exists. The campaign runner already generates both figure families automatically; these commands are for a later visual-only re-render. Main figures are in `figures/`; story candidates and their provenance manifest are in `story_candidates/`.

After a completed Stage-5 analysis, render the separate five-figure narrative
set from its saved tables and source runs:

```bash
MPLCONFIGDIR=/tmp/cognitive-mpl python -m scripts.plot_visibility_story \
    --analysis-dir results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis
```

Use the full campaign's `<tag>_analysis` directory when it is available. The
script writes PDF, SVG, and 300-dpi PNG candidates, individual caption notes,
and `story_manifest.json` to `<analysis-dir>/story_candidates/`. It emits JSON
events for input failures, each figure, and completion. A complete manifest is
written only after all five figures succeed. The script checks source coverage,
observer-link counts, paired contrast coverage, and consistency between raw
fresh-evaluation means and the saved analysis; the manifest hashes its input
files for provenance. It does not rerun simulations or
replace the analysis figures. The original [visual-story specification](docs/ideas/stage-5-visual-story.md)
provides the five reader questions; the [capacity-matched specification](docs/specs/capacity-matched-visibility-study.md)
defines their new ecology comparisons and stock measures.

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

## Archived Stage-4 minimal example

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

## Archived Stage-4 campaign workflow

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

## Archived Stage-4 analysis

Analyze one or more compatible Stage-4 runs:

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

- [Stage-5 visibility study specification](docs/specs/visibility-bounded-search-study.md): current questions, hypotheses, matrix, signals and figures.
- [Stage-5 implementation verification](docs/reviews/visibility-study-implementation-2026-09-30.md): calibration, smoke results and limits.
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
