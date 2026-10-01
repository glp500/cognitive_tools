# Cognitive tools

This project studies how resource geography and restricted social observation
shape behavior in a renewable common-pool resource game. The current study
compares three **capacity-matched** landscapes: uniform, dispersed, and
segregated. All have total carrying capacity 75; the latter two also have the
same 50 low-capacity and 50 high-capacity cells, arranged differently.

The active research questions, treatment design, outcomes, and inferential
units are in the [study specification](docs/specs/capacity-matched-visibility-study.md).
The separate [commons-payoff gate](docs/reviews/balanced-capacity-payoff-gate-2026-10-01.md)
supports the social-dilemma endpoint criterion for the tested policy pair in
all three landscapes, for summed and discounted returns. It does not establish
that learning agents cooperate.

The completed learning study contains a ten-replicate exploratory pilot and an
independent 100-replicate full run. The full analysis finds that the primary
H1 association has the opposite sign from the prediction, H2 remains
unresolved, and H3 shows a small reduction in local-view error under adaptive
observation. See the [full-run audit](docs/reviews/balanced-full-run-recovery-2026-10-01.md)
for estimates, health checks, and the input-run provenance limitation. The
[current slide guide](presentations/draft4_slide_revision_plan_2026-10-01.md)
uses the full-run figures and results.

## Install and check

Use Python 3.12 from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff format --check .
ruff check .
pytest -q
```

## Reproduce the validation gate

A new gate run needs a new output name. The existing `balanced_gate_v1` is
complete and its six verdicts can be checked directly:

```bash
python -m scripts.validate_balanced_gate \
    results/payoff_validation/balanced_gate_v1/analysis
```

To reproduce the gate from scratch under a new tag:

```bash
python -m cognitive_tools.payoff \
    --output results/payoff_validation/balanced_gate_next/run \
    --purpose validation --workers 14 --population 64 --replicates 100 \
    --replicate-start 4000 --seed 20261014 --focal-count 4 --assignments 1 \
    --compositions 0 63 --horizons 1000 --gamma 0.95 \
    --reward-mode capped_harvest \
    --scenarios balanced_uniform balanced_dispersed balanced_segregated
python -m cognitive_tools.payoff_analysis \
    --run results/payoff_validation/balanced_gate_next/run \
    --output results/payoff_validation/balanced_gate_next/analysis --resamples 5000
python -m scripts.validate_balanced_gate \
    results/payoff_validation/balanced_gate_next/analysis
```

Choose an unused tag before running these commands.

## Run the learning study

The campaign runner requires a clean committed source tree and checks that the
Git revision does not change between treatments. Each mode includes the three
landscapes and eleven treatments (five initial visibility profiles crossed
with fixed/adaptive observation, plus a no-observation baseline).

```bash
# Fast mechanics check: N=8, two short replicates per treatment.
PYTHON_BIN=python WORKERS=2 bash scripts/run_balanced_campaign.sh smoke balanced_smoke_next

# Figure pipeline check: N=64, two short replicates per treatment.
PYTHON_BIN=python WORKERS=8 bash scripts/run_balanced_campaign.sh figure_smoke balanced_figures_next

# Scientific pilot: N=64, ten replicates per treatment and landscape.
PYTHON_BIN=python WORKERS=8 bash scripts/run_balanced_campaign.sh pilot balanced_pilot_next

# Full study: N=64, 100 replicates per treatment and landscape.
CONFIRM_FULL=YES BALANCED_PAYOFF_GATE=results/payoff_validation/balanced_gate_v1/analysis \
    PYTHON_BIN=python WORKERS=14 bash scripts/run_balanced_campaign.sh full balanced_full_next
```

Use a different tag for each campaign. `RESUME=1` resumes an interrupted
campaign only with its original tag, frozen configuration, and code revision.
A full run has 3,300 condition-replicates. The runner writes the campaign
freeze, landscape audit, eleven experiment directories, analysis tables and
figures, and five narrative story figures.

If all simulations complete but combined analysis stops, use the analysis-only
helper; it checks all eleven completion markers and uses the campaign's frozen
bootstrap count:

```bash
PYTHON_BIN=python bash scripts/analyze_balanced_campaign.sh balanced_full_next
```

The recorded `balanced_full_v1` run spans two commits whose committed changes
were presentation-only. For **that existing campaign**, the recovery command is:

```bash
PYTHON_BIN=python bash scripts/analyze_balanced_campaign.sh balanced_full_v1 \
    --allow-presentation-only-commit-drift
```

The option verifies changed paths with Git and retains the original commit IDs
and dirty-worktree warnings; it does not certify uncommitted input-run source.
The [recovery audit](docs/reviews/balanced-full-run-recovery-2026-10-01.md)
explains this limit.

## Retained results

Generated results are local and ignored by Git. The cleaned tree retains only:

- `results/payoff_validation/balanced_gate_v1/` — validation run, analysis,
  verdicts, and provenance.
- `results/q_learning_baseline/campaigns/balanced_pilot_v1/` and
  `balanced_full_v1/` — frozen designs and landscape checks.
- `results/q_learning_baseline/experiments/balanced_pilot_v1_*` and
  `balanced_full_v1_*` — all eleven completed treatments per campaign.
- `results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/`
  and `balanced_full_v1_analysis/` — tables, figures, health reports, and manifests.

The full-run figures live under `balanced_full_v1_analysis/figures/` and
`balanced_full_v1_analysis/story_candidates/`. To re-render them from saved
analysis without rerunning simulations:

```bash
MPLCONFIGDIR=/tmp/cognitive-mpl python -m scripts.plot_visibility_main \
    --analysis-dir results/q_learning_baseline/social_analysis/balanced_full_v1_analysis
MPLCONFIGDIR=/tmp/cognitive-mpl python -m scripts.plot_visibility_story \
    --analysis-dir results/q_learning_baseline/social_analysis/balanced_full_v1_analysis
```

The [figure audit](docs/reviews/visibility-figure-audit-2026-10-01.md) explains
visual conventions; the [cleanup record](docs/reviews/repository-cleanup-2026-10-02.md)
lists what was retired. Git history remains available for the older tracked
campaigns, but this checkout is scoped to the current study. Scientific
citations remain in [references.bib](docs/references.bib).
