# Pilot figure review and slide rationale

## Intended use

Seven slides for supervisor feedback on the clarity of the research question,
hypothesis, and project structure. Individual observations and rewiring should
connect visibly to population-level patterns and shared-resource outcomes.
This deck replaces the status/cleanup narrative of the earlier seven-slide
update. It preserves the longer reference deck's 20 × 11.25-inch format,
white background, bold black Arial headings, generous spacing, and large figures.
Diagrams, labels, and text remain editable PowerPoint objects. Scientific figures
are embedded at high resolution and also supplied as PNG, SVG, and PDF.
The original PowerPoints and archived pilot outputs are unchanged.

## What the original figures needed

The pilot's original visibility figure uses six tall panels and an eight-series
legend that overlaps the bottom axis. Long run labels force the audience to
translate IDs while comparing trajectories. The original paired-effect figure
uses different axis ranges across panels and diagonal treatment labels. That
makes visually similar effect sizes incomparable across ecologies. The full
figure suite also includes one-row theta × mu heatmaps: the pilot only contains
mu = 0.10, so a heatmap cannot establish a two-dimensional phase boundary.

The revised figures use horizontal labels, one consistent scope palette,
solid/dashed adaptive/control lines, and shared quantitative scales. They show
all three ecologies and all paired sustainability comparisons. Dense diagnostic
views remain available outside the seven-slide argument.

## How related work informed the visualization

- [Schrama, Tilman & Vasconcelos (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12226385/),
  *Majority illusion drives the spontaneous emergence of alternative states in
  common-pool resource games with network-based information*, iScience,
  DOI [10.1016/j.isci.2025.112831](https://doi.org/10.1016/j.isci.2025.112831).
  Figure 2 pairs individual resource trajectories with an outcome distribution;
  Figures 3–4 connect perception to network degree. The applicable design lesson
  is to preserve replicate variation and distinguish local perception from
  population outcomes. Our companion distribution plot shows all ten replicate
  values instead of a smooth density. Neither their bistability result nor their
  incentive mechanism is attributed to this Q-learning pilot.

- [Oh & Schauf (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12618610/),
  *Self-organizing group structure through rewiring for collective decision-making
  in evolving environments*, Scientific Reports,
  DOI [10.1038/s41598-025-23634-3](https://doi.org/10.1038/s41598-025-23634-3).
  Figure 1 separates local and global replacement; Figure 6 separately tracks
  performance, influence concentration, and reallocation. We use a simple
  mechanism diagram and distinguish visibility concentration from sustainability.
  Their correctness-based DeGroot model differs from our surprise-triggered
  Q-learning model. All graphics here were drawn from our own data or as labeled
  schematics; no paper figures were copied or adapted.

The paired forest plots are an analysis-driven choice for this project's
matched-R0 design. They show the actual within-replicate contrast and its
uncertainty, rather than inviting inference from overlapping treatment means.

## Figure choices

| Figure | Question answered | Design and scope |
|---|---|---|
| `organization_N64` | What population patterns emerge during training? | Visibility Gini and low-extraction share over time; three ecologies, all six rewiring runs, ten-replicate means. Solid adaptive and dashed matched R0. Descriptive lines, no interval claim. |
| `organization_N32` | Does the smaller population show the same pattern? | Companion with identical axes and encodings. Kept outside the deck for readability; selection of N=64 is explicit. |
| `sustainability` | Does the adaptive trigger improve the shared resource beyond turnover? | All 18 paired contrasts; common -5 to +5 pp scale; 95% bootstrap intervals and zero reference. |
| `split_tradeoffs` | Does a resource change mean better welfare and equality? | All six paired contrasts in the split ecology, where resource effects change sign. Resource, reserve welfare, and final wealth Gini kept distinct. |
| `welfare`, `inequality` | Are the secondary outcomes consistent across ecologies? | Full all-ecology companions, preserving every population/scope comparison. |
| `resource_distributions_all_conditions` | What does the replicate spread look like? | All nine runs, three ecologies, and both populations; ten raw points and a mean mark per cell. No smoothed density or multimodality claim. |
| `capacity_*` | Which ecological settings were tested? | Replicate-0 capacity maps, seed 42, shared viridis scale 0–1. Exact scenario construction; illustrative maps, not outcome figures. |

Blue = local, teal = mixed, ochre = global. Population has a redundant shape
encoding (open circles N=32, filled squares N=64) and explicit row labels. No
red/green significance scheme. Zero is visible on every difference plot.
The inequality axis is in Gini units; fractional resource/welfare differences
are in percentage points, not relative percentages.

## What can be said

Local search concentrates visibility, and the matched random controls frequently
show similar concentration. Thus structural organization alone cannot establish
an effect specific to prediction error. Population behavior is displayed as the
share of low-extraction actions, not as an unqualified cooperation or consensus
score. Visibility Gini is an attention-distribution measure, not eigenvector
influence, modularity, or proof of coordinated collective action.

Fifteen of 18 resource intervals cross zero. The estimated differences span
about -2.10 to +1.50 percentage points. That does not establish equivalence or a
general advantage. Three individual intervals excluding zero are exploratory,
not multiplicity-adjusted discoveries. The pilot does not prove alternative
stable states, a structural mediation mechanism, or a fairer collective outcome.

The split-ecology close-up is disclosed as a subset; it includes negative,
positive, and uncertain contrasts. All-ecology welfare and inequality figures
are supplied. R2/N64's positive resource and reserve-welfare differences do not
establish a wealth-inequality improvement.

## Evidence and reproducibility

Inputs were restored byte-for-byte from the verified archive:
`/home/gavinl/Projects/cognitive-tools-archive/pre-cleanup-b0340b10/results.tar.gz`.
Only `pilot_v1_*` runs and their existing analysis were restored under ignored
`results/`. The original source commit is
`b0340b10dd33499a714af5c22ab124b05f680a3a`; all nine recorded worktrees were clean.

Primary rows: `strategy=q_learning`, `evaluation_mode=fresh_reset`, with terminal
training graphs carried forward for social runs; policies and graphs frozen.
Resource and reserve welfare average the 1,000-step evaluation; wealth inequality
uses the final step. Training trajectories are a different phase and are labeled.
The ten independent replicate differences are the sampling units.

`data_audit.json` records input hashes and checks. The build independently
reconstructs raw paired means and the archived 2,000-resample percentile
bootstrap intervals (95%; deterministic seeds derived from 1729). Source schedule
hashes and every matched checkpoint count are verified. `data/` contains plotted
source rows and aggregate values. No simulations were rerun to obtain results.

Rebuild from the repository root:

```bash
MPLCONFIGDIR=/tmp/cognitive-mpl python results/presentation/build_data_round.py
```

Requires the project environment plus `python-pptx`; the supplied longer deck
must remain available at its original path, or update `REF` in the script.
Figures are regenerated first, then the seven-slide PowerPoint. PDF export uses
LibreOffice. The accompanying audit checks the source evidence before authoring.

## Delivery checks

Completed: all seven exported slides were visually reviewed for legibility,
clipping, and arrow placement. Slide count, object bounds, and presenter notes
were checked. The PDF and PowerPoint contain the same seven-slide narrative.
All 54 plotted paired estimates and intervals were independently reproduced;
15 of the 18 sustainability intervals cross zero. The original decks remain
unchanged, and the repository is clean on main.
