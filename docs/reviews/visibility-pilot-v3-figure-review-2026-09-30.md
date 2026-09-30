# Stage-5 pilot results and figure review — 30 September 2026

This is an exploratory review of `visibility_pilot_v3`, not a confirmatory
analysis. The frozen pilot ran 3 ecologies × (5 initial visibility profiles ×
2 network dynamics + B0) × 10 independent replicates, for 330 completed
condition-replicates. Each social condition used 64 agents, four attention
sources, 5,000 training steps and 1,000 fresh-evaluation steps. The final
1,000 training steps form the primary window. The pilot's producer revision was
`ecc7cf9dd8a8288c5da75918fcc7fce9b97534c3`; all 11 source runs were clean.
The analysis found no missing treatments or primary windows, verified 50
profile/replicate starting-graph identities across ecologies and dynamics,
and reported no warnings. All summary estimates used 10 replicates. The
3,030 undefined majority windows in B0 are expected because B0 has no social
view; every social treatment reported zero undefined windows. Adaptive runs
recorded 26,452 successful rewires in total.

## What the pilot shows

| Study claim | Descriptive pilot result | Interpretation |
|---|---|---|
| Initial manipulation | Mean initial visibility Gini: equal **0.000**, random **0.265**, normal **0.315**, low-majority **0.386**, high-majority **0.297**. Initially invisible fractions: **0**, **0.017**, **0.034**, **0.080**, **0.025**, respectively. | The manipulation operated, but normal and high-majority realized concentration overlap. Profile contrasts are about the whole propensity distribution, not ordered Gini doses. |
| H1: ecology and perception | Only **2 of 20** pointwise 95% intervals for the planned ecology effects on mean absolute perception error exclude zero. The largest examples point in opposite directions: patchy−uniform for random/fixed **−0.0144** [−0.0229, −0.0066]; split−uniform for normal/adaptive **+0.0131** [+0.0055, +0.0212]. | No stable pilot-wide ecological direction. The study's H1 is nondirectional, but this pilot does not establish a general ecology effect. |
| H2: initial profile and perception | After correcting the planned **random−equal** sign, this contrast is **+0.0085** in uniform high, **−0.0113** in patchy high and **+0.0061** in split high/low. Normal−random in patchy high is **+0.0097** [+0.0045, +0.0150]; most other pointwise intervals include zero. | Effects depend on ecology and are small relative to the baseline error around 0.18–0.20. Do not summarize H2 as a single monotone visibility effect. |
| H3: adaptive versus fixed | Visibility Gini differences are positive in **15/15** cells (range **+0.0152** to **+0.2833**); **14/15** pointwise intervals exclude zero. Perception-error differences cross zero in **12/15** intervals and change sign among profiles/ecologies. | Adaptive search concentrates attention in this pilot. Its effect on misperception is heterogeneous; H3's two outcomes should be reported separately. Fixed-graph Gini cannot change by construction, so the within-adaptive trajectory and magnitude are the informative parts. |
| Companion behavior and consequences | Low-extraction share has **1/15** adaptive−fixed intervals excluding zero. Fresh-evaluation resource fraction and reserve welfare each have **1/15**; final wealth Gini has **0/15**. | No consistent downstream improvement or deterioration is visible at this pilot size. The secondary effects should remain secondary. |

These counts describe **pointwise** replicate-bootstrap intervals among many
correlated comparisons; they are not multiplicity-adjusted discoveries or a
formal hypothesis verdict. Ten replicates are insufficient for a confident
null claim. Do not tune the frozen visibility profiles or select favorable
ecology cells because of these pilot outcomes. The full 100-replicate campaign
should keep the predefined comparisons and report all cells, including zero
crossings.

An analysis bug affected the first saved H2 table: it emitted `equal-random`
although the protocol specified `random-equal`. The sign and label have been
corrected in `visibility_analysis.py` and covered by a regression test. The
pilot's **source simulations were not rerun**; only their derived analysis is
regenerated. The other H2 comparisons and H1/H3 tables are unchanged.

## Figure audit and recommended manuscript sequence

The five existing figures are useful exploratory exports but should not go
into a manuscript unchanged. At a reduced page width the 3×3/5×3 panels,
rotated treatment labels and repeated axes become difficult to read. The
absolute-mean panels also force readers to infer paired effects by comparing
overlapping marginal intervals, which is not the analysis the protocol planned.

| Existing figure | Finding | Manuscript action |
|---|---|---|
| `01_mechanism_and_design` | The ecology maps are real, but lack a capacity scale; the lower panels contain many words, cryptic E/R/N/L/H abbreviations and no training/evaluation loop. | Rebuild as a concise visual ODD: initialization (landscape, agents, initial attention), repeated update/rewire schedule, frozen-network evaluation, and observed outputs. Show N, k, spatial extent and time scale in the diagram or caption. |
| `02_initial_visibility_manipulation` | Gini and invisible-fraction intervals are clear. Identical-position scatter dots hide the *frequency* of observer counts. | Replace the dot cloud with a per-profile count distribution or frequency heatmap. Keep Gini and invisible fraction as replicate estimates; label agent counts descriptive. |
| `03_population_perception` | Nine panels of absolute means conceal H1/H2/H3 paired comparisons. Error, non-tie mismatch and tie rate compete for attention. | Use the new H1/H2/H3 contrast figures for primary claims. Retain absolute perception, mismatch and tie plots as supplementary diagnostics with definitions in captions. |
| `04_collective_organization` | Initial profile separation and adaptive Gini increase are visible, but Gini's range visually dominates the small extraction differences. | Lead with paired adaptive−fixed Gini and a selected trajectory showing growth. Give low-extraction share a separate companion panel. |
| `05_secondary_consequences` | Correct paired effects and zero lines, but nine panels fill a page for mostly zero-crossing intervals and omit the absolute B0 ecological context. | Move to supplement. If consequences become central, add a compact absolute-outcome table including B0, with paired effects alongside it. |
| Gini decomposition and trajectories | G0/GT/Δ is valuable; the 5×3 trajectory small multiples are too large for a main figure. | Keep all as supplements; show one or two prespecified representative profile trajectories in the main text if space permits. |

The following four **candidate** figure sets render from saved replicate
contrast summaries without new inferential choices. They use horizontal
estimates and intervals, explicit zero lines, readable category labels, a
consistent ecology order, a non-color-only fixed/adaptive label, and a compact
parameter/replicate note. PDF and SVG remain editable/vector; PNG is 300 dpi.
The publication candidates are separate from the original five-figure study
contract so the pilot cannot silently replace prespecified figures.

1. [H1 ecological contrasts](../../results/q_learning_baseline/social_analysis/visibility_pilot_v3_analysis/publication_candidates/H1_ecology_perception.pdf): patchy−uniform and split−uniform, paired within each profile and network dynamic.
2. [H2 profile contrasts](../../results/q_learning_baseline/social_analysis/visibility_pilot_v3_analysis/publication_candidates/H2_profile_perception.pdf): the four frozen fixed-network comparisons, faceted by ecology.
3. [H3 primary adaptive contrasts](../../results/q_learning_baseline/social_analysis/visibility_pilot_v3_analysis/publication_candidates/H3_adaptation_primary.pdf): paired perception error and visibility Gini with distinct scales, faceted by ecology.
4. [Companion extraction contrasts](../../results/q_learning_baseline/social_analysis/visibility_pilot_v3_analysis/publication_candidates/S3_adaptation_extraction.pdf): paired low-extraction share by ecology and profile.

For the main manuscript, prioritize a visual ODD, the corrected manipulation
figure, and the three H1/H2/H3 contrasts. Treat the extraction and downstream
outcomes as supplements unless the full experiment changes their importance.
This selection follows the research questions and frozen contrasts rather than
pointwise pilot intervals. For a further mechanism supplement, an initial-to-
terminal observer-count transition heatmap by profile could show whether
attention concentration comes from a few winners or broad redistribution.
Both endpoints are already present in `initial_visibility.csv` and
`agent_social_summary.csv`; no new simulations are needed. A spatial
perception-error map could be exploratory, but it should average matched
agent positions across replicates and label ecological regions, not depict one
attractive replicate as typical.

## JASSS preparation standard

I interpret the intended venue as the *Journal of Artificial Societies and
Social Simulation* (JASSS). A [JASSS-published assessment of simulation
figures](https://www.jasss.org/18/4/16.html) identifies five basic reporting
attributes: labeled axes; numeric scales and units; a clear simulation basis;
the parameters that generated the data; and the number of simulations. The
candidate contrast figures include these elements, though the full manuscript
caption should define perception error, Gini, the training window, pairing and
the exploratory pointwise interval method. A [recent JASSS visual-ODD
article](https://www.jasss.org/27/4/1.html) recommends showing initialization,
process scheduling, observations and spatial/temporal scales in model overview
figures; this is the best model-figure template for this project. These are
published research recommendations, not a verified fixed figure-size or file-
format rule. Check the [current JASSS submission page](https://www.jasss.org/admin/submit.html)
and final manuscript layout before submission, especially actual text size after
scaling, caption placement and accepted file formats. The strongest submission
improvement is transparent model/experiment documentation alongside the plots,
consistent with the [ODD protocol discussion in JASSS](https://www.jasss.org/23/2/7.html).

Rebuild the candidates after analysis with:

```bash
MPLCONFIGDIR=/tmp/cognitive-mpl python scripts/plot_visibility_hypotheses.py \
  --analysis-dir results/q_learning_baseline/social_analysis/visibility_pilot_v3_analysis
```

The source CSVs, campaign freeze, input hashes and analysis manifest remain
under `results/q_learning_baseline/`. Figure captions should cite this study
identity and the final analysis commit; pilot figures should not be labeled
as the full 100-replicate study.
