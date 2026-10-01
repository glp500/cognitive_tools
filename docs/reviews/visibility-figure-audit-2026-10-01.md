# Visibility figure audit — 1 October 2026

Scope: the **eight analysis figures** and **five story candidates** first reviewed in the revised ten-replicate balanced pilot. The completed full campaign uses the same plot code. These are visual and semantic changes; no simulations or inferential estimates were changed.

## Shared visual contract

- **Landscape:** uniform teal `#197A73`, dispersed rust `#B06435`, segregated violet `#6654A4`, always with circle/square/triangle markers when the three appear together.
- **Observation dynamics in analysis grids:** fixed blue `#245B78` with circles/solid lines; adaptive orange `#C56A32` with triangles/dashed trajectories. Initial-versus-terminal bars use blue-grey versus adaptive orange.
- **Continuous capacity:** a separate sequential green scale with fixed limits 0–1. This communicates quantity rather than a fourth treatment category.
- **Uncertainty:** point plus 95% interval, with zero reference for contrasts. Replicate is the inferential unit; condition cells are descriptive and their intervals are pointwise. Captions carry method detail instead of crowding the data area.

## Analysis figures

| Figure | Audit and revision | Best use |
|---|---|---|
| [01 Mechanism and design](../../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/figures/01_mechanism_and_design.pdf) | Corrected the candidate-search wording to “75% two-hop / 25% global,” and brought the RQ2 outcomes into line with the revised spec. The three maps are clear; the schematic text remains intentionally method-heavy. | Methods or appendix; use story 01 for a presentation. |
| [02 Initial visibility](../../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/figures/02_initial_visibility_manipulation.pdf) | Replaced faint overplotted agent dots with aligned observer-count boxplots and replicate-interval summaries. Profile names are now horizontal and the quantity “actual observers per agent” is explicit. | Manipulation check; usable at full-page width. |
| [03 Population perception](../../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/figures/03_population_perception.pdf) | Kept the complete 3×3 outcome grid for auditability, but removed repeated angled profile labels from upper rows and added shape encoding for fixed/adaptive. Renamed the primary axis “local-view error.” | Detailed RQ1 results; not a single-slide talk figure. |
| [04 Collective organization](../../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/figures/04_collective_organization.pdf) | Same grid cleanup and treatment encoding. The different y-scales across its two outcome rows are explicit; the plot supports comparison within a row, not visual comparison of numeric heights across metrics. | Detailed RQ2 grid. |
| [05 Secondary consequences](../../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/figures/05_secondary_consequences.pdf) | Preserved paired adaptive-minus-fixed intervals and a zero baseline. Removed redundant upper-row category labels. The violet estimates are contrasts, not a segregated-landscape color code here. | Welfare/wealth appendix. |
| [Supplement Gini decomposition](../../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/figures/supplement_gini_decomposition.pdf) | Aligned fixed/adaptive colors and markers with the other grids; category labels appear once at the bottom. Keep the zero-valued fixed-network change visible as a structural check. | Technical appendix. |
| [S2 Visibility trajectories](../../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/figures/S2_visibility_gini_trajectories.pdf) | Fixed is solid blue; adaptive is dashed orange, readable in grayscale as well as color. The 5×3 panel grid is necessary for exhaustive profile-by-ecology coverage. | Supplement, not a projected slide. |
| [S2 Extraction trajectories](../../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/figures/S2_low_extraction_trajectories.pdf) | Same trajectory treatment encoding. The crowded but complete grid is appropriate for checking time patterns; use a selected profile or summary contrast for a talk. | Supplement. |

## Story candidates

| Figure | Audit and revision | Best use |
|---|---|---|
| [01 Uneven ground](../../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/01_uneven_ground.pdf) | Equal total K=75 is legible above every map, and the shared sequential scale distinguishes capacity from treatment color. The process strip remains short. | Main narrative or study-design slide. |
| [02 Who gets seen](../../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/02_who_gets_seen.pdf) | Harmonized initial/terminal colors and removed connecting lines on the top-six-share panel, since the six agents are re-ranked at each snapshot. Five rows remain dense but necessary to show manipulation heterogeneity. | Full-width manuscript figure or backup slide. |
| [03 Different windows](../../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/03_different_windows.pdf) | Reduced the left panel to the ten **primary** segregated-minus-dispersed H2 contrasts. Replaced the dense H1 checkpoint cloud with raw and time-adjusted correlation intervals, making the pilot sign reversal directly visible. Both zero lines and contrast directions are labeled. | Main H1/H2 manuscript figure; split across two talk slides. |
| [04 Stock and wellbeing](../../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/04_stock_and_wellbeing.pdf) | The color/marker legend differentiates geography and B0/fixed/adaptive. The first two panels are fresh-evaluation condition means; the regional panel is explicitly training-end. Its axes are intentionally zoomed to the observed range, so read exact scales. | RQ2 manuscript figure; for a talk, show the first two panels and put the regional gap in backup. |
| [05 Adaptation pathway](../../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/05_adaptation_pathway.pdf) | Reworded the middle heading from “What is perceived?” to “What do four peers reveal?” because local-view error is available information, not elicited belief. The three panels follow visibility → local view → collective extraction; they do **not** prove causal mediation. | Full-width manuscript figure or a one-row talk cut. |

## Remaining presentation limits

The analysis grids and candidate 02/05 contain many treatment cells. They are legible as PDF pages and useful for audit or manuscript supplement, but shrink too far on a single projected slide. For a mixed-audience talk, use story 01, split story 03 into H1 and H2, and show a selected row or pooled contrast from story 05. Do not infer a universal welfare effect from story 04's descriptive scatterplots. Candidate 03's intervals are still pilot estimates; the full campaign must be interpreted under the same frozen estimands.

## Verification

The eight analysis figures and five story figures were rendered from the saved pilot analysis into temporary QA directories, visually inspected, then regenerated in their normal pilot output directories. `tests/test_visibility_analysis.py` and `tests/test_visibility_hypotheses.py` passed (10 tests). Main and story rendering can be rerun independently with the commands in the README. The full campaign script already invokes analysis-figure generation and story rendering after its 3,300 simulations finish.
