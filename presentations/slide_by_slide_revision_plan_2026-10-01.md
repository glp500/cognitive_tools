# Draft 2: slide-by-slide revision plan

**Audience and length:** mixed audience, 10–12 minutes. Use 13 main slides and keep detailed plots in backup. This is an editing guide for [Gavin Lip. Cogntive Tools Project Draft-2.pdf](</home/gavinl/Downloads/Gavin Lip. Cogntive Tools Project Draft-2.pdf>), not a request to build another deck. Keep each slide to one claim, one visual, and a short source/status line. Put definitions, methods, and caveats in speaker notes.

**Evidence labels:** The independent held-out payoff gate supports the **commons-dilemma structure** for the tested policy pair and evaluation design. The ten-replicate **learning pilot is exploratory**; it does not confirm H1–H3. Use “Exploratory pilot · n=10” on every pilot-result slide. Replace those slides with the independent full-campaign results only after the frozen analysis and health checks complete.

The five saved candidate figures are in [`story_candidates`](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/story_manifest.json). Prefer their PDF or SVG versions when placing them in slides. The PNG links below are convenient previews.

**Exact prospective hypotheses:** H1: Across training, greater inequality in actual observer counts is associated with greater local-view error. H2: Averaged equally over five initial profiles and both network dynamics, segregated resource layout produces greater final-window local-view error than dispersed layout. H3: Averaged equally over three landscapes and five initial profiles, adaptive bounded rewiring reduces final-window local-view error relative to fixed observation starting from the same graph. The concise slide headlines below preserve those meanings; their full estimands are in the [study specification](../docs/specs/capacity-matched-visibility-study.md).

| Draft 2 page | Change | Revised slide |
|---|---|---|
| 1 | Keep the project identity; simplify title and hero visual. | 1 |
| 2–4 | Condense motivation, progress, and broad framing into one causal story. | 2 |
| 5 | Replace the other Commons Game's apples with this model's validation result. | 3 |
| 8 | Replace old RQ/H wording with the frozen revised questions. | 4 |
| 6 and 9 | Separate matched-capacity geography from the factorial design. | 5 and 7 |
| 7 | Correct and simplify fixed versus adaptive observation mechanics. | 6 |
| 10–12 | Fill the empty results slides with the revised H1–H3, one per slide. | 8–10 |
| 13 | Replace the outdated migration/uncertain-dilemma conclusion. | 13 |
| New | Add RQ2 consequences before the conclusion. | 11–12 |

## 1. Title — 20–30 seconds

**On slide:** “Who gets seen in a shared-resource world?” Subtitle: “Ecology, local observation, and collective outcomes.” Include name and date.

**Figure:** A small three-map strip based on [candidate 1](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/01_uneven_ground.png), or no figure. If using a stylized strip, label it “illustrative” and use the actual maps on slide 5.

**Speaker notes:** This is a controlled agent-based commons study. The talk asks whether spatial resource arrangement and who agents observe change what information is available and what the population does. Do not claim the slides measure human beliefs.

## 2. The story in one line — 40–50 seconds

**On slide:** `Resource geography → who gets seen → local view and collective outcomes`. Under the last step, name only “extraction · stock · welfare · inequality.”

**Figure:** Three large boxes and arrows. This replaces Draft 2's tiny multi-part framing diagram. Keep the “technological mediation” motivation in notes rather than as a list on the slide.

**Speaker notes:** Each agent observes four peers. The observation network determines whose extraction choices become visible. Local-view error compares those four peers with the population. Resource geography is manipulated while total configured capacity is fixed. The study then checks extraction, resource persistence, reserve welfare, and inequality.

## 3. Why this is a commons dilemma — 45–60 seconds

**On slide:** “Held-out payoff gate: supported in all three landscapes.” Show three rows—uniform, dispersed, segregated—with a check for **summed** and **discounted** return. Footer: “6/6 ecology × return verdicts supported.”

**Figure:** A simple six-cell verdict grid, built from the [payoff validation report](../results/payoff_validation/balanced_gate_v1/analysis/validation_report.md). Do **not** use Draft 2's screenshot of another game's apples as if it were this model.

**Speaker notes:** The gate tested N=64, horizon 1,000, 100 held-out replicates per landscape, always-low versus always-high policies, and capped-harvest utility. Collective gain, exploitation gap, and fear had positive simultaneous lower bounds for both return measures in all three ecologies. This establishes the criterion for the tested policies and reset distribution, not learning convergence or H1–H3.

## 4. Revised research questions — 45–60 seconds

**On slide:** Use these exact two questions, in separate large cards:

> **RQ1:** How do capacity-matched resource layouts, initial visibility profiles, and adaptive observation affect how accurately agents' four observed peers represent population extraction behavior?

> **RQ2:** How do those conditions affect collective extraction, inequality in actual observer counts, resource persistence, reserve welfare, and wealth inequality?

**Figure:** None. The questions are the visual. Avoid putting H1–H3 on this slide as additional small paragraphs.

**Speaker notes:** “Actual observer counts” means how many agents selected each source as a peer. Local-view error is an information-availability measure: the absolute difference between the low-extraction share among four observed peers and among all other agents, averaged over agents. It is not a reported belief or psychological “misperception.”

## 5. Same capacity, different geography — 50–60 seconds

**On slide:** “Same total capacity (K=75); different spatial organization.” Name the three maps: uniform, dispersed, segregated.

**Figure:** [Candidate 1: three capacity maps](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/01_uneven_ground.png), large and full width. It already contains a shared color scale. If cropping, preserve all three map titles and the scale.

**Speaker notes:** The 10×10 uniform map uses K=0.75 per cell. Both unequal maps contain fifty K=0.55 and fifty K=0.95 cells; only arrangement differs. Regeneration, initial resource fraction, and spatial coupling are held fixed. These controls isolate **configured spatial organization**, but they do not guarantee identical realized unharvested stock. The figure shows replicate 0, not a mean map.

## 6. What “adaptive observation” means — 50–60 seconds

**On slide:** Two paths from the same starting graph: **Fixed: retain four sources** and **Adaptive: sometimes replace one source**. Add a compact rule: “Every 50 steps → error > .25 → 10% rewire chance.”

**Figure:** Redraw Draft 2's fixed/adaptive diagram with at most four source dots and one replacement arrow. Avoid the old line “25% chance to have a source outside local view.”

**Speaker notes:** Five initial visibility profiles are crossed with fixed and adaptive dynamics. Adaptive candidate search is 75% two-hop and 25% global *conditional on a rewire attempt*. The 25% is neither the chance of rewiring at each step nor a guarantee that the selected peer is outside the old local view. The fixed and adaptive pair begins from the same observation graph.

## 7. Study design and measures — 50–60 seconds

**On slide:** A large `3 landscapes × (5 profiles × 2 dynamics + B0)` design line. Beneath it, show `N=64 · 5,000 training steps · 1,000 fresh-evaluation steps · paired seeds`. Separate small labels: “pilot n=10” and “independent full campaign n=100.”

**Figure:** One clean matrix or flow strip. Do not repeat the maps at thumbnail size. A bottom row can name the five readouts: local-view error, observer-count Gini, low-extraction share, resource fraction, reserve welfare/wealth Gini.

**Speaker notes:** Profiles are equal, random, normal-centered, low-propensity majority, and high-propensity majority. B0 is an ecological baseline, not a social-observation network treatment. Paired seeds support matched contrasts; the independent replicate, not an agent or checkpoint, is the inferential unit. The primary sustainability measure is **total resource / total capacity** on fresh evaluation, not mean local fill fraction.

## 8. H1 — unequal attention and local-view error — 60–75 seconds

**On slide:** “**H1:** Across training, greater inequality in actual observer counts is associated with greater local-view error.” Show two large interval estimates: within-run `r = −0.079 [−0.143, −0.021]`; after linear time adjustment `r = +0.036 [+0.011, +0.063]`. Put a visible zero reference and conclusion: “Sign reverses after time adjustment.”

**Figure:** Replot these two estimates from [H1 checkpoint summary](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/data/h1_checkpoint_association_summary.csv) as two horizontal intervals. The raw checkpoint hexbin in [candidate 3](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/03_different_windows.png) belongs in backup because it does not itself show the time-adjustment sensitivity.

**Speaker notes:** The primary estimate centers observer-count Gini and mean absolute local-view error within each adaptive run, then pools matched training checkpoints and bootstraps replicate IDs as clusters. The raw pilot association is opposite the predicted positive sign; time adjustment reverses it. A separate between-run mean estimate is +0.094 [−0.101, +0.294]. Do not call H1 supported or refuted from this pilot, and do not describe an association as a causal effect of visibility.

## 9. H2 — spatial segregation — 50–65 seconds

**On slide:** “**H2:** Segregated layout produces greater final-window local-view error than dispersed layout.” Show one large `segregated − dispersed` interval: `+0.0011 [−0.0016, +0.0040]`; add “uncertain in pilot.”

**Figure:** Replot the pooled paired contrast from [pooled hypothesis summary](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/data/pooled_hypothesis_summary.csv) as a horizontal interval with zero clearly labeled. Keep the ten treatment-level contrasts from [candidate 3](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/03_different_windows.png) in backup.

**Speaker notes:** H2 averages equally across five initial profiles and fixed/adaptive dynamics. For each replicate, first pair segregated minus dispersed within each social-treatment cell, then average the ten differences; bootstrap replicates. The interval includes zero. Both unequal maps have the same total capacity and capacity histogram. H2's directional wording was set after the exploratory pilot existed, so the pilot cannot confirm it.

## 10. H3 — adaptive versus fixed observation — 50–65 seconds

**On slide:** “**H3:** Adaptive bounded rewiring reduces final-window local-view error relative to fixed observation.” Show `adaptive − fixed = −0.0029 [−0.0050, −0.0010]`; add “pilot-aligned; independent test pending.”

**Figure:** Replot the pooled paired contrast from [pooled hypothesis summary](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/data/pooled_hypothesis_summary.csv) with a zero line. Use a matching visual scale and color convention with slide 9 where possible.

**Speaker notes:** H3 averages the fifteen ecology × profile paired adaptive-minus-fixed differences within each replicate. The two conditions start from the same graph. This pilot interval is below zero, but the directional prediction is pilot-informed and therefore not confirmatory. The detailed effect varies by cell, as [candidate 5](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/05_adaptation_pathway.png) shows.

## 11. RQ2 — from attention to collective extraction — 60 seconds

**On slide:** “What changes when attention adapts?” Use three labeled steps: **who is seen → local-view error → low-extraction share**. Under each, show one concise adaptive-minus-fixed summary or one emphasized example row. Avoid asking the audience to read all 15 treatment cells live.

**Figure:** Make a presentation cut of [candidate 5](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/05_adaptation_pathway.png) from [adaptive–fixed summaries](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/data/adaptive_fixed_summary.csv). Preserve the full three-panel figure for backup. If using the existing figure unchanged, give it the entire slide and explicitly trace **one** profile left to right.

**Speaker notes:** Adaptive observation often increases visibility concentration, especially from an initially equal graph, while local-view error can fall. Changes in collective extraction are heterogeneous and do not support a universal directional story from the pilot. The full figure's cell intervals are pointwise and descriptive. The plotted sequence is a narrative ordering of measures, not a demonstrated causal mediation chain.

## 12. RQ2 — sustainability, welfare, and wealth — 60 seconds

**On slide:** “What happens to the commons and to agents?” Put **resource persistence** on the horizontal axis, with **reserve welfare** and **wealth Gini** as two clearly separated outcomes. If showing the high/low-region comparison, label it “segregated world, training end.”

**Figure:** [Candidate 4: stock and wellbeing](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/04_stock_and_wellbeing.png). For live presentation, the strongest version is a two-panel cut of its first two panels; move its regional-gap panel to backup. Keep the same landscape colors and B0/fixed/adaptive symbols. The source is [outcome summary](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/data/outcome_summary.csv).

**Speaker notes:** The first two panels show 33 condition means from fresh evaluation, not a pooled regression. The regional-gap panel uses training-end reserves and a different comparison unit; do not silently combine it with fresh-evaluation outcomes. B0 is the ecological baseline. These plots help describe the joint outcome space; they do not prove that segregation reduces welfare or that changes in visibility cause wealth inequality.

## 13. Conclusion — 40–50 seconds

**On slide:** Two large statements: **Established:** “Commons-payoff criterion supported in all six ecology × return cells.” **Open:** “H1 time-sensitive; H2 uncertain; H3 pilot-aligned.” Final line: “The independent full campaign tests the frozen learning hypotheses.”

**Figure:** None. Do not reuse Draft 2's old conclusion that the ecology prevented evaluation as a dilemma, or imply that a Melting Pot/SocialJAX migration is needed to establish the criterion.

**Speaker notes:** The structural gate and the learning study answer different questions. The gate supports a commons-dilemma interpretation for its tested policies, horizon, and reset distribution. It does not guarantee that learning agents will cooperate. The independent full campaign is needed for prospective H1–H3 evaluation. A framework migration would be a separate comparability or engineering decision.

## Backup slides (show only for questions)

1. [Candidate 3: treatment-level H2 contrasts and H1 checkpoints](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/03_different_windows.png). Use for questions about heterogeneous effects and the pooled estimands; remind listeners that the plotted H1 checkpoint slope changes sign after time adjustment.
2. [Candidate 2: who gets seen](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/02_who_gets_seen.png). Use for questions about whether the visibility manipulation actually changed observer counts. Its “top six” agents are re-ranked at each snapshot, so connected points are **not** trajectories of the same six individuals.
3. The full [candidate 5](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/05_adaptation_pathway.png) and the segregated-region panel of [candidate 4](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/04_stock_and_wellbeing.png), if you simplify those for slides 11–12.

## Visual rules for the edit

- Use a title that states the claim or question. Put no more than one main figure on a slide; allow full width for landscape and outcome figures.
- Use one consistent color per landscape—uniform teal, dispersed rust, segregated violet—and separate symbol shapes for B0, fixed, and adaptive where needed.
- On hypothesis slides, show estimate, interval, zero reference, and a plain-language interpretation. State the contrast direction in the title or axis label.
- Keep source and evidence-status labels small but readable. The details of bootstrap, pairing, time windows, and measurement formulas belong in speaker notes.
- Avoid the words “belief” and “misperception” unless explaining that local-view error measures available information. Avoid causal verbs for H1 or for cross-outcome scatterplots.
- Check projection readability at the actual room size. The existing full candidate figures 2, 3, and 5 are better as backup or re-rendered presentation cuts because their many rows become hard to follow at a distance.
