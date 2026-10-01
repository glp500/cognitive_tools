# Updated slide edit plan — 1 October 2026

For [Draft 2](</home/gavinl/Downloads/Gavin Lip. Cogntive Tools Project Draft-2.pdf>). This revision incorporates the [figure audit](../docs/reviews/visibility-figure-audit-2026-10-01.md) and the regenerated pilot figures. It is an **editing plan**, not a new slide deck. Aim for **13 main slides in 10–12 minutes** for a mixed audience. Use one visual and one take-home point per slide; move methods and caveats to speaker notes.

The independent full campaign is still running. Slides 8–12 below use **exploratory pilot examples (n=10)**. Mark them that way on the slide. When the full analysis passes its health checks, update the values and figures without changing the pre-specified comparisons or wording. The held-out commons-payoff gate on slide 3 is a separate completed validation.

## Figure choices at a glance

| Figure | Use in the talk? | Reason |
|---|---|---|
| [Story 01 · three capacity maps](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/01_uneven_ground.pdf) | **Yes, slide 5.** | The three maps and shared scale show the matched resource budget immediately. |
| [Story 02 · who gets seen](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/02_who_gets_seen.pdf) | **Backup.** | Five stacked profiles are valuable evidence but too much to read during a short talk. |
| [Story 03 · local views](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/03_different_windows.pdf) | **Yes, split its panels across slides 8–9.** | The right panel shows H1's sign reversal; the left shows the ten primary H2 contrasts. Do not shrink the whole two-panel figure onto one slide. |
| [Story 04 · stock and wellbeing](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/04_stock_and_wellbeing.pdf) | **Yes, first two panels on slide 12.** | Resource persistence is on the common horizontal axis. Put the separate training-end regional panel in backup. |
| [Story 05 · adaptation pathway](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/05_adaptation_pathway.pdf) | **One selected row or a simplified cut on slide 11; full figure in backup.** | The complete 15-cell display is too dense when projected. |
| [Analysis 02 · initial visibility](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/figures/02_initial_visibility_manipulation.pdf) | **Backup.** | Clear manipulation check; use if asked whether the initial profiles produced unequal actual observer counts. |
| Other [analysis grids](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/figures/03_population_perception.pdf) | **Backup or appendix.** | They document every condition but do not give a quick visual answer to one hypothesis. |

Use PDF or SVG exports when editing so labels remain sharp. Landscape colors are **uniform teal, dispersed rust, segregated violet**; the analysis grids use **fixed blue and adaptive orange** with different marker shapes. Keep the same key when drawing a new summary interval.

## Slide-by-slide edits

### 1. Title — 20 seconds

**Replace Draft 2 slide 1 title with:** “Who gets seen in a shared-resource world?” Subtitle: “Ecology, local observation, and collective outcomes.” Add your name and date.

**Figure:** No statistical chart. A small crop of the three maps from story 01 is optional; the maps get their full explanation on slide 5.

**Speaker notes:** This is an agent-based commons study about what agents can observe and what the group does. It does not measure human beliefs.

### 2. Why this question matters — 40 seconds

**On-slide sentence:** “What agents see depends on where resources are and whose actions become visible.”

**Figure:** Draw three large boxes: `resource geography → four observed peers → group outcomes`. Under the last box, use only `extraction · stock · welfare · inequality`. Condense Draft 2 slides 2–4 into this one visual.

**Speaker notes:** The “technological mediation” motivation from Draft 2 belongs here in spoken form. The observation network shapes the information available to an agent; ecology shapes the context in which extraction occurs. This diagram organizes the study, but arrows alone do not establish causal mediation.

### 3. Is this a commons dilemma? — 45 seconds

**On-slide sentence:** “The tested incentives pass the commons-dilemma gate in all three landscapes.”

**Figure:** A 3×2 check grid: rows `uniform / dispersed / segregated`; columns `summed return / discounted return`. Show `6 of 6 supported`. Build it from the [held-out payoff report](../results/payoff_validation/balanced_gate_v1/analysis/validation_report.md). Replace Draft 2 slide 5's screenshot of a different Commons Game.

**Speaker notes:** The gate tested always-low and always-high policies, N=64, horizon 1,000, and 100 held-out replicates per landscape. Collective gain, exploitation gap, and fear met the required simultaneous bounds. This validates the tested incentive structure; it does **not** say what learned agents will choose.

### 4. The revised questions — 50 seconds

**Put only these two questions on the slide:**

> **RQ1:** How do capacity-matched resource layouts, initial visibility profiles, and adaptive observation affect how accurately agents' four observed peers represent population extraction behavior?

> **RQ2:** How do those conditions affect collective extraction, inequality in actual observer counts, resource persistence, reserve welfare, and wealth inequality?

**Figure:** None. Use two spacious text cards. Replace the old RQ/H wording on Draft 2 slide 8; do not squeeze the hypotheses onto this slide.

**Speaker notes:** “Actual observer count” is how many agents select a given agent as a source. “Local-view error” is the difference between the low-extraction share among four observed peers and the share among all other agents, averaged in absolute value. It measures available information, **not** an elicited belief.

### 5. The resource manipulation — 50 seconds

**On-slide title:** “The worlds have the same capacity, arranged differently.”

**Figure:** Use [story 01](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/01_uneven_ground.pdf) nearly full width. Preserve the three map names, shared color scale, and `K=75` labels. This replaces the small maps on Draft 2 slide 9.

**Speaker notes:** Uniform has K=0.75 in every cell. Both unequal maps have fifty K=0.55 and fifty K=0.95 cells, so dispersed versus segregated differs in arrangement, not total capacity or capacity histogram. Other resource parameters are fixed. The displayed map is replicate 0; matching configured capacity does not force identical realized stock.

### 6. How observation can change — 50 seconds

**On-slide title:** “Adaptive agents can replace one observed peer.”

**Figure:** Redraw Draft 2 slide 7 as two simple paths from the **same starting graph**: `fixed: keep four peers` and `adaptive: check → sometimes replace one`. Under adaptive, print `every 50 steps · error > .25 · 10% rewire chance`.

**Speaker notes:** Conditional on a rewire, candidate search uses a 75% two-hop / 25% global branch. The 25% is **not** a 25% chance of having an outside observer on every step. Both dynamics begin from the same graph. Keep the five profile names for the next slide or in notes.

### 7. The comparison design — 55 seconds

**On-slide title:** “We change geography and observation, then compare matched runs.”

**Figure:** A simple matrix: `3 landscapes × [5 initial profiles × fixed/adaptive + B0]`. Under it: `64 agents · 5,000 training steps · 1,000-step fresh evaluation · paired seeds`. Use `pilot: 10 replicates` and `independent full study: 100` as separate status labels. Do not use a multi-panel results plot here.

**Speaker notes:** The profiles are equal, random, normal-centered, low-propensity majority, and high-propensity majority. B0 is an ecological baseline, not an observation treatment. RQ1's primary outcome is local-view error. RQ2 records low-extraction share, observer-count Gini, total resource/total capacity, mean reserve welfare, and final wealth Gini. Independent replicates are the inferential units.

### 8. H1: unequal attention — 65 seconds

**On-slide title:** “H1: More unequal attention should accompany worse local views.”

**Figure:** Use **only the right panel** of the updated [story 03](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/03_different_windows.pdf). Keep both intervals and the zero line. Add a large takeaway below: **“The pilot association changes sign after time adjustment.”**

**Pilot values:** within-run `r = −0.079 [−0.143, −0.021]`; after linear time adjustment `r = +0.036 [+0.011, +0.063]`.

**Speaker notes:** Exact H1: “Across training, greater inequality in actual observer counts is associated with greater local-view error.” The primary longitudinal estimate uses adaptive-run checkpoints and clusters the interval by replicate. The sign reversal makes the pilot inconclusive for a stable directional interpretation. This is an association, not a causal effect. Mark the slide **Exploratory pilot · n=10**.

### 9. H2: segregated versus dispersed — 60 seconds

**On-slide title:** “H2: Does spatial segregation make the local view less accurate?”

**Figure:** Use **only the left panel** of updated [story 03](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/03_different_windows.pdf). Circle or otherwise highlight the pooled line in its axis label; do not explain all ten rows aloud. If the audience is distant, replace the panel with a single large pooled interval drawn from the [summary table](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/data/pooled_hypothesis_summary.csv).

**Pilot value:** `segregated − dispersed = +0.0011 [−0.0016, +0.0040]`; the interval includes zero.

**Speaker notes:** Exact H2: “Averaged equally over five initial profiles and both network dynamics, segregated resource layout produces greater final-window local-view error than dispersed layout.” The ten cell contrasts are paired by replicate and then averaged. The pilot does not establish a difference. Its directional wording was chosen after seeing pilot data. Mark the slide **Exploratory pilot · n=10**.

### 10. H3: adaptive versus fixed — 60 seconds

**On-slide title:** “H3: Does adapting attention improve the local view?”

**Figure:** Draw one large horizontal estimate-and-interval chart from the [pooled H3 row](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/data/pooled_hypothesis_summary.csv). Label `adaptive − fixed` and zero. **Do not** paste the full 15-cell story 05 here; it obscures the primary result.

**Pilot value:** `−0.0029 [−0.0050, −0.0010]`. Add “pilot-aligned; prospective test pending,” not “confirmed.”

**Speaker notes:** Exact H3: “Averaged equally over three landscapes and five initial profiles, adaptive bounded rewiring reduces final-window local-view error relative to fixed observation starting from the same graph.” The analysis averages 15 paired cell differences within each replicate. The direction was pilot-informed. Mark the slide **Exploratory pilot · n=10**.

### 11. What else changes with adaptation? — 60 seconds

**On-slide title:** “Changing attention can concentrate visibility without a uniform extraction effect.”

**Figure:** Make a **talk cut** of [story 05](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/05_adaptation_pathway.pdf): show one named initial profile across its three panels, or show three pooled/landscape summaries. Keep `who is seen → what four peers reveal → low-extraction share` as the reading order. The full five-profile figure belongs in backup.

**Speaker notes:** The pilot often shows more concentrated actual observer counts after adaptive rewiring; local-view error can improve, while low-extraction-share changes vary by condition. The plot is a comparison of related outcomes, **not proof of a causal pathway**. The full figure uses pointwise cell intervals. Mark the slide **Exploratory pilot · n=10**.

### 12. What happens to the resource and to agents? — 60 seconds

**On-slide title:** “Resource persistence, welfare, and wealth can move differently.”

**Figure:** Show the **first two panels only** from [story 04](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/04_stock_and_wellbeing.pdf): resource/total capacity on the common x-axis, reserve welfare on the left, wealth Gini on the right. Preserve landscape colors and B0/fixed/adaptive symbol key. Put its regional-gap panel in backup because it is measured at training end, not fresh evaluation.

**Speaker notes:** The first two panels are descriptive means for 33 conditions in fresh evaluation, not a pooled regression or a test of mediation. B0 is the ecological baseline. The regional panel compares high- and low-capacity areas in the segregated world at **training end**. Avoid claiming that segregation necessarily lowers welfare. Mark the slide **Exploratory pilot · n=10**.

### 13. Closing answer and next step — 40 seconds

**On-slide title:** “The commons criterion passed; the learning questions remain open.”

**On-slide text:** Two cards: **Established:** “Held-out commons gate: 6/6 supported.” **Learning pilot:** “H1 time-sensitive · H2 uncertain · H3 pilot-aligned.” End with “The independent full campaign tests the frozen hypotheses.” No figure needed.

**Speaker notes:** The payoff gate addresses the game's incentives for a particular policy pair and evaluation setup. It does not guarantee cooperation by learned agents. Replace the pilot assessments only when the independent full analysis is complete. Remove Draft 2's old conclusion that the model could not be evaluated as a dilemma; framework migration is a separate decision.

## Backup slides and editing rules

Keep [story 02](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/02_who_gets_seen.pdf) and [analysis 02](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/figures/02_initial_visibility_manipulation.pdf) for questions about whether visibility profiles worked. Keep the **full** [story 05](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/05_adaptation_pathway.pdf) for condition heterogeneity and the **regional panel** of [story 04](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/04_stock_and_wellbeing.pdf) for place-based welfare. The top-six panel in story 02 re-ranks agents at each snapshot; it is not a trajectory of the same six people.

At projected size, use large titles and axes, one comparison per slide, and no paragraph captions inside plots. Explain intervals, pairing, and limitations in notes. Keep `local-view error` as the measured term; avoid “belief” or “misperception” unless explicitly distinguishing them from available information. Use “associated with” for H1 and “paired difference” for H2/H3. After full results arrive, retain the same figure selections and contrasts but replace every pilot number, status label, and source path with the verified full-analysis version.
