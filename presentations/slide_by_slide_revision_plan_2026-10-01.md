> **Historical Draft-3/pilot plan.** Its pilot numbers and pending-full-run language are superseded. Use the [current 11-slide Draft-4 plan with full-run results](draft4_slide_revision_plan_2026-10-01.md) for the presentation.

# Draft-3 slide revision plan — 1 October 2026

Edit [Draft-3](</home/gavinl/Downloads/Gavin Lip. Cogntive Tools Project Draft-3.pdf>) into a **13-slide, 10–12-minute talk for a mixed audience**. This is an edit guide, not a replacement presentation. Results here are from the **exploratory ten-replicate pilot**. The independent 100-replicate full run is still in progress; replace pilot values and plots only after its analysis passes the saved health checks. The held-out commons-payoff validation is complete and separate from the learning results.

## How the questions and results fit together

The previous plan *did* give RQ1 three result slides, but called them only H1–H3. That hid the connection. RQ1 also asks about **initial visibility profiles**, which deserves its own descriptive slide. Draft-3's present “H2” slide shows wealth inequality, not H2's local-view error; its “H3” slide shows observer inequality and extraction, not H3's local-view error. Relabel and move those secondary outcomes under RQ2.

| Question | Plain-language version for the screen | Evidence in this talk |
|---|---|---|
| **RQ1** | Do four observed peers give agents an accurate picture of the population? | Slide 7: starting profiles; slide 8: H1, attention inequality and view accuracy; slide 9: H2, resource layout; slide 10: H3, fixed versus adaptive observation. |
| **RQ2** | Who gets seen, what do agents choose, and what happens to resources and wellbeing? | Slide 11: actual observer counts and low-extraction choices; slide 12: remaining stock, reserve welfare, and wealth inequality. |

The same comparison can inform both questions, but a slide's **headline must name the outcome actually plotted**. H1–H3 specifically test local-view error; observer-count Gini, extraction, stock, and welfare are RQ2 outcomes. A closely related RQ2 panel can sit next to an RQ1 panel if both labels remain explicit, but the RQ2 panel is not the evidence for an H2/H3 claim.

Use these meanings consistently on screen and in speech:

- **Local-view error:** the absolute gap between the share choosing low extraction among an agent's four observed peers and that share among all other agents. Lower means the four peers are more representative. It measures available information, not what an agent believes.
- **Visibility inequality:** Gini of **actual observer counts**—how many agents observe each individual. Higher means attention is less equally distributed; 0 means equal counts.
- **Low-extraction share:** the share of agents choosing the more restrained action. This is the observed choice measure, not a direct welfare measure.
- **Resource persistence:** resource stock divided by total capacity during the 1,000-step fresh evaluation. It is a finite-horizon outcome, not a claim of indefinite sustainability.
- **Reserve welfare** and **wealth Gini:** separate agent outcomes. Higher wealth Gini means more unequal wealth. Do not call wealth Gini “welfare inequality.”

On pilot slides with intervals, identify the pilot sample and **95% intervals** in a small footer. Slide 12's descriptive scatter has no intervals; label it **descriptive condition means** instead. Use “suggests” or “is compatible with” for pilot results. For H2 and H3, say whether the plotted difference is `segregated − dispersed` or `adaptive − fixed`, and show a zero line. Keep inferential methods and exact hypothesis wording in speaker notes.

## Figure library and layout rule

| Short name | Existing source | Best role |
|---|---|---|
| **Maps** | [Story 01: three capacity maps](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/01_uneven_ground.pdf) | Main slide 4. |
| **Starting attention** | [Analysis 02: initial observer counts](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/figures/02_initial_visibility_manipulation.pdf) | Small crop on slide 7 or backup. |
| **Profile outcomes** | [Analysis 03: local-view outcomes](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/figures/03_population_perception.pdf) | **Top row only**, enlarged on slide 7; full grid in backup. |
| **Profile contrasts** | [Saved paired profile contrasts](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/data/visibility_profile_contrast_summary.csv) | Alternative simple slide-7 chart; filter to `social_perception_error`. |
| **H1/H2** | [Story 03: H2 left, H1 right](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/03_different_windows.pdf) | Split across slides 8 and 9. |
| **H3 pooled** | [Saved pooled H3 summary](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/data/pooled_hypothesis_summary.csv) | Draw one large interval on slide 10. |
| **Attention and choices** | [Story 05: adaptation outcomes](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/05_adaptation_pathway.pdf) | Cut selected panels/rows for slide 11; full figure in backup. |
| **Stock and wellbeing** | [Story 04: stock, welfare, wealth](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/04_stock_and_wellbeing.pdf) | First two panels on slide 12; regional panel in backup. |
| **Full RQ2 grid** | [Analysis 04: visibility and extraction](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/figures/04_collective_organization.pdf) | Backup, or crop a single outcome row. |

“Primary” below means the figure to use in the 10–12-minute talk. “Companion” is an **optional second small visual** that reinforces the same claim; “alternative” replaces the primary. Avoid two complete multi-panel figures on one slide. Use PDF/SVG exports, keep labels readable from the back of a room, and preserve landscape colors (uniform teal, dispersed rust, segregated violet). Fixed is blue and adaptive is orange in the detailed grids.

## Replacement slide sequence

The **Paste onto slide** blocks below contain every new text box outside the plotted figure. Copy the wording as written; the figure's own axis and legend labels stay inside the linked PDF/SVG. The quoted “Say” and “Notes” text is for delivery, not for the screen. The final section transcribes the current Draft-3 text boxes so you can compare page by page.

### 1. Title — 15 seconds

**Paste onto slide:**

```text
Who gets seen in a shared-resource world?
Resource geography, local observation, and collective outcomes
Gavin Lip · 1 October 2026
```

**Figure:** None. A narrow crop of **Maps** is optional. Replace Draft-3's broad “Technology, Access, and Welfare” title so the audience knows the actual study question.

**Say:** “I study what a four-person view reveals about a 64-agent commons, and what changes when resources or observation links change.”

### 2. Why the question matters — 35 seconds

**Paste onto slide:**

```text
Four peers are a small window onto a shared world.
64 agents share resources; social agents observe four peers.
Resource layout → Four observed peers → Choices and outcomes
```

**Primary visual:** Three plain boxes: `resource layout → four observed peers → choices and outcomes`. **Companion:** a simple `4 of 63` icon. Replace Draft-3's empty motivation slide and fold its abstract question diagram into this one idea.

**Say:** “A local sample may miss what most agents are doing. Where resources sit and who becomes visible could change that sample.” The arrows organize the study; they are not a claim of proven mediation.

### 3. A tested commons dilemma — 45 seconds

**Paste onto slide:**

```text
The tested incentives form a commons dilemma.
Held-out payoff gate: 6 of 6 checks supported
Landscape        Summed return        Discounted return
Uniform                 ✓                         ✓
Dispersed               ✓                         ✓
Segregated              ✓                         ✓
```

**Primary visual:** One minimal individual/group incentive diagram plus a 3×2 check grid from the [held-out payoff report](../results/payoff_validation/balanced_gate_v1/analysis/validation_report.md). **Alternative:** one clear payoff-gate interval panel from the [validation figures](../results/payoff_validation/balanced_gate_v1/analysis/endpoints_1000.pdf). Replace Draft-3's screenshot of a different Commons Game and its concluding doubt about whether this model is a social dilemma.

**Notes:** The separate held-out gate tested always-low and always-high policies, 100 replicates per landscape, and both summed and discounted returns. Collective gain, exploitation gap, and fear passed the required simultaneous bounds. This validates the tested incentive structure, not the behavior learned later.

### 4. The three resource worlds — 45 seconds

**Paste onto slide:**

```text
Same capacity. Different spatial arrangement.
Each world has 75 capacity units.
```

**Primary visual:** **Maps**, large, on a shared color scale. **Companion:** a tiny `50 low + 50 high cells` label under dispersed and segregated. **Alternative:** retain Draft-3's three maps only if enlarged and given one common legend.

**Notes:** Uniform has capacity 0.75 in every cell. Dispersed and segregated each have 50 cells at 0.55 and 50 at 0.95; these two differ only in spatial arrangement. The three match total **capacity**, not necessarily realized stock during play.

### 5. What agents do and see — 50 seconds

**Paste onto slide:**

```text
Agents choose how much to take and whom to watch.
Low or high extraction · Four peers per social agent
Fixed: keep the same peers
Adaptive: sometimes replace one peer
```

**Primary visual:** Simplify Draft-3's learning and social-observation diagrams into one agent loop. **Companion:** two four-node mini-networks, one fixed and one with a single changed link. Remove the tiny prose and the incorrect implication that 25% is the chance of having an outside source at any step.

**Notes:** Adaptive search checks every 50 steps; if prediction error exceeds 0.25, a rewire has a 10% chance, then replaces one source. Candidate search is 75% two-hop and 25% global. Fixed and adaptive begin from the same graph. Resource renewal is spatial and depends on remaining local stock; explain this verbally, not in dense on-slide text.

### 6. Questions and comparison design — 55 seconds

**Paste onto slide:**

```text
Two questions guide 33 study conditions.
RQ1  Do four observed peers reflect what the population does?
RQ2  How do geography and observation affect choices, resources, and inequality?
3 landscapes × (5 starting profiles × 2 network rules + no-observation baseline)
64 agents · 5,000 training steps · 1,000 fresh-evaluation steps
```

**Primary visual:** A spacious 3-by-11 condition matrix or the equation above. **Companion:** a slim timeline `5,000 training steps → 1,000 fresh-evaluation steps`. Replace Draft-3's crowded RQ/H paragraph slide and tiny design text. Do not read every hypothesis before showing evidence.

**Notes:** Exact RQ1: “How do capacity-matched resource layouts, initial visibility profiles, and adaptive observation affect how accurately agents' four observed peers represent population extraction behavior?” Exact RQ2: “How do those conditions affect collective extraction, inequality in actual observer counts, resource persistence, reserve welfare, and wealth inequality?” N=64; B0 is an ecological baseline without a social-observation network. The five starting profiles are equal, random, normal-centered, low-propensity majority, and high-propensity majority. H1–H3 appear on slides 8–10.

### 7. RQ1: starting profiles — 50 seconds

**Paste onto slide:**

```text
RQ1 · Starting profiles
No clear accuracy ranking emerged across starting profiles in the pilot.
Local-view error: lower means four peers better reflect the population.
Exploratory pilot · n=10 per condition · pointwise 95% intervals
```
**Primary visual:** Crop and enlarge **only the top row** of **Profile outcomes** (local-view error across five profiles, fixed/adaptive, three landscapes). **Companion:** one small initial-observer-count strip cropped from **Starting attention** to show that the starting profiles really differ. **Alternative:** draw three small landscape panels from **Profile contrasts**, each showing the four fixed-network profile differences with a zero line. That contrast plot is easier for a projected talk if prepared in Canva.

**Notes:** This is the part of RQ1 omitted from the older plan. The pilot's 12 fixed-network profile contrasts are heterogeneous; 11 pointwise intervals cross zero. Do not elevate the remaining single cell to a general profile effect or rank all five profiles. Explain that the starting pattern was manipulated, then show the measured local-view result.

### 8. RQ1 / H1: unequal attention and local views — 60 seconds

**Paste onto slide:**

```text
RQ1 · H1: unequal attention and local views
The pilot points in opposite directions before and after accounting for time.
Within-run r = −0.079    After time adjustment r = +0.036
Exploratory pilot · n=10 per condition · 95% intervals
```

**Primary visual:** **Right panel only** of **H1/H2**. Keep the two intervals, labels, and zero line. **Companion:** a small `higher visibility Gini = fewer agents get much of the attention` icon, not another statistical chart. **Alternative:** redraw just two horizontal intervals at presentation scale.

**Result to say:** “Within a run, the unadjusted association was negative. After a simple time adjustment, it was slightly positive. I cannot give this pilot a stable directional reading.” Pilot correlations: `−0.079 [−0.143, −0.021]` and `+0.036 [+0.011, +0.063]`.

**Notes:** Exact H1: “Across training, greater inequality in actual observer counts is associated with greater local-view error.” The plotted analysis uses repeated training checkpoints, with replicate-cluster intervals. It is an association, not a causal effect. Do not say “there is no association”; say the **direction depends on the adjustment**.

### 9. RQ1 / H2: resource clustering and local views — 55 seconds

**Paste onto slide:**

```text
RQ1 · H2: resource layout and local views
The pilot shows no clear accuracy difference between segregated and dispersed worlds.
Segregated − dispersed: +0.11 percentage points of local-view error
95% interval: −0.16 to +0.40 percentage points
Exploratory pilot · n=10 per condition
```

**Primary visual:** **Left panel only** of **H1/H2**, with the pooled result highlighted. **Companion:** two small map thumbnails from **Maps** above the plot, if space permits. **Alternative:** one large pooled interval from [the H2 summary row](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/data/pooled_hypothesis_summary.csv); move the ten treatment-cell intervals to backup.

**Result to say:** “The estimated difference is about **+0.11 percentage points** of error for segregated minus dispersed, but its interval includes no difference.” Pilot 95% interval: `−0.16 to +0.40 percentage points`.

**Notes:** Exact H2: “Averaged equally over five initial profiles and both network dynamics, segregated resource layout produces greater final-window local-view error than dispersed layout.” The ten fixed/adaptive-by-profile paired contrasts are averaged equally. H2 compares **local-view error**, not wealth inequality. Its directional wording was pilot-informed. Do not turn an interval crossing zero into proof that the landscapes are equivalent.

### 10. RQ1 / H3: changing peers — 55 seconds

**Paste onto slide:**

```text
RQ1 · H3: changing peers
Changing peers slightly improved local views in the pilot.
Adaptive − fixed: −0.29 percentage points of local-view error
95% interval: −0.50 to −0.10 percentage points
Exploratory pilot · n=10 per condition
```
**Primary visual:** Draw one large interval from **H3 pooled**: `adaptive − fixed = −0.29 percentage points` of local-view error, 95% interval `−0.50 to −0.10`. Label the negative side **“more representative with adaptive”**. **Companion:** one small fixed/adaptive network icon from slide 5. **Alternative:** crop the **middle panel only** of **Attention and choices**, but that panel shows condition heterogeneity and does not replace the pooled primary estimate.

**Notes:** Exact H3: “Averaged equally over three landscapes and five initial profiles, adaptive bounded rewiring reduces final-window local-view error relative to fixed observation starting from the same graph.” H3 averages 15 matched landscape-by-profile contrasts equally. This is a **pilot-aligned** result, not a confirmed prospective test. Draft-3's current H3 figure is about visibility inequality and extraction and belongs under RQ2 instead.

### 11. RQ2: who gets seen and what agents choose — 60 seconds

**Paste onto slide:**

```text
RQ2 · Attention and choices
Adaptive observation concentrated attention; low-extraction changes varied.
Who gets seen?                    Who chooses low extraction?
Higher observer-count Gini = less equal attention
Exploratory pilot · n=10 per condition · pointwise 95% intervals
```

**Primary visual:** Make a two-panel talk cut of **Attention and choices**: left `adaptive − fixed visibility Gini`, right `adaptive − fixed low-extraction share`. Show one **named** starting profile across all three landscapes, or draw a compact all-cell summary with large labels. State clearly if a row is illustrative. **Companion:** a one-line badge, `higher observer-count Gini = greater visibility inequality`. **Alternative:** crop the visibility and extraction rows from the **Full RQ2 grid**, enlarged enough to read.

**Notes:** In the pilot, visibility Gini rises under adaptation in all 15 landscape-by-profile cells, while low-extraction-share differences vary by condition. The plotted association of these outcomes is not proof that concentrated attention caused a change in extraction. This is where Draft-3's current “H3” discussion belongs. Avoid “adaptation made agents extract more overall” unless the independent full analysis supports a defined pooled estimate.

### 12. RQ2: stock, reserve welfare, and wealth — 65 seconds

**Paste onto slide:**

```text
RQ2 · Resources and wellbeing
In the pilot, segregated worlds retained more resources and had more unequal wealth.
Reserve welfare was slightly higher, too.
Descriptive condition means · Fresh evaluation · n=10 per condition
```

**Primary visual:** Use the **first two panels** of **Stock and wellbeing**, enlarged: remaining resource fraction versus reserve welfare, and remaining resource fraction versus wealth Gini. **Companion:** a three-word reading key: `stock = environment · reserve = wellbeing · Gini = inequality`. **Alternative:** make three aligned landscape dot plots from [the saved outcome summary](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/data/outcome_summary.csv), one scale per metric. The latter is better for direct fixed/adaptive comparisons than a descriptive scatter.

**Result to say:** “Across the pilot's 11 treatment means, segregated worlds had more remaining resource than dispersed worlds, slightly higher reserve welfare, and more unequal wealth. These are descriptive comparisons; this scatter does not show that stock caused either agent outcome.”

**Notes:** The scatter has 33 condition means. Equal-weight descriptive means across the 11 treatments per world are: remaining resource fraction `0.218` segregated versus `0.187` dispersed; reserve welfare `0.861` versus `0.841`; wealth Gini `0.446` versus `0.435`. These are not pooled uncertainty intervals. Stock, reserve welfare, and wealth Gini in the first two panels come from fresh-reset evaluation. The rightmost regional panel of the full story figure uses **training-end** reserve welfare and belongs in backup, with that phase named. Do not call this a long-run sustainability test. Draft-3's current “H2: welfare inequality ~2% higher” title is unsupported as H2 and should be removed.

### 13. Answer and next evidence — 40 seconds

**Paste onto slide:**

```text
What the study tells us so far
Commons incentives: passed in all three worlds.
RQ1 pilot: H1 changes with time adjustment; H2 unclear; H3 slightly lower error.
RQ2 pilot: attention became less equal; choices and resources varied.
Full 100-replicate analysis: pending.
```

**Figure:** None; optionally repeat the three resource-world thumbnails. Remove Draft-3's statement that the current game may not qualify as a social dilemma. Do not frame migration to Melting Pot or SocialJAX as necessary because the validated gate passed; discuss platform migration only as a separate future engineering choice if asked.

**Say:** “The payoff test answers whether the configured game has a commons dilemma for the tested policies. The learning study asks what agents actually do with local information. The pilot tells us what to inspect; the independent full campaign will carry the main inference.”

## Presentation edits and full-run handoff

1. Replace Draft-3 slides 2–9 with the shorter setup on slides 2–6 above. This creates room for **six** result slides (7–12), so RQ1 and RQ2 both receive direct answers.
2. Replace Draft-3 results slides 10–12. Its H1 graphic can be retained after enlargement and clearer wording; its H2 wealth plot moves to RQ2 slide 12; its H3 visibility/extraction material moves to RQ2 slide 11. Add the missing RQ1 starting-profile and H3 pooled-local-view plots.
3. When `results/q_learning_baseline/social_analysis/balanced_full_v1_analysis/analysis_health.json` exists and reports complete conditions, swap **all** pilot estimates, intervals, result headlines, and figure links to the full analysis together. Do not mix a pilot interval with a full-run figure. Keep the estimands and slide question labels unchanged.
4. Keep full grids, [Story 02: who gets seen](../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/02_who_gets_seen.pdf), the full **Attention and choices** figure, and Story 04's regional training-end panel as backup. The top-six-share in Story 02 re-ranks agents at each snapshot; it is not the history of the same six agents.

At projected size, prefer a short takeaway headline, one main comparison, direct labels, and a small pilot footer. Put definitions, pairing, phases, and limitations in speaker notes. Never use “perception” as a synonym for elicited belief; “what four peers reveal” or “local-view accuracy” is clearer and closer to what was measured.

## Current Draft-3 text boxes, by PDF page

This is the selectable **text in the current PDF**, transcribed to let you find each slide quickly. It preserves the current wording and spelling; line wrapping and placement may differ. Text baked into plotted images is not reliably extractable, so compare those figures visually with the PDF. Because the new sequence reallocates setup pages to results, a current page and its proposed page number need not cover the same topic.

**Current page 1**

```text
Technology, Access, and Welfare in Artificial Societies
Social-Technical Multi-Agent Environments
```

**Current page 2**

```text
Motivation
```

**Current page 3**

```text
So far
1. baseline resource world, (September)**
2. Inheritance, tool-building, trade, and social learning.(October)**
Implemented q-learning algorithm
Did more research on social dilemma games and frameworks
Revised experiment and constrained research questions
Implemented social network configurations
Performed full experiment run
Now reflecting on results, design of study, and future directions
```

**Current page 4**

```text
How does ecology shape perception and collective organization?
Local perception
What do selected peers do?
How representative is that view?
Collective organization
Who becomes visible?
Which actions become common?
Consequences
Resource condition
Welfare and inequality
* Middle figure taken from (schrama, MajorityIllusionDrives, 2025)
```

**Current page 5**

```text
Commons Game
Players collect shared resources (“apples”).
Apple regrowth depends on nearby unharvested apples.
If a local area is fully depleted, apples cannot regrow until reset.
Individual incentives favor rapid harvesting, while collective welfare requires restraint.
Simultaneous harvesting increases the risk of irreversible local depletion.
https://github.com/Danfoa/commons_game/tree/master
https://papers.nips.cc/paper_files/paper/2017/hash/2b0f658cbffd284984fb11d90254081f-Abstract.html
```

**Current page 6**

```text
Commons Game with Ecological Mechanics
Learn extraction
Local resource state + observed peer behavior
Resource renewal
Low or high extraction; renewal and spatial coupling
Replace a source
If surprise is high: local or global search
Predict and compare
Expected vs. observed peer behavior
```

**Current page 7**

```text
Social Observations
FIXED
Observe peers
ADAPTIVE BOUNDED
Every 50 steps → surprise > .25 → 10% rewire chance → replace 1 source
25% chance to have a source outside of local view
k = 4 sources per observer
```

**Current page 8**

```text
Research Questions & Hypotheses
RQ1: To what effect does ecological and social conditions manipulate local observations represention of population extraction behavior?
RQ2: How do ecology and social-observation conditions shape inequality, welfare and population extraction behavior?
H1 Across training, greater inequality in actual observer counts is associated with greater error in agents’ local views of population extraction.
H2 Across observation conditions, segregated resource environments produce greater local-perception error than dispersed environments with the same total resources.
H3 Across resource environments and starting visibility conditions, social adaptation reduces local-perception error compared with an otherwise identical fixed network.
```

**Current page 9**

```text
Design of Study
3 landscapes × (5 profiles × 2 dynamics + B0)
Profiles
Dynamics
N=64 · 5,000 training steps · 1,000 evaluation steps
```

**Current page 10**

```text
(H1) Currently there are no stable assications in observer counts with greater local-view errors.
Low bi-direction association
```

**Current page 11**

```text
H2 Welfare inequality is around ~2% higher for segregated worlds
```

**Current page 12**

```text
(H3) Attention adaptation actually increased visiblity inequality and marginally decreased overall low-extraction behavior in population
```

**Current page 13**

```text
Conclusions
Perception
Ecology and social observation jointly shape what agents see.
Visibility
Visibility only slighty changed collective outcomes
Adaptation
One bounded rewiring rule gives a clean Fixed-vs-Adaptive comparison.
Some current thoughts:
Ecological mechanics are complex have made it difficult to evaluate the framework as a true social dilemma games
There are some interesting frameworks I found later on that might better to migrate to (i.e Melting Pot, SocialJAX)
```
