# Draft-4 slide revision plan — 1 October 2026

This guide follows the **11-page order** of [Draft-4](</home/gavinl/Downloads/Gavin Lip. Cogntive Tools Project Draft-4.pdf>): title, question, commons model, mechanics, observation, questions/hypotheses, design, H1, H2, H3, conclusion. It is an editing guide, not a new deck. Each `Paste onto slide` block is the **complete proposed slide text outside the figure**. Figure axes and legends remain in the linked figure. The numbered edits describe exactly what to change on the matching PDF page; presenter notes stay off the slide.

The 100-replicate full study has now been analyzed and its recorded health checks pass. All result numbers and linked figures below use that **independent full study**. The separate held-out commons-payoff gate passed six of six checks. The [recovery audit](../docs/reviews/balanced-full-run-recovery-2026-10-01.md) explains the two recorded Git commits and three runs whose original worktree was dirty; keep that provenance qualification in presenter notes and any manuscript report.

The two research questions share the three result slides. **H1–H3 test local-view error for RQ1.** Observer-count inequality, extraction choices, remaining resources, reserve welfare, and wealth inequality answer RQ2. A slide may show one result from each question, but they must have separate panel labels. Higher observer-count Gini means **more unequal visibility**; lower local-view error means four peers better represent the other agents' low-extraction choices. Resource persistence is measured over a 1,000-step fresh evaluation, not indefinitely.

## Slide 1 — Title

**Current page:** “Technology, Access, and Welfare in Artificial Societies,” with images from an older set of worlds.

**Paste onto slide:**

```text
Who gets seen in a shared-resource world?
Resource geography, local observation, and collective outcomes
Gavin Lip · 1 October 2026
```

**Figure:** None is needed. Optional: a slim crop of the [three matched-capacity maps](../results/q_learning_baseline/social_analysis/balanced_full_v1_analysis/story_candidates/01_uneven_ground.pdf).

**Edit steps:**

1. Replace both existing title lines with the copy block.
2. Remove the six-world strip and older resource image; if retaining an image, show only uniform, dispersed, and segregated.
3. Keep the title larger than any image label.

**Presenter note:** “This is an agent-based commons study about what four observed peers reveal in a 64-agent world.”

## Slide 2 — Why local views matter

**Current page:** “How does ecology shape perception and collective organization?” with an external middle figure and three busy columns.

**Paste onto slide:**

```text
A four-person view may miss what the group is doing.
Resource layout → Four observed peers → Choices and outcomes
64 agents share the resource; each social agent observes four peers.
```

**Figure:** Redraw the current three columns as **one large three-box flow diagram**. Optional small companion: a `4 of 63` icon. Remove the Schram figure and its citation if the figure is removed.

**Edit steps:**

1. Keep the current left-to-right story, but use the exact three box labels in the copy block.
2. Remove the small embedded plots and the external figure credit.
3. Use “four-person view” rather than “belief” or “perception” on screen: the metric measures available information.

**Presenter note:** The arrows organize the comparisons; the later plots do not prove a causal path from attention through choices to welfare.

## Slide 3 — Commons-game validation

**Current page:** “Commons Game,” apples and screenshots from another game, with two URLs.

**Paste onto slide:**

```text
The tested incentives form a commons dilemma.
Agents choose low or high extraction from a shared, renewing resource.
Held-out payoff gate: 6 of 6 checks supported
Landscape        Summed return        Discounted return
Uniform                 ✓                         ✓
Dispersed               ✓                         ✓
Segregated              ✓                         ✓
```

**Figure:** A simple shared-resource loop beside the grid above. Build the grid from the [held-out gate report](../results/payoff_validation/balanced_gate_v1/analysis/validation_report.md). Alternative: one enlarged [endpoint figure](../results/payoff_validation/balanced_gate_v1/analysis/endpoints_1000.pdf) with a `6 of 6` badge.

**Edit steps:**

1. Delete the unrelated game's screenshot, apple wording, and URLs.
2. Draw your own low/high extraction and local-regrowth loop.
3. Add the six-check grid; keep technical payoff criteria in notes.

**Presenter note:** Always-low versus always-high policies were tested in 100 held-out replicates per landscape. Collective gain, exploitation gap, and fear passed simultaneous bounds for summed and discounted returns. This validates the tested incentives, not the choices learned by agents.

## Slide 4 — Model mechanics

**Current page:** “Commons Game with Ecological Mechanics,” four small cards and a hard-to-read spatial screenshot.

**Paste onto slide:**

```text
Agents choose; the shared resource responds.
Choose: low or high extraction
Regrow: local stock renews when resources remain
Learn: past rewards guide later choices
```

**Figure:** Three large stages in a loop: `choose → resource changes → learn`. A small crop of one [matched-capacity map](../results/q_learning_baseline/social_analysis/balanced_full_v1_analysis/story_candidates/01_uneven_ground.pdf) may locate the agents. The existing screenshot is too small for a projected talk.

**Edit steps:**

1. Replace the four tiny cards with the three stages above.
2. Move “replace a source” to slide 5, where network change is explained.
3. Remove the screenshot or label it clearly as an illustrative spatial state.

**Presenter note:** Q-learning uses local resource state and observed peer behavior; regrowth depends on remaining nearby resources. The payoff gate on slide 3, rather than this mechanism diagram, establishes the tested commons criterion.

## Slide 5 — Observation rules

**Current page:** Fixed/adaptive diagram with “25% chance to have a source outside of local view,” which misstates the rule.

**Paste onto slide:**

```text
Every social agent observes four peers.
Fixed: keep the starting peers
Adaptive: sometimes replace one peer
Check every 50 steps → error > 0.25 → 10% chance to rewire
When replacing: 75% nearby search · 25% global search
```

**Figure:** Enlarge the existing fixed/adaptive cartoons. Start both branches from **the same four-peer graph**; mark only one replacement arrow in the adaptive branch.

**Edit steps:**

1. Replace all tiny network labels with the copy block.
2. Correct the 25% claim: it is the candidate-search branch **conditional on replacement**, not a per-step chance of having an outside observer.
3. Show the shared starting graph visually before the two branches.

**Presenter note:** The 10% rewire chance applies only after a threshold-crossing check. Candidate search is 75% two-hop and 25% global.

## Slide 6 — Research questions and hypotheses

**Current page:** Two long questions and three paragraph-length hypotheses. RQ2 says “social conditions” and omits resource layout.

**Paste onto slide:**

```text
Two questions, three local-view hypotheses
RQ1  When do four peers represent population extraction?
RQ2  How do layout and observation affect choices, visibility, resources, welfare, and inequality?
H1  More unequal observer counts are associated with higher local-view error
H2  Segregated layout → higher local-view error than dispersed layout
H3  Adaptive observation → lower local-view error than fixed observation
```

**Figure:** No result plot. Make two spacious RQ cards above a three-row H1/H2/H3 list. Group the hypotheses under RQ1.

**Edit steps:**

1. Replace the current paragraphs with the exact short copy above and enlarge the type.
2. Group H1–H3 under RQ1; reserve RQ2 outcome definitions for the result slides.
3. Keep the exact protocol wording in notes, rather than shrinking it to fit on screen.

**Presenter note:** Exact RQ1: “How do capacity-matched resource layouts, initial visibility profiles, and adaptive observation affect how accurately agents' four observed peers represent population extraction behavior?” Exact RQ2: “How do those conditions affect collective extraction, inequality in actual observer counts, resource persistence, reserve welfare, and wealth inequality?” H1 is an association; H2 averages ten profile-by-dynamics contrasts; H3 averages fifteen landscape-by-profile contrasts. H2 and H3's directional wording was pilot-informed.

## Slide 7 — Comparison design

**Current page:** “Design of Study,” three maps, and profile/dynamics text too small to read.

**Paste onto slide:**

```text
Three worlds. The same total capacity.
Uniform · Dispersed · Segregated     K = 75 in each world
3 landscapes × (5 starting profiles × 2 network rules + no-observation baseline)
64 agents · 5,000 training steps · 1,000 fresh-evaluation steps
Full study: 100 new replicates per condition · 3,300 simulations
```

**Figure:** Use the [three matched-capacity maps](../results/q_learning_baseline/social_analysis/balanced_full_v1_analysis/story_candidates/01_uneven_ground.pdf) on one common color scale. Below them, use the single design equation or a thin 3×11 matrix. Optional tiny timeline: training → fresh evaluation.

**Edit steps:**

1. Replace or enlarge the three maps so their shared K=75 and color scale are readable.
2. Replace the tiny “Profiles” and “Dynamics” lists with the large design equation.
3. Use one full-study sample-size badge; identify the ten-replicate pilot only in presenter notes if needed.

**Presenter note:** Uniform has 100 cells at K=.75. Dispersed and segregated each have 50 at K=.55 and 50 at K=.95; only the arrangement differs between those two. B0 is the no-social-observation baseline. Equal capacity need not mean equal realized stock.

## Slide 8 — H1 and starting profiles

**Current page:** “Currently there are no stable assications…” with dotted arrows and “Low bi-direction association.”

**Paste onto slide:**

```text
RQ1 / H1 · Who gets seen and what four peers reveal
Primary H1 result runs opposite the prediction.
Within-run r = −0.088    Time-adjusted r = +0.035
Starting-profile accuracy effects varied by world.
Full study · n=100 per condition · 95% intervals
```

**Primary figure:** Enlarge the **right H1 panel** of [the H1/H2 story figure](../results/q_learning_baseline/social_analysis/balanced_full_v1_analysis/story_candidates/03_different_windows.pdf). Optional companion: a **small profile-contrast strip** from [the paired profile table](../results/q_learning_baseline/social_analysis/balanced_full_v1_analysis/data/visibility_profile_contrast_summary.csv), or a top-row crop of [local-view outcomes](../results/q_learning_baseline/social_analysis/balanced_full_v1_analysis/figures/03_population_perception.pdf). Keep [initial observer counts](../results/q_learning_baseline/social_analysis/balanced_full_v1_analysis/figures/02_initial_visibility_manipulation.pdf) as backup; they show the manipulation, not accuracy.

**Edit steps:**

1. Replace the current headline with the primary-result statement; show the time-adjusted sign change as a sensitivity result.
2. Remove the dotted arrows and tiny side annotation; retain the two intervals and zero line.
3. Add the starting-profile sentence because RQ1 includes those profiles. Omit the optional second plot if it shrinks H1 below readable size.

**Presenter note:** The primary raw within-run correlation is `−0.088 [−0.111, −0.066]`, opposite H1. After linear time adjustment it is `+0.035 [+0.026, +0.046]`; a secondary comparison of run averages centered within treatment is `+0.114 [+0.044, +0.184]`. Repeated checkpoints use replicate-cluster intervals. These are associations, not causal effects. Ten of 12 fixed-network profile-contrast pointwise intervals cross zero, so do not claim a universal profile ranking.

## Slide 9 — H2 and resource outcomes

**Current page:** “H2 Welfare inequality is around ~2% higher…,” although H2 concerns local-view error. The stock/wealth scatter is a separate RQ2 outcome.

**Paste onto slide:**

```text
Resource layout changes more than one outcome.
RQ1 / H2 · No clear difference in local-view error
Segregated − dispersed: +0.10 percentage points [−0.01, +0.22]
RQ2 · More resource remains in the segregated world; wealth is less equal
Full study · n=100 per condition · RQ2 values are descriptive
```

**Primary figures:** Left: **one large pooled H2 interval** from [the pooled estimate table](../results/q_learning_baseline/social_analysis/balanced_full_v1_analysis/data/pooled_hypothesis_summary.csv), with zero. Right: a **three-row segregated-versus-dispersed dot comparison** from [the outcome table](../results/q_learning_baseline/social_analysis/balanced_full_v1_analysis/data/outcome_summary.csv): remaining resource, reserve welfare, wealth Gini, each on its **own labeled scale**. Existing-figure alternative: use only the **left H2 panel** of the [H1/H2 story figure](../results/q_learning_baseline/social_analysis/balanced_full_v1_analysis/story_candidates/03_different_windows.pdf) at full width and move RQ2 to backup. Do not squeeze the complete [Stock and wellbeing](../results/q_learning_baseline/social_analysis/balanced_full_v1_analysis/story_candidates/04_stock_and_wellbeing.pdf) figure beside H2.

**Edit steps:**

1. Delete the “welfare inequality ~2%” headline; wealth Gini is not H2.
2. Replace the dense ten-row H2 panel with a pooled interval, or enlarge it to full width if using the existing PDF.
3. Label the RQ2 panel as **descriptive condition means**, not as a hypothesis test.
4. If two panels cannot be read from the back of the room, keep H2 alone and move the outcome comparison to backup.

**Presenter note:** H2's full-study interval crosses zero; that does not prove equivalence. Across the 11 treatment means per world, segregated versus dispersed means are resource fraction `0.221 vs 0.192`, reserve welfare `0.860 vs 0.857`, and wealth Gini `0.443 vs 0.420`. These are descriptive, not pooled inferential estimates. Resource and welfare are measured in 1,000-step fresh evaluation, not long-run sustainability.

## Slide 10 — H3 and collective organization

**Current page:** H3 title and 3×2 grid describe visibility inequality and low-extraction behavior, not H3's primary local-view-error result.

**Paste onto slide:**

```text
Changing peers made local views slightly more representative.
RQ1 / H3 · Adaptive − fixed error: −0.12 percentage points
95% interval: −0.20 to −0.04 percentage points
RQ2 · Attention became less equal; low-extraction changes varied
Full study · n=100 per condition
```

**Primary figures:** Left: **one pooled H3 interval** from [the pooled estimate table](../results/q_learning_baseline/social_analysis/balanced_full_v1_analysis/data/pooled_hypothesis_summary.csv). Label the negative side “more representative with adaptive.” Right: a compact two-line RQ2 callout, `visibility Gini rose in 15/15 cells` and `low-extraction effects varied`, or one named-profile row from [Attention and choices](../results/q_learning_baseline/social_analysis/balanced_full_v1_analysis/story_candidates/05_adaptation_pathway.pdf). Full 15-cell figure in backup.

**Edit steps:**

1. Replace the current headline with the local-view-error claim and interval.
2. Remove the 3×2 grid from the main slide; its small plots are hard to read and do not show the pooled H3 result.
3. Place visibility and extraction in a **separately labeled RQ2 panel**; do not present them as proof of H3.
4. Use consistent fixed-blue/adaptive-orange colors and a zero line on the pooled contrast.

**Presenter note:** H3 averages 15 paired landscape-by-profile differences from the same starting graphs. Its directional wording was pilot-informed; the 100 new replicates independently estimate that contrast. Higher observer-count Gini means more unequal visibility. In the full study, adaptation increased that Gini in all 15 cells, while low-extraction-rate contrasts varied and all 15 pointwise intervals included zero.

## Slide 11 — Conclusions

**Current page:** Vague perception/visibility/adaptation cards, doubt about whether the model is a dilemma, and platform-migration speculation.

**Paste onto slide:**

```text
What the full study shows
Commons incentives: 6 of 6 held-out checks passed.
RQ1: Primary H1 runs opposite the prediction; H2 is unresolved; H3 shows slightly lower error.
RQ2: Attention became less equal; extraction effects varied across conditions.
Resource layout changed remaining stock and wealth inequality.
```

**Figure:** No chart. Use three spacious cards—`Commons incentives`, `Four-person views`, `Collective outcomes`—or a small repeat of the three resource maps.

**Edit steps:**

1. Replace the current three cards with the evidence-led text above.
2. Remove the claim that the model cannot be evaluated as a social dilemma; the held-out payoff gate has evaluated this configured incentive structure.
3. Move Melting Pot/SocialJAX to spoken future-work discussion only if asked; migration is not a result of this study.
4. Keep the provenance caveat in notes: three input runs recorded a dirty worktree, and their uncommitted contents cannot be reconstructed from metadata.

**Presenter note:** The gate concerns policy incentives, whereas the learning experiment concerns agents' behavior with limited local information. The full analysis passed its recorded health checks. A clean-worktree replication would remove the provenance limitation described in the recovery audit.

## Draft-4 page check

| PDF page | Existing text to locate | Proposed section |
|---|---|---|
| 1 | “Technology, Access, and Welfare in Artificial Societies” | Slide 1 |
| 2 | “How does ecology shape perception and collective organization?” | Slide 2 |
| 3 | “Commons Game”; “apples” | Slide 3 |
| 4 | “Commons Game with Ecological Mechanics” | Slide 4 |
| 5 | “Social Observations”; “25% chance to have a source outside of local view” | Slide 5 |
| 6 | “Research Questions & Hypotheses” | Slide 6 |
| 7 | “Design of Study” | Slide 7 |
| 8 | “Currently there are no stable assications…” | Slide 8 |
| 9 | “H2 Welfare inequality is around ~2% higher…” | Slide 9 |
| 10 | “Attention adaptation actually increased visiblity inequality…” | Slide 10 |
| 11 | “Conclusions”; framework migration bullets | Slide 11 |

For slides 9–10, the local-view interval is the **primary H2/H3 evidence**. Stock, welfare, wealth, visibility, and extraction are separate RQ2 outcomes. If the companion plot makes the slide crowded, keep only the primary interval on the main slide and move the companion to backup.
