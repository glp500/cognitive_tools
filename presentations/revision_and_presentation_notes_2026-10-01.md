# Study update deck: revision guide and critique

The editable deck is `cognitive_tools_study_update_2026-10-01.pptx`. Its PDF is a visual preview; the PowerPoint carries speaker notes on every slide. Slides 1–13 form the 10–12 minute talk for a mixed audience. Slides 14–15 are backup detail. The deck uses the five saved story candidates as either main-slide figures (1, 4, 5) or backup figures (2, 3), while redrawing the three headline estimates in larger, simpler form.

## Suggested spoken route

| Slides | Time | Point to make |
|---|---:|---|
| 1–2 | 1 min | Local observation mediates the link between resource geography and collective outcomes. |
| 3 | 1 min | The frozen, held-out payoff gate supports the commons-dilemma criterion for this model and tested policies. |
| 4–7 | 3 min | State both revised questions; explain equal-capacity maps, paired observation treatments, and the five readouts. |
| 8–10 | 3 min | H1 changes sign with time adjustment; H2 is inconclusive; H3 aligns in the exploratory pilot only. |
| 11–12 | 2 min | Follow the pathway from visibility to error to extraction, then stock, reserve welfare, and wealth inequality. Describe patterns without claiming causal mediation. |
| 13 | 1 min | Separate validated commons incentives from still-open learning hypotheses. |

## Revised wording to preserve

- **RQ1:** How do capacity-matched resource layouts, initial visibility profiles, and adaptive observation affect how accurately agents' four observed peers represent population extraction behavior?
- **RQ2:** How do those conditions affect collective extraction, inequality in actual observer counts, resource persistence, reserve welfare, and wealth inequality?
- **H1:** Across training, greater inequality in actual observer counts is associated with greater local-view error.
- **H2:** Averaged equally over five initial profiles and both network dynamics, segregated resource layout produces greater final-window local-view error than dispersed layout.
- **H3:** Averaged equally over three landscapes and five initial profiles, adaptive bounded rewiring reduces final-window local-view error relative to fixed observation starting from the same graph.

The source of truth for exact estimands, pairing, intervals, and measurement definitions is [`docs/specs/capacity-matched-visibility-study.md`](../docs/specs/capacity-matched-visibility-study.md). “Local-view error” is the mean absolute difference between low-extraction shares among four observed peers and among all other agents. It is an **available-information measure**, not an elicited belief or psychological misperception.

## What Draft 2 needed corrected

1. Slides 10–12 were effectively empty and their H1/H2 headings described earlier hypotheses. The new slides 8–10 use the revised numbering and state one prediction and one primary estimate each.
2. Its “Commons Game” example showed a different game's apples and regrowth, which could imply those were this model's mechanics. The revision presents this project's own held-out commons-payoff gate and actual capacity maps instead.
3. The “25% chance to have a source outside of local view” line overstated the adaptive search rule. The 25% is the global **candidate-search branch conditional on rewiring**, not a guarantee about the chosen source or an event at every observation. This distinction is in slide 6 notes.
4. The old conclusion said the ecology made it difficult to establish a true dilemma and speculated about migrating to another framework. The new held-out gate supports the tested commons criterion in all six ecology × return cells. A framework migration may have other merits, but that old rationale is no longer supported by the frozen evidence.
5. The old RQ/H slide mixed “uncertainty,” “observation,” and “misperception.” The revision uses measured observer counts and local-view error. H1 is associational; none of these figures by themselves establish a causal effect of attention on welfare.
6. The original study-design text and charts were too small for live presentation. The revised deck reserves dense, treatment-level plots for backup and puts the primary estimates in large interval displays.

## Interpretation and figure cautions

- The pilot has **10 independent replicates** and informed the directional H2/H3 wording; it cannot confirm those hypotheses. The 100-replicate independent learning campaign is the prospective test. Keep the “exploratory pilot” label until those outputs replace it.
- H1's raw within-run estimate is `r = −0.0788` (95% interval `−0.1429, −0.0215`), but the linear-time-adjusted estimate is `r = +0.0362` (`+0.0108, +0.0625`). This sign reversal is the result to say aloud, rather than describing H1 as supported or refuted.
- H2's pooled segregated-minus-dispersed local-view error is `+0.00114` (`−0.00158, +0.00400`): the interval spans zero. H3's pooled adaptive-minus-fixed error is `−0.00286` (`−0.00500, −0.00102`): pilot-aligned, not confirmatory.
- Candidate figure 4 combines **fresh-evaluation condition means** (left and center) with a **training-end regional gap** (right); call out the different time bases. Its stock metric is total resource divided by total capacity, not mean local fill.
- Candidate figure 5 has 15 paired cells per outcome. During the talk, point to the left-to-right progression and a single example row; let the speaker notes carry the full qualification. Its cell intervals are pointwise and descriptive.
- Candidate figure 2's top-six shares re-rank agents at each snapshot. Do not describe its connectors as trajectories of the same six agents.
- The capacity maps hold the configured total and the unequal-map capacity histogram fixed. They do not force identical realized unharvested stocks under spatial dynamics.

## Update procedure after the full campaign

First check the new analysis-health report, pairing, and payoff-gate freeze. Then replace the pilot values and labels on slides 8–12 using the **same estimands and directions**, and regenerate the five story figures from the independent campaign. Preserve the notes' distinction between payoff validation and learning evidence. The deck can be rebuilt with `python presentations/build_study_update.py` in an environment containing `python-pptx`; source figures are read from the saved local story-candidate directory.
