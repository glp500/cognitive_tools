# Capacity-matched visibility study

Status: implemented design, pending new payoff validation and learning results. This is the prospective primary environment design. The earlier [Stage-5 study](visibility-bounded-search-study.md) and its results remain historical.

## Research questions and hypotheses

The social-observation treatment remains Stage-5: five initial visibility profiles, fixed versus adaptive bounded attention, plus a B0 ecological baseline. The new ecology factor isolates spatial resource organization while holding total capacity fixed.

- **RQ1:** Does spatial resource organization alter population misperception and who receives attention?
- **RQ2:** How do resource organization and attention affect collective extraction, capacity-weighted resource persistence, regional welfare and wealth inequality?
- **H1 primary:** Segregated minus dispersed differs in final-window social perception error, holding capacity total, local capacity distribution and social treatment fixed. This is nondirectional. Dispersed minus uniform and segregated minus uniform are contextual comparisons that also change local inequality.
- **H2:** The four initial-visibility-profile contrasts in fixed networks remain as specified in the earlier Stage-5 protocol.
- **H3:** Adaptive minus fixed attention changes population misperception and visibility Gini, with low-extraction share as a companion outcome.

No hypothesis asserts that more resource segregation necessarily reduces welfare. Resource dynamics and fixed agent positions make the sign uncertain. Do not select a direction after seeing pilot outcomes.

## Frozen landscape contract

The new scenario IDs are `balanced_uniform`, `balanced_dispersed`, and `balanced_segregated`. On a 10×10 grid, each has total K=75. Uniform has K=0.75 in all cells. The two unequal maps each have fifty K=0.55 and fifty K=0.95 cells; the dispersed map uses the landscape seed to permute positions, and the segregated map arranges low and high capacity by column. The unequal maps have the same capacity histogram in every replicate. All maps use regeneration r=0.05, equilibrium fraction q=0.70, initial resource 0.5K, and neighbor coupling c=0.10. Landscape seed is campaign seed plus replicate; other paired seeds and all social parameters follow the earlier Stage-5 study.

The new `environment_design=balanced_capacity_v1` identity is stored in every run configuration and campaign freeze. Analyses reject mixed environment designs and differing scenario sets. Historical scenario definitions and reference hashes remain untouched.

## Validation gates and measurements

1. **Structural audit:** verify total K within numerical tolerance, identical unequal-map capacity multisets, constant r/q, and more unlike-neighbor edges in the dispersed map. Record per-replicate map SHA-256 values and no-harvest stock at steps 100 and 1000 in `landscape_validation.csv`. This baseline measures any physical supply effect of spatial exchange and clipping before agents act.
2. **Commons-dilemma gate:** run the independent population-payoff experiment with the new scenarios, capped-harvest reward, N=64, H=1000, C=always-low, D=always-high, and 100 held-out replicates. Require collective gain G>0, exploitation gap E>0, and greed or fear >0 for both summed and discounted utility in each ecology, using the existing simultaneous bootstrap family. The full learning runner requires all six verdicts to be `supported`.
3. **Learning smoke and pilot:** use paired condition seeds and the unchanged 11 treatments per ecology. Mechanical smoke has N=8 and two replicates. Figure smoke has N=64, two short replicates and renders the candidate story. The exploratory pilot uses N=64, ten replicates and the full training/evaluation lengths. The confirmatory full campaign uses 100 new replicates per ecology/treatment after the gate.

The primary aggregate sustainability metric is `total_resource / total_capacity` on fresh evaluation. `mean(R_i/K_i)` is retained as a separate mean local fill fraction. Total resource, total capacity, final-window low-extraction share, mean reserve welfare, and final wealth Gini remain visible. Agent-region summaries distinguish high- and low-capacity exposure in the segregated map. Replicates, not agents, are the inferential units. H1 reports segregated−dispersed first, with the two uniform comparisons clearly labeled as contextual. Intervals remain pointwise replicate-bootstrap 95% for learning contrasts, not familywise significance tests.

## Figures and interpretation

The five story candidates show (1) the three capacity maps with a common scale and equal total K, (2) who gets seen, (3) paired ecology/profile perception contrasts with the primary segregated−dispersed comparison, (4) fresh-evaluation collective stock and local fill against welfare plus a separately labeled training-end regional gap, and (5) adaptive−fixed differences in visibility, perception and extraction. They render from the saved analysis and source runs, export PDF/SVG/PNG, and write captions plus a provenance manifest. These are candidate manuscript figures; the analysis also retains full factor grids and contrast tables.

The new design licenses a claim about **capacity-matched spatial organization**, conditional on the tested policy pair, resource dynamics, grid and agent positions. It does not guarantee equal unharvested realized stock or validate the commons dilemma until the new payoff gate passes.
