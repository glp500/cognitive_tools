# Capacity-matched spatial archetypes

## Problem Statement

How might we test whether resource location changes visibility, welfare, inequality, and collective sustainability when every landscape offers the same total carrying capacity?

## Recommended Direction

Use three new 10×10 environments with total carrying capacity 75: uniform cells at 0.75; a dispersed random arrangement of fifty 0.55 and fifty 0.95 cells; and a segregated arrangement of those same fifty low and fifty high cells. Dispersed versus segregated is the primary spatial-arrangement contrast. Uniform is an equal-access benchmark and also changes local inequality.

All three use regeneration 0.05, equilibrium fraction 0.70, initial stock 0.5K, and neighbor coupling 0.10. Equal total capacity and initial stock do not guarantee equal later stock because exchange and local clipping depend on arrangement. Treat this unharvested difference as a measured spatial mechanism, not an unmeasured nuisance.

## Key Assumptions to Validate

- [ ] Every replicate has capacity 75 within floating-point tolerance and matched regeneration maps.
- [ ] The two unequal maps have the same multiset of local capacities, with more unlike-neighbor edges in the dispersed map.
- [ ] No-harvest trajectories expose any layout-driven difference in realized stock.
- [ ] The new maps independently pass the held-out population commons-dilemma gate for capped-harvest utility before confirmatory learning runs.
- [ ] Differences are interpretable without selecting a favorable landscape seed or tuning maps against learning results.

## MVP Scope

Create new scenario IDs and study identity while preserving archived maps. Record capacity-weighted stock fraction and mean local fill fraction separately. Produce a structural/no-harvest audit, payoff gate, paired-seed smoke and pilot campaigns, planned spatial contrasts, and five story-candidate figures from saved runs.

## Not Doing (and Why)

- Editing historical maps — earlier results need stable inputs.
- Calling total capacity realized supply — spatial exchange makes these distinct.
- Adding an abundance-by-pattern factorial — it increases run cost before the primary contrast is tested.
- Choosing maps from favorable agent outcomes — that would invalidate the confirmatory comparison.

## Open Questions

- Do all six payoff-gate verdicts support a commons dilemma under the new maps?
- How much of the observed stock difference exists without harvesting?
- Does the pilot provide enough separation to motivate the frozen full campaign?
