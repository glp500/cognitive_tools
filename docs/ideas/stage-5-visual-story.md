# Stage-5 visual story

## Problem Statement

How might five figures let a broad social-science reader see who receives attention, how unequal the resource environment is, and whether changes in observation affect perception, welfare, and collective outcomes?

## Recommended Direction

Tell a population-first story: uneven ground, who gets seen, different local views, resource stock and wellbeing, then the adaptive-versus-fixed outcome sequence. Prototype with the ten-replicate Stage-5 pilot and regenerate with the frozen full campaign. The pilot remains exploratory; its pointwise intervals are not a final hypothesis verdict. These story figures are publication candidates, separate from the frozen study figure contract in [the Stage-5 spec](../specs/visibility-bounded-search-study.md).

| Figure | Reader question | Minimum visual | Source and interpretation guardrail |
|---|---|---|---|
| 1. Uneven ground | What world do agents inhabit? | Three capacity maps on one color scale and a short observation/action/rewiring process strip. | Reconstruct a labeled replicate-0 landscape from the frozen configuration; an example map is not an average landscape. |
| 2. Who gets seen? | Does attention spread or concentrate? | Initial and terminal adaptive observer-count distributions, plus top-six attention share by profile. | Initial graphs are matched across ecologies and dynamics. Re-rank the six most-observed agents at each snapshot; do not claim the same people remain leaders. |
| 3. Different windows | Do local views misrepresent population behavior? | Prespecified paired H1 ecology and H2 fixed-network profile differences with zero lines and pointwise intervals. | Show every planned contrast; no post-hoc selection of favorable cells. |
| 4. Stock and wellbeing | Does a resource measure tell the welfare story? | Fresh-evaluation absolute stock and remaining-fraction views against reserve welfare, with ecology and B0 visible. | Resource fraction alone is not sustainability. Keep any training-end regional welfare annotation separate from fresh-evaluation means. |
| 5. What did adaptation change? | Did rewiring alter the group? | Paired adaptive-minus-fixed contrasts for attention Gini, perception error, and low-extraction share, with resource and welfare consequences linked or shown in Figure 4. | Use independent replicates as inference units, not agents; show all ecology/profile cells and pointwise uncertainty. |

## Key Assumptions to Validate

- [ ] The attention story persists across ecologies and replicates; inspect all 15 ecology/profile cells.
- [ ] Absolute resource stock and capacity contextualize normalized resource fraction; check whether the interpretation changes.
- [ ] Broad readers can state each figure's finding at manuscript width and in grayscale without relying on the caption.
- [ ] Each plotted number traces to a saved source row or paired-contrast table; training and fresh-evaluation outcomes are never silently mixed.

## MVP Scope

Build a separate, deterministic plot script consuming the completed Stage-5 analysis manifest and saved CSVs. Export PDF, SVG, and 300-dpi PNG files plus concise interpretation notes. Validate the pilot prototypes visually at manuscript width, then use the same script for the full campaign. Keep the original exploratory grids and complete contrast tables as supplements.

The batch renderer must answer three operational questions: **which completed
study and source files generated these figures; did every planned cell and
observer count pass validation; and which figure failed if rendering stopped?**
Its output manifest, stable JSON events, and fail-fast source checks provide
those answers. HTTP metrics, distributed traces, and paging have no useful
endpoint in this local batch workflow.

## Not Doing (and Why)

- Network hairballs: 64 nodes and 256 directed edges conceal the count distribution.
- Individual-agent visual diaries: the pilot does not record per-step local views.
- A causal chain from attention to welfare: the planned comparisons do not identify every mediating link.
- A single sustainability score from remaining resource fraction: capacity and absolute stock differ among ecologies.
- Selected attractive replicates or significance stars: both risk overstating the exploratory pilot.

## Open Questions

- Does Figure 5 remain legible at the intended journal column width? If not, retain its three primary/companion outcomes and move secondary resource/welfare contrasts to the supplement.
- Which figure sizes and file formats does the final JASSS layout require? Check its current submission instructions when preparing the manuscript.
