# Stage-5 visual story prototype review — 1 October 2026

The five story figures generated from `visibility_pilot_v3_analysis` are
**exploratory publication candidates**. They do not replace the five frozen
Stage-5 figures or alter simulation results. The ten-replicate pilot is a design
test; planned pointwise intervals are not a formal hypothesis verdict. The
source study and statistical review remain in
[the pilot review](visibility-pilot-v3-figure-review-2026-09-30.md).

| Figure | Narrative role | Pilot reading and scientific limit |
|---|---|---|
| [1. Uneven ground](../../results/q_learning_baseline/social_analysis/visibility_pilot_v3_analysis/story_candidates/01_uneven_ground.pdf) | Actual replicate-0 capacity maps on one scale, followed by the observation/action/update sequence. | Uniform and patchy maps each total 75.0 capacity units in this displayed replicate; the split map totals 53.6. These are configured example landscapes, not across-replicate estimates. |
| [2. Who gets seen?](../../results/q_learning_baseline/social_analysis/visibility_pilot_v3_analysis/story_candidates/02_who_gets_seen.pdf) | Observer-count distributions and top-six attention share expose concentration without a network hairball. | The equal profile starts at exactly four observers per source, then broadens under adaptive search. Initial graphs are deduplicated across matched ecologies. The top six are re-ranked at each endpoint, so this does not show persistent winners. |
| [3. Different windows](../../results/q_learning_baseline/social_analysis/visibility_pilot_v3_analysis/story_candidates/03_different_windows.pdf) | H1 ecological and H2 fixed-network profile comparisons share a readable zero-centered difference scale. | Every planned comparison is shown. Pilot effects are small and heterogeneous; do not infer a general ecological direction or ordered profile dose. |
| [4. Stock and wellbeing](../../results/q_learning_baseline/social_analysis/visibility_pilot_v3_analysis/story_candidates/04_stock_and_wellbeing.pdf) | The same 33 condition means appear against absolute stock and normalized remaining fraction; the split ecology's regional training-end welfare is a separate third panel. | Split-world conditions have relatively more *fraction* remaining than some high-capacity conditions while showing lower welfare and less absolute stock. This is descriptive. The regional panel is training-end welfare; the scatter panels are fresh-evaluation means. |
| [5. Adaptation pathway](../../results/q_learning_baseline/social_analysis/visibility_pilot_v3_analysis/story_candidates/05_adaptation_pathway.pdf) | H3 and its extraction companion align what changed in attention, perception, and group behavior. | Pilot adaptive-minus-fixed visibility Gini is positive in all 15 ecology/profile cells; perception and extraction contrasts are heterogeneous. Each point is a paired-replicate estimate with a pointwise interval. |

PDF and SVG are editable vector exports; PNG is 300 dpi. Color is paired with
marker shape for ecology in the inferential panels. Titles ask reader-facing
questions; one figure carries one main idea. The detailed run count, contrast
direction, phase, and caveats are in each `.caption.txt` file instead of
crowding the plotting area.

The renderer logs `story_start`, `story_figure_saved`, `story_figure_failed`,
`story_input_failed`, `story_provenance_failed`, and `story_complete` as JSON. Its
[`story_manifest.json`](../../results/q_learning_baseline/social_analysis/visibility_pilot_v3_analysis/story_candidates/story_manifest.json)
records run IDs, configuration hashes, SHA-256 hashes of all figure input files,
source tables, fixed study dimensions,
and completed validation checks. A deliberate missing-input run produced a
structured `story_input_failed` event; a fixture with an empty H1 table
produced `story_figure_failed` for Figure 3 and no complete manifest. The full
pilot render completed five figures with no scientific coverage errors. Source
checks include 64 agents
and 256 links per attention snapshot, all H1/H2/H3 cells, all 33
ecology/treatment fresh-evaluation means, and agreement between raw means and
the saved analysis.

Before submission, inspect these at the actual final layout width, run a
five-reader comprehension check with non-modelers, and rerender from the full
100-replicate campaign. Retain the planned full contrast tables and original
study figures as supplements. If Figure 5 is too dense at journal width,
move secondary consequences to the supplement. The current pilot prototype
does not establish causal mediation from attention through perception to
welfare. JASSS-published [visualization guidance](https://www.jasss.org/12/2/1.html)
and [figure assessment](https://www.jasss.org/18/4/16.html) support clear
scales, model basis, parameters, and simulation counts; check the journal's
current submission instructions before final export.
