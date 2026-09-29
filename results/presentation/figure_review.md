# Figure and presentation review — full run

The current seven-slide data round replaces the pilot presentation. Original files are preserved in `archive/pilot_v1_before_full_run_revision/`.

## Narrative

1. Confirmed RQ1 (population perception), RQ2 (collective organization), and three frozen hypotheses. The older hypothesis of a necessary sustainability improvement is removed.
2. Observation–learning–resource feedback; population accuracy may not help stationary agents manage their local resources.
3. Completed design: 3 ecologies × 9 rules × 100 replicates, N=64; matched turnover controls observer selection conditional on scope and event counts.
4. Perception error, non-tie mismatch and ties together; paired adaptive improvement and the supplementary finite-sample benchmark.
5. Visibility concentration beside extraction behavior; local search concentrates visibility even under random turnover.
6. Absolute ecological outcomes; split ecology has lower welfare and higher wealth inequality. Fresh-reset outcomes are distinguished from continuation.
7. Supervisor feedback on question clarity, organization measures and one bounded follow-up. SocialJax Commons Harvest is a candidate, not an implemented migration.

## Visual choices and evidence

The local PowerPoint preserves the reference deck's white background, Arial typography, dimensions, masters, and blue/teal/ochre accents. Text and conceptual diagrams remain editable. Statistical figures are embedded at high resolution; PDF and editable SVG versions are supplied.

The three result charts use consistent ecology colors and shapes, treatment order and pointwise 95% bootstrap intervals. Zoomed dot-plot axes disclose small differences; no confidence interval is cropped. Captions distinguish absolute outcomes from paired adaptive-minus-control contrasts.

Sources are the completed `ecology_perception_parallel_v1` campaign. Training outcomes average the 20 checkpoints in `4000 < t ≤ 5000` within each replicate. Secondary outcomes use frozen policies and terminal networks in a 1,000-step fresh reset. There are 100 independent replicates per condition and 5,000 bootstrap resamples. Intervals are pointwise, not multiplicity-adjusted; crossing zero does not establish equivalence.

Slide 3's capacity maps are explicitly illustrative seed-42 maps from the same scenario definitions, not full-run replicate observations. Split bundles changes in capacity, regeneration, equilibrium and layout. All relevant treatments and all three ecologies appear in each main results figure.

## Checks and reproduction

- Source tables: `data/full_run/`; hashes: `data_audit.json`.
- Underlying primary values and paired means were verified in `docs/reviews/full-run-review-2026-09-29.md`.
- The seven-page PDF was rendered from the replacement PowerPoint and visually inspected.
- Canva was inspected separately; imported chart cropping and lost diagram arrows were corrected.
- Notes are embedded in PowerPoint and saved in `presenter_notes.md`.
- No simulations, primary analysis definitions or model parameters were changed.

Rebuild from the repository root:

```bash
MPLCONFIGDIR=/tmp/cognitive-mpl /tmp/cognitive-tools-clean/bin/python results/presentation/build_data_round.py
```

Requires NumPy, pandas, matplotlib, Pillow and python-pptx. PDF export uses LibreOffice. The full review contains primary-source links for related work and MARL candidates.
