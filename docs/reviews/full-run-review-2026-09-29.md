# Full-run review: ecology, perception and collective organization

Reviewed 29 September 2026. Campaign: `ecology_perception_parallel_v1`; simulation source commit `2ac3a887f271874d9ff71726a3d0ed9e83601efc`.

**The run is informative, but its strongest conclusion is a separation between perception, network structure and material outcomes.** Ecology substantially changes welfare and inequality. Observation rules change perception and visibility concentration. Adaptive selection modestly improves perception without a consistent downstream benefit. This supports a narrower conclusion than either “ecology and networks do not matter” or “better information improves collective welfare.”

The supplied critique is largely supported by the implementation. Its proposed extensions are possibilities, not requirements for completing this study. No simulation, reward, ecological parameter, learning rule or frozen primary analysis was changed during this review.

## What the results actually show

All 27 conditions have 100 replicates: 2,700 condition-replicates. The final training window is `4000 < t ≤ 5000`, with 20 recorded checkpoints averaged within each replicate. Secondary outcomes use frozen-policy, frozen-network, 1,000-step **fresh-reset** evaluation.

| Question | Evidence from the completed campaign | Interpretation |
|---|---|---|
| Does ecology affect population perception? | Uniform-versus-split differences in mean absolute perception error are below 0.24 percentage points across the eight social treatments. | Ecological differences in this perception metric are small for these scenarios and this learner. This is not evidence that ecology generally has no effect. |
| Do network rules affect perception? | S2 minus S1 increases error by **4.07–4.26 percentage points** and non-tie majority mismatch by **2.15–2.87 points**. | A clear treatment difference, but S2 changes attention capacity as well as topology. It does not isolate visibility concentration. |
| Does search scope affect organization? | Adaptive local versus global search increases visibility Gini by **0.116–0.123**. Matched random turnover produces a similar **0.114–0.123** difference. | The candidate-search rule largely structures concentration. It should not all be credited to surprise-driven observer selection. |
| Does adaptive selection improve perception? | Adaptive minus matched turnover reduces non-tie mismatch by **1.35–2.16 percentage points**; all nine pointwise intervals exclude zero. Mean absolute error falls by **0.095–0.443 points**, with seven of nine intervals excluding zero. | A modest, consistent direction of improvement. Tie rates also decrease; mismatch must remain paired with the tie measure. |
| Does that improve behavior and welfare? | All nine low-extraction and wealth-Gini contrast intervals include zero. Only one of nine resource contrasts and one of nine welfare contrasts excludes zero. | There is no consistent downstream improvement. These are not equivalence tests, and isolated pointwise intervals are not a family-wide finding. |
| Does ecology affect material outcomes? | Compared with uniform high, split ecology reduces fresh-reset welfare by **9.49–11.92 percentage points** and increases wealth Gini by **0.169–0.175**, across social treatments. | Substantial ecological scenario effects are present. The original outcome figure hid these because it only showed adaptive-minus-control differences. |

Ranges summarize treatment-specific estimates, not pooled effects or confidence intervals. The frozen analysis uses 5,000-resample percentile bootstrap intervals; intervals are pointwise and do not control multiplicity. The analysis did not define a smallest meaningful effect, so “little consistent benefit” is more defensible than “no meaningful effect.”

## Two useful checks using existing data

**Sampling explains most raw absolute error at fixed attention.** I calculated an exact finite-population benchmark for each recorded checkpoint: randomly sample four distinct peers from the other 63 agents, retaining the checkpoint's actual population action count. The expected error is **17.55–17.81 percentage points**. Observed minus expected error ranges from **−0.273 to +0.210 points** across the seven fixed-attention treatments and three ecologies. S1 is close to this benchmark; adaptive conditions are slightly below it.

This benchmark is a post-hoc diagnostic, not a replacement primary outcome or a counterfactual training run. A population can still display majority mismatches when samples are unbiased but small. S2 is excluded because the data needed to match each observer's varying attention and current action jointly are not available in the saved aggregate checkpoints. Do not interpret the benchmark as proving the absence of network effects or as an irreducible lower bound.

For reproducibility, if `L` of `N=64` agents choose low extraction, the focal action is `a∈{0,1}` with weights `L/N` and `1−L/N`. Compute `X ~ Hypergeom(N−1, L−a, 4)` and average `|X/4 − (L−a)/(N−1)|` over focal actions and `X`. Then average checkpoints within replicate and bootstrap replicate means. This exactly matches the implementation's exclusion of the focal agent from the comparison population.

**Fresh-reset welfare is not sustained welfare.** Across all 27 condition means, equally weighted, mean reserve welfare is **72.3%** in fresh-reset evaluation versus **26.2%** in continuation evaluation. The reset restores ecological and agent starting states, including full energy reserves. This difference cannot be attributed to reserves alone. It shows why a finite evaluation starting from a replenished state must not be described as long-run sustainability. Continuation remains a diagnostic; it does not replace the prespecified endpoint.

The existing evaluation logs also show the ecological state is classified as “scarce” for **97.1%** of agent-steps in fresh-reset evaluation and **99.7%** in continuation, averaged equally across conditions. The richer environmental differences are therefore largely compressed into the same learner state. This supports the critique of the representation, without proving that it caused the weak downstream effects.

## Design and architecture assessment

The module separation is useful: [ecology](../../cognitive_tools/ecology.py) defines resource dynamics; [scenarios](../../cognitive_tools/scenarios.py) constructs landscapes; [model](../../cognitive_tools/model.py) implements allocation and reserves; [social](../../cognitive_tools/social.py) defines observation and rewiring; [qlearning](../../cognitive_tools/qlearning.py) supplies the learner; [experiment](../../cognitive_tools/experiment.py) manages treatments, checkpoints and evaluation. There is no reason to rebuild this architecture merely because some effects are small. The main limitations concern what the model identifies and what information can influence behavior.

1. **Ecology is a bundled intervention.** Uniform and patchy high share nominal mean capacity, recovery and equilibrium settings. Split changes capacity, recovery, equilibrium and spatial pattern together; its high half also differs from the all-high scenarios. Report a scenario effect, not a pure fragmentation or heterogeneity effect. A future comparison should hold total opportunity/productivity fixed while changing spatial arrangement.
2. **The information target and the decision problem differ.** Perception is judged against population-wide extraction, but each stationary agent harvests its own tile. Network-local search follows sources-of-sources, not geographic or ecological proximity. A representative population estimate may have little value for predicting local returns. This is the central scientific gap.
3. **Learning and welfare are only indirectly coupled.** Reward is realized harvest; bounded energy welfare does not determine reward, survival or ability. Under scarcity, allocation is proportional to requested extraction, giving larger requests a larger share. Agents cannot relocate. Inequality can therefore reflect fixed ecological opportunity and co-location alongside learned behavior. The current data do not establish their separate contributions.
4. **The representation and horizon are restrictive.** Social learners use three resource bins × three social bins, without local capacity, regeneration, competitors or trends. `gamma=0.95` discounts delayed consequences over a characteristic horizon of about 20 steps. This is a plausible limitation, not proof that a longer horizon or neural network would help. Neither 5,000 steps nor a smooth curve establishes convergence.
5. **Controls identify specific contrasts.** R0 matches the adaptive event-count schedule and search scope; it assesses observer selection conditional on that schedule. B0 versus social treatments also changes state representation. Frozen evaluation removes ongoing adaptation. These are legitimate estimands, but narrower than an unrestricted “benefit of adaptive cognition.”

The attached report's points about deterministic regeneration, depletion and unrecorded clipping are also valid limits. Shocks, resilience tests, movement and richer rewards would each introduce another research problem. They should not all be added now.

One further caution: “low extraction” is not automatically a demonstrated cooperative policy. In S1's existing fresh-reset controls, averaged across ecologies, always-low agents have higher reserve welfare (**97.0% versus 59.7%**) and resource fraction (**60.1% versus 7.5%**) than always-high agents, but lower accumulated harvest (**1.93 versus 2.34 per agent**). These homogeneous-policy comparisons do not establish the policy-level incentive structure of a sequential social dilemma. That requires comparing unilateral policy incentives against collective returns, as in [Leibo et al.'s sequential social dilemmas](https://arxiv.org/abs/1702.03037).

## Revised figures

[Open the six-page figure PDF](../../results/review/full_run_2026-09-29/revised_figures.pdf). Each page is also available as PNG, PDF and editable SVG in the same folder.

| Figure | Purpose and revision |
|---|---|
| 1. Population perception | Put error, non-tie mismatch and ties together; consistent ecology colors and shapes; include replicate uncertainty. |
| 2. Collective organization | Place concentration beside extraction behavior; group adaptive and matched controls by search scope. This makes their dissociation visible. |
| 3. Ecology and outcomes | Show absolute resource fraction, reserve welfare and wealth Gini. This restores the major ecological differences missing from the previous figure. |
| 4. Adaptive minus matched | Show paired effects directly, with zero references, percentage-point units and explicit evaluation windows. |
| 5. Sampling diagnostic | Separate ordinary four-peer sampling error from excess error. Clearly marked post-hoc. |
| 6. Evaluation diagnostic | Show fresh-reset and continuation welfare on identical 0–100% scales. Clearly marked post-hoc. |

For the supervisor discussion, use Figures **1–4** as the main sequence and **5–6** as backup. Forest/dot plots convey treatment estimates and uncertainty more clearly than dense nine-line trajectories here. Zoomed axes are disclosed, no confidence interval is cropped, and ecological scenarios retain the same order and styling throughout. Original figures remain available for provenance.

This presentation follows the substantive distinction in [Lerman et al.'s majority-illusion work](https://arxiv.org/abs/1506.03022): local population impressions, network structure and actual population prevalence are related but distinct quantities. Visibility Gini alone is not a measure of majority illusion; absolute sampling error alone does not establish systematic network bias.

## Platforms for a future iteration

Repository documentation checked 29 September 2026. These are source-based fit assessments, not local installation or performance benchmarks.

| Candidate | Fit and effort | Recommendation |
|---|---|---|
| [SocialJax](https://github.com/cooperativex/SocialJax) — Commons Harvest Open | JAX social-dilemma environments with GPU-oriented simulation and IPPO/MAPPO baselines. Closest combination of renewable commons and accelerated learning. Port the project's social observation channel and diagnostics into its functional state; this is a substantive port, not an environment-name substitution. | **First candidate to assess for a new GPU-oriented study.** Use one Harvest environment and individual rewards. Pin a reviewed commit: the README explicitly notes that post-v1.1.0 fixes to agent overlap can change dynamics. |
| [Melting Pot](https://github.com/google-deepmind/meltingpot) — Commons Harvest Open | Established social-environment benchmark with [Harvest source configuration](https://github.com/google-deepmind/meltingpot/blob/main/meltingpot/configs/substrates/commons_harvest__open.py). DeepMind Lab2D and custom substrate work add integration complexity. [Extensions](https://github.com/google-deepmind/meltingpot/blob/main/docs/extending.md) are documented. | **Best alternative when canonical benchmark provenance matters most.** Use a single substrate, not its entire benchmark suite. |
| [BenchMARL](https://github.com/facebookresearch/BenchMARL) | TorchRL training framework with Melting Pot and PettingZoo support. It is a trainer/integration layer, not a replacement ecological model. The existing PettingZoo-style wrapper provides a starting interface, but a custom task adapter is still needed. | A route to retain the current ecology while comparing a stronger learner, or to train Melting Pot. It does not fix causal design or automatically put a Python environment on GPU. |
| [JaxMARL](https://github.com/bold-lab-ai/JaxMARL) | General JAX MARL platform with STORM, Coin Game, Overcooked and standard learners. Its listed tasks are less directly aligned with a renewable population commons than SocialJax Harvest. | Useful infrastructure or a deliberately simpler mechanism study; a less direct primary destination. |
| [Sequential Social Dilemma Games](https://github.com/eugenevinitsky/sequential_social_dilemma_games) | Historical Harvest/Cleanup implementation. Its README explicitly deprecates this route and recommends Melting Pot; installation includes older Ray patches. | Read for context; **do not choose as the foundation of a new migration.** |

Cleanup is a credible later option, but changes the problem from extraction restraint to costly public-good provision. Fully cooperative tasks such as Overcooked remove the individual-versus-collective incentive tension that motivates this project. Neither is the closest first move.

GPU-native execution is a reason to consider SocialJax, not a guarantee of a faster complete experiment on your hardware. Measure compilation, steady-state throughput and memory with the actual observations, population size and learner. There is no measured speedup or ETA for this project's port yet.

## A constrained direction, without another broad redevelopment

Keep the current campaign as the completed study of **ecology, observation rules and the separation between population perception and outcomes**. Present the limitations alongside the results; small effects do not invalidate it.

For a prospective follow-up, preserve the original order of questions:

- **RQ1:** How does ecological arrangement alter population misperception beyond finite-sample error, when productivity and attention capacity are held constant?
- **RQ2:** When do changes in information improve collective resource outcomes and individual welfare, rather than only population representativeness?

The key hypothesis is conditional: **information improves material outcomes when it predicts the ecological consequences relevant to the observer's decisions; population-wide accuracy alone may not be sufficient.** Treat the existing scope/concentration relationship as an established descriptive result to explain, not a reason to add more rewiring variants.

A bounded next study could use **one Commons Harvest substrate, one learner, two productivity-matched ecological arrangements and two matched-capacity information conditions**: ecologically informative versus a shuffled-information control. Keep reward and policy architecture fixed. This tests decision-relevant information without simultaneously introducing movement rules, new welfare rewards, several learners and another large rewiring factorial. It is a proposed new design, not an extension to the frozen campaign.

Before committing to migration, resolve one conceptual issue: Harvest has movement and harvesting actions, not the current binary low/high decision. Define population behavior over a fixed time window and account for harvesting opportunity; inactivity must not automatically count as cooperation. Preserve a meaningful population-perception measure rather than importing the current binary labels unchanged. Also specify whether “individual welfare” means material return or a separately justified reserve measure.

The minimum entry checks are to reproduce a substrate baseline, establish the relevant policy-level dilemma, and verify that the information manipulation preserves attention and changes the intended signal. If those fail, refine the mechanism before launching another full grid. A larger platform will not itself repair the link between information and decisions.

## Reproduction and verification

[Reproduction script](../../scripts/review_full_campaign.py):

```bash
MPLCONFIGDIR=/tmp/cognitive-mpl /tmp/cognitive-tools-clean/bin/python scripts/review_full_campaign.py
```

Use another Python environment with NumPy, pandas and matplotlib if the temporary environment is removed. The script reads saved CSVs only. It independently reconstructs every primary replicate value and selected outcome value from raw tables, checks all summary and paired-contrast means, validates complete primary windows, and records input SHA-256 hashes in [review_manifest.json](../../results/review/full_run_2026-09-29/review_manifest.json). Figures 1–4 retain the frozen analysis's intervals; supplementary figures bootstrap across 100 replicates, never across individual agents or checkpoints. The exported figures were visually inspected. No new simulations or migration were performed.
