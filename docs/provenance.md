# Scientific and implementation provenance

Mechanism definitions belong in [experiment.md](experiment.md); data contracts
belong in [analysis.md](analysis.md). This document distinguishes scientific
lineage from source-code reuse. Citation metadata is in [references.bib](references.bib).

## Scientific sources

| Component | Lineage | Source and implementation boundary |
|---|---|---|
| Renewable ecology | Inspired | Tilman, Plotkin & Akcay (2020), DOI `10.1038/s41467-020-14531-6`; Tu et al. (2025), DOI `10.1007/s41748-024-00489-8`. Spatial resource feedback is independently implemented in `ecology.py`. |
| Tabular Q-learning | Reproduced algorithm | Watkins & Dayan (1992), DOI `10.1007/BF00992698`. Independent implementation in `qlearning.py`. |
| Restricted observation | Inspired | Schrama, Tilman & Vasconcelos (2025), DOI `10.1016/j.isci.2025.112831`. Previous source actions form the social state; this is not their full HSM, asynchronous update, payoff, or memory model. |
| Fixed BA-style visibility | Adapted/inspired | Schrama et al. motivate skewed information access. `social.py` independently implements preferential attachment with project-specific density. |
| Fixed attention and local/global replacement | Adapted | Oh & Schauf (2025), DOI `10.1038/s41598-025-23634-3`. Fixed attention, one-source replacement, and sources-of-sources search are retained structural ideas. |
| Social forecast | Adapted idea | EWMA forecasting is a project implementation related to adaptive expectations; it does not reproduce HSM. |
| Error-triggered rewiring | Original | Local social surprise replaces an objective-correctness trigger. This is not Oh & Schauf's DeGroot decision model or a misinformation model. |
| Scenarios, state bins, welfare, matched controls, evaluation, diagnostics and analysis | Original project constructions | Repository-local choices and code; see the protocol for definitions. |

Qin et al. (2026), DOI `10.48550/arXiv.2606.05867`, is retained in the
bibliography as related network-CPR reinforcement-learning work; the project
does not claim to reproduce its deep RL mechanisms.

Landscape primitives were consolidated from the project's former ecological
scan during ecological unification. Mesa, PettingZoo, Gymnasium, NumPy, and
Matplotlib are software dependencies, not scientific mechanism sources.

## External code reuse

No scientific mechanism source code from the cited papers has been copied into
this repository. Implementations are independently written. Any future direct
reuse must identify source repository, commit, file/function, license, and
modifications both here and at the reuse site.

## Implementation milestones

History is preserved. These commit identifiers retain their original meaning:

| Commit | Scientific implementation provenance |
|---|---|
| `b31478295ce7987db72066b592278995e5d297eb` | Unified ecological dynamics |
| `b8baef4bf041982d10ada37830c187cf6cec821f` | State/action-agnostic tabular learner preserving B0 |
| `298bdbea483b336eb1b8249518f713a7101d2b88` | Source attribution and bibliography |
| `6e7d7cc9fb89a641e77fd116c646ba7988ee2aad` | Fixed social observation and joint state |
| `286cb3a5c2c8ae6929dc0ec0109830655012f4a4` | Decentralized replacement, forecasting and error trigger |
| `8703f07b84709bfc206ca4539a30236408e2d169` | Git/runtime run metadata |
| `be78a42845f75498a6da5e4a42e8399b85d60a8f` | Fixed BA control and matched random schedules |
| `f159c0ab16f0d0a76e922105d85e73ea0839ceed` | Network evaluation decomposition |
| `af38b411073aa5564e18cf4dfe60a962ca7f2dfb`, `1fb513f5f20b6feae7d60cf6c5f8f0799386519e` | Measurement schema, tie handling, network checkpoints and policy occupancy |
| `b0340b10dd33499a714af5c22ab124b05f680a3a` | Pre-cleanup snapshot including cross-treatment analysis and campaign preparation |

## Reproducibility contract

Separate deterministic streams cover landscapes, agent placement, learners,
initial topology, training rewiring, evaluation random policies, and adaptive
evaluation rewiring. Structural cleanup preserves seed arithmetic and RNG
consumption in supported treatments. Exact scenario references were captured
before removing the old experiment, for all eight scenarios and three
width/height/seed combinations, including region labels.

`config.json` records Git commit, branch, dirty state, creation time, command,
Python/platform and package versions, scenario definitions, and matched schedule
hash. The analysis manifest records input hashes and pairing. Archive whole runs
with their complete environment inventory; direct package pins alone do not
freeze transitive dependencies or platform numerics.

The output labels `fresh_reset`, `stage4_v1`, and the `q_learning_baseline`
result path remain current contracts used by analysis and archived runs.
Programmatic experiment helpers retain defaults for omitted `network_eval` and
`social_network` attributes so existing minimal scientific test configurations
continue to mean frozen evaluation and random-k topology. The CLI always
supplies these settings. The analysis pairing fallback and its warning are
documented in [analysis.md](analysis.md#exact-adaptiver0-pairing).

The baseline-only plotting campaign is frozen in the pre-cleanup tag. B0,
fixed-policy controls, and all supported ecological scenarios remain active.
No published archive URL was supplied; the verified local backup and source
snapshot are recorded in [cleanup.md](cleanup.md). No history was rewritten.
