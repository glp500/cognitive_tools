# Baseline 1 — Independent Ecological Q-Learning

**Purpose:** establish the no-communication baseline for the project.  
**Primary outcomes:** **welfare** and **wealth inequality**.  
**Observed explanatory variables:** environment, population pressure, resource condition, low-extraction/cooperation proxy, collective order, state occupancy, and learned-policy heterogeneity.

> Causal claims are strongest for manipulated variables (environment, population, fixed-policy controls). Cooperation level, collective order, and policy heterogeneity are learned/endogenous outcomes and should be interpreted as associations unless explicitly manipulated.

## 1. Study question

Can stationary, independently learning agents develop low-extraction behavior from local ecological feedback alone, and how do ecological conditions and population pressure shape **welfare** and **inequality**?

No movement, communication, sanctioning, transfers, social learning, or shared Q-table are allowed.

## 2. Ecology

Each grid cell has resource stock \(R(x,t)\), carrying capacity \(K(x)\), regeneration \(r(x)\), equilibrium fraction \(q(x)\), and spatial coupling \(c\):

\[
R_{x,t+1}=R_{x,t}+r_xR_{x,t}\left(1-\frac{R_{x,t}}{K_x}\right)-r_x(1-q_x)R_{x,t}+c(\bar R_{N(x),t}-R_{x,t})-H_{x,t}.
\]

Without harvesting/diffusion, \(R_x^*\approx q_xK_x\).

| Setting | Value |
|---|---:|
| Grid | `10 x 10` |
| Initial resource | `R0 = 0.50 K` |
| Coupling | `0.10` |
| Boundary | no-wrap / reflecting-like |

### Environments

| Scenario | Structure |
|---|---|
| `uniform_high` | uniform, high productive potential |
| `patchy_high` | patchy, high productive potential |
| `decentralized_high` | multiple high-resource centers |
| `centralized_low` | centralized, low/slow ecology |
| `patchy_high_central_low` | high patchy background + low/slow center |
| `central_high_patchy_low` | low patchy background + high/fast center |
| `split_high_low` | high patchy half + low fragmented half |
| `decentralized_high_in_low` | high islands + low background |

[Environment figure](results/q_learning_baseline/experiments/baseline_validation_v1/figures/00_environment_catalog.png)

## 3. Initialization

| Setting | Value |
|---|---:|
| Populations | `8, 16, 32` |
| Replicates | `20` |
| Base seed | `42` |
| Agent positions | random, fixed after initialization |
| Position matching | same replicate seed across ecological scenarios |
| Landscape seed | same replicate seed across scenarios |
| Co-location | allowed |

## 4. Actions and learning

Agents choose one action every timestep:

| Action | Meaning | Requested harvest |
|---|---|---:|
| `L` | low extraction / cooperation proxy | `0.002` |
| `H` | high extraction / defection proxy | `0.020` |

Each agent has an independent tabular Q-learner. The learner uses only local \(R/K\):

| State | Definition |
|---|---|
| Scarce | `R/K < 1/3` |
| Moderate | `1/3 <= R/K < 2/3` |
| Abundant | `R/K >= 2/3` |

| Q-learning setting | Value |
|---|---:|
| \(\alpha\) | `0.10` |
| \(\gamma\) | `0.95` |
| Initial \(\epsilon\) | `0.20` |
| Minimum \(\epsilon\) | `0.02` |
| Decay | `0.9995` |
| Training | `5,000` steps |
| Evaluation | `1,000` steps |

**Reward:** realized individual harvest only. Welfare, equality, sustainability, and cooperation are not directly rewarded.

## 5. Welfare model — primary outcome

Metabolism is treated as a per-step subsistence requirement.

| Welfare setting | Value |
|---|---:|
| Initial energy | `1.0` |
| Energy capacity | `1.0` |
| Metabolic need / step | `0.002` |

Harvest enters a bounded energy reserve. Metabolism then consumes up to `0.002`:

\[
\text{need satisfaction}_{i,t}=\frac{\text{metabolic consumption}_{i,t}}{0.002}\in[0,1].
\]

Primary welfare measures:

- **Reserve welfare:** \(E_i/E_{max}\in[0,1]\)
- **Need satisfaction:** fraction of metabolic need met
- **Deprivation rate:** fraction of agent-steps with unmet metabolic need
- **Metabolic shortfall:** amount of unmet need

Raw energy is retained as a secondary resilience/buffer measure.

[Primary welfare result](results/q_learning_baseline/experiments/baseline_validation_v1/figures/01_primary_welfare.png)

## 6. Inequality — primary outcome

Material accumulation is measured as cumulative harvested wealth. Distributional inequality is measured with wealth Gini.

[Primary inequality result](results/q_learning_baseline/experiments/baseline_validation_v1/figures/02_primary_inequality.png)

The main joint outcome view is welfare versus inequality:

[Welfare–inequality state space](results/q_learning_baseline/experiments/baseline_validation_v1/figures/03_welfare_inequality_state_space.png)

## 7. Validation design

For every **environment x population x replicate**:

1. Train independent Q-learning for 5,000 steps.
2. Freeze Q-values and evaluate greedily for 1,000 steps in the ecology produced during training.
3. Reset ecology, wealth, and welfare state and evaluate the same learned policy again from a common initial condition.
4. From that same fresh initial condition evaluate three controls:
   - always low extraction;
   - always high extraction;
   - random 50/50 extraction.
5. Record ecological state occupancy and realized \(P(L\mid state)\).

This distinguishes learned-policy quality from damage accumulated during training and provides fixed-policy counterfactuals.

## 8. Outcomes and explanatory measures

| Role | Variables | Result |
|---|---|---|
| **Primary outcome** | reserve welfare, need satisfaction, deprivation | [Welfare](results/q_learning_baseline/experiments/baseline_validation_v1/figures/01_primary_welfare.png) |
| **Primary outcome** | wealth Gini | [Inequality](results/q_learning_baseline/experiments/baseline_validation_v1/figures/02_primary_inequality.png) |
| Joint outcome | welfare x inequality | [State space](results/q_learning_baseline/experiments/baseline_validation_v1/figures/03_welfare_inequality_state_space.png) |
| Behavior | low-extraction/cooperation proxy | [Low extraction](results/q_learning_baseline/experiments/baseline_validation_v1/figures/04_low_extraction_cooperation_proxy.png) |
| Ecology | mean resource fraction `R/K` | [Resource condition](results/q_learning_baseline/experiments/baseline_validation_v1/figures/05_resource_condition.png) |
| Policy counterfactual | Q-learning vs fixed policies: welfare | [Control comparison](results/q_learning_baseline/experiments/baseline_validation_v1/figures/06_fixed_policy_welfare.png) |
| Policy counterfactual | Q-learning vs fixed policies: inequality | [Control comparison](results/q_learning_baseline/experiments/baseline_validation_v1/figures/07_fixed_policy_inequality.png) |
| Policy counterfactual | Q-learning vs fixed policies: ecology | [Control comparison](results/q_learning_baseline/experiments/baseline_validation_v1/figures/08_fixed_policy_resource.png) |
| Experience/behavior | state occupancy + `P(L|state)` | [State-conditioned behavior](results/q_learning_baseline/experiments/baseline_validation_v1/figures/09_state_occupancy_and_behavior.png) |
| Evaluation validity | continuation vs fresh reset | [Reset comparison](results/q_learning_baseline/experiments/baseline_validation_v1/figures/10_continuation_vs_fresh_reset.png) |
| Organization | pairwise learned-policy Hamming distance | [Policy heterogeneity](results/q_learning_baseline/experiments/baseline_validation_v1/figures/11_policy_heterogeneity.png) |
| Dynamics | welfare + inequality during training | [Training trajectories](results/q_learning_baseline/experiments/baseline_validation_v1/figures/12_training_welfare_inequality.png) |

## 9. Generated data

- [Configuration](results/q_learning_baseline/experiments/baseline_validation_v1/config.json)
- [Evaluation summary](results/q_learning_baseline/experiments/baseline_validation_v1/data/evaluation_summary.csv)
- [Aggregate summary + 95% CIs](results/q_learning_baseline/experiments/baseline_validation_v1/data/aggregate_summary.csv)
- [Training trajectories](results/q_learning_baseline/experiments/baseline_validation_v1/data/training_timeseries.csv)
- [Evaluation trajectories](results/q_learning_baseline/experiments/baseline_validation_v1/data/evaluation_timeseries.csv)
- [Run-level policy metrics](results/q_learning_baseline/experiments/baseline_validation_v1/data/policy_summary.csv)
- [Individual policies + Q-values](results/q_learning_baseline/experiments/baseline_validation_v1/data/agent_policies.csv)

## 10. Interpretation rules

- Welfare and inequality are the two principal outcomes.
- Environment and population are manipulated treatments.
- Fixed-policy controls directly test consequences of extraction strategy.
- Low-extraction rate is a **cooperation proxy**, not proof of successful cooperation.
- Collective order can represent either uniform restraint or uniform over-extraction.
- Policy heterogeneity and cooperation levels are endogenous learned properties; associations with welfare/inequality are not automatically causal.
- Mixed environments change several ecological quantities together; causal attribution to a single ecological parameter requires matched follow-up experiments.

## 11. Core hypothesis structure

\[
\text{environment + population}
\rightarrow
\text{experienced ecological states}
\rightarrow
\text{learned policy / collective organization}
\rightarrow
\boxed{\text{welfare + inequality}}
\]

Resource persistence is an ecological outcome and potential mediator connecting behavior to welfare and inequality.
