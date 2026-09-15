# Cognitive Tools

A research codebase for studying how ecological structure, renewable-resource dynamics, and learning shape individual and collective outcomes in common-pool resource environments.

The project is organized as a sequence of increasingly complex baselines:

1. **Ecology-only environments** — spatial renewable resources without agents.
2. **Random-agent baseline** — agents interact with the resource system without learning.
3. **Independent Q-learning baseline** — stationary agents learn whether to extract cooperatively or defectively from local ecological feedback, without communication.
4. **Future extensions** — information, communication networks, exchange, obligation, and richer multi-agent learning.

The broader goal is to understand how ecological constraints become differentiated opportunities and how learning and information transform those opportunities into welfare, inequality, cooperation, dependence, and resilience.

---

## Model overview

### Ecological dynamics

The environment is a spatial grid. Each tile has:

- `K(x)` — carrying capacity / structural ecological opportunity
- `R(x,t)` — current resource stock
- `r(x)` — regeneration rate
- `q(x)` — equilibrium resource fraction
- `c` — spatial coupling / diffusion rate

The resource dynamics are approximately:

```text
R(x,t+1)
=
R(x,t)
+ r(x) R(x,t) [1 - R(x,t)/K(x)]
- d(x) R(x,t)
+ c [mean_neighbor_resource - R(x,t)]
- harvest(x,t)
```

with:

```text
d(x) = r(x) [1 - q(x)]
```

so that an isolated, unharvested tile tends toward:

```text
R*(x) = q(x) K(x)
```

The reaction term governs local renewable-resource dynamics. The diffusion term redistributes resources between neighboring tiles.

---

## Environment types

The project currently uses several spatial resource structures:

- `uniform`
- `weak_patchy`
- `patchy`
- `centralized`
- `decentralized`
- `fragmented`

The Q-learning baseline also includes mixed environments in which different parts of the same world have different ecological conditions, for example:

- patchy high-abundance background + centralized low-abundance region
- centralized high-abundance region + patchy low-abundance background
- high-abundance patchy region + low-abundance fragmented region
- decentralized high-abundance islands + low-abundance background

These mixed environments allow stationary agents in the same simulation to experience different local ecological feedback.

---

## Agent baselines

### Random agents

The random-agent baseline is a null model. It asks what inequality, welfare, and ecological outcomes arise without strategic learning.

### Independent Q-learning agents

In the Q-learning baseline, agents are stationary. They do not move and they do not communicate.

Each agent chooses between two extraction actions:

```text
0 = COOPERATE = low extraction
1 = DEFECT    = high extraction
```

The current starting calibration is:

```text
cooperative harvest = 0.002
defective harvest   = 0.020
```

These values are calibration parameters rather than universal constants.

Each agent learns its own tabular Q-function:

```text
Q_i(state, action)
```

There is no shared Q-table and no centralized controller.

The local ecological state is discretized using `R/K`:

```text
scarce:    R/K < 1/3
moderate:  1/3 <= R/K < 2/3
abundant:  R/K >= 2/3
```

The default reward is the agent's realized harvest. Agents are not explicitly rewarded for cooperation, equality, or resource preservation. Any learned restraint must emerge from the relationship between current extraction and future resource availability.

---

## Main outcome measures

### Ecological measures

- mean resource
- resource fraction `R/K`
- resource Gini
- scarcity
- Moran's I / spatial organization
- recovery after disturbance

### Agent measures

- cumulative wealth
- wealth Gini
- mean energy as a welfare proxy
- mean wealth

### Cooperation and collective organization

The Q-learning experiment measures:

- cooperation rate
- collective action entropy
- collective order
- action-switching rate
- state-conditioned cooperation
- learned policy diversity
- policy entropy
- pairwise policy Hamming distance

A learned policy such as:

```text
(C, C, D)
```

means:

```text
scarce    -> cooperate
moderate  -> cooperate
abundant  -> defect
```

This can be interpreted as state-dependent ecological restraint.

---

## Network analysis

The current Q-learning baseline does **not** contain a communication network.

Instead, it can construct an **ecological interaction network** in which two stationary agents are linked when they occupy the same tile or nearby tiles. This represents potential interaction through the shared local resource system.

Current network measures include:

- network density
- connected components
- largest-component fraction
- average clustering
- policy similarity among connected agents
- policy assortativity

This provides a baseline for comparison with future explicit communication networks.

---

## Repository structure

After adding the Q-learning baseline files, the project should look approximately like this:

```text
cognitive_tools/
│
├── README.md
├── environment.yml
├── run.py
│
├── ecology_run.py
├── ecology_experiment.py
├── ecology_patch_scan.py
├── agent_environment_experiment.py
├── qlearning_experiment.py
├── organize_results.py
│
├── Cognitive_tools/
│   ├── __init__.py
│   ├── model.py
│   ├── env.py
│   ├── ecology.py
│   ├── qlearning.py
│   └── visualization.py
│
└── results/
```

---

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate eco-marl
```

If the environment already exists:

```bash
conda activate eco-marl
```

For GIF generation:

```bash
pip install imageio pillow
```

For MP4 export, install `ffmpeg` separately through your operating system.

---

## Basic checks

Before running a large experiment:

```bash
python -m py_compile \
    Cognitive_tools/model.py \
    Cognitive_tools/env.py \
    Cognitive_tools/qlearning.py \
    qlearning_experiment.py
```

No output means the syntax check passed.

---

# Running experiments

## Interactive visualization

From the repository root:

```bash
PYTHONPATH="$PWD" python -m solara run Cognitive_tools.visualization
```

## Basic agent run

```bash
python run.py \
    --steps 1000 \
    --regen 0.05 \
    --metabolism 0.05
```

## Ecology-only single environment

```bash
python ecology_run.py \
    --steps 2000 \
    --distribution patchy \
    --mean-capacity 0.6 \
    --heterogeneity 0.5 \
    --regen 0.05 \
    --depletion 0.01 \
    --coupling 0.10
```

Run the ecology sweep:

```bash
python ecology_run.py \
    --steps 5000 \
    --sweep
```

## Ecology archetype experiment

```bash
python ecology_experiment.py \
    --steps 1000 \
    --seed 42 \
    --coupling 0.10
```

## Patch structure experiment

```bash
python ecology_patch_scan.py \
    --steps 5000
```

## Random-agent baseline

```bash
python agent_environment_experiment.py \
    --steps 1000 \
    --replicates 10 \
    --populations 4 8 16 32
```

## Q-learning smoke test

Always run a small test before a full experiment:

```bash
python qlearning_experiment.py \
    --run-name smoke_test \
    --scenarios uniform_high patchy_high_central_low \
    --populations 8 \
    --replicates 1 \
    --training-steps 300 \
    --evaluation-steps 100 \
    --record-every 20
```

## Main Q-learning baseline

```bash
python qlearning_experiment.py \
    --run-name qlearning_baseline_v1 \
    --populations 8 16 32 \
    --replicates 10 \
    --training-steps 5000 \
    --evaluation-steps 1000 \
    --record-every 50
```

## Mixed-environment Q-learning experiment

```bash
python qlearning_experiment.py \
    --run-name mixed_environment_baseline \
    --scenarios \
        patchy_high_central_low \
        central_high_patchy_low \
        split_high_low \
        decentralized_high_in_low \
    --populations 8 16 32 \
    --replicates 20 \
    --training-steps 5000 \
    --evaluation-steps 1000
```

## Learning GIF

```bash
python qlearning_experiment.py \
    --run-name gif_patchy_central \
    --make-gif patchy_high_central_low \
    --gif-population 16 \
    --gif-replicate 0 \
    --gif-steps 1000 \
    --frame-every 5 \
    --fps 8 \
    --save-frames \
    --gif-only
```

In the GIF:

```text
circle = cooperative extraction
x      = defective extraction
```

---

# Results organization

## Recommended long-term structure

For new experiments, write outputs directly into an organized hierarchy:

```text
results/
│
├── ecology_archetypes/
│   └── <run_name>/
│       ├── config.json
│       ├── data/
│       ├── figures/
│       ├── gifs/
│       └── frames/
│
├── ecology_patch_scan/
│   └── <run_name>/
│       ├── config.json
│       ├── data/
│       ├── figures/
│       ├── gifs/
│       └── frames/
│
├── random_agent_baseline/
│   └── <run_name>/
│       ├── config.json
│       ├── data/
│       ├── figures/
│       ├── gifs/
│       └── frames/
│
└── q_learning_baseline/
    └── <run_name>/
        ├── config.json
        ├── data/
        ├── figures/
        ├── gifs/
        └── frames/
```

This is preferable to permanently relying on a cleanup script.

### Why use run-specific folders?

A run folder preserves:

- the data produced by that experiment
- figures derived from the data
- the parameter configuration
- GIFs and frames
- separation from previous exploratory runs

Use descriptive run names such as:

```text
qlearning_baseline_v1
mixed_environment_baseline
gamma_sensitivity_v1
population_scan_v2
```

or a timestamp when no explicit name is supplied.

## Organizing existing historical results

`organize_results.py` is best treated as a **one-time migration tool** for results that already exist in the flat `results/` directory.

Preview the changes:

```bash
python organize_results.py
```

Nothing is moved during the dry run.

After reviewing the proposed destinations:

```bash
python organize_results.py --apply
```

For future experiments, prefer changing the experiment scripts so they write directly to their own run directories instead of repeatedly reorganizing files afterward.

---

# Suggested experiment workflow

```text
1. Run a small smoke test
2. Inspect ecological and behavioral trajectories
3. Check learned policies
4. Run multiple replicates
5. Aggregate across replicates
6. Compare environments and population sizes
7. Run sensitivity analyses
8. Preserve each run in its own results directory
```

Important Q-learning sensitivity parameters include:

- learning rate `alpha`
- discount factor `gamma`
- exploration rate `epsilon`
- exploration decay
- cooperative harvest amount
- defective harvest amount
- recovery rate
- equilibrium fraction
- coupling rate
- population size
- training duration

---

# Interpretation notes

### Cooperation is not the same as equality

A population can have low wealth inequality while collectively over-extracting the commons.

```text
wealth Gini != cooperation
```

### Welfare is not the same as cooperation

Mean energy is currently used as a simple welfare proxy. A high-welfare outcome may still be ecologically unsustainable in the short run.

### Collective order is not automatically cooperation

An all-defect population is highly ordered. Always interpret collective order together with cooperation rate.

### "Chaos" should be used cautiously

The current analysis measures behavioral disorder, diversity, switching, and collective organization. Formal dynamical chaos would require additional analysis such as sensitivity to initial conditions or Lyapunov-style measures.

---

# Reproducibility

Use fixed random seeds when comparing environmental treatments.

For final experiments:

- use multiple replicates
- preserve the seed and configuration
- avoid overwriting previous runs
- report means and variability across replicates
- keep raw data separate from generated figures

---

# Current development direction

The present Q-learning baseline is intentionally limited:

```text
ecology
    ↓
local environmental observation
    ↓
independent learning
    ↓
cooperate / defect
```

There is no communication.

This makes it possible to ask whether collective organization can emerge from shared ecological feedback alone.

Later extensions can add:

```text
information quality
communication networks
social learning
resource transfers
exchange
obligation
sanctioning
coordination
```

while retaining the ecology-learning baseline as a comparison condition.
