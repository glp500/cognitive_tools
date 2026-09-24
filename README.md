# cognitive_tools

`cognitive_tools` is a research codebase for studying how local learning,
social information, and decentralized adaptation interact with renewable
common-pool-resource dynamics.

The current baseline asks:

> Can resource users learn extraction behavior from local ecological
> feedback alone?

The next model extension asks:

> Can agents that locally predict the behavior of their information
> sources use prediction error to adapt whom they observe, and how does
> that decentralized rewiring interact with ecological conditions?

---

## Current model

The current implemented model contains:

1. a spatial renewable resource;
2. stationary resource users;
3. low- and high-extraction actions;
4. independent tabular Q-learning;
5. ecological state information based on local `R/K`;
6. no social-information network yet.

The social layer is intentionally being added only after the ecological
and learning baselines are mechanically stable.

---

## Architecture

```text
cognitive_tools/
│
├── README.md
├── PROVENANCE.md
├── REFERENCES.bib
├── BASELINE_EXPERIMENT.md
├── environment.yml
│
├── Cognitive_tools/
│   ├── __init__.py
│   ├── ecology.py
│   ├── model.py
│   ├── env.py
│   ├── qlearning.py
│   └── visualization.py
│
├── baseline_validation_experiment.py
├── baseline_validation_figures.py
│
├── qlearning_experiment.py
├── agent_environment_experiment.py
├── run.py
│
└── tests/
    ├── test_ecology.py
    ├── test_qlearning.py
    └── test_imports.py