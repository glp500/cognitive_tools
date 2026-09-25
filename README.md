# cognitive_tools

`cognitive_tools` is a research codebase for studying how ecological
feedback, local learning, social information, and decentralized
adaptation interact in renewable common-pool-resource systems.

The central social-network question is:

> Can agents that locally predict the behavior of their information
> sources use prediction error to adapt whom they observe, and can that
> decentralized rewiring improve or destabilize common-pool-resource
> sustainability?

## Current model

The implemented model now contains:

1. a spatial renewable resource;
2. stationary resource users;
3. low- and high-extraction actions;
4. independent tabular Q-learning;
5. a three-state ecological representation based on local `R/K`;
6. an optional directed fixed-k social-information network;
7. previous-action social observation;
8. a nine-state ecology x social learner;
9. optional decentralized network rewiring.

Ecological dynamics and social information remain separate.

The physical environment owns:

```text
resource dynamics
extraction
reward
wealth
welfare