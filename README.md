# cognitive_tools

`cognitive_tools` is a research codebase for studying how ecological
feedback, individual reinforcement learning, social information, and
decentralized adaptation interact in renewable common-pool-resource
systems.

The central social-network question is:

> Can agents that locally predict the behavior of their information
> sources use prediction error to adapt whom they observe, and can that
> decentralized rewiring improve or destabilize common-pool-resource
> sustainability?

The project keeps physical resource dynamics and social information
mechanically separate.

---

## Current implementation

The implemented model contains:

1. a spatial renewable resource;
2. stationary resource users;
3. low- and high-extraction actions;
4. independent tabular Q-learning;
5. a three-state ecological representation based on local `R/K`;
6. an optional directed fixed-attention social-information network;
7. previous-action social observation;
8. a nine-state ecology x social learner;
9. random social-network turnover;
10. prediction-error-triggered decentralized rewiring;
11. local/global replacement search;
12. ecological, welfare, inequality, perception, and network diagnostics.

The current implementation supports the following core treatments:

| Treatment | Social information | Network dynamics |
|---|---|---|
| `B0` | none | none |
| `S1` | previous peer actions | fixed random directed network |
| `R0` | previous peer actions | random rewiring |
| `R1` | previous peer actions | prediction-error rewiring, local search |
| `R2` | previous peer actions | prediction-error rewiring, mixed local/global search |
| `R3` | previous peer actions | prediction-error rewiring, global search |

`R1`, `R2`, and `R3` use the same prediction-error mechanism and differ
through the global-search probability `theta`:

```text
R1: theta = 0.00
R2: theta = 0.25
R3: theta = 1.00