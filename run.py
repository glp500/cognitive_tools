import argparse

import numpy as np

from Cognitive_tools import EcoEnv


def gini(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    if (
        len(values) == 0
        or values.sum() == 0
    ):
        return 0.0

    values = np.sort(values)

    n = len(values)

    index = np.arange(
        1,
        n + 1,
    )

    return (
        2
        * np.sum(index * values)
        / (n * values.sum())
        - (n + 1) / n
    )


def print_summary(env):

    model = env.model

    wealth = [
        agent.wealth
        for agent
        in model.by_name.values()
    ]

    energy = [
        agent.energy
        for agent
        in model.by_name.values()
    ]

    print(
        f"step={model.steps:>7} | "
        f"resource="
        f"{model.resource.sum():>8.3f} | "
        f"wealth="
        f"{sum(wealth):>8.3f} | "
        f"gini="
        f"{gini(wealth):.3f} | "
        f"mean_energy="
        f"{np.mean(energy):.3f} | "
        f"zero_energy="
        f"{sum(value <= 0.0 for value in energy)}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--steps",
        type=int,
        default=1_000,
    )

    parser.add_argument(
        "--regen",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--metabolism",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--initial-energy",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--agents",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--report-every",
        type=int,
        default=100,
        help=(
            "Print metrics every N steps. "
            "Use 0 to disable."
        ),
    )

    args = parser.parse_args()

    env = EcoEnv(
        n_agents=args.agents,
        max_steps=args.steps,
        regeneration_rate=args.regen,
        metabolism_rate=args.metabolism,
        initial_energy=(
            args.initial_energy
        ),
    )

    env.reset(
        seed=args.seed
    )

    while env.agents:

        actions = {
            agent:
            env.action_space(
                agent
            ).sample()
            for agent
            in env.agents
        }

        env.step(actions)

        if (
            args.report_every > 0
            and
            env.model.steps
            % args.report_every
            == 0
        ):
            print_summary(env)

    print(
        "\nFinal state"
    )

    print_summary(env)

    print(
        "\nAgents"
    )

    for name, agent in (
        env.model.by_name.items()
    ):

        print(
            f"{name}: "
            f"wealth="
            f"{agent.wealth:.3f}, "
            f"energy="
            f"{agent.energy:.3f}, "
            f"position="
            f"{tuple(agent.position)}"
        )


if __name__ == "__main__":
    main()