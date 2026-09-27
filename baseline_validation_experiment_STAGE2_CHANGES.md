# Stage 2 changes for `baseline_validation_experiment.py`

This file describes every change required to the current `main` version of
`baseline_validation_experiment.py` at commit
`8703f07b84709bfc206ca4539a30236408e2d169`.

All code not named below remains unchanged.

## 1. Imports

Add these standard-library imports:

```python
import hashlib
from functools import lru_cache
```

Keep the existing imports.

Replace the `Cognitive_tools.social` import block with:

```python
from Cognitive_tools.social import (
    REWIRING_MODES,
    SOCIAL_NETWORK_MODES,
    SOCIAL_STATE_NAMES,
    copy_sources,
    init_barabasi_albert_attention,
    init_random_attention,
    joint_state,
    network_turnover,
    observed_low_fraction,
    prediction_errors,
    rewire_epoch,
    social_bin,
    social_metrics,
    social_observations,
    update_forecasts,
    visibility_counts,
)
```

## 2. Add these helper functions after `build_run_metadata()`

```python
def sha256_file(
    path: str | Path,
) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as file:
        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


@lru_cache(
    maxsize=None
)
def load_matched_rewire_schedule(
    path_string: str,
) -> dict[
    tuple[
        str,
        int,
        int,
        int,
    ],
    dict[str, object],
]:
    """
    Load an adaptive rewiring schedule for a matched-random R0 run.

    Rows are keyed by:

        (scenario, population, replicate, time)
    """

    path = (
        Path(
            path_string
        )
        .expanduser()
        .resolve()
    )

    if not path.is_file():
        raise ValueError(
            "Matched rewiring schedule does not exist: "
            f"{path}"
        )

    with path.open(
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        required = {
            "scenario",
            "population",
            "replicate",
            "time",
            "rewiring",
            "rewire_theta",
            "rewire_every",
            "base_seed",
            "social_network",
            "social_k",
            "training_steps",
            "successful_rewires",
        }

        fieldnames = set(
            reader.fieldnames
            or []
        )

        missing = (
            required
            - fieldnames
        )

        if missing:
            raise ValueError(
                "Matched rewiring schedule is missing columns: "
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )

        schedule = {}

        for row in reader:
            key = (
                str(
                    row[
                        "scenario"
                    ]
                ),
                int(
                    row[
                        "population"
                    ]
                ),
                int(
                    row[
                        "replicate"
                    ]
                ),
                int(
                    row[
                        "time"
                    ]
                ),
            )

            if key in schedule:
                raise ValueError(
                    "Duplicate matched rewiring schedule row for "
                    f"{key}."
                )

            successful_rewires = int(
                row[
                    "successful_rewires"
                ]
            )

            if successful_rewires < 0:
                raise ValueError(
                    "successful_rewires must be non-negative."
                )

            schedule[
                key
            ] = {
                "rewiring": str(
                    row[
                        "rewiring"
                    ]
                ),
                "rewire_theta": float(
                    row[
                        "rewire_theta"
                    ]
                ),
                "rewire_every": int(
                    row[
                        "rewire_every"
                    ]
                ),
                "base_seed": int(
                    row[
                        "base_seed"
                    ]
                ),
                "social_network": str(
                    row[
                        "social_network"
                    ]
                ),
                "social_k": int(
                    row[
                        "social_k"
                    ]
                ),
                "training_steps": int(
                    row[
                        "training_steps"
                    ]
                ),
                "successful_rewires": (
                    successful_rewires
                ),
            }

    return schedule


def matched_rewire_target(
    args,
    *,
    scenario_name: str,
    population: int,
    replicate: int,
    time: int,
) -> int:
    """Return the exact adaptive event count for one paired R0 checkpoint."""

    if not args.matched_rewire_schedule:
        raise ValueError(
            "random_matched rewiring requires "
            "--matched-rewire-schedule."
        )

    resolved_path = str(
        Path(
            args.matched_rewire_schedule
        )
        .expanduser()
        .resolve()
    )

    schedule = (
        load_matched_rewire_schedule(
            resolved_path
        )
    )

    key = (
        scenario_name,
        int(
            population
        ),
        int(
            replicate
        ),
        int(
            time
        ),
    )

    if key not in schedule:
        raise ValueError(
            "Matched rewiring schedule has no row for "
            f"scenario={scenario_name}, "
            f"population={population}, "
            f"replicate={replicate}, "
            f"time={time}."
        )

    row = schedule[
        key
    ]

    if (
        row[
            "rewiring"
        ]
        != "prediction_error"
    ):
        raise ValueError(
            "Matched R0 schedule must come from a "
            "prediction_error run."
        )

    if not np.isclose(
        float(
            row[
                "rewire_theta"
            ]
        ),
        float(
            args.rewire_theta
        ),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Matched R0 theta does not match the adaptive schedule: "
            f"current={args.rewire_theta}, "
            f"source={row['rewire_theta']}."
        )

    if int(row["rewire_every"]) != int(args.rewire_every):
        raise ValueError(
            "Matched R0 rewire interval does not match the adaptive "
            "schedule."
        )

    if int(row["base_seed"]) != int(args.seed):
        raise ValueError(
            "Matched R0 base seed does not match the adaptive schedule."
        )

    if row["social_network"] != "random_k":
        raise ValueError(
            "Matched R0 schedule must come from a random_k adaptive run."
        )

    if int(row["social_k"]) != int(args.social_k):
        raise ValueError(
            "Matched R0 social_k does not match the adaptive schedule."
        )

    if int(row["training_steps"]) != int(args.training_steps):
        raise ValueError(
            "Matched R0 training length does not match the adaptive "
            "schedule."
        )

    return int(
        row[
            "successful_rewires"
        ]
    )


def treatment_name(
    args,
) -> str:
    """Return a compact treatment identifier for run metadata."""

    if (
        args.social_mode
        == "none"
    ):
        return "B0"

    if (
        args.social_network
        == "ba"
    ):
        return "S2"

    if (
        args.rewiring
        == "none"
    ):
        return "S1"

    if (
        args.rewiring
        == "random_matched"
    ):
        return "R0"

    if (
        args.rewiring
        == "random"
    ):
        return "R0_unmatched"

    if (
        args.rewiring
        == "prediction_error"
    ):
        if np.isclose(
            args.rewire_theta,
            0.0,
        ):
            return "R1"

        if np.isclose(
            args.rewire_theta,
            0.25,
        ):
            return "R2"

        if np.isclose(
            args.rewire_theta,
            1.0,
        ):
            return "R3"

        return "R_adaptive"

    raise ValueError(
        f"Unknown rewiring mode: "
        f"{args.rewiring}"
    )


def validate_configuration(
    args,
) -> None:
    """Validate ecological, social-network, and rewiring configuration."""

    for scenario in (
        args.scenarios
    ):
        if (
            scenario
            not in SCENARIOS
        ):
            raise ValueError(
                f"Unknown scenario: "
                f"{scenario}"
            )

    if not (
        0.0
        <= args.rewire_theta
        <= 1.0
    ):
        raise ValueError(
            "--rewire-theta must be between 0 and 1."
        )

    if not (
        0.0
        <= args.rewire_mu
        <= 1.0
    ):
        raise ValueError(
            "--rewire-mu must be between 0 and 1."
        )

    if (
        args.rewire_every
        <= 0
    ):
        raise ValueError(
            "--rewire-every must be positive."
        )

    if not (
        0.0
        <= args.rewire_threshold
        <= 1.0
    ):
        raise ValueError(
            "--rewire-threshold must be between 0 and 1."
        )

    if not (
        0.0
        <= args.forecast_alpha
        <= 1.0
    ):
        raise ValueError(
            "--forecast-alpha must be between 0 and 1."
        )

    if (
        args.record_network_every
        <= 0
    ):
        raise ValueError(
            "--record-network-every must be positive."
        )

    if (
        args.social_mode
        == "none"
    ):
        if (
            args.rewiring
            != "none"
        ):
            raise ValueError(
                "Rewiring requires --social-mode fixed."
            )

        if args.matched_rewire_schedule:
            raise ValueError(
                "--matched-rewire-schedule requires "
                "--social-mode fixed and "
                "--rewiring random_matched."
            )

        return

    if (
        args.social_mode
        != "fixed"
    ):
        raise ValueError(
            f"Unknown social mode: "
            f"{args.social_mode}"
        )

    if (
        args.social_network
        not in SOCIAL_NETWORK_MODES
    ):
        raise ValueError(
            f"Unknown social network: "
            f"{args.social_network}"
        )

    if (
        args.social_network
        == "random_k"
    ):
        if (
            args.social_k
            < 1
        ):
            raise ValueError(
                "--social-k must be at least 1."
            )

        for population in (
            args.populations
        ):
            if (
                args.social_k
                >= population
            ):
                raise ValueError(
                    "--social-k must be smaller than every population "
                    "size. "
                    f"Received k={args.social_k}, "
                    f"population={population}."
                )

            if (
                args.rewiring
                != "none"
                and args.social_k
                >= population - 1
            ):
                raise ValueError(
                    "Rewiring requires at least one unobserved replacement "
                    "candidate for every observer, so --social-k must be "
                    "at most population - 2."
                )

    elif (
        args.social_network
        == "ba"
    ):
        if (
            args.rewiring
            != "none"
        ):
            raise ValueError(
                "BA social networks are fixed S2 controls in the current "
                "experiment and cannot be rewired."
            )

        if (
            args.ba_m
            < 1
        ):
            raise ValueError(
                "--ba-m must be at least 1."
            )

        for population in (
            args.populations
        ):
            if (
                args.ba_m
                >= population
            ):
                raise ValueError(
                    "--ba-m must be smaller than every population size. "
                    f"Received m={args.ba_m}, "
                    f"population={population}."
                )

    if (
        args.rewiring
        == "random_matched"
    ):
        if not args.matched_rewire_schedule:
            raise ValueError(
                "random_matched rewiring requires "
                "--matched-rewire-schedule."
            )

        # Validate the file before the experiment begins.
        load_matched_rewire_schedule(
            str(
                Path(
                    args.matched_rewire_schedule
                )
                .expanduser()
                .resolve()
            )
        )

    elif args.matched_rewire_schedule:
        raise ValueError(
            "--matched-rewire-schedule is only valid with "
            "--rewiring random_matched."
        )
```

## 3. Replace `make_social_sources()` with this function

```python
def make_social_sources(
    env: EcoEnv,
    replicate: int,
    args,
) -> dict[str, list[str]] | None:
    """Create the requested initial social-information network."""

    if (
        args.social_mode
        == "none"
    ):
        return None

    if (
        args.social_mode
        != "fixed"
    ):
        raise ValueError(
            f"Unknown social mode: "
            f"{args.social_mode}"
        )

    rng = np.random.default_rng(
        social_seed(
            replicate,
            len(
                env.possible_agents
            ),
            args.seed,
        )
    )

    agents = list(
        env.possible_agents
    )

    if (
        args.social_network
        == "random_k"
    ):
        return init_random_attention(
            agents,
            args.social_k,
            rng,
        )

    if (
        args.social_network
        == "ba"
    ):
        return (
            init_barabasi_albert_attention(
                agents,
                args.ba_m,
                rng,
            )
        )

    raise ValueError(
        f"Unknown social network: "
        f"{args.social_network}"
    )
```

## 4. Replace `train_q_learning()` with this function

```python
def train_q_learning(
    scenario_name: str,
    population: int,
    replicate: int,
    args,
):
    total_steps = (
        args.training_steps
        + args.evaluation_steps
    )

    (
        env,
        observations,
        regions,
    ) = make_environment(
        scenario_name,
        population,
        replicate,
        total_steps,
        args,
    )

    sources = make_social_sources(
        env,
        replicate,
        args,
    )

    learners = make_learners(
        env,
        replicate,
        args,
    )

    timeseries = []
    network_timeseries = []
    rewiring_schedule = []

    previous_actions = None

    rewire_counts = {
        name: 0
        for name
        in env.possible_agents
    }

    prediction_error_sums = {
        name: 0.0
        for name
        in env.possible_agents
    }

    prediction_error_counts = {
        name: 0
        for name
        in env.possible_agents
    }

    if sources is None:
        forecasts = None
        rewire_rng = None
        previous_recorded_sources = None

    else:
        forecasts = {
            name: 0.5
            for name
            in env.possible_agents
        }

        rewire_rng = np.random.default_rng(
            rewiring_seed(
                replicate,
                population,
                args.seed,
            )
        )

        previous_recorded_sources = (
            copy_sources(
                sources
            )
        )

        network_timeseries.append(
            make_network_record(
                scenario_name=scenario_name,
                population=population,
                replicate=replicate,
                time=0,
                args=args,
                sources=sources,
                previous_recorded_sources=(
                    previous_recorded_sources
                ),
                actions=None,
                prediction_error_values=None,
                events_since_record=[],
                cumulative_rewires=0,
            )
        )

    events_since_record: list[
        dict[str, object]
    ] = []

    cumulative_rewires = 0

    for time in range(
        1,
        args.training_steps
        + 1,
    ):
        # -------------------------------------------------------------
        # (G_t, a_(t-1), R_t) -> s_t
        # -------------------------------------------------------------

        states = encode_states(
            observations,
            social_mode=(
                args.social_mode
            ),
            sources=sources,
            previous_actions=(
                previous_actions
            ),
        )

        # -------------------------------------------------------------
        # s_t -> a_t
        # -------------------------------------------------------------

        actions = {
            name: learners[
                name
            ].choose_action(
                states[
                    name
                ],
                explore=True,
            )
            for name
            in env.agents
        }

        # -------------------------------------------------------------
        # a_t -> R_(t+1)
        # -------------------------------------------------------------

        (
            next_observations,
            rewards,
            terminations,
            truncations,
            _,
        ) = env.step(
            actions
        )

        current_prediction_errors = None

        # -------------------------------------------------------------
        # Observe a_t through G_t, then optionally form G_(t+1).
        # -------------------------------------------------------------

        if sources is not None:
            assert forecasts is not None
            assert rewire_rng is not None

            observed_before_rewire = (
                social_observations(
                    sources,
                    actions,
                )
            )

            current_prediction_errors = (
                prediction_errors(
                    forecasts,
                    observed_before_rewire,
                )
            )

            for name, error in (
                current_prediction_errors.items()
            ):
                prediction_error_sums[
                    name
                ] += error

                prediction_error_counts[
                    name
                ] += 1

            events = []

            if (
                time
                % args.rewire_every
                == 0
            ):
                target_count = None

                if (
                    args.rewiring
                    == "random_matched"
                ):
                    target_count = (
                        matched_rewire_target(
                            args,
                            scenario_name=(
                                scenario_name
                            ),
                            population=(
                                population
                            ),
                            replicate=(
                                replicate
                            ),
                            time=time,
                        )
                    )

                events = rewire_epoch(
                    sources,
                    mode=args.rewiring,
                    theta=(
                        args.rewire_theta
                    ),
                    mu=args.rewire_mu,
                    rng=rewire_rng,
                    prediction_error_values=(
                        current_prediction_errors
                    ),
                    threshold=(
                        args.rewire_threshold
                    ),
                    target_count=(
                        target_count
                    ),
                )

                if (
                    args.rewiring
                    != "none"
                ):
                    rewiring_schedule.append(
                        {
                            "scenario": (
                                scenario_name
                            ),
                            "population": (
                                population
                            ),
                            "replicate": (
                                replicate
                            ),
                            "time": time,
                            "rewiring": (
                                args.rewiring
                            ),
                            "rewire_theta": (
                                args.rewire_theta
                            ),
                            "rewire_every": (
                                args.rewire_every
                            ),
                            "base_seed": (
                                args.seed
                            ),
                            "social_network": (
                                args.social_network
                            ),
                            "social_k": (
                                args.social_k
                            ),
                            "training_steps": (
                                args.training_steps
                            ),
                            "target_rewires": (
                                ""
                                if target_count
                                is None
                                else target_count
                            ),
                            "successful_rewires": (
                                len(
                                    events
                                )
                            ),
                        }
                    )

                for event in events:
                    observer = str(
                        event[
                            "observer"
                        ]
                    )

                    rewire_counts[
                        observer
                    ] += 1

                cumulative_rewires += len(
                    events
                )

                events_since_record.extend(
                    events
                )

            # Forecast update uses what was observed before rewiring.
            update_forecasts(
                forecasts,
                observed_before_rewire,
                alpha=(
                    args.forecast_alpha
                ),
            )

        # -------------------------------------------------------------
        # R_(t+1), G_(t+1), a_t -> s_(t+1)
        # -------------------------------------------------------------

        next_states = encode_states(
            next_observations,
            social_mode=(
                args.social_mode
            ),
            sources=sources,
            previous_actions=actions,
        )

        # -------------------------------------------------------------
        # Q update
        # -------------------------------------------------------------

        for name in states:
            done = (
                terminations[
                    name
                ]
                or truncations[
                    name
                ]
            )

            learner_next_state = (
                states[
                    name
                ]
                if done
                else next_states[
                    name
                ]
            )

            learners[
                name
            ].update(
                states[
                    name
                ],
                actions[
                    name
                ],
                rewards[
                    name
                ],
                learner_next_state,
                done=done,
            )

            learners[
                name
            ].decay_exploration()

        # -------------------------------------------------------------
        # Standard training diagnostics
        # -------------------------------------------------------------

        if (
            time == 1
            or time
            % args.record_every
            == 0
            or time
            == args.training_steps
        ):
            timeseries.append(
                {
                    "scenario": (
                        scenario_name
                    ),
                    "population": (
                        population
                    ),
                    "replicate": (
                        replicate
                    ),
                    "social_mode": (
                        args.social_mode
                    ),
                    "rewiring": (
                        args.rewiring
                    ),
                    "time": time,
                    **system_metrics(
                        env,
                        actions,
                        sources=sources,
                    ),
                    "mean_epsilon": float(
                        np.mean(
                            [
                                learner.epsilon
                                for learner
                                in learners.values()
                            ]
                        )
                    ),
                }
            )

        # -------------------------------------------------------------
        # Network diagnostics
        # -------------------------------------------------------------

        if (
            sources is not None
            and (
                time == 1
                or time
                % args.record_network_every
                == 0
                or time
                == args.training_steps
            )
        ):
            assert (
                previous_recorded_sources
                is not None
            )

            network_timeseries.append(
                make_network_record(
                    scenario_name=(
                        scenario_name
                    ),
                    population=population,
                    replicate=replicate,
                    time=time,
                    args=args,
                    sources=sources,
                    previous_recorded_sources=(
                        previous_recorded_sources
                    ),
                    actions=actions,
                    prediction_error_values=(
                        current_prediction_errors
                    ),
                    events_since_record=(
                        events_since_record
                    ),
                    cumulative_rewires=(
                        cumulative_rewires
                    ),
                )
            )

            previous_recorded_sources = (
                copy_sources(
                    sources
                )
            )

            events_since_record = []

        previous_actions = (
            actions.copy()
        )

        observations = (
            next_observations
        )

    mean_prediction_errors = {}

    for name in env.possible_agents:
        count = (
            prediction_error_counts[
                name
            ]
        )

        if count > 0:
            mean_prediction_errors[
                name
            ] = (
                prediction_error_sums[
                    name
                ]
                / count
            )

        else:
            mean_prediction_errors[
                name
            ] = float(
                "nan"
            )

    return (
        env,
        observations,
        regions,
        learners,
        timeseries,
        network_timeseries,
        sources,
        previous_actions,
        rewire_counts,
        mean_prediction_errors,
        rewiring_schedule,
    )
```

## 5. Replace `run_condition()` with this function

```python
def run_condition(
    scenario_name: str,
    population: int,
    replicate: int,
    args,
):
    (
        trained_env,
        trained_observations,
        regions,
        learners,
        training_timeseries,
        network_timeseries,
        sources,
        final_training_actions,
        rewire_counts,
        mean_prediction_errors,
        rewiring_schedule,
    ) = train_q_learning(
        scenario_name,
        population,
        replicate,
        args,
    )

    (
        policy_summary,
        agent_policy_rows,
    ) = policy_diagnostics(
        trained_env,
        regions,
        learners,
        scenario_name,
        population,
        replicate,
        social_mode=(
            args.social_mode
        ),
        rewiring=(
            args.rewiring
        ),
        sources=sources,
        rewire_counts=(
            rewire_counts
        ),
        mean_prediction_errors=(
            mean_prediction_errors
        ),
    )

    (
        continuation_summary,
        continuation_ts,
    ) = evaluate_policy(
        trained_env,
        trained_observations,
        scenario_name=(
            scenario_name
        ),
        population=population,
        replicate=replicate,
        strategy="q_learning",
        evaluation_mode=(
            "continuation"
        ),
        learners=learners,
        sources=sources,
        previous_actions=(
            final_training_actions
        ),
        args=args,
    )

    (
        fresh_env,
        fresh_obs,
        _,
    ) = make_environment(
        scenario_name,
        population,
        replicate,
        args.evaluation_steps,
        args,
    )

    (
        fresh_summary,
        fresh_ts,
    ) = evaluate_policy(
        fresh_env,
        fresh_obs,
        scenario_name=(
            scenario_name
        ),
        population=population,
        replicate=replicate,
        strategy="q_learning",
        evaluation_mode=(
            "fresh_reset"
        ),
        learners=learners,
        sources=sources,
        previous_actions=None,
        args=args,
    )

    control_summaries = []
    control_timeseries = []

    for strategy in (
        "always_low",
        "always_high",
        "random_50",
    ):
        (
            control_env,
            control_obs,
            _,
        ) = make_environment(
            scenario_name,
            population,
            replicate,
            args.evaluation_steps,
            args,
        )

        (
            summary,
            timeseries,
        ) = evaluate_policy(
            control_env,
            control_obs,
            scenario_name=(
                scenario_name
            ),
            population=population,
            replicate=replicate,
            strategy=strategy,
            evaluation_mode=(
                "fresh_reset"
            ),
            learners=None,
            sources=sources,
            previous_actions=None,
            args=args,
        )

        control_summaries.append(
            summary
        )

        control_timeseries.extend(
            timeseries
        )

    return {
        "evaluation_summary": [
            continuation_summary,
            fresh_summary,
            *control_summaries,
        ],
        "training_timeseries": (
            training_timeseries
        ),
        "evaluation_timeseries": [
            *continuation_ts,
            *fresh_ts,
            *control_timeseries,
        ],
        "policy_summary": (
            policy_summary
        ),
        "agent_policies": (
            agent_policy_rows
        ),
        "network_timeseries": (
            network_timeseries
        ),
        "rewiring_schedule": (
            rewiring_schedule
        ),
    }
```

## 6. Add these CLI arguments in `main()` under the social-information section

Insert after `--social-mode` and before `--social-k`:

```python
    parser.add_argument(
        "--social-network",
        choices=SOCIAL_NETWORK_MODES,
        default="random_k",
        help=(
            "Initial social-network family. "
            "'random_k' gives the directed fixed-attention network used "
            "by S1 and rewiring treatments. "
            "'ba' gives the fixed visibility-skew S2 control."
        ),
    )
```

Insert after `--social-k`:

```python
    parser.add_argument(
        "--ba-m",
        type=int,
        default=2,
        help=(
            "Preferential-attachment parameter for --social-network ba. "
            "m=2 gives mean degree close to 4 for moderate/large N."
        ),
    )
```

## 7. Add this CLI argument under the rewiring section

Insert after `--rewiring`:

```python
    parser.add_argument(
        "--matched-rewire-schedule",
        default=None,
        help=(
            "Path to an adaptive run's data/rewiring_schedule.csv. "
            "Required only for --rewiring random_matched."
        ),
    )
```

Also replace the `--rewiring` help text with:

```python
        help=(
            "'none' gives a fixed network; "
            "'random' retains the legacy independent-probability random "
            "turnover control; "
            "'random_matched' gives the confirmatory R0 control using a "
            "paired adaptive event-count schedule; "
            "'prediction_error' gives adaptive rewiring."
        ),
```

## 8. Replace the existing validation block in `main()`

Delete the block beginning with:

```python
    # -----------------------------------------------------------------
    # Validate ecological treatments
```

through the final `--record-network-every` validation.

Replace it with:

```python
    # -----------------------------------------------------------------
    # Validate experiment configuration
    # -----------------------------------------------------------------

    validate_configuration(
        args
    )
```

## 9. Extend the `config` block

Immediately after:

```python
    config[
        "learner_n_states"
    ] = (
        learner_state_count(
            args.social_mode
        )
    )
```

add:

```python
    config[
        "treatment"
    ] = treatment_name(
        args
    )

    config[
        "social_network_semantics"
    ] = (
        "none"
        if args.social_mode
        == "none"
        else (
            "directed_fixed_k"
            if args.social_network
            == "random_k"
            else "fixed_symmetric_ba"
        )
    )
```

Immediately after the existing `fresh_network` block add:

```python
    config[
        "matched_rewire_schedule_sha256"
    ] = (
        sha256_file(
            args.matched_rewire_schedule
        )
        if args.matched_rewire_schedule
        else None
    )
```

Leave the existing `run_metadata` block in place.

## 10. Add one accumulator in `main()`

After:

```python
    network_rows = []
```

add:

```python
    rewiring_schedule_rows = []
```

Inside the condition loop, after extending `network_rows`, add:

```python
                rewiring_schedule_rows.extend(
                    output[
                        "rewiring_schedule"
                    ]
                )
```

## 11. Write the rewiring schedule

After the existing `network_timeseries.csv` write, add:

```python
    write_csv(
        data_dir
        / "rewiring_schedule.csv",
        rewiring_schedule_rows,
    )
```

Because `write_csv()` returns immediately for an empty list, B0/S1/S2 runs
will not create a meaningless empty schedule file.

## 12. Update the run-status prints

After printing `Social mode`, add:

```python
    if (
        args.social_mode
        == "fixed"
    ):
        print(
            f"Social network: "
            f"{args.social_network}"
        )
```

Replace the current `Attention capacity k` print block with:

```python
    if (
        args.social_mode
        == "fixed"
        and args.social_network
        == "random_k"
    ):
        print(
            f"Attention capacity k: "
            f"{args.social_k}"
        )

    if (
        args.social_mode
        == "fixed"
        and args.social_network
        == "ba"
    ):
        print(
            f"BA attachment m: "
            f"{args.ba_m}"
        )
```

Keep the existing theta print for social runs, or preferably restrict it to
rewiring runs:

```python
    if (
        args.rewiring
        != "none"
    ):
        print(
            f"Search scope theta: "
            f"{args.rewire_theta}"
        )
```

For matched R0 add:

```python
    if (
        args.rewiring
        == "random_matched"
    ):
        print(
            "Matched schedule: "
            f"{args.matched_rewire_schedule}"
        )
```

No other functions in `baseline_validation_experiment.py` need to change for
Stage 2.
