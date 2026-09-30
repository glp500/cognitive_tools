import pytest

from cognitive_tools import experiment
from cognitive_tools.social import visibility_counts
from cognitive_tools.visibility import PROTOCOL, load_visibility_spec


def args(profile="equal", dynamics="fixed"):
    _, sha = load_visibility_spec()
    parsed = experiment.build_parser().parse_args(
        [
            "--study-protocol",
            PROTOCOL,
            "--reward-mode",
            "capped_harvest",
            "--visibility-profile",
            profile,
            "--network-dynamics",
            dynamics,
            "--social-mode",
            "fixed",
            "--social-network",
            "random_k",
            "--social-k",
            "4",
            "--attention-k",
            "4",
            "--rewiring",
            "prediction_error" if dynamics == "adaptive_bounded" else "none",
            "--scenarios",
            "uniform_high",
            "--populations",
            "8",
            "--replicates",
            "1",
            "--training-steps",
            "4",
            "--evaluation-steps",
            "2",
            "--record-every",
            "2",
            "--record-network-every",
            "2",
            "--rewire-every",
            "50",
        ]
    )
    parsed.visibility_profile_spec_sha256 = sha
    return parsed


@pytest.mark.parametrize(
    "profile",
    ["equal", "random", "normal_centered", "low_propensity_majority", "high_propensity_majority"],
)
def test_stage5_pair_and_ecology_share_exact_initialization(profile):
    fixed = args(profile)
    adaptive = args(profile, "adaptive_bounded")
    experiment.validate_configuration(fixed)
    experiment.validate_configuration(adaptive)
    seen = []
    for configuration, scenario in (
        (fixed, "uniform_high"),
        (adaptive, "uniform_high"),
        (fixed, "patchy_high"),
    ):
        env, _, _ = experiment.make_environment(scenario, 8, 0, 6, configuration)
        seen.append(experiment.make_social_sources(env, 0, configuration))
    assert seen[0] == seen[1] == seen[2]
    assert all(len(row) == 4 for row in seen[0].values())
    if profile == "equal":
        assert set(visibility_counts(seen[0]).values()) == {4}


def test_stage5_rejects_search_sweep_and_wrong_reward():
    configuration = args()
    configuration.rewire_theta = 0
    with pytest.raises(ValueError, match="theta"):
        experiment.validate_configuration(configuration)
    configuration.rewire_theta = 0.25
    configuration.reward_mode = "harvest"
    with pytest.raises(ValueError, match="capped"):
        experiment.validate_configuration(configuration)


def test_stage5_condition_records_initial_visibility():
    configuration = args()
    result = experiment.run_condition("uniform_high", 8, 0, configuration)
    rows = result["initial_visibility"]
    assert len(rows) == 8
    assert {row["initial_visibility_count"] for row in rows} == {4}
    assert len({row["initial_network_sha256"] for row in rows}) == 1
    assert len({row["initial_propensity_sha256"] for row in rows}) == 1
    assert result["network_timeseries"][0]["visibility_gini"] == 0
