import numpy as np
import pytest

from cognitive_tools.env import EcoEnv


def test_capped_reward_preserves_physics_and_accounts_for_surplus():
    envs = [EcoEnv(n_agents=2, max_steps=4, reward_mode=m) for m in ("harvest", "capped_harvest")]
    for env in envs:
        env.reset(seed=123)
    for _ in range(4):
        outputs = [env.step({"agent_0": 1, "agent_1": 0}) for env in envs]
        np.testing.assert_array_equal(envs[0].model.resource, envs[1].model.resource)
        for name in envs[0].possible_agents:
            gross = outputs[0][1][name]
            utility = outputs[1][1][name]
            assert utility == min(gross, 0.002)
            info = outputs[1][4][name]
            assert info["harvested"] == gross
            assert info["utility_reward"] == utility
            assert info["uncredited_harvest"] == pytest.approx(gross - utility)
            for field in ("wealth", "energy"):
                assert getattr(envs[0].model.by_name[name], field) == getattr(
                    envs[1].model.by_name[name], field
                )
            if outputs[0][0]:
                for key in outputs[0][0][name]:
                    np.testing.assert_array_equal(
                        outputs[0][0][name][key], outputs[1][0][name][key]
                    )


def test_full_reserves_do_not_reward_zero_harvest():
    env = EcoEnv(n_agents=1, initial_resource_fraction=0, reward_mode="capped_harvest")
    env.reset(seed=1)
    _, rewards, _, _, infos = env.step({"agent_0": 1})
    assert rewards["agent_0"] == 0
    assert infos["agent_0"]["energy"] > 0


@pytest.mark.parametrize(
    "mode,need",
    [
        ("unknown", 0.002),
        ("capped_harvest", 0),
        ("capped_harvest", float("nan")),
        ("capped_harvest", float("inf")),
    ],
)
def test_invalid_reward_contract(mode, need):
    with pytest.raises(ValueError):
        EcoEnv(reward_mode=mode, metabolism_rate=need)
