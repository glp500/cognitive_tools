import json

import pytest

from cognitive_tools import experiment


@pytest.mark.parametrize("mode", ["none", "prediction_error", "random_matched"])
def test_workers_preserve_all_outputs_and_resume(tmp_path, monkeypatch, mode):
    args = experiment.build_parser().parse_args(
        [
            "--scenarios",
            "uniform_high",
            "--populations",
            "8",
            "--replicates",
            "2",
            "--training-steps",
            "8",
            "--evaluation-steps",
            "3",
            "--record-every",
            "2",
            "--record-network-every",
            "2",
            "--rewire-every",
            "2",
            "--rewire-threshold",
            "0",
            "--rewire-mu",
            "1",
            "--social-mode",
            "fixed",
            "--rewiring",
            mode,
        ]
    )
    if mode == "random_matched":
        args.rewiring = "prediction_error"
        schedule = []
        for replicate in range(2):
            schedule.extend(
                experiment.run_condition("uniform_high", 8, replicate, args)["rewiring_schedule"]
            )
        path = tmp_path / "schedule.csv"
        experiment.write_csv(path, schedule)
        args.matched_rewire_schedule = str(path)
        args.rewiring = mode
    directories = [tmp_path / "serial", tmp_path / "parallel"]
    outputs = []
    for workers, directory in zip((1, 2), directories):
        directory.mkdir()
        args.workers = workers
        outputs.append(json.dumps(list(experiment.run_conditions(args, directory)), sort_keys=True))
    assert outputs[0] == outputs[1]
    assert len(list((directories[1] / "conditions").glob("*.json"))) == 2
    # A fully recovered run must not dispatch any simulation work.
    monkeypatch.setattr(experiment, "condition_checkpoint", lambda _: pytest.fail("recomputed"))
    assert (
        json.dumps(list(experiment.run_conditions(args, directories[1])), sort_keys=True)
        == outputs[0]
    )


def test_condition_resume_rejects_changed_reward(tmp_path):
    args = experiment.build_parser().parse_args(
        [
            "--scenarios",
            "uniform_high",
            "--populations",
            "8",
            "--replicates",
            "1",
            "--training-steps",
            "2",
            "--evaluation-steps",
            "2",
            "--workers",
            "1",
        ]
    )
    list(experiment.run_conditions(args, tmp_path))
    args.reward_mode = "capped_harvest"
    with pytest.raises(ValueError, match="configuration"):
        list(experiment.run_conditions(args, tmp_path))
