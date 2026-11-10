import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from reticular_diploid.data.frozen_diploid import sha256
from reticular_diploid.experiments import run_public_distillation as campaign
from reticular_diploid.experiments.run_public_distillation import trial_grid


def test_grid_is_paired_across_arms_and_excludes_only_preexisting_primary_trials():
    recipe = {"training_people": [32, 128, 400], "seeds": [20261023, 20261024],
              "effect_variants": [16, 128, 1024], "arms": ["scratch", "pretrained", "joint"]}
    full = trial_grid(recipe, [400, 128, 32], recipe["seeds"])
    remaining = trial_grid(recipe, [400, 128, 32], recipe["seeds"], True)
    assert len(full) == 54 and len(remaining) == 45
    assert set(full) - set(remaining) == {(400, 20261023, effects, arm)
                                        for effects in recipe["effect_variants"] for arm in recipe["arms"]}
    for values in [[16], [400, 400]]:
        with pytest.raises(ValueError):
            trial_grid(recipe, values, recipe["seeds"])
    with pytest.raises(ValueError):
        trial_grid(recipe, [400], [1])


def test_runner_canaries_resumes_and_skips_only_verified_completed_trials(tmp_path, monkeypatch):
    ntp = tmp_path / "ntp"
    ntp.mkdir()
    (ntp / "selected.pt").write_bytes(b"synthetic NTP fixture")
    selected = {"path": "selected.pt", "sha256": sha256(ntp / "selected.pt")}
    (ntp / "selected-checkpoint.json").write_text(json.dumps(selected))
    recipe = {"training_people": [32], "seeds": [7], "effect_variants": [16],
              "arms": ["scratch", "pretrained", "joint"], "training": {"steps": 1000},
              "resources": {"maximum_worker_hours": .5}, "checkpoint_retention": "latest_and_best"}
    config = tmp_path / "config.json"
    config.write_text(json.dumps(recipe))
    args = SimpleNamespace(config=config, output=tmp_path / "students", ntp_run=ntp,
                           corpus=tmp_path / "corpus", teachers=tmp_path / "teachers",
                           training_people=None, seeds=None, exclude_primary_full=False, maximum_hours=1.)
    monkeypatch.setattr(campaign, "clean_commit", lambda: "synthetic-source")
    commands = []

    def child(command, **kwargs):
        commands.append(command)
        def value(key):
            return command[command.index("--" + key) + 1]
        output = Path(value("output"))
        output.mkdir(exist_ok=True)
        step = int(value("stop-after"))
        contract = {"code_commit": "synthetic-source", "recipe_sha256": sha256(config),
                    "pretrained_sha256": selected["sha256"], "arm": value("arm"),
                    "seed": 7, "training_people": 32, "effect_variants": 16}
        (output / "contract.json").write_text(json.dumps(contract))
        for name in ["latest.pt", "best.pt"]:
            (output / name).write_bytes(f"synthetic step{step}".encode())
        with (output / "learning-curve.jsonl").open("a") as handle:
            handle.write(json.dumps({"step": step, "total_worker_seconds": step / 1000}) + "\n")
        (output / "selected-checkpoint.json").write_text(json.dumps({"path": "best.pt",
            "sha256": sha256(output / "best.pt"), "step": step}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(campaign.subprocess, "run", child)
    campaign.run(args)
    assert len(commands) == 6
    assert all("--resume" not in cmd for cmd in commands[::2])
    assert all(Path(cmd[cmd.index("--resume") + 1]).name == "latest.pt" for cmd in commands[1::2])
    campaign.run(args)
    assert len(commands) == 6
    path = args.output / "A1-scratch-n32-s7/contract.json"
    contract = json.loads(path.read_text())
    path.write_text(json.dumps({**contract, "code_commit": "unrelated-source"}))
    with pytest.raises(ValueError, match="immutable"):
        campaign.run(args)
    assert len(commands) == 6
