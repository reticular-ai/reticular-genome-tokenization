import json
from types import SimpleNamespace

import pytest

from reticular_diploid.data.frozen_diploid import sha256
from reticular_diploid.experiments.package_public_students import package, tensor_fingerprint


def fixture(tmp_path, monkeypatch, bad=None):
    import torch
    from torch import nn

    from reticular_diploid.experiments import package_public_students as export
    from reticular_diploid.experiments.distill_public_diploid import ContextualRegressor
    from reticular_diploid.models import public_kimi

    class Fake(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1., 2.]))

    monkeypatch.setattr(public_kimi, "NoPECausalTransformer", Fake)
    monkeypatch.setattr(export, "clean_commit", lambda: "independently-synthetic")
    protocol = {"cohort_sha256": "cohort", "tokenizer_sha256": "tokenizer", "canonical_sha256": "canonical",
                "teacher_scores_sha256": "scores", "teacher_weights_sha256": "weights",
                "student_grid": {"effect_variants": [16], "training_people": [4], "seeds": [7],
                                 "arms": ["pretrained"], "optimizer_steps": 2}}
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    contract = {"schema_version": "reticular-public-distillation/v1", "effect_variants": 16,
                "corpus": {k: protocol[k] for k in ["cohort_sha256", "tokenizer_sha256", "canonical_sha256"]}
                          | {"public_data_only": bad != "controlled", "locked_test_used": False},
                "pretrained_sha256": "parent", "teachers": {"scores_sha256": "scores", "weights_sha256": "weights"},
                "ntp_parent": {"model_config": {"vocabulary_size": 13, "d_model": 2}}}
    model = ContextualRegressor(Fake(None), 2)
    if bad == "nonfinite":
        with torch.no_grad():
            model.backbone.weight[0] = torch.nan
    root = tmp_path / "students"
    name = "A1-pretrained-n4-s7"
    source = root / name
    source.mkdir(parents=True)
    (source / "contract.json").write_text(json.dumps(contract))
    checkpoint = source / "best.pt"
    torch.save({"model": model.state_dict(), "contract": contract, "step": 2, "exposure": 8,
                "optimizer": {"private_auxiliary": torch.arange(12)}, "torch_rng": torch.get_rng_state(),
                "unrelated_field": "INDEPENDENTLY_SYNTHETIC_EXCLUDED_FIXTURE"}, checkpoint)
    metric = {"r_squared": .2, "mse": .8, "pearson_r": .5, "people": 55}
    selected = {"path": "best.pt", "sha256": sha256(checkpoint), "step": 2, "validation": metric}
    (source / "selected-checkpoint.json").write_text(json.dumps(selected))
    point = {"step": 2, "person_draws": 4, "region_targets_seen": 8, "total_worker_seconds": 1.,
             "validation": metric, "peak_allocated_bytes": None, "training_batch_mse": 123.}
    (source / "learning-curve.jsonl").write_text(json.dumps(point) + "\n")
    row = {"id": name, "sha256": selected["sha256"], "selected_step": 2, "validation": metric,
           "effect_variants": 16, "training_people": 4, "seed": 7, "arm": "pretrained", "contract": contract,
           "final": point}
    certificate = tmp_path / "certificate.json"
    certificate.write_text(json.dumps({"schema_version": "reticular-public-selection-certificate/v1",
        "protocol_sha256": sha256(protocol_path), "cohort_sha256": "cohort", "ntp_checkpoint_sha256": "parent",
        "students": [row]}))
    args = SimpleNamespace(output=tmp_path / "review", protocol=protocol_path,
                           selection_certificate=certificate, student_roots=[root])
    return args, checkpoint


def test_export_preserves_tensors_and_selection_without_optimizer_rng_or_batch_logs(tmp_path, monkeypatch):
    import torch

    args, original = fixture(tmp_path, monkeypatch)
    package(args)
    source = torch.load(original, weights_only=True)
    checkpoint = args.output / "A1-pretrained-n4-s7/model.pt"
    state = torch.load(checkpoint, weights_only=True)
    assert tensor_fingerprint(state["model"]) == tensor_fingerprint(source["model"])
    assert set(state) == {"model", "contract", "step", "exposure", "export_schema", "source_training_checkpoint_sha256"}
    assert state["source_training_checkpoint_sha256"] == sha256(original)
    selected = json.loads((checkpoint.parent / "selected-checkpoint.json").read_text())
    assert selected["sha256"] == sha256(checkpoint) and selected["step"] == 2
    point = json.loads((checkpoint.parent / "learning-curve.jsonl").read_text())
    assert "training_batch_mse" not in point
    receipt = json.loads((args.output / "bundle.json").read_text())
    assert receipt["publication_authorized"] is False and receipt["all_selected_tensors_preserved"] is True
    assert (args.output / "_SUCCESS.json").is_file()


@pytest.mark.parametrize("bad", ["controlled", "nonfinite"])
def test_nonpublic_or_nonfinite_weights_cannot_receive_a_success_receipt(tmp_path, monkeypatch, bad):
    args, _ = fixture(tmp_path, monkeypatch, bad)
    with pytest.raises(ValueError, match="provenance|nonfinite"):
        package(args)
    assert not (args.output / "_SUCCESS.json").exists()


def test_curve_changes_after_selection_are_rejected_before_export(tmp_path, monkeypatch):
    args, checkpoint = fixture(tmp_path, monkeypatch)
    curve = checkpoint.parent / "learning-curve.jsonl"
    point = json.loads(curve.read_text())
    point["validation"]["mse"] = .1
    curve.write_text(json.dumps(point) + "\n")
    with pytest.raises(ValueError, match="original completed learning curve"):
        package(args)
    assert not args.output.exists()
