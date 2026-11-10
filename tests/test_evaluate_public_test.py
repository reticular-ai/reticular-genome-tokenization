import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from reticular_diploid.data.frozen_diploid import FrozenDiploidTokenizer, sha256
from reticular_diploid.experiments.evaluate_public_test import (
    encode_partition,
    joined_canonical,
    selected_baselines,
    selected_students,
    verify_validation_canary,
)


def metadata():
    receipt = {"public_data_only": True, "canonical_sha256": "canonical"}
    teachers = {"scores_sha256": "scores", "weights_sha256": "weights",
                "teachers": [{"effect_variants": 2, "training_mean": .1, "training_std": .2}]}
    protocol = {"student_grid": {"effect_variants": [2], "training_people": [4], "seeds": [7],
                                  "arms": ["scratch", "pretrained"], "expected_models": 2,
                                  "optimizer_steps": 2, "allowed_recipe_sha256": ["recipe"]},
                "student_context": {"training": 16},
                "baselines": {"methods": ["genotype", "pca"], "alphas": [.1, 1.]}}
    return receipt, teachers, protocol


def students(root, receipt, teachers):
    for arm in ["scratch", "pretrained"]:
        out = root / arm
        out.mkdir(parents=True)
        contract = {"schema_version": "reticular-public-distillation/v1", "effect_variants": 2,
                    "training_people": 4, "seed": 7, "arm": arm, "recipe_sha256": "recipe",
                    "pretrained_sha256": "parent", "context": 16, "corpus": receipt, "teachers": teachers}
        (out / "contract.json").write_text(json.dumps(contract))
        rows = [{"step": 1, "validation": {"mse": 1.}}, {"step": 2, "validation": {"mse": .5}}]
        (out / "learning-curve.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        (out / "best.pt").write_bytes(b"synthetic checkpoint fixture")
        (out / "selected-checkpoint.json").write_text(json.dumps({"path": "best.pt", "step": 2,
            "sha256": sha256(out / "best.pt"), "validation": rows[-1]["validation"]}))


def test_unsealing_requires_every_model_and_exact_selected_identity(tmp_path, monkeypatch):
    receipt, teachers, protocol = metadata()
    root = tmp_path / "students"
    students(root, receipt, teachers)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Preflight must not load cohort arrays"))
    records = selected_students([root], protocol, receipt, teachers, "parent")
    assert len(records) == 2
    with pytest.raises(ValueError, match="incomplete"):
        selected_students([tmp_path / "empty"], protocol, receipt, teachers, "parent")
    with pytest.raises(ValueError, match="duplicate"):
        selected_students([root, root], protocol, receipt, teachers, "parent")
    (root / "scratch/best.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="identity"):
        selected_students([root], protocol, receipt, teachers, "parent")


def test_incomplete_schedule_and_changed_normalization_cannot_unseal(tmp_path):
    receipt, teachers, protocol = metadata()
    root = tmp_path / "students"
    students(root, receipt, teachers)
    p = root / "scratch/contract.json"
    contract = json.loads(p.read_text())
    contract["teachers"]["teachers"][0]["training_std"] = .3
    p.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="frozen"):
        selected_students([root], protocol, receipt, teachers, "parent")
    contract["teachers"]["teachers"][0]["training_std"] = .2
    p.write_text(json.dumps(contract))
    (root / "scratch/learning-curve.jsonl").write_text(json.dumps({"step": 1, "validation": {"mse": 1.}}))
    with pytest.raises(ValueError, match="finish"):
        selected_students([root], protocol, receipt, teachers, "parent")


def test_baseline_penalty_is_frozen_before_evaluation(tmp_path):
    receipt, teachers, protocol = metadata()
    root = tmp_path / "baseline"
    root.mkdir()
    records = [{"baseline": name, "effect_variants": 2, "training_people": 4, "selected_alpha": 1.,
                "validation": {"mse": 1.}, "alpha_validation": [{"mse": 2.}, {"mse": 1.}]}
               for name in ["genotype", "pca"]]
    data = {"corpus": receipt, "teachers": teachers, "locked_test_used": False,
            "seed": 7, "alphas": [.1, 1.], "records": records}
    path = root / "baselines.json"
    path.write_text(json.dumps(data))
    assert len(selected_baselines([root], protocol, receipt, teachers)) == 2
    records[0]["selected_alpha"] = .1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="selection"):
        selected_baselines([root], protocol, receipt, teachers)


def test_heldout_encoding_and_join_keep_original_partition_and_all_alleles():
    codec = FrozenDiploidTokenizer(np.arange(5) + 100, np.array(["A"] * 5), np.array(["C"] * 5),
                                   np.array([0]), np.array([], np.int64), np.array([], np.int64),
                                   np.array([], np.int64))
    g = np.array([[0, 1, 2, 0, 1], [1, 0, 0, 2, 0]], np.uint8)
    heldout, summary = encode_partition(codec, g, [0, 1], 2)
    dev, _ = encode_partition(codec, g, [0], 0)
    joined = joined_canonical(dev, heldout)
    assert joined["participant_split"].tolist() == [0, 2, 2]
    assert len(joined["target_indptr"]) == 4
    assert summary["people"] == 2
    for row in range(2):
        a, b = heldout["target_indptr"][row:row + 2]
        np.testing.assert_array_equal(codec.decode(heldout["target_ids"][a:b]), g[row])


def test_reserved_test_requires_a_completed_validation_replay():
    with pytest.raises(ValueError, match="completed validation"):
        verify_validation_canary(None, "protocol", "parent", [], [], "commit")


def test_complete_synthetic_evaluation_certifies_before_opening_genotypes(tmp_path, monkeypatch):
    import torch
    from torch import nn

    from reticular_diploid.experiments import evaluate_public_test as final
    from reticular_diploid.experiments.distill_public_diploid import ContextualRegressor, evaluate_student
    from reticular_diploid.experiments.public_teacher_baselines import regression_metrics, ridge_predictions
    from reticular_diploid.models import public_kimi

    class FakeCausal(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.embedding = nn.Embedding(config.vocabulary_size, config.d_model)
            self.output = nn.Linear(config.d_model, config.vocabulary_size, bias=False)

        def forward_hidden(self, ids, padding_mask=None, coordinate_ids=None):
            return self.embedding(ids).cumsum(1)

        def forward(self, ids, coordinate_ids=None):
            return self.output(self.forward_hidden(ids, coordinate_ids=coordinate_ids))

    monkeypatch.setattr(public_kimi, "NoPECausalTransformer", FakeCausal)
    monkeypatch.setattr(final, "clean_commit", lambda: "independently-synthetic")
    prepared, corpus, teacher_root = [tmp_path / name for name in ["prepared", "corpus", "teachers"]]
    for p in [prepared, corpus, teacher_root]:
        p.mkdir()
    codec = FrozenDiploidTokenizer(np.arange(5) + 100, np.array(["A"] * 5), np.array(["C"] * 5),
                                   np.array([0]), np.array([], np.int64), np.array([], np.int64),
                                   np.array([], np.int64))
    vocab = tmp_path / "vocab.npz"
    codec.save(vocab, {"public_data_only": True})
    g = np.array([[0, 0, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [2, 1, 0, 0, 1],
                  [1, 1, 0, 0, 1], [0, 2, 1, 0, 0], [1, 2, 0, 0, 0], [0, 0, 1, 1, 0]], np.uint8)
    split = np.array([0, 0, 0, 0, 1, 1, 2, 2], np.uint8)
    cohort = prepared / "cohort.npz"
    np.savez_compressed(cohort, dosages=g, split=split, family_groups=np.array(list("abcdefgh")),
                        positions=codec.positions, refs=codec.refs, alts=codec.alts)
    (prepared / "preparation.json").write_text(json.dumps({"public_data_only": True, "cohort_sha256": sha256(cohort)}))
    dev, _ = encode_partition(codec, g, np.arange(6), 0)
    dev.update(participant_split=split[:6], participant_indices=np.arange(6))
    np.savez_compressed(corpus / "canonical.npz", **dev)
    np.savez_compressed(corpus / "windows.npz", independently_synthetic=np.zeros(1))
    receipt = {"public_data_only": True, "locked_test_used": False, "cohort_sha256": sha256(cohort),
               "tokenizer_sha256": sha256(vocab), "canonical_sha256": sha256(corpus / "canonical.npz"),
               "windows_sha256": sha256(corpus / "windows.npz"), "context": 8, "stride": 3,
               "content_size": codec.content_size, "vocabulary_size": codec.content_size + 6,
               "special_tokens": codec.special_tokens}
    (corpus / "corpus.json").write_text(json.dumps(receipt))
    weights = teacher_root / "weights.tsv"
    weights.write_text("teacher\tchromosome\tposition\tREF\tALT\tbeta\n"
                       "2\t22\t100\tA\tC\t1\n2\t22\t101\tA\tC\t2\n")
    raw = g[:, 0].astype(float) + 2 * g[:, 1]
    mean, std = float(raw[:4].mean()), float(raw[:4].std())
    scores = ((raw[:6] - mean) / std)[:, None]
    np.savez_compressed(teacher_root / "scores.npz", participant_indices=np.arange(6),
                        participant_split=split[:6], scores=scores)
    teachers = {"public_data_only": True, "locked_test_scores_generated": False,
                "cohort_sha256": sha256(cohort), "tokenizer_sha256": sha256(vocab),
                "scores_sha256": sha256(teacher_root / "scores.npz"), "weights_sha256": sha256(weights),
                "teachers": [{"effect_variants": 2, "training_mean": mean, "training_std": std}]}
    (teacher_root / "teachers.json").write_text(json.dumps(teachers))
    config = public_kimi.ModelConfig(vocabulary_size=codec.content_size + 6, d_model=4,
                                     maximum_position_embeddings=32)
    parent_contract = json.loads(json.dumps({"schema_version": "reticular-public-ntp/v1",
        "recipe_sha256": "ntp-recipe", "corpus": receipt, "model_config": asdict(config)}))
    backbone = FakeCausal(config)
    parent = tmp_path / "ntp.pt"
    torch.save({"step": 2, "contract": parent_contract, "model": backbone.state_dict()}, parent)
    root = tmp_path / "students"
    students(root, receipt, teachers)
    student = ContextualRegressor(backbone, 4)
    validation, _ = evaluate_student(student, dev, receipt, scores[:, 0], np.array([4, 5]),
                                     16, torch.device("cpu"), 2)
    for path in root.glob("*/contract.json"):
        contract = json.loads(path.read_text())
        contract.update(pretrained_sha256=sha256(parent), ntp_parent=parent_contract)
        path.write_text(json.dumps(contract))
        checkpoint = path.parent / "best.pt"
        torch.save({"step": 2, "contract": contract, "model": student.state_dict()}, checkpoint)
        selection = json.loads((path.parent / "selected-checkpoint.json").read_text())
        selection["sha256"] = sha256(checkpoint)
        selection["validation"] = validation
        (path.parent / "selected-checkpoint.json").write_text(json.dumps(selection))
        (path.parent / "learning-curve.jsonl").write_text(json.dumps({"step": 2, "validation": validation}) + "\n")
    _, _, protocol = metadata()
    protocol.update(cohort_sha256=sha256(cohort), tokenizer_sha256=sha256(vocab),
                    canonical_sha256=receipt["canonical_sha256"], teacher_scores_sha256=teachers["scores_sha256"],
                    teacher_weights_sha256=teachers["weights_sha256"],
                    partition_counts={"training": 4, "validation": 2, "test": 2},
                    ntp={"checkpoint_step": 2, "recipe_sha256": "ntp-recipe"},
                    uncertainty={"draws": 20, "seed": 7},
                    student_context={"training": 16, "backbone_maximum": 32},
                    baselines={"methods": ["direct_genotype_ridge"], "alphas": [1.]})
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    baseline_prediction = ridge_predictions(g[:4], g[4:6], scores[:4], [1.])[0, :, 0]
    baseline_validation = regression_metrics(scores[4:6, 0], baseline_prediction)
    (baseline / "baselines.json").write_text(json.dumps({"locked_test_used": False, "corpus": receipt,
        "teachers": teachers, "seed": 7, "alphas": [1.], "records": [{"baseline": "direct_genotype_ridge",
        "effect_variants": 2, "training_people": 4, "selected_alpha": 1., "validation": baseline_validation,
        "alpha_validation": [baseline_validation]}]}))
    output = tmp_path / "validation"
    original_load = np.load

    def guarded_load(path, *args, **kwargs):
        if str(path) == str(cohort):
            assert (output / "selection-certificate.json").is_file()
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded_load)
    args = SimpleNamespace(prepared=prepared, corpus=corpus, teachers=teacher_root, tokenizer=vocab,
                           ntp_checkpoint=parent, protocol=protocol_path, output=output,
                           student_roots=[root], baseline_roots=[baseline], canary_validation=True,
                           validation_canary=None, device="cpu", ntp_batch_size=2, student_batch_size=2)
    final.run(args)
    args.validation_canary = output
    output = tmp_path / "evaluation"
    args.output, args.canary_validation = output, False
    final.run(args)
    result = json.loads((output / "evaluation.json").read_text())
    assert result["partition"] == "reserved_test" and result["people"] == 2
    assert result["exact_reconstruction_rate"] == 1 and result["ntp"]["scored_targets"] == 10
    assert len(result["records"]) == 4 and max(result["oracle_max_absolute_errors"]) < 1e-12
    assert (output / "_SUCCESS.json").is_file()
    certificate = json.loads((args.validation_canary / "selection-certificate.json").read_text())
    with pytest.raises(ValueError, match="exact frozen"):
        verify_validation_canary(args.validation_canary, "changed-protocol", sha256(parent),
                                 certificate["students"], certificate["baselines"], "independently-synthetic")
