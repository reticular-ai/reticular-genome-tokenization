"""Independently reload a public student and evaluate training/validation fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from reticular_diploid.data.frozen_diploid import sha256
from reticular_diploid.data.public_diploid_corpus import load_corpus, same_corpus
from reticular_diploid.experiments.distill_public_diploid import ContextualRegressor, evaluate_student
from reticular_diploid.experiments.prepare_public_1000g import clean_commit, public_output
from reticular_diploid.experiments.public_additive_teachers import load_scores
from reticular_diploid.experiments.public_teacher_baselines import training_subset
from reticular_diploid.models.public_kimi import ModelConfig, NoPECausalTransformer


def verify_contract(contract, receipt, teachers):
    if (contract["schema_version"] != "reticular-public-distillation/v1"
            or not same_corpus(contract["corpus"], receipt)
            or contract["ntp_parent"]["schema_version"] != "reticular-public-ntp/v1"
            or not same_corpus(contract["ntp_parent"]["corpus"], receipt)):
        raise ValueError("Student and public development corpus provenance differ")
    def bound(metadata):
        result = {k: v for k, v in metadata.items() if k != "code_commit"}
        # Descriptive floating-point QC is not part of target semantics. Keep
        # exact score/weight hashes, selection rules and training normalization.
        diagnostics = {"oracle_max_absolute_error", "ld_pairs_measured", "ld_mean_r_squared", "ld_max_absolute_r"}
        result["teachers"] = [{k: v for k, v in row.items() if k not in diagnostics}
                              for row in metadata.get("teachers", [])]
        return result

    expected, actual = bound(contract["teachers"]), bound(teachers)
    if expected != actual:
        raise ValueError("Student and frozen teacher semantics or hashes differ")


def run(args):
    commit = clean_commit()
    public_output(args.output)
    if args.output.exists():
        raise ValueError("Student evaluation attempt exists; preserve immutable results")
    if sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Student checkpoint checksum differs")
    receipt, canonical, _ = load_corpus(args.corpus)
    teachers, scores = load_scores(args.teachers, canonical, receipt)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    contract = state["contract"]
    verify_contract(contract, receipt, teachers)
    trait = [t["effect_variants"] for t in teachers["teachers"]].index(contract["effect_variants"])
    targets = scores[:, trait]
    config = ModelConfig(**contract["ntp_parent"]["model_config"])
    device = torch.device(args.device)
    torch.set_num_threads(4)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(.2)
    model = ContextualRegressor(NoPECausalTransformer(config), config.d_model).to(device)
    model.load_state_dict(state["model"], strict=True)
    training = training_subset(canonical["participant_split"], contract["training_people"], contract["seed"])
    validation = np.flatnonzero(canonical["participant_split"] == 1)
    metrics, predictions = {}, {}
    for name, rows in [("supervised_training", training), ("validation", validation)]:
        metrics[name], predictions[name] = evaluate_student(
            model, canonical, receipt, targets, rows, contract["context"], device, args.batch_size)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "predictions.npz", **predictions,
                        training_canonical_rows=training, validation_canonical_rows=validation)
    result = {"schema_version": "reticular-public-student-evaluation/v1", "code_commit": commit,
              "checkpoint_sha256": args.checkpoint_sha256, "checkpoint_step": state["step"],
              "training_contract": contract, "metrics": metrics, "locked_test_used": False,
              "predictions_sha256": sha256(args.output / "predictions.npz")}
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"checkpoint_step": state["step"], "metrics": metrics, "locked_test_used": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["checkpoint", "corpus", "teachers", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
