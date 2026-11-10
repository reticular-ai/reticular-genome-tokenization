"""Freeze selected public artifacts, then evaluate the reserved pedigree holdout."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from reticular_diploid.analysis.public_distillation_uncertainty import family_intervals
from reticular_diploid.data.frozen_diploid import FrozenDiploidTokenizer, sha256
from reticular_diploid.data.public_diploid_corpus import load_corpus, same_corpus
from reticular_diploid.experiments.benchmark_public_diploid import summarize
from reticular_diploid.experiments.build_public_diploid_corpus import causal_coordinates
from reticular_diploid.experiments.evaluate_public_diploid import (
    bigram_baseline,
    evaluate_causal,
    unigram_baseline,
)
from reticular_diploid.experiments.prepare_public_1000g import clean_commit, public_output
from reticular_diploid.experiments.public_additive_teachers import (
    load_frozen_effects,
    load_scores,
    score_frozen_effects,
)
from reticular_diploid.experiments.public_teacher_baselines import (
    genotype_pca_features,
    regression_metrics,
    ridge_predictions,
    token_counts,
    training_subset,
)


def selected_students(roots, protocol, receipt, teachers, parent_sha):
    grid = protocol["student_grid"]
    expected = set(itertools.product(grid["effect_variants"], grid["training_people"], grid["seeds"], grid["arms"]))
    if len(expected) != grid["expected_models"]:
        raise ValueError("Final protocol model count differs from its Cartesian grid")
    result = {}
    for root in roots:
        public_output(root)
        for path in sorted(root.glob("*/contract.json")):
            contract = json.loads(path.read_text())
            key = tuple(contract[k] for k in ["effect_variants", "training_people", "seed", "arm"])
            if key not in expected or key in result:
                raise ValueError("Student grid has an unexpected or duplicate model")
            if (contract["schema_version"] != "reticular-public-distillation/v1"
                    or contract["recipe_sha256"] not in grid["allowed_recipe_sha256"]
                    or contract["pretrained_sha256"] != parent_sha
                    or contract["context"] != protocol["student_context"]["training"]
                    or not same_corpus(contract["corpus"], receipt)
                    or any(contract["teachers"][k] != teachers[k] for k in ["scores_sha256", "weights_sha256"])
                    or [(t["effect_variants"], t["training_mean"], t["training_std"])
                        for t in contract["teachers"]["teachers"]] !=
                       [(t["effect_variants"], t["training_mean"], t["training_std"]) for t in teachers["teachers"]]):
                raise ValueError("Selected student lies outside the frozen public protocol")
            rows = [json.loads(line) for line in (path.parent / "learning-curve.jsonl").read_text().splitlines()]
            if not rows or rows[-1]["step"] != grid["optimizer_steps"]:
                raise ValueError("Every predeclared student must finish before test unsealing")
            chosen = min(rows, key=lambda row: row["validation"]["mse"])
            selected = json.loads((path.parent / "selected-checkpoint.json").read_text())
            if selected["step"] != chosen["step"] or selected["validation"] != chosen["validation"]:
                raise ValueError("Selected checkpoint differs from validation-only selection")
            name = selected["path"]
            checkpoint = path.parent / name
            if Path(name).name != name or checkpoint.is_symlink() or sha256(checkpoint) != selected["sha256"]:
                raise ValueError("Selected checkpoint identity differs")
            result[key] = {"id": path.parent.name, "checkpoint": str(checkpoint.absolute()),
                           "sha256": selected["sha256"], "selected_step": selected["step"],
                           "effect_variants": key[0], "training_people": key[1], "seed": key[2], "arm": key[3],
                           "contract": contract, "validation": selected["validation"], "final": rows[-1]}
    if set(result) != expected:
        raise ValueError(f"Final selection is incomplete: {len(result)} of {len(expected)} students")
    return [result[key] for key in sorted(result)]


def selected_baselines(roots, protocol, receipt, teachers):
    grid = protocol["student_grid"]
    expected = set(itertools.product(protocol["baselines"]["methods"], grid["effect_variants"],
                                     grid["training_people"], grid["seeds"]))
    result = {}
    for root in roots:
        metadata = json.loads((root / "baselines.json").read_text())
        if (metadata.get("locked_test_used") is not False or not same_corpus(metadata["corpus"], receipt)
                or metadata["alphas"] != protocol["baselines"]["alphas"]
                or any(metadata["teachers"][k] != teachers[k] for k in ["scores_sha256", "weights_sha256"])):
            raise ValueError("Baseline selection provenance differs")
        for row in metadata["records"]:
            key = row["baseline"], row["effect_variants"], row["training_people"], metadata["seed"]
            if key not in expected or key in result or row["selected_alpha"] not in protocol["baselines"]["alphas"]:
                raise ValueError("Baseline grid or selected penalty differs")
            best = int(np.argmin([m["mse"] for m in row["alpha_validation"]]))
            if (metadata["alphas"][best] != row["selected_alpha"]
                    or row["validation"] != row["alpha_validation"][best]):
                raise ValueError("Baseline penalty is not the frozen validation selection")
            result[key] = {**row, "seed": metadata["seed"], "id": f"{key[0]}-M{key[1]}-n{key[2]}-s{key[3]}"}
    if set(result) != expected:
        raise ValueError("Final baseline selection is incomplete")
    return [result[key] for key in sorted(result)]


def encode_partition(codec, dosages, rows, split):
    ids, coords, lengths, motifs, residuals = [], [], [], [], []
    for row in rows:
        g = dosages[row]
        encoded = codec.encode_genotype(g)
        if not np.array_equal(codec.decode(encoded), g):
            raise ValueError("Held-out genotype reconstruction failed")
        ids.append(encoded)
        coords.append(causal_coordinates(codec, encoded))
        lengths.append(len(encoded))
        motifs.append(int(np.sum(encoded < codec.motif_entries)))
        residuals.append(2 * np.count_nonzero(g))
    return {"target_indptr": np.r_[0, np.cumsum(lengths)].astype(np.int64),
            "target_ids": np.concatenate(ids), "target_coordinate_ids": np.concatenate(coords),
            "participant_split": np.full(len(rows), split, np.uint8)}, summarize(lengths, motifs, residuals)


def joined_canonical(development, heldout):
    return {"target_indptr": np.r_[development["target_indptr"],
                                   development["target_indptr"][-1] + heldout["target_indptr"][1:]],
            "target_ids": np.r_[development["target_ids"], heldout["target_ids"]],
            "target_coordinate_ids": np.r_[development["target_coordinate_ids"], heldout["target_coordinate_ids"]],
            "participant_split": np.r_[development["participant_split"], heldout["participant_split"]]}


def verify_validation_canary(path, protocol_sha, parent_sha, models, baselines, commit):
    """Require a successful replay of these exact selections before test scoring."""
    if path is None:
        raise ValueError("Reserved-test evaluation requires a completed validation canary")
    success = json.loads((path / "_SUCCESS.json").read_text())
    evaluation = json.loads((path / "evaluation.json").read_text())
    certificate = json.loads((path / "selection-certificate.json").read_text())
    if (success["locked_test_used"] is not False
            or success["evaluation_sha256"] != sha256(path / "evaluation.json")
            or success["selection_certificate_sha256"] != sha256(path / "selection-certificate.json")
            or evaluation["selection_certificate_sha256"] != success["selection_certificate_sha256"]
            or evaluation["partition"] != "validation_canary" or evaluation["locked_test_used"] is not False
            or evaluation["code_commit"] != commit or certificate["code_commit"] != commit
            or evaluation["protocol_sha256"] != protocol_sha or certificate["protocol_sha256"] != protocol_sha
            or certificate["validation_canary"] is not True or certificate["test_opened"] is not False
            or certificate["ntp_checkpoint_sha256"] != parent_sha
            or certificate["students"] != models or certificate["baselines"] != baselines):
        raise ValueError("Validation canary does not certify these exact frozen selections")
    return {"evaluation_sha256": success["evaluation_sha256"],
            "selection_certificate_sha256": success["selection_certificate_sha256"]}


def run(args):
    import torch

    from reticular_diploid.experiments.distill_public_diploid import ContextualRegressor, evaluate_student
    from reticular_diploid.models.public_kimi import ModelConfig, NoPECausalTransformer

    commit = clean_commit()
    public_output(args.output)
    if args.output.exists():
        raise ValueError("Final evaluation attempt exists; preserve it and its selection certificate")
    protocol = json.loads(args.protocol.read_text())
    receipt, development, _ = load_corpus(args.corpus)
    teachers, development_targets = load_scores(args.teachers, development, receipt)
    if (receipt["cohort_sha256"] != protocol["cohort_sha256"]
            or receipt["canonical_sha256"] != protocol["canonical_sha256"]
            or receipt["tokenizer_sha256"] != protocol["tokenizer_sha256"]
            or teachers["scores_sha256"] != protocol["teacher_scores_sha256"]
            or teachers["weights_sha256"] != protocol["teacher_weights_sha256"]):
        raise ValueError("Final evaluation data or teacher identity differs")
    parent_sha = sha256(args.ntp_checkpoint)
    parent = torch.load(args.ntp_checkpoint, map_location="cpu", weights_only=True)
    if (parent["contract"]["schema_version"] != "reticular-public-ntp/v1"
            or parent["step"] != protocol["ntp"]["checkpoint_step"]
            or parent["contract"]["recipe_sha256"] != protocol["ntp"]["recipe_sha256"]
            or not same_corpus(parent["contract"]["corpus"], receipt)):
        raise ValueError("Frozen NTP endpoint differs")
    models = selected_students(args.student_roots, protocol, receipt, teachers, parent_sha)
    if any(row["contract"]["ntp_parent"] != parent["contract"] for row in models):
        raise ValueError("Students do not share the exact frozen NTP parent")
    baselines = selected_baselines(args.baseline_roots, protocol, receipt, teachers)
    canary_receipt = None if args.canary_validation else verify_validation_canary(
        args.validation_canary, sha256(args.protocol), parent_sha, models, baselines, commit)
    codec = FrozenDiploidTokenizer.load(args.tokenizer, protocol["tokenizer_sha256"])
    preparation = json.loads((args.prepared / "preparation.json").read_text())
    cohort = args.prepared / "cohort.npz"
    if preparation.get("public_data_only") is not True or sha256(cohort) != protocol["cohort_sha256"]:
        raise ValueError("Final cohort provenance differs")
    config = ModelConfig(**parent["contract"]["model_config"])
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA must be available before reserved-test evaluation")
        torch.cuda.set_per_process_memory_fraction(.2)
    backbone = NoPECausalTransformer(config)
    backbone.load_state_dict(parent["model"], strict=True)
    if any(not bool(torch.isfinite(v).all()) for v in parent["model"].values() if v.is_floating_point()):
        raise ValueError("NTP checkpoint has nonfinite tensors")
    student = ContextualRegressor(backbone, config.d_model)
    # Validate every selected tensor payload before creating the unsealing certificate.
    for selected in models:
        state = torch.load(selected["checkpoint"], map_location="cpu", weights_only=True)
        if state["step"] != selected["selected_step"] or state["contract"] != selected["contract"]:
            raise ValueError("Selected student payload differs from its certificate metadata")
        student.load_state_dict(state["model"], strict=True)
        if any(not bool(torch.isfinite(v).all()) for v in state["model"].values() if v.is_floating_point()):
            raise ValueError("Selected model has nonfinite tensors")
    student.to(device)
    args.output.mkdir(parents=True)
    certificate = {"schema_version": "reticular-public-selection-certificate/v1", "code_commit": commit,
                   "protocol_sha256": sha256(args.protocol), "cohort_sha256": protocol["cohort_sha256"],
                   "ntp_checkpoint_sha256": parent_sha, "students": models, "baselines": baselines,
                   "validation_canary": args.canary_validation, "prior_validation_canary": canary_receipt,
                   "test_opened": False}
    certificate_path = args.output / "selection-certificate.json"
    certificate_path.write_text(json.dumps(certificate, indent=2) + "\n")
    certificate_sha = sha256(certificate_path)
    split = 1 if args.canary_validation else 2
    (args.output / "partition-opened.json").write_text(json.dumps({
        "selection_certificate_sha256": certificate_sha,
        "partition": "validation_canary" if args.canary_validation else "reserved_test"}, indent=2) + "\n")
    # No cohort genotype array is loaded before the immutable certificate above.
    with np.load(cohort, allow_pickle=False) as z:
        if not all(np.array_equal(z[k], getattr(codec, k)) for k in ["positions", "refs", "alts"]):
            raise ValueError("Final allele universe differs")
        if any(np.count_nonzero(z["split"] == i) != protocol["partition_counts"][name]
               for i, name in enumerate(["training", "validation", "test"])):
            raise ValueError("Final partition counts differ")
        rows = np.flatnonzero(z["split"] == split)
        dosages = z["dosages"]
        families = z["family_groups"][rows]
        effects = load_frozen_effects(args.teachers / "weights.tsv", z["positions"], z["refs"], z["alts"], teachers)
        dev_genotypes = dosages[development["participant_indices"]]
        eval_genotypes = dosages[rows]
        heldout, lengths = encode_partition(codec, dosages, rows, split)
    np.testing.assert_allclose(score_frozen_effects(dev_genotypes, effects), development_targets, rtol=0, atol=1e-12)
    context = protocol["student_context"]["training"]
    maximum = int(np.diff(heldout["target_indptr"]).max()) + 3
    if maximum > context:
        context = protocol["student_context"]["backbone_maximum"]
    if maximum > context or context > config.maximum_position_embeddings:
        raise ValueError("Complete held-out stream exceeds the frozen context policy")
    targets = score_frozen_effects(eval_genotypes, effects)
    backbone.load_state_dict(parent["model"], strict=True)
    started = time.monotonic()
    ntp = evaluate_causal(backbone, receipt, heldout, codec, device, args.ntp_batch_size, split=split)
    combined = development if args.canary_validation else joined_canonical(development, heldout)
    ntp_baselines = {"unigram": unigram_baseline(receipt, combined, split=split),
                     "bigram": bigram_baseline(receipt, combined, split=split)}
    predictions, records = [], []
    sizes = [t["effect_variants"] for t in teachers["teachers"]]
    for selected in models:
        if sha256(Path(selected["checkpoint"])) != selected["sha256"]:
            raise ValueError("Selected checkpoint changed after test unsealing")
        state = torch.load(selected["checkpoint"], map_location="cpu", weights_only=True)
        student.load_state_dict(state["model"], strict=True)
        trait = sizes.index(selected["effect_variants"])
        metric, predicted = evaluate_student(student, heldout, receipt, targets[:, trait],
                                             np.arange(len(rows)), context, device, args.student_batch_size)
        if args.canary_validation:
            if any(abs(metric[k] - selected["validation"][k]) > 1e-6
                   for k in ["r_squared", "mse", "pearson_r"] if metric[k] is not None):
                raise ValueError("Validation canary does not reproduce the selected student")
        records.append({k: selected[k] for k in ["id", "effect_variants", "training_people", "seed", "arm"]}
                       | {"method": selected["arm"], "metrics": metric, "checkpoint_sha256": selected["sha256"]})
        predictions.append(predicted)
    all_training = np.flatnonzero(development["participant_split"] == 0)
    pca = genotype_pca_features(np.r_[dev_genotypes, eval_genotypes], all_training, width=128)
    feature_pairs = {"direct_genotype_ridge": (dev_genotypes, eval_genotypes),
                     "token_count_ridge": (token_counts(development, receipt["content_size"]),
                                            token_counts(heldout, receipt["content_size"])),
                     "genotype_pca128_ridge": (pca[:len(dev_genotypes)], pca[len(dev_genotypes):])}
    for selected in baselines:
        trait = sizes.index(selected["effect_variants"])
        if selected["baseline"] == "effect_locus_genotype_ridge":
            ranks = effects[trait]["ranks"]
            train_features, eval_features = dev_genotypes[:, ranks], eval_genotypes[:, ranks]
        else:
            train_features, eval_features = feature_pairs[selected["baseline"]]
        training = training_subset(development["participant_split"], selected["training_people"], selected["seed"])
        predicted = ridge_predictions(train_features[training], eval_features,
                                      development_targets[training, trait:trait + 1], [selected["selected_alpha"]])[0, :, 0]
        metric = regression_metrics(targets[:, trait], predicted)
        if args.canary_validation and abs(metric["r_squared"] - selected["validation"]["r_squared"]) > 1e-6:
            raise ValueError("Validation canary does not reproduce the selected baseline")
        records.append({k: selected[k] for k in ["id", "effect_variants", "training_people", "seed"]}
                       | {"method": selected["baseline"], "selected_alpha": selected["selected_alpha"], "metrics": metric})
        predictions.append(predicted)
    oracle_errors = []
    for trait, effect in enumerate(effects):
        raw = np.sum(eval_genotypes[:, effect["ranks"]] * effect["beta"], axis=1)
        oracle = (raw - effect["mean"]) / effect["std"]
        error = float(np.max(np.abs(oracle - targets[:, trait])))
        if not np.isfinite(error) or error > 1e-12:
            raise ValueError("Independent supplied-beta genotype oracle failed")
        oracle_errors.append(error)
        records.append({"id": f"oracle-M{sizes[trait]}", "effect_variants": sizes[trait],
                        "method": "supplied_beta_genotype_oracle", "metrics": regression_metrics(targets[:, trait], oracle)})
        predictions.append(oracle)
    predictions = np.stack(predictions)
    uncertainty = {}
    for trait, size in enumerate(sizes):
        indices = [i for i, r in enumerate(records) if r["effect_variants"] == size]
        local = {records[i]["id"]: j for j, i in enumerate(indices)}
        pairs = []
        for count, seed, arm in itertools.product(protocol["student_grid"]["training_people"],
                                                 protocol["student_grid"]["seeds"],
                                                 [a for a in protocol["student_grid"]["arms"] if a != "scratch"]):
            first = next(r["id"] for r in records if r.get("arm") == arm and r["effect_variants"] == size
                         and r["training_people"] == count and r["seed"] == seed)
            second = next(r["id"] for r in records if r.get("arm") == "scratch" and r["effect_variants"] == size
                          and r["training_people"] == count and r["seed"] == seed)
            pairs.append((local[first], local[second]))
        uncertainty[str(size)] = {"model_ids": [records[i]["id"] for i in indices],
                                  **family_intervals(targets[:, trait], predictions[indices], families, pairs=pairs,
                                                     draws=protocol["uncertainty"]["draws"],
                                                     seed=protocol["uncertainty"]["seed"])}
    np.savez_compressed(args.output / "predictions.npz", predictions=predictions, targets=targets,
                        cohort_rows=rows, model_ids=np.asarray([r["id"] for r in records]))
    result = {"schema_version": "reticular-public-final-evaluation/v1", "code_commit": commit,
              "selection_certificate_sha256": certificate_sha, "protocol_sha256": sha256(args.protocol),
              "partition": "validation_canary" if args.canary_validation else "reserved_test",
              "locked_test_used": not args.canary_validation, "people": len(rows),
              "tokenization": lengths, "exact_reconstruction_rate": 1., "student_context": context,
              "ntp": ntp, "ntp_baselines": ntp_baselines, "records": records, "uncertainty": uncertainty,
              "oracle_max_absolute_errors": oracle_errors, "elapsed_seconds": time.monotonic() - started,
              "predictions_sha256": sha256(args.output / "predictions.npz")}
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "_SUCCESS.json").write_text(json.dumps({
        "evaluation_sha256": sha256(args.output / "evaluation.json"),
        "selection_certificate_sha256": certificate_sha, "predictions_sha256": result["predictions_sha256"],
        "locked_test_used": not args.canary_validation}, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["partition", "people", "exact_reconstruction_rate", "ntp", "oracle_max_absolute_errors"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["prepared", "corpus", "teachers", "tokenizer", "ntp-checkpoint", "protocol", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--student-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--canary-validation", action="store_true")
    parser.add_argument("--validation-canary", type=Path,
                        help="Completed canary directory required before reserved-test scoring")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--ntp-batch-size", type=int, default=8)
    parser.add_argument("--student-batch-size", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
