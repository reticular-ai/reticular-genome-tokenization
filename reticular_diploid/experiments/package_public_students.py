"""Prepare selected public student weights for review without optimizer states."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

from reticular_diploid.data.frozen_diploid import sha256
from reticular_diploid.experiments.prepare_public_1000g import clean_commit, public_output


def tensor_fingerprint(state):
    """Hash sorted tensor names, dtype, shape and contiguous CPU bytes."""
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        descriptor = [name, str(value.dtype), list(value.shape)]
        digest.update(json.dumps(descriptor, separators=(",", ":")).encode() + b"\n")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def package(args):
    import torch

    from reticular_diploid.experiments.distill_public_diploid import ContextualRegressor
    from reticular_diploid.models.public_kimi import ModelConfig, NoPECausalTransformer

    commit = clean_commit()
    public_output(args.output)
    if args.output.exists():
        raise ValueError("Weight-review attempt exists; preserve it")
    protocol = json.loads(args.protocol.read_text())
    certificate = json.loads(args.selection_certificate.read_text())
    if (certificate["schema_version"] != "reticular-public-selection-certificate/v1"
            or certificate["protocol_sha256"] != sha256(args.protocol)
            or certificate["cohort_sha256"] != protocol["cohort_sha256"]):
        raise ValueError("Weight export must match the frozen selection protocol")
    grid = protocol["student_grid"]
    expected = set(itertools.product(grid["effect_variants"], grid["training_people"], grid["seeds"], grid["arms"]))
    models = certificate["students"]
    keys = [tuple(row[k] for k in ["effect_variants", "training_people", "seed", "arm"]) for row in models]
    if len(keys) != len(expected) or set(keys) != expected or len({r["id"] for r in models}) != len(models):
        raise ValueError("Weight export must retain every predeclared student")
    paths = []
    for row in models:
        name = row["id"]
        if Path(name).name != name:
            raise ValueError("Student ID must be a directory basename")
        roots = [root / name for root in args.student_roots if (root / name / "contract.json").is_file()]
        if len(roots) != 1 or roots[0].is_symlink():
            raise ValueError("Selected student must have exactly one immutable source")
        root = roots[0]
        public_output(root)
        selected = json.loads((root / "selected-checkpoint.json").read_text())
        path = root / selected["path"]
        if (Path(selected["path"]).name != selected["path"] or path.is_symlink()
                or selected["sha256"] != row["sha256"] or sha256(path) != row["sha256"]
                or selected["step"] != row["selected_step"] or selected["validation"] != row["validation"]):
            raise ValueError("Selected source checkpoint identity differs")
        points = [json.loads(line) for line in (root / "learning-curve.jsonl").read_text().splitlines()]
        if (not points or points[-1] != row["final"] or points[-1]["step"] != grid["optimizer_steps"]):
            raise ValueError("Weight export requires the original completed learning curve")
        chosen = min(points, key=lambda point: point["validation"]["mse"])
        if chosen["step"] != selected["step"] or chosen["validation"] != selected["validation"]:
            raise ValueError("Weight export cannot change validation-only checkpoint selection")
        paths.append((row, root, path, selected, points))
    args.output.mkdir(parents=True)
    started = time.monotonic()
    records = []
    torch.set_num_threads(4)
    for row, root, path, selected, points in paths:
        state = torch.load(path, map_location="cpu", weights_only=True)
        contract = state["contract"]
        corpus = contract["corpus"]
        if (state["step"] != row["selected_step"] or contract != row["contract"]
                or contract["schema_version"] != "reticular-public-distillation/v1"
                or corpus["public_data_only"] is not True or corpus["locked_test_used"] is not False
                or any(corpus[key] != protocol[key] for key in ["cohort_sha256", "tokenizer_sha256", "canonical_sha256"])
                or contract["pretrained_sha256"] != certificate["ntp_checkpoint_sha256"]
                or contract["teachers"]["scores_sha256"] != protocol["teacher_scores_sha256"]
                or contract["teachers"]["weights_sha256"] != protocol["teacher_weights_sha256"]):
            raise ValueError("Student weight provenance differs from the public selection")
        config = ModelConfig(**contract["ntp_parent"]["model_config"])
        model = ContextualRegressor(NoPECausalTransformer(config), config.d_model)
        model.load_state_dict(state["model"], strict=True)
        for key, tensor in model.state_dict().items():
            if not torch.equal(tensor, state["model"][key]) or (tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())):
                raise ValueError("Weight export changed a tensor or encountered nonfinite values")
        fingerprint = tensor_fingerprint(state["model"])
        destination = args.output / row["id"]
        destination.mkdir()
        output = destination / "model.pt"
        torch.save({"model": model.state_dict(), "contract": contract, "step": state["step"],
                    "exposure": state["exposure"], "export_schema": "reticular-public-inference-checkpoint/v1",
                    "source_training_checkpoint_sha256": row["sha256"]}, output)
        restored = torch.load(output, map_location="cpu", weights_only=True)
        if tensor_fingerprint(restored["model"]) != fingerprint:
            raise ValueError("Serialized inference weights differ from the original selected tensors")
        exported_sha = sha256(output)
        (destination / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
        (destination / "selected-checkpoint.json").write_text(json.dumps({
            **selected, "path": "model.pt", "sha256": exported_sha,
            "source_training_checkpoint_sha256": row["sha256"]}, indent=2) + "\n")
        fields = ["step", "person_draws", "region_targets_seen", "total_worker_seconds", "validation", "peak_allocated_bytes"]
        (destination / "learning-curve.jsonl").write_text("".join(
            json.dumps({k: point[k] for k in fields}) + "\n" for point in points))
        records.append({"id": row["id"], "selected_step": row["selected_step"],
                        "source_training_checkpoint_sha256": row["sha256"], "inference_sha256": exported_sha,
                        "tensor_sha256": fingerprint, "bytes": output.stat().st_size,
                        "source_bytes": path.stat().st_size})
    receipt = {"schema_version": "reticular-public-student-weight-review/v1", "code_commit": commit,
               "protocol_sha256": sha256(args.protocol), "source_selection_certificate_sha256": sha256(args.selection_certificate),
               "required_original_ntp_parent_sha256": certificate["ntp_checkpoint_sha256"],
               "public_data_only": True, "publication_authorized": False, "optimizer_or_rng_included": False,
               "all_selected_tensors_preserved": True, "records": records,
               "elapsed_seconds": time.monotonic() - started}
    (args.output / "bundle.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (args.output / "_SUCCESS.json").write_text(json.dumps({"bundle_sha256": sha256(args.output / "bundle.json"),
                                                          "publication_authorized": False}, indent=2) + "\n")
    print(json.dumps({"models_prepared": len(records), "inference_bytes": sum(r["bytes"] for r in records),
                      "source_bytes": sum(r["source_bytes"] for r in records), "publication_authorized": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["selection-certificate", "protocol", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--student-roots", type=Path, nargs="+", required=True)
    package(parser.parse_args())


if __name__ == "__main__":
    main()
