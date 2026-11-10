"""Full-region contextual readout for frozen public additive teachers."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reticular_diploid.data.frozen_diploid import sha256
from reticular_diploid.data.public_diploid_corpus import load_corpus, same_corpus
from reticular_diploid.experiments.prepare_public_1000g import clean_commit, public_output
from reticular_diploid.experiments.public_additive_teachers import load_scores
from reticular_diploid.experiments.public_teacher_baselines import regression_metrics, training_subset
from reticular_diploid.models.public_kimi import ModelConfig, NoPECausalTransformer
from reticular_diploid.experiments.train_public_diploid import (
    masked_ntp,
    precision,
    restore_checkpoint,
    runtime_metadata,
    save_checkpoint,
    synchronize,
)
from reticular_diploid.models.haplotype_optimizers import hybrid_muon_adamw


def full_region_batch(canonical, rows, receipt, maximum_length, device):
    """BOS, direction, all content, true EOS, then suffix padding; never crop."""
    special = receipt["special_tokens"]
    ptr = canonical["target_indptr"]
    lengths = np.asarray([int(ptr[r + 1] - ptr[r]) + 3 for r in rows])
    if np.any(lengths > maximum_length):
        raise ValueError("Whole-region teacher input exceeds the frozen context; cropping is forbidden")
    ids = np.full((len(rows), maximum_length), special["pad"], np.int64)
    coords = np.zeros_like(ids)
    pool = np.zeros_like(ids, bool)
    for local, row in enumerate(rows):
        start, end = ptr[row:row + 2]
        length = lengths[local]
        ids[local, :2] = [special["bos"], special["forward"]]
        ids[local, 2:length - 1] = canonical["target_ids"][start:end]
        ids[local, length - 1] = special["eos"]
        coords[local, 2:length - 1] = canonical["target_coordinate_ids"][start:end]
        coords[local, length - 1] = coords[local, length - 2] if length > 3 else 0
        pool[local, 2:length] = True
    tensors = [torch.as_tensor(a, device=device) for a in [ids, coords, pool]]
    padding = torch.arange(maximum_length, device=device)[None, :] >= torch.as_tensor(lengths, device=device)[:, None]
    return *tensors, padding


class ContextualRegressor(nn.Module):
    def __init__(self, backbone, width):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, ids, coords, pool, padding, *, ntp=False):
        hidden = self.backbone.forward_hidden(ids, padding_mask=padding, coordinate_ids=coords)
        # Every content position and true EOS contributes after causal contextualization.
        pooled = (hidden.float() * pool[..., None]).sum(1) / pool.sum(1, keepdim=True)
        prediction = self.head(pooled).squeeze(-1)
        loss = None
        if ntp:
            # At input index1 the model predicts the first content ID. EOS is scored once.
            logits = self.backbone.output(hidden[:, :-1])
            loss = masked_ntp(logits, ids[:, 1:], pool[:, 1:])
        return prediction, loss


@torch.no_grad()
def evaluate_student(model, canonical, receipt, targets, rows, maximum_length, device, batch_size):
    model.eval()
    predictions = []
    for start in range(0, len(rows), batch_size):
        batch = full_region_batch(canonical, rows[start:start + batch_size], receipt, maximum_length, device)
        with precision(device):
            predicted, _ = model(*batch)
        predictions.extend(predicted.float().cpu().numpy())
    predictions = np.asarray(predictions, np.float64)
    return regression_metrics(targets[rows], predictions), predictions


def run(args):
    commit = clean_commit()
    public_output(args.output)
    recipe = json.loads(args.config.read_text())
    if args.arm not in recipe["arms"] or args.training_people not in recipe["training_people"] or args.seed not in recipe["seeds"]:
        raise ValueError("Student arm, sample size or seed is outside the frozen comparison")
    receipt, canonical, _ = load_corpus(args.corpus)
    teachers, targets = load_scores(args.teachers, canonical, receipt)
    if any(teachers[k] != recipe[k] for k in ["scores_sha256", "weights_sha256"]):
        raise ValueError("Frozen teacher artifact differs")
    if sha256(args.pretrained) != args.pretrained_sha256:
        raise ValueError("NTP checkpoint checksum differs")
    state = torch.load(args.pretrained, map_location="cpu", weights_only=True)
    parent = state["contract"]
    if (parent["schema_version"] != "reticular-public-ntp/v1" or not same_corpus(parent["corpus"], receipt)
            or parent["recipe_sha256"] != recipe["ntp_recipe_sha256"]):
        raise ValueError("NTP initialization is outside the frozen public-data protocol")
    trait = [t["effect_variants"] for t in teachers["teachers"]].index(args.effect_variants)
    targets = targets[:, trait]
    training = training_subset(canonical["participant_split"], args.training_people, args.seed)
    validation = np.flatnonzero(canonical["participant_split"] == 1)
    learning, resources = recipe["training"], recipe["resources"]
    retention = recipe.get("checkpoint_retention", "all_step_snapshots")
    if retention not in {"all_step_snapshots", "latest_and_best"}:
        raise ValueError("Unknown checkpoint retention policy")
    device = torch.device(args.device)
    torch.set_num_threads(resources["cpu_threads"])
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(resources["gpu_memory_fraction"])
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(args.seed)
    config = ModelConfig(**parent["model_config"])
    maximum_length = recipe["context"]
    if maximum_length > config.maximum_position_embeddings:
        raise ValueError("Student context exceeds backbone capacity")
    backbone = NoPECausalTransformer(config)
    model = ContextualRegressor(backbone, config.d_model).to(device)
    if args.arm != "scratch":
        backbone.load_state_dict(state["model"], strict=True)
    del state
    optimizer = hybrid_muon_adamw(backbone, model.head, learning_rate=learning["learning_rate"],
                                  weight_decay=learning["weight_decay"])
    factors = [g["lr"] / learning["learning_rate"] for g in optimizer.param_groups]
    rng = np.random.default_rng(args.seed)
    contract = {"schema_version": "reticular-public-distillation/v1", "code_commit": commit,
                "recipe_sha256": sha256(args.config), "corpus": receipt, "teachers": teachers,
                "arm": args.arm, "effect_variants": args.effect_variants, "seed": args.seed,
                "training_people": args.training_people, "pretrained_sha256": args.pretrained_sha256,
                "ntp_parent": parent, "readout": "masked mean contextual content plus true EOS; layernorm and linear",
                "context": maximum_length, "device_type": device.type, "torch_version": str(torch.__version__),
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "runtime": runtime_metadata(device)}
    contract = json.loads(json.dumps(contract))
    step = exposure = 0
    if args.resume:
        step, exposure = restore_checkpoint(args.resume, model, optimizer, rng, contract, device)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("Existing student output belongs to another immutable run")
    path.write_text(json.dumps(contract, indent=2) + "\n")
    history_path = args.output / "learning-curve.jsonl"
    if history_path.exists() and not args.resume:
        raise ValueError("Student attempt exists; resume its exact checkpoint")
    history = [json.loads(line) for line in history_path.read_text().splitlines()] if history_path.exists() else []
    if history and history[-1]["step"] != step:
        raise ValueError("Resume must use the latest recorded student checkpoint")
    best = min([r["validation"]["mse"] for r in history], default=math.inf)
    previous_seconds = history[-1]["total_worker_seconds"] if history else 0.
    endpoint = args.stop_after or learning["steps"]
    if not step < endpoint <= learning["steps"]:
        raise ValueError("Student endpoint must advance inside the frozen schedule")
    started = time.monotonic()
    for current in range(step + 1, endpoint + 1):
        elapsed = previous_seconds + time.monotonic() - started
        if elapsed / 3600 >= resources["maximum_worker_hours"]:
            raise ValueError("Student resource cap reached; preserve the last checkpoint")
        model.train()
        rows = rng.choice(training, learning["batch_size"], replace=True)
        batch = full_region_batch(canonical, rows, receipt, maximum_length, device)
        labels = torch.as_tensor(targets[rows], device=device, dtype=torch.float32)
        ratio = .1 + .9 * (1 + math.cos(math.pi * current / learning["steps"])) / 2
        for group, factor in zip(optimizer.param_groups, factors, strict=True):
            group["lr"] = learning["learning_rate"] * ratio * factor
        optimizer.zero_grad(set_to_none=True)
        with precision(device):
            prediction, ntp = model(*batch, ntp=args.arm == "joint")
            mse = F.mse_loss(prediction.float(), labels)
            loss = mse if ntp is None else mse + learning["joint_ntp_weight"] * ntp
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite teacher-recovery loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), learning["gradient_clip"])
        if not bool(torch.isfinite(norm)):
            raise ValueError("Nonfinite teacher-recovery gradient")
        optimizer.step()
        exposure += int(batch[2].sum())
        if current == step + 1 or current % 50 == 0:
            print(json.dumps({"stage": "training_progress", "step": current,
                              "training_batch_mse": float(mse.detach()),
                              "training_batch_ntp_ce": float(ntp.detach()) if ntp is not None else None,
                              "total_worker_seconds": previous_seconds + time.monotonic() - started}), flush=True)
        if current % learning["evaluate_every"] == 0 or current == endpoint:
            synchronize(device)
            metrics, predicted = evaluate_student(model, canonical, receipt, targets, validation,
                                                  maximum_length, device, learning["batch_size"])
            record = {"step": current, "training_batch_mse": float(mse.detach()),
                      "training_batch_ntp_ce": float(ntp.detach()) if ntp is not None else None,
                      "validation": metrics, "region_targets_seen": exposure,
                      "person_draws": current * learning["batch_size"],
                      "total_worker_seconds": previous_seconds + time.monotonic() - started,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
            with history_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            checkpoint = args.output / (f"step-{current:06d}.pt" if retention == "all_step_snapshots" else "latest.pt")
            save_checkpoint(checkpoint, model, optimizer, rng, current, exposure, contract,
                            replace=retention == "latest_and_best")
            if metrics["mse"] < best:
                best = metrics["mse"]
                if retention == "latest_and_best":
                    selected = args.output / "best.pt"
                    partial = selected.with_suffix(".partial")
                    if partial.exists():
                        raise ValueError("Partial best checkpoint exists; preserve it for recovery")
                    shutil.copyfile(checkpoint, partial)
                    partial.replace(selected)
                    checkpoint = selected
                np.savez_compressed(args.output / "selected-validation-predictions.npz", predictions=predicted,
                                    canonical_rows=validation)
                selection = {
                    "path": checkpoint.name, "sha256": sha256(checkpoint), "step": current,
                    "selection": "minimum validation teacher MSE", "validation": metrics}
                (args.output / "selected-checkpoint.json").write_text(json.dumps(selection, indent=2) + "\n")
                if retention == "latest_and_best":
                    with (args.output / "selection-history.jsonl").open("a") as handle:
                        handle.write(json.dumps(selection) + "\n")
            print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["corpus", "teachers", "config", "pretrained", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pretrained-sha256", required=True)
    parser.add_argument("--arm", choices=["scratch", "pretrained", "joint"], required=True)
    parser.add_argument("--effect-variants", type=int, choices=[16, 128, 1024], required=True)
    parser.add_argument("--training-people", type=int, choices=[32, 128, 400], default=400)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
