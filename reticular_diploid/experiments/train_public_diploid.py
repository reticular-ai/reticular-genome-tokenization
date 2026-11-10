"""Public-data causal NTP adapter using the existing Kimi model and optimizer."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reticular_diploid.data.diploid_windows import materialize_window
from reticular_diploid.data.frozen_diploid import FrozenDiploidTokenizer, sha256
from reticular_diploid.data.public_diploid_corpus import load_corpus
from reticular_diploid.experiments.prepare_public_1000g import clean_commit, public_output
from reticular_diploid.models.public_kimi import ModelConfig, NoPECausalTransformer
from reticular_diploid.models.haplotype_optimizers import hybrid_muon_adamw


def collate_windows(canonical, view, rows, receipt, device):
    windows = [materialize_window(canonical, view, int(row), receipt["content_size"],
                                  receipt["context"]) for row in rows]
    ids = np.stack([w["input_ids"] for w in windows])
    masks = np.stack([w["target_mask"] for w in windows])
    coordinates = np.stack([w["coordinate_ids"] for w in windows])
    return (torch.as_tensor(ids[:, :-1], device=device, dtype=torch.long),
            torch.as_tensor(ids[:, 1:], device=device, dtype=torch.long),
            torch.as_tensor(masks[:, 1:], device=device, dtype=torch.bool),
            torch.as_tensor(coordinates[:, :-1], device=device, dtype=torch.long), windows)


def precision(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def masked_ntp(logits, targets, mask):
    if not bool(mask.any()):
        raise ValueError("A training batch must contain scored content or true-end EOS")
    return F.cross_entropy(logits[mask].float(), targets[mask])


@torch.no_grad()
def evaluate(model, receipt, canonical, view, device, batch_size=8, *, tokenizer=None):
    """Use prefix-only ownership when a frozen tokenizer is explicitly supplied.

    Historical callers retain the legacy window diagnostic for exact replay.
    """
    if tokenizer is not None:
        from reticular_diploid.experiments.evaluate_public_diploid import evaluate_causal

        return evaluate_causal(model, receipt, canonical, tokenizer, device, batch_size)
    model.eval()
    rows = np.flatnonzero(view["participant_split"] == 1)
    seen = set()
    total, targets = 0.0, 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        ids, labels, mask, coords, windows = collate_windows(canonical, view, batch, receipt, device)
        with precision(device):
            logits = model(ids, coordinate_ids=coords)
        for local, (row, window) in enumerate(zip(batch, windows, strict=True)):
            person = int(view["canonical_local_person"][row])
            ownership = np.zeros(receipt["context"] - 1, dtype=bool)
            for index, (position, kind) in enumerate(zip(window["canonical_token_positions"][1:],
                                                       window["target_kinds"][1:], strict=True)):
                key = (person, int(position), int(kind))
                if kind >= 0 and key not in seen:
                    seen.add(key)
                    ownership[index] = True
            chosen = torch.as_tensor(ownership, device=device)
            if bool(chosen.any()):
                total += float(F.cross_entropy(logits[local, chosen].float(), labels[local, chosen],
                                                reduction="sum"))
                targets += int(chosen.sum())
    validation = np.flatnonzero(canonical["participant_split"] == 1)
    expected = int(sum(np.diff(canonical["target_indptr"])[validation] + 1))
    if targets != expected:
        raise ValueError("Validation ownership does not reconcile with canonical targets plus EOS")
    ce = total / targets
    return {"cross_entropy_nats": ce, "perplexity": math.exp(ce),
            "bits_per_person": total / (len(validation) * math.log(2)),
            "scored_targets": targets, "people": len(validation), "nll_nats": total}


def save_checkpoint(path, model, optimizer, rng, step, exposure, contract, *, replace=False):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "numpy_generator": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
             "step": step, "exposure": exposure, "contract": contract}
    if next(model.parameters()).device.type == "cuda":
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    elif next(model.parameters()).device.type == "mps":
        state["mps_rng"] = torch.mps.get_rng_state()
    temporary = path.with_suffix(".partial")
    if temporary.exists() or (path.exists() and not replace):
        raise ValueError("Checkpoint identity already exists")
    torch.save(state, temporary)
    if replace:
        temporary.replace(path)
    else:
        temporary.rename(path)


def restore_checkpoint(path, model, optimizer, rng, contract, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["contract"] != contract:
        raise ValueError("Resume contract differs; this is not an exact continuation")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    rng.bit_generator.state = state["numpy_generator"]
    torch.set_rng_state(state["torch_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    elif device.type == "mps":
        torch.mps.set_rng_state(state["mps_rng"])
    return int(state["step"]), int(state["exposure"])


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def runtime_metadata(device):
    versions = {}
    for name in ["numpy", "torch", "fla-core", "triton"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {"packages": versions, "device_type": device.type,
              "cuda_build": torch.version.cuda, "training_precision": "bfloat16" if device.type == "cuda" else "float32",
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(gpu_name=properties.name, gpu_total_memory_bytes=properties.total_memory,
                      compute_capability=[properties.major, properties.minor],
                      matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    return result


def train(args):
    commit = clean_commit()
    public_output(args.output)
    recipe = json.loads(args.config.read_text())
    receipt, canonical, view = load_corpus(args.corpus)
    if any(receipt[key] != recipe[key] for key in
           ["tokenizer_sha256", "canonical_sha256", "windows_sha256"]):
        raise ValueError("Training recipe binds other tokenizer or corpus bytes")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(recipe["resources"]["gpu_memory_fraction"])
        torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(recipe["resources"]["cpu_threads"])
    torch.manual_seed(recipe["seed"])
    config = ModelConfig(vocabulary_size=receipt["vocabulary_size"],
                         coordinate_vocabulary_size=receipt["coordinate_vocabulary_size"],
                         **recipe["model"])
    model = NoPECausalTransformer(config).to(device)
    learning = recipe["training"]
    codec = None
    if learning["validation_ownership"] == "prefix_causal":
        if args.tokenizer is None:
            raise ValueError("Prefix-only training evaluation requires the frozen tokenizer")
        codec = FrozenDiploidTokenizer.load(args.tokenizer, receipt["tokenizer_sha256"])
    elif learning["validation_ownership"] != "earliest containing canonical window; content and true-end EOS once":
        raise ValueError("Unknown validation ownership protocol")
    optimizer = hybrid_muon_adamw(model, nn.Module(), learning_rate=learning["learning_rate"],
                                  weight_decay=learning["weight_decay"])
    lr_factors = [group["lr"] / learning["learning_rate"] for group in optimizer.param_groups]
    rng = np.random.default_rng(recipe["seed"])
    contract = {"schema_version": "reticular-public-ntp/v1", "code_commit": commit,
                "recipe_sha256": sha256(args.config), "corpus": receipt,
                "model_config": asdict(config), "device_type": device.type,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "torch_version": str(torch.__version__), "runtime": runtime_metadata(device)}
    contract = json.loads(json.dumps(contract))
    step = exposure = 0
    if args.resume is not None:
        step, exposure = restore_checkpoint(args.resume, model, optimizer, rng, contract, device)
    args.output.mkdir(parents=True, exist_ok=True)
    contract_path = args.output / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Existing output belongs to another immutable run")
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    history_path = args.output / "learning-curve.jsonl"
    if history_path.exists() and args.resume is None:
        raise ValueError("Training attempt exists; use its exact checkpoint to resume")
    training_rows = np.flatnonzero(view["participant_split"] == 0)
    started = time.monotonic()
    history = [json.loads(line) for line in history_path.read_text().splitlines()] if history_path.exists() else []
    if history and history[-1]["step"] != step:
        raise ValueError("Resume must use the latest recorded NTP checkpoint")
    best = min([r["validation"]["cross_entropy_nats"] for r in history], default=math.inf)
    previous_seconds = history[-1]["total_worker_seconds"] if history else 0.
    endpoint = args.stop_after or learning["steps"]
    if not step < endpoint <= learning["steps"]:
        raise ValueError("Run endpoint must advance within the frozen schedule")
    for current in range(step + 1, endpoint + 1):
        if (previous_seconds + time.monotonic() - started) / 3600 > recipe["resources"]["maximum_worker_gpu_hours"]:
            raise ValueError("Frozen worker time budget exhausted; preserve the latest checkpoint")
        model.train()
        rows = rng.choice(training_rows, learning["batch_size"], replace=True)
        ids, labels, mask, coords, _ = collate_windows(canonical, view, rows, receipt, device)
        warmup = learning["warmup_steps"]
        ratio = (current / warmup if current <= warmup else
                 learning["minimum_lr_ratio"] + (1 - learning["minimum_lr_ratio"]) *
                 (1 + math.cos(math.pi * (current - warmup) / (learning["steps"] - warmup))) / 2)
        for group, factor in zip(optimizer.param_groups, lr_factors, strict=True):
            group["lr"] = learning["learning_rate"] * ratio * factor
        optimizer.zero_grad(set_to_none=True)
        with precision(device):
            loss = masked_ntp(model(ids, coordinate_ids=coords), labels, mask)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite NTP loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), learning["gradient_clip"])
        if not bool(torch.isfinite(norm)):
            raise ValueError("Nonfinite NTP gradient")
        optimizer.step()
        exposure += int(mask.sum())
        if current % learning["evaluate_every"] == 0 or current == endpoint:
            synchronize(device)
            metrics = evaluate(model, receipt, canonical, view, device, learning["batch_size"], tokenizer=codec)
            record = {"step": current, "training_batch_ce": float(loss.detach()),
                      "target_tokens_seen": exposure, "validation": metrics,
                      "attempt_elapsed_seconds": time.monotonic() - started,
                      "total_worker_seconds": previous_seconds + time.monotonic() - started,
                      "worker_gpu_hours": (previous_seconds + time.monotonic() - started) / 3600 if device.type == "cuda" else 0,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
            with history_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            checkpoint = args.output / f"step-{current:06d}.pt"
            save_checkpoint(checkpoint, model, optimizer, rng, current, exposure, contract)
            if metrics["cross_entropy_nats"] < best:
                best = metrics["cross_entropy_nats"]
                (args.output / "selected-checkpoint.json").write_text(json.dumps({
                    "path": checkpoint.name, "sha256": sha256(checkpoint),
                    "selection": "minimum full canonical validation CE", "step": current,
                    "validation": metrics}, indent=2) + "\n")
            print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["corpus", "config", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--tokenizer", type=Path, help="Frozen dictionary for prefix-only validation")
    parser.add_argument("--stop-after", type=int)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
