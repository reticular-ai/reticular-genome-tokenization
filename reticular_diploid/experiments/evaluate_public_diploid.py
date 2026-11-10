"""Evaluate a public NTP checkpoint with canonical target ownership."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from reticular_diploid.data.causal_diploid_evaluation import evaluation_groups, materialize_group
from reticular_diploid.data.frozen_diploid import FrozenDiploidTokenizer, sha256
from reticular_diploid.data.public_diploid_corpus import load_corpus, same_corpus
from reticular_diploid.experiments.prepare_public_1000g import clean_commit, public_output


@contextlib.contextmanager
def causal_precision(device):
    """Use FP32 inference; restore flags so teacher replay keeps its precision."""
    import torch

    previous = None
    if device.type == "cuda":
        previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.autocast("cuda", enabled=False) if device.type == "cuda" else contextlib.nullcontext():
            yield
    finally:
        if previous is not None:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def verify_prefix_independence(model, receipt, device):
    """Probe the actual network for future-token and other-person leakage."""
    import torch

    length = receipt["context"] - 1
    cut = max(1, length // 2)
    ids = (torch.arange(2 * length, device=device).reshape(2, length) * 73) % receipt["content_size"]
    coords = torch.zeros_like(ids)
    future = ids.clone()
    future[:, cut:] = (future[:, cut:] + 1) % receipt["content_size"]
    future_coords = coords.clone()
    future_coords[:, cut:] = 1
    other = ids.clone()
    other[1] = (other[1] + 1) % receipt["content_size"]
    model.eval()
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Causal evaluation requires FP32 model weights")
    with torch.no_grad(), causal_precision(device):
        reference = model(ids, coordinate_ids=coords)
        changed_future = model(future, coordinate_ids=future_coords)
        changed_person = model(other, coordinate_ids=coords)
    # Expert GEMM row counts depend on suffix routing. Synthetic all-checkpoint
    # probes measured at most 3.15e-5 FP32 roundoff (vs .14 in BF16), so retain
    # an explicit 1e-4 CUDA bound and report the actual discrepancy. This is
    # numerical near-invariance, not a claim of bitwise GPU independence.
    tolerance = 1e-4 if device.type == "cuda" else 1e-6
    comparisons = [(reference[:, :cut], changed_future[:, :cut]), (reference[:1], changed_person[:1])]
    errors = []
    for original, changed in comparisons:
        if not bool(torch.isfinite(original).all() & torch.isfinite(changed).all()):
            raise ValueError("Synthetic prefix check produced nonfinite logits")
        errors.append(float((original.float() - changed.float()).abs().max()))
        if errors[-1] > tolerance:
            raise ValueError("NTP model depends on future tokens or another person's input")
    return {"synthetic_only": True, "future_suffix_max_logit_difference": errors[0],
            "other_person_max_logit_difference": errors[1], "absolute_tolerance": tolerance,
            "prefix_length": cut, "input_length": length, "evaluation_dtype": "float32", "tf32": False}


def evaluate_causal(model, receipt, canonical, tokenizer, device, batch_size=8, *, split=1):
    import torch
    from torch.nn import functional as F

    groups = evaluation_groups(canonical, tokenizer, receipt["context"], receipt["stride"], split=split)
    people = np.flatnonzero(canonical["participant_split"] == split)
    if not len(people) or batch_size <= 0:
        raise ValueError("Causal evaluation requires selected people and a positive batch size")
    if tokenizer.content_size != receipt["content_size"]:
        raise ValueError("Evaluation tokenizer namespace differs")
    independence = verify_prefix_independence(model, receipt, device)
    total, targets = 0., 0
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(groups), batch_size):
            rows = [materialize_group(canonical, row, receipt["special_tokens"], receipt["context"])
                    for row in groups[offset:offset + batch_size]]
            ids, coords, labels, mask = [torch.as_tensor(np.stack([r[key] for r in rows]), device=device,
                                                       dtype=dtype) for key, dtype in
                                        [("input_ids", torch.long), ("coordinate_ids", torch.long),
                                         ("labels", torch.long), ("target_mask", torch.bool)]]
            with causal_precision(device):
                logits = model(ids, coordinate_ids=coords)
            losses = F.cross_entropy(logits[mask].float(), labels[mask], reduction="none")
            if not bool(torch.isfinite(losses).all()):
                raise ValueError("Causal evaluation produced nonfinite likelihoods")
            total += float(losses.double().sum())
            targets += int(mask.sum())
    expected = int(np.sum(np.diff(canonical["target_indptr"])[people] + 1))
    if targets != expected:
        raise ValueError("Prefix-only ownership does not reconcile with canonical content plus EOS")
    ce = total / targets
    return {"cross_entropy_nats": ce, "perplexity": math.exp(ce),
            "bits_per_person": total / (len(people) * math.log(2)),
            "scored_targets": targets, "people": len(people), "nll_nats": total,
            "prefix_independence_canary": independence, "evaluation_dtype": "float32", "tf32": False}


def training_unigram(receipt, canonical, smoothing):
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("Unigram smoothing must be positive and finite")
    counts = np.full(receipt["vocabulary_size"], smoothing, np.float64)
    ptr, ids = canonical["target_indptr"], canonical["target_ids"]
    eos = receipt["special_tokens"]["eos"]
    for person in np.flatnonzero(canonical["participant_split"] == 0):
        counts += np.bincount(ids[ptr[person]:ptr[person + 1]], minlength=len(counts))
        counts[eos] += 1
    return counts / counts.sum()


def unigram_baseline(receipt, canonical, smoothing=.5, *, split=1):
    """Fit only on training content and true EOS; score each validation target once."""
    logp = np.log(training_unigram(receipt, canonical, smoothing))
    ptr, ids = canonical["target_indptr"], canonical["target_ids"]
    eos = receipt["special_tokens"]["eos"]
    total, targets = 0., 0
    if split not in {1, 2}:
        raise ValueError("Unigram evaluation requires a held-out partition")
    people = np.flatnonzero(canonical["participant_split"] == split)
    for person in people:
        content = ids[ptr[person]:ptr[person + 1]]
        total -= float(logp[content].sum() + logp[eos])
        targets += len(content) + 1
    if not len(people) or not targets:
        raise ValueError("Unigram evaluation requires validation people and targets")
    ce = total / targets
    return {"cross_entropy_nats": ce, "perplexity": math.exp(ce),
            "bits_per_person": total / (len(people) * math.log(2)),
            "scored_targets": targets, "people": len(people), "nll_nats": total,
            "smoothing": smoothing, "fit_partition": "training only"}


def bigram_baseline(receipt, canonical, concentration=10., smoothing=.5, *, split=1):
    """Fixed-strength bigram interpolation backs off to the training unigram.

    This CPU control captures simple token grammar without neural pretraining.
    All validation content and true EOS targets are counted exactly once.
    """
    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError("Bigram concentration must be positive and finite")
    prior = training_unigram(receipt, canonical, smoothing)
    ptr, ids = canonical["target_indptr"], canonical["target_ids"]
    eos, forward = (receipt["special_tokens"][k] for k in ["eos", "forward"])
    pairs, contexts = Counter(), Counter()
    for person in np.flatnonzero(canonical["participant_split"] == 0):
        previous = forward
        for token in [*ids[ptr[person]:ptr[person + 1]], eos]:
            token = int(token)
            pairs[previous, token] += 1
            contexts[previous] += 1
            previous = token
    total, targets = 0., 0
    if split not in {1, 2}:
        raise ValueError("Bigram evaluation requires a held-out partition")
    people = np.flatnonzero(canonical["participant_split"] == split)
    for person in people:
        previous = forward
        for token in [*ids[ptr[person]:ptr[person + 1]], eos]:
            token = int(token)
            probability = (pairs[previous, token] + concentration * prior[token]) / (contexts[previous] + concentration)
            total -= math.log(probability)
            targets += 1
            previous = token
    if not len(people) or not targets:
        raise ValueError("Bigram evaluation requires validation targets")
    ce = total / targets
    return {"cross_entropy_nats": ce, "perplexity": math.exp(ce),
            "bits_per_person": total / (len(people) * math.log(2)),
            "scored_targets": targets, "people": len(people), "nll_nats": total,
            "concentration": concentration, "unigram_smoothing": smoothing,
            "fit_partition": "training only", "validation_tuning": False}


def rescore_curve(args):
    """Rescore saved training endpoints without selecting another checkpoint."""
    import torch

    from reticular_diploid.models.public_kimi import ModelConfig, NoPECausalTransformer

    commit = clean_commit()
    public_output(args.output)
    if args.output.exists() or args.tokenizer is None or args.ownership != "prefix_causal":
        raise ValueError("Curve rescoring requires a new output, frozen tokenizer and prefix-only ownership")
    receipt, canonical, _ = load_corpus(args.corpus)
    codec = FrozenDiploidTokenizer.load(args.tokenizer, receipt["tokenizer_sha256"])
    contract = json.loads((args.checkpoint_directory / "contract.json").read_text())
    if contract["schema_version"] != "reticular-public-ntp/v1" or not same_corpus(contract["corpus"], receipt):
        raise ValueError("NTP curve and development corpus provenance differ")
    paths = sorted(args.checkpoint_directory.glob("step-*.pt"))
    if not paths:
        raise ValueError("Curve rescoring requires preserved step checkpoints")
    selection = json.loads((args.checkpoint_directory / "selected-checkpoint.json").read_text())
    selected = args.checkpoint_directory / selection["path"]
    if Path(selection["path"]).name != selection["path"] or sha256(selected) != selection["sha256"]:
        raise ValueError("Frozen NTP selection identity differs")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(.2)
    model = NoPECausalTransformer(ModelConfig(**contract["model_config"])).to(device)
    rows = []
    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["contract"] != contract or (rows and state["step"] <= rows[-1]["step"]):
            raise ValueError("Curve checkpoint contract or step order differs")
        model.load_state_dict(state["model"], strict=True)
        row = {"step": state["step"], "target_tokens_seen": state["exposure"],
               "checkpoint_sha256": sha256(path),
               "validation": evaluate_causal(model, receipt, canonical, codec, device, args.batch_size)}
        if getattr(args, "training_diagnostics", False):
            # Evaluate the same prefix-only targets in eval mode on both splits;
            # a noisy minibatch training loss cannot establish overfitting.
            row["training"] = evaluate_causal(model, receipt, canonical, codec, device, args.batch_size, split=0)
        rows.append(row)
    if selection["sha256"] not in {row["checkpoint_sha256"] for row in rows}:
        raise ValueError("Frozen selected checkpoint is absent from the rescored curve")
    result = {"schema_version": "reticular-public-ntp-curve/v1", "code_commit": commit,
              "evaluation_ownership": "prefix_causal", "training_contract": contract,
              "frozen_selection": selection, "records": rows, "locked_test_used": False,
              "training_unigram_baseline": unigram_baseline(receipt, canonical),
              "training_bigram_baseline": bigram_baseline(receipt, canonical)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"rescored_checkpoints": len(rows), "locked_test_used": False,
                      "selection_changed": False, "final": rows[-1]}))


def run(args):
    import torch

    from reticular_diploid.experiments.train_public_diploid import evaluate
    from reticular_diploid.models.public_kimi import ModelConfig, NoPECausalTransformer

    commit = clean_commit()
    public_output(args.output)
    if args.output.exists():
        raise ValueError("Evaluation output already exists")
    if sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Checkpoint checksum differs")
    receipt, canonical, view = load_corpus(args.corpus)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    contract = state["contract"]
    if contract["schema_version"] != "reticular-public-ntp/v1" or not same_corpus(contract["corpus"], receipt):
        raise ValueError("Checkpoint and development corpus provenance differ")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(.2)
    model = NoPECausalTransformer(ModelConfig(**contract["model_config"])).to(device)
    model.load_state_dict(state["model"], strict=True)
    if args.ownership == "prefix_causal":
        if args.tokenizer is None:
            raise ValueError("Prefix-only evaluation requires the frozen tokenizer")
        codec = FrozenDiploidTokenizer.load(args.tokenizer, receipt["tokenizer_sha256"])
        metrics = evaluate_causal(model, receipt, canonical, codec, device, args.batch_size)
    else:
        metrics = evaluate(model, receipt, canonical, view, device, args.batch_size)
    result = {"schema_version": "reticular-public-ntp-evaluation/v2", "code_commit": commit,
              "checkpoint_sha256": args.checkpoint_sha256, "checkpoint_step": state["step"],
              "training_contract": contract, "torch_version": str(torch.__version__),
              "evaluation_ownership": args.ownership,
              "checkpoint_selection": "retained from the frozen training protocol; no reselection",
              "validation": metrics,
              "training_unigram_baseline": unigram_baseline(receipt, canonical),
              "training_bigram_baseline": bigram_baseline(receipt, canonical),
              "locked_test_used": False}
    if getattr(args, "training_diagnostics", False):
        if args.ownership != "prefix_causal":
            raise ValueError("Training-fit diagnostics require prefix-only ownership")
        result["training"] = evaluate_causal(model, receipt, canonical, codec, device, args.batch_size, split=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["validation", "training_unigram_baseline", "training_bigram_baseline", "locked_test_used"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--checkpoint-directory", type=Path,
                        help="Rescore every preserved step checkpoint; retain the original selection")
    for name in ["corpus", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--ownership", choices=["prefix_causal", "legacy"], default="prefix_causal")
    parser.add_argument("--tokenizer", type=Path,
                        help="Frozen dictionary required for prefix-only event-aligned starts")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--training-diagnostics", action="store_true",
                        help="Also score training people with the identical deterministic eval-mode protocol")
    args = parser.parse_args()
    if args.checkpoint is not None and args.checkpoint_sha256 is None:
        parser.error("Single-checkpoint evaluation requires --checkpoint-sha256")
    (rescore_curve if args.checkpoint_directory else run)(args)


if __name__ == "__main__":
    main()
