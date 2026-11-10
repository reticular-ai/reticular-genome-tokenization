"""Synthetic CUDA canaries for the public Kimi adapter and whole-region readout."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reticular_diploid.data.frozen_diploid import sha256
from reticular_diploid.experiments.distill_public_diploid import ContextualRegressor
from reticular_diploid.experiments.prepare_public_1000g import clean_commit, public_output
from reticular_diploid.models.public_kimi import ModelConfig, NoPECausalTransformer
from reticular_diploid.experiments.train_public_diploid import (
    masked_ntp,
    precision,
    restore_checkpoint,
    save_checkpoint,
)
from reticular_diploid.models.haplotype_optimizers import hybrid_muon_adamw


def run(args):
    commit = clean_commit()
    public_output(args.output)
    if args.output.exists():
        raise ValueError("Synthetic canary attempt already exists")
    args.output.mkdir(parents=True)
    recipe = json.loads(args.config.read_text())
    torch.set_num_threads(4)
    device = torch.device("cuda")
    free, _ = torch.cuda.mem_get_info()
    if free < 10 * 1024 ** 3:
        raise ValueError("Insufficient free GPU capacity for this bounded canary")
    torch.cuda.set_per_process_memory_fraction(.2)
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(recipe["seed"])
    # These dimensions belong to the frozen 8,192-ID public dictionary.
    config = ModelConfig(vocabulary_size=8192, coordinate_vocabulary_size=287, **recipe["model"])
    backbone = NoPECausalTransformer(config).to(device)
    optimizer = hybrid_muon_adamw(backbone, nn.Module(), learning_rate=.0003, weight_decay=.1)
    rng = np.random.default_rng(recipe["seed"])
    contract = {"synthetic_only": True, "code_commit": commit, "recipe_sha256": sha256(args.config)}
    records = []
    started = time.monotonic()

    def ntp_update():
        before = time.monotonic()
        ids = torch.as_tensor(rng.integers(0, 8186, (8, 256)), device=device)
        coords = torch.zeros_like(ids)
        mask = torch.ones_like(ids[:, 1:], dtype=torch.bool)
        optimizer.zero_grad(set_to_none=True)
        with precision(device):
            loss = masked_ntp(backbone(ids[:, :-1], coordinate_ids=coords[:, :-1]), ids[:, 1:], mask)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite synthetic NTP loss")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(backbone.parameters(), 1.)
        if not bool(torch.isfinite(norm)):
            raise ValueError("Nonfinite synthetic NTP gradient")
        optimizer.step()
        torch.cuda.synchronize()
        record = {"stage": "synthetic_ntp", "loss": float(loss.detach()),
                  "seconds": time.monotonic() - before}
        print(json.dumps(record), flush=True)
        records.append(record)
        return float(loss.detach())

    backbone.train()
    ntp_update()
    path = args.output / "synthetic-state.pt"
    save_checkpoint(path, backbone, optimizer, rng, 1, 2040, contract)
    expected_loss = ntp_update()
    expected = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
    restore_checkpoint(path, backbone, optimizer, rng, contract, device)
    actual_loss = ntp_update()
    differences = [float((v.detach().cpu().double() - expected[k].double()).abs().max())
                   for k, v in backbone.state_dict().items() if v.numel()]
    maximum = max(differences)
    # Report exactness separately. Parallel CUDA reductions can differ at roundoff.
    for key, value in backbone.state_dict().items():
        torch.testing.assert_close(value.detach().cpu(), expected[key], rtol=1e-5, atol=1e-6)
    print(json.dumps({"stage": "synthetic_resume", "loss_difference": abs(actual_loss - expected_loss),
                      "state_max_absolute_difference": maximum, "bit_exact": maximum == 0}), flush=True)
    del optimizer, expected
    model = ContextualRegressor(backbone, config.d_model).to(device)
    optimizer = hybrid_muon_adamw(backbone, model.head, learning_rate=.0003, weight_decay=.01)
    # Include realistic unequal lengths, both prefixes, complete content and true EOS.
    base_ids = torch.randint(0, 8186, (2, 2048), device=device)
    # Different unpadded totals test the variable-length kernel path, not just
    # repeated updates on one convenient shape.
    for joint, sizes in [(False, [1988, 1378]), (True, [1200, 1700])]:
        ids = base_ids.clone()
        lengths = torch.tensor(sizes, device=device)
        padding = torch.arange(2048, device=device)[None, :] >= lengths[:, None]
        ids[padding] = 8186
        ids[:, 0], ids[:, 1] = 8187, 8190
        ids[torch.arange(2, device=device), lengths - 1] = 8189
        pool = ~padding
        pool[:, :2] = False
        coords = torch.zeros_like(ids)
        model.train()
        before = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with precision(device):
            prediction, ntp = model(ids, coords, pool, padding, ntp=joint)
            mse = F.mse_loss(prediction.float(), torch.tensor([-.7, 1.2], device=device))
            loss = mse if ntp is None else mse + .1 * ntp
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite synthetic whole-region loss")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not bool(torch.isfinite(norm)):
            raise ValueError("Nonfinite synthetic whole-region gradient")
        optimizer.step()
        torch.cuda.synchronize()
        record = {"stage": "synthetic_full_region_joint" if joint else "synthetic_full_region_regression",
                  "loss": float(loss.detach()), "seconds": time.monotonic() - before,
                  "complete_stream_lengths": sizes, "unpadded_total": sum(sizes)}
        print(json.dumps(record), flush=True)
        records.append(record)
    result = {"synthetic_only": True, "code_commit": commit, "config_sha256": sha256(args.config),
              "parameters": sum(p.numel() for p in backbone.parameters()), "stages": records,
              "resume_state_max_absolute_difference": maximum, "resume_bit_exact": maximum == 0,
              "elapsed_seconds": time.monotonic() - started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__)}
    (args.output / "canary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"synthetic_canary_passed": True, "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
