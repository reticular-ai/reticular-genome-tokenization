import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from reticular_diploid.data.diploid_windows import window_arrays
from reticular_diploid.experiments.train_public_diploid import (
    collate_windows,
    evaluate,
    masked_ntp,
    restore_checkpoint,
    runtime_metadata,
    save_checkpoint,
)


def corpus():
    # Repeated synthetic IDs isolate window ownership; this is not a genotype fixture.
    ids = np.zeros(20, dtype=np.int32)
    canonical = {"target_ids": ids, "target_indptr": np.array([0, 10, 20]),
                 "participant_indices": np.array([3, 9]), "participant_split": np.array([0, 1]),
                 "target_coordinate_ids": np.zeros(20, np.uint16),
                 "target_genetic_cm": np.zeros(20), "genomic_positions": np.zeros(20)}
    codec = SimpleNamespace(selected=range(2), first=2, second=2, positions=np.arange(4))
    view = window_arrays(canonical, codec, context=7, stride=2)
    return {"context": 7, "content_size": 8}, canonical, view


def test_causal_shift_and_prefix_mask():
    receipt, canonical, view = corpus()
    ids, targets, mask, _, windows = collate_windows(canonical, view, [0], receipt, torch.device("cpu"))
    assert ids[0, :2].tolist() == [9, 12]
    assert targets[0, 0] == 12 and not mask[0, 0]
    assert targets[0, 1] == 0 and mask[0, 1]
    assert not windows[0]["target_mask"][:2].any()
    loss = masked_ntp(torch.zeros(1, 6, 14), targets, mask)
    assert float(loss) == pytest.approx(math.log(14))


def test_constant_logits_validate_canonical_target_count_not_window_exposure():
    class Uniform(nn.Module):
        def forward(self, ids, coordinate_ids=None):
            return torch.zeros(*ids.shape, 14)

    receipt, canonical, view = corpus()
    result = evaluate(Uniform(), receipt, canonical, view, torch.device("cpu"), batch_size=2)
    assert result["scored_targets"] == 11
    assert result["people"] == 1
    assert result["perplexity"] == pytest.approx(14, rel=1e-6)
    assert result["bits_per_person"] == pytest.approx(11 * math.log2(14), rel=1e-6)


def test_explicit_frozen_tokenizer_uses_prefix_only_evaluation():
    class Uniform(nn.Module):
        def forward(self, ids, coordinate_ids=None):
            return torch.zeros(*ids.shape, 14)

    receipt, canonical, view = corpus()
    receipt.update(stride=2, special_tokens={"pad": 8, "bos": 9, "forward": 12, "eos": 11})
    codec = SimpleNamespace(motif_entries=2, first=2, second=2, positions=np.arange(4), content_size=8)
    result = evaluate(Uniform(), receipt, canonical, view, torch.device("cpu"), batch_size=2, tokenizer=codec)
    assert result["scored_targets"] == 11 and result["people"] == 1
    assert result["evaluation_dtype"] == "float32" and result["tf32"] is False
    assert result["prefix_independence_canary"]["future_suffix_max_logit_difference"] == 0
    assert result["cross_entropy_nats"] == pytest.approx(math.log(14), rel=1e-6)


def test_exact_checkpoint_restores_rng_optimizer_and_next_update(tmp_path):
    torch.manual_seed(41)
    model = nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    rng = np.random.default_rng(8)

    def update():
        x = torch.randn(4, 3) + float(rng.random())
        optimizer.zero_grad()
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach().clone()

    update()
    path = tmp_path / "synthetic.pt"
    contract = {"source": "synthetic", "schedule": [1, 2, 3]}
    save_checkpoint(path, model, optimizer, rng, 1, 4, contract)
    expected_loss = update()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    assert restore_checkpoint(path, model, optimizer, rng, contract, torch.device("cpu")) == (1, 4)
    actual_loss = update()
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="contract differs"):
        restore_checkpoint(path, model, optimizer, rng, {}, torch.device("cpu"))


def test_runtime_receipt_contains_versions_and_no_machine_identity():
    receipt = runtime_metadata(torch.device("cpu"))
    assert receipt["packages"]["torch"]
    assert receipt["packages"]["numpy"]
    assert receipt["training_precision"] == "float32"
    assert "hostname" not in receipt and "gpu_uuid" not in receipt


def test_latest_checkpoint_replacement_is_explicit_and_preserves_exact_state(tmp_path):
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    rng = np.random.default_rng(7)
    path = tmp_path / "latest.pt"
    save_checkpoint(path, model, optimizer, rng, 1, 2, {"retention": "latest_and_best"})
    with pytest.raises(ValueError, match="already exists"):
        save_checkpoint(path, model, optimizer, rng, 2, 4, {})
    with torch.no_grad():
        model.weight.add_(1)
    expected = model.weight.detach().clone()
    save_checkpoint(path, model, optimizer, rng, 2, 4, {"retention": "latest_and_best"}, replace=True)
    with torch.no_grad():
        model.weight.zero_()
    assert restore_checkpoint(path, model, optimizer, rng, {"retention": "latest_and_best"}, torch.device("cpu")) == (2, 4)
    torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)
