import numpy as np
import pytest
import torch
from torch import nn

from reticular_diploid.experiments.distill_public_diploid import ContextualRegressor, full_region_batch


def fixture():
    canonical = {"target_indptr": np.array([0, 4, 6]), "target_ids": np.array([1, 2, 3, 4, 2, 3]),
                 "target_coordinate_ids": np.array([0, 0, 1, 1, 0, 1])}
    receipt = {"special_tokens": {"pad": 10, "bos": 11, "forward": 12, "eos": 13}}
    return canonical, receipt


def test_full_region_includes_final_variant_and_true_eos_without_truncation():
    canonical, receipt = fixture()
    ids, coords, pool, padding = full_region_batch(canonical, [0, 1], receipt, 8, "cpu")
    assert ids.tolist() == [[11, 12, 1, 2, 3, 4, 13, 10], [11, 12, 2, 3, 13, 10, 10, 10]]
    assert pool.sum(1).tolist() == [5, 3]
    assert coords[0, 6] == 1
    assert padding.sum(1).tolist() == [1, 3]
    with pytest.raises(ValueError, match="cropping is forbidden"):
        full_region_batch(canonical, [0], receipt, 6, "cpu")


def test_contextual_readout_uses_all_region_and_ignores_suffix_padding():
    class CausalSynthetic(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(32, 4)
            self.output = nn.Linear(4, 32)

        def forward_hidden(self, ids, padding_mask, coordinate_ids):
            return self.embedding(ids).cumsum(1)

    torch.manual_seed(7)
    canonical, receipt = fixture()
    model = ContextualRegressor(CausalSynthetic(), 4)
    batch = full_region_batch(canonical, [0, 1], receipt, 8, "cpu")
    prediction, loss = model(*batch, ntp=True)
    assert prediction.shape == (2,) and torch.isfinite(loss)
    (loss + prediction.square().mean()).backward()
    assert model.backbone.embedding.weight.grad[1:5].abs().sum() > 0
    padded = batch[0].clone()
    padded[batch[3]] = 31
    actual, _ = model(padded, *batch[1:])
    torch.testing.assert_close(actual, prediction)
    changed = batch[0].clone()
    changed[0, 5] = 5  # Last content event must affect the region readout.
    actual, _ = model(changed, *batch[1:])
    assert not torch.allclose(actual[0], prediction[0])
