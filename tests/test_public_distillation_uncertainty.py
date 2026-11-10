import numpy as np
import pytest

from reticular_diploid.analysis.public_distillation_uncertainty import cluster_resamples, family_intervals


def test_cluster_draws_preserve_every_family_and_are_reproducible():
    groups = np.array(["a", "a", "b", "c", "c", "c"])
    first, second = cluster_resamples(groups, 30, 7), cluster_resamples(groups, 30, 7)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
        counts = np.bincount(left, minlength=len(groups))
        assert counts[0] == counts[1]
        assert counts[3] == counts[4] == counts[5]


def test_oracle_and_paired_constant_error_have_exact_intervals():
    truth = np.array([-1., 0., 1., 2.])
    groups = np.array([0, 0, 1, 1])
    predicted = np.stack([truth, truth + 1, np.zeros_like(truth)])
    result = family_intervals(truth, predicted, groups, pairs=[(0, 1)], draws=80, seed=7)
    assert result["intervals"][0]["mse"] == {"lower": 0., "upper": 0., "valid_draws": 80}
    assert result["intervals"][0]["r_squared"]["lower"] == 1
    assert result["paired"][0]["interval"]["mse"] == {"lower": -1., "upper": -1., "valid_draws": 80}
    assert result["intervals"][2]["pearson_r"]["valid_draws"] == 0
    assert result["intervals"][2]["pearson_r"]["lower"] is None
    assert result["points"][2]["pearson_r"] is None
    with pytest.raises(ValueError, match="aligned"):
        family_intervals(truth, predicted[:, :2], groups)


def test_constant_target_resamples_are_reported_without_invented_r_squared():
    truth = np.array([0., 0., 1., 1.])
    result = family_intervals(truth, truth[None], [0, 0, 1, 1], draws=100, seed=7)
    assert 0 < result["intervals"][0]["r_squared"]["valid_draws"] < 100
    assert result["intervals"][0]["mse"]["valid_draws"] == 100
