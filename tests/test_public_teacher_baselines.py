import numpy as np
import pytest

from reticular_diploid.experiments.public_teacher_baselines import (
    genotype_pca_features,
    regression_metrics,
    ridge_predictions,
    teacher_variant_ranks,
    training_subset,
)


def test_pca_uses_only_training_covariance_and_transforms_heldout_people():
    features = np.array([[0, 0], [1, 0], [0, 2], [1, 2], [2, 3]], dtype=float)
    rows = np.arange(4)
    projected = genotype_pca_features(features, rows, width=2)
    centered = features - features[rows].mean(axis=0)
    np.testing.assert_allclose(projected @ projected.T, centered @ centered.T, atol=1e-12)
    changed = features.copy()
    changed[-1] = [10000, -9000]
    alternate = genotype_pca_features(changed, rows, width=2)
    np.testing.assert_allclose(projected[rows], alternate[rows], atol=1e-12)
    with pytest.raises(ValueError, match="positive-variance"):
        genotype_pca_features(np.ones((4, 2)), rows)


def test_direct_dosage_fit_recovers_linear_teacher_and_training_center():
    rng = np.random.default_rng(12)
    x = rng.integers(0, 3, (30, 4))
    beta = np.array([[1.], [-2.], [.3], [0.]])
    y = x @ beta + 9
    predicted = ridge_predictions(x[:20], x[20:], y[:20], [1e-9])[0, :, 0]
    assert regression_metrics(y[20:, 0], predicted)["r_squared"] == pytest.approx(1)
    np.testing.assert_allclose(predicted, y[20:, 0], atol=1e-8)
    constant = ridge_predictions(np.zeros((3, 2)), np.ones((2, 2)), np.array([[2.], [4.], [6.]]))[0]
    np.testing.assert_equal(constant, [[4.], [4.]])


def test_nested_subsets_exclude_validation_and_test_and_metrics_reject_nan():
    split = np.array([0] * 20 + [1] * 5 + [2] * 5)
    np.testing.assert_equal(training_subset(split, 5, 7), training_subset(split, 10, 7)[:5])
    assert np.max(training_subset(split, 20, 7)) < 20
    assert regression_metrics([1, 2, 3], [1, 2, 3])["pearson_r"] == pytest.approx(1)
    with pytest.raises(ValueError):
        regression_metrics([1, 2], [1, np.nan])


def test_effect_locus_control_uses_ref_alt_identity_not_position_alone(tmp_path):
    path = tmp_path / "weights.tsv"
    path.write_text("teacher\tchromosome\tposition\tREF\tALT\n1\t22\t10\tA\tG\n")
    assert teacher_variant_ranks(path, [10, 10], ["A", "A"], ["C", "G"]) == {1: [1]}
    with pytest.raises(ValueError, match="absent"):
        teacher_variant_ranks(path, [10], ["A"], ["C"])
