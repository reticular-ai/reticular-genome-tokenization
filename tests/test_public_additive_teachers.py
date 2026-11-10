import json

import numpy as np
import pytest

from reticular_diploid.data.frozen_diploid import sha256
from reticular_diploid.experiments.public_additive_teachers import (
    load_frozen_effects,
    load_scores,
    make_teachers,
    score_frozen_effects,
)


def test_frozen_effects_use_exact_alleles_and_existing_training_normalization(tmp_path):
    path = tmp_path / "weights.tsv"
    path.write_text("teacher\tchromosome\tposition\tREF\tALT\tbeta\n"
                    "2\t22\t100\tA\tG\t-2\n2\t22\t100\tA\tC\t1\n")
    metadata = {"teachers": [{"effect_variants": 2, "training_mean": 3., "training_std": 2.}]}
    effects = load_frozen_effects(path, [100, 100], ["A", "A"], ["C", "G"], metadata)
    values = np.array([[1, 0], [0, 2], [1, 1]], np.uint8)
    np.testing.assert_array_equal(score_frozen_effects(values, effects)[:, 0], [-1., -3.5, -2.])
    with pytest.raises(ValueError, match="unknown"):
        load_frozen_effects(path, [100, 100], ["A", "A"], ["C", "T"], metadata)
    with pytest.raises(ValueError, match="integer"):
        score_frozen_effects(values.astype(float), effects)


def test_public_teachers_are_nested_deterministic_and_training_normalized():
    rng = np.random.default_rng(11)
    training = rng.binomial(2, .25, size=(40, 64)).astype(np.uint8)
    development = np.r_[training, rng.binomial(2, .25, size=(8, 64))].astype(np.uint8)
    positions = np.arange(64) * 100 + 1000
    motifs = np.arange(0, 64, 2)
    teachers, scores = make_teachers(training, development, positions, motifs, sizes=(4, 8, 16))
    again, repeated = make_teachers(training, development, positions, motifs, sizes=(4, 8, 16))
    np.testing.assert_array_equal(scores, repeated)
    np.testing.assert_allclose(scores[:40].mean(0), 0, atol=1e-14)
    np.testing.assert_allclose(scores[:40].std(0), 1, atol=1e-14)
    for j, (teacher, other) in enumerate(zip(teachers, again, strict=True)):
        np.testing.assert_array_equal(teacher["ranks"], other["ranks"])
        assert teacher["oracle_max_absolute_error"] < 1e-12
        assert teacher["motif_covered_variants"] > 0
        assert teacher["residual_only_variants"] > 0
        if j:
            np.testing.assert_array_equal(teacher["ranks"][:len(teachers[j-1]["ranks"])],
                                          teachers[j-1]["ranks"])
    changed_validation = development.copy()
    changed_validation[40:] = 0
    changed, _ = make_teachers(training, changed_validation, positions, motifs, sizes=(4, 8, 16))
    for before, after in zip(teachers, changed, strict=True):
        np.testing.assert_array_equal(before["ranks"], after["ranks"])
        np.testing.assert_array_equal(before["beta"], after["beta"])
        assert before["training_mean"] == after["training_mean"]
        assert before["training_std"] == after["training_std"]


def test_nonfinite_scores_cannot_pass_the_oracle_comparison():
    rng = np.random.default_rng(18)
    g = rng.binomial(2, .25, size=(30, 64)).astype(float)
    development = g.copy()
    development[-1] = np.nan
    with pytest.raises(ValueError, match="oracle differs"):
        make_teachers(g, development, np.arange(64), np.arange(0, 64, 2), sizes=(4, 8, 16))


def test_score_loader_binds_hashes_and_exact_people_order(tmp_path):
    canonical = {"participant_indices": np.array([3, 9]), "participant_split": np.array([0, 1])}
    receipt = {"cohort_sha256": "public-fixture-cohort", "tokenizer_sha256": "public-fixture-codec"}
    scores = tmp_path / "scores.npz"
    np.savez_compressed(scores, **canonical, scores=np.array([[-1.], [1.]]))
    weights = tmp_path / "weights.tsv"
    weights.write_text("independently synthetic fixture\n")
    metadata = {**receipt, "public_data_only": True, "locked_test_scores_generated": False,
                "scores_sha256": sha256(scores), "weights_sha256": sha256(weights),
                "teachers": [{"effect_variants": 1}]}
    (tmp_path / "teachers.json").write_text(json.dumps(metadata))
    _, actual = load_scores(tmp_path, canonical, receipt)
    np.testing.assert_equal(actual, [[-1.], [1.]])
    reordered = {**canonical, "participant_indices": np.array([9, 3])}
    with pytest.raises(ValueError, match="rows or partitions differ"):
        load_scores(tmp_path, reordered, receipt)
    weights.write_text("changed\n")
    with pytest.raises(ValueError, match="provenance differs"):
        load_scores(tmp_path, canonical, receipt)
