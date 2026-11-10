import numpy as np
import pytest
from reticular_diploid.analysis.motif_frequency import count_motifs


def fixture():
    return dict(target_ids=np.array([0, 0, 4, 5, 1, 2, 4, 5]),
                target_indptr=np.array([0, 4, 8]), participant_split=np.array([0, 1]),
                participant_indices=np.array([10, 20]))


def test_counts_occurrences_without_residuals_or_windows():
    counts = count_motifs(fixture(), 4)
    np.testing.assert_array_equal(counts['training'], [2, 0, 0, 0])
    np.testing.assert_array_equal(counts['validation'], [0, 1, 1, 0])


def test_rejects_test_and_duplicate_people():
    c = fixture()
    c['participant_split'][1] = 2
    with pytest.raises(ValueError):
        count_motifs(c, 4)
    c = fixture()
    c['participant_indices'][1] = 10
    with pytest.raises(ValueError):
        count_motifs(c, 4)
