import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from reticular_diploid.data.causal_diploid_evaluation import evaluation_groups, materialize_group

CODEC = SimpleNamespace(motif_entries=2, first=4, second=4, positions=np.arange(16))
SPECIAL = {"pad": 14, "bos": 15, "forward": 18, "eos": 17}


def canonical(ids, split=1):
    return {"target_ids": np.asarray(ids, np.int32), "target_indptr": np.array([0, len(ids)]),
            "target_coordinate_ids": np.arange(len(ids)), "participant_split": np.array([split])}


def owner(c, position):
    return next(g for g in evaluation_groups(c, CODEC, 8, 3) if g[2] <= position < g[3])


def test_identical_prefix_has_identical_context_for_motif_residual_and_eos():
    alternatives = [canonical([0] * 12), canonical([0] * 4 + [2, 10] + [0] * 6), canonical([0] * 4)]
    prefixes = []
    for c in alternatives:
        group = owner(c, 4)
        assert group[1] == 0
        row = materialize_group(c, group, SPECIAL, 8)
        query = int(row["query_indices"][np.flatnonzero(row["canonical_positions"] == 4)[0]])
        prefixes.append((row["input_ids"][:query + 1], row["coordinate_ids"][:query + 1]))
    for ids, coords in prefixes[1:]:
        np.testing.assert_array_equal(ids, prefixes[0][0])
        np.testing.assert_array_equal(coords, prefixes[0][1])


@pytest.mark.parametrize("ids", [[], [0] * 5, [0] * 4 + [2, 10] + [0] * 13])
@pytest.mark.parametrize("split", [0, 1, 2])
def test_every_target_and_eos_once_with_only_consumed_prefix_inputs(ids, split):
    c = canonical(ids, split)
    seen, labels = [], []
    for group in evaluation_groups(c, CODEC, 8, 3, split=split):
        row = materialize_group(c, group, SPECIAL, 8)
        assert row["input_length"] <= 7
        for target, query in zip(row["canonical_positions"], row["query_indices"], strict=True):
            assert row["input_ids"][query] == (ids[target - 1] if target else SPECIAL["forward"])
            if target:
                assert row["coordinate_ids"][query] == target - 1
        seen.extend(row["canonical_positions"])
        labels.extend(row["labels"][row["target_mask"]])
    assert seen == list(range(len(ids) + 1))
    assert labels == [*ids, SPECIAL["eos"]]


def test_future_suffix_does_not_change_any_earlier_owned_context():
    prefix = [0] * 4 + [2, 10] + [0] * 10
    short, long = canonical(prefix), canonical(prefix + [2, 11] + [0] * 8)
    for position in range(len(prefix) + 1):
        assert owner(short, position)[1] == owner(long, position)[1]
    with pytest.raises(ValueError):
        evaluation_groups(short, CODEC, 8, 5)
    with pytest.raises(ValueError):
        evaluation_groups(short, CODEC, 8, 3, split=3)


def test_training_diagnostics_do_not_change_default_validation_partition():
    c = canonical([0, 0, 0])
    c["target_indptr"] = np.arange(4)
    c["participant_split"] = np.array([0, 1, 2])
    assert set(evaluation_groups(c, CODEC, 8, 3)[:, 0]) == {1}
    assert set(evaluation_groups(c, CODEC, 8, 3, split=0)[:, 0]) == {0}


def test_exhaustive_small_event_prefixes_have_future_independent_ownership():
    events = [(0,), (1,), (2, 10)]
    for count in range(5):
        for selected in itertools.product(events, repeat=count):
            prefix = [token for event in selected for token in event]
            for context, stride in [(5, 1), (8, 3)]:
                a = evaluation_groups(canonical(prefix), CODEC, context, stride)
                for suffix in events:
                    b = evaluation_groups(canonical(prefix + list(suffix)), CODEC, context, stride)
                    for position in range(len(prefix) + 1):
                        left = next(g[1] for g in a if g[2] <= position < g[3])
                        right = next(g[1] for g in b if g[2] <= position < g[3])
                        assert left == right
