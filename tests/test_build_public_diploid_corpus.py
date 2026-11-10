from types import SimpleNamespace

import numpy as np

from reticular_diploid.data.diploid_windows import materialize_window, window_arrays
from reticular_diploid.experiments.build_public_diploid_corpus import causal_coordinates


def test_coordinates_are_prefix_only_and_windows_score_each_target_once():
    codec = SimpleNamespace(motif_entries=2, first=2, second=2,
                            motifs=[np.array([0, 1])], selected=range(2),
                            positions=np.array([10, 20, 30, 40]))
    # Motif, then coarse-address 0/fine-address 1, then coarse 1/fine 0.
    ids = np.array([0, 2, 7, 3, 6], dtype=np.int32)
    coordinates = causal_coordinates(codec, ids)
    assert coordinates.tolist() == [0, 0, 0, 1, 1]
    changed = ids.copy()
    changed[2] = 6
    np.testing.assert_array_equal(causal_coordinates(codec, changed)[:2], coordinates[:2])
    canonical = {"target_ids": ids, "target_indptr": np.array([0, len(ids)]),
                 "participant_indices": np.array([5]), "participant_split": np.array([1]),
                 "target_coordinate_ids": coordinates, "target_genetic_cm": np.zeros(len(ids)),
                 "genomic_positions": np.zeros(len(ids))}
    view = window_arrays(canonical, codec, context=7, stride=2)
    seen = set()
    for row in range(len(view["window_lengths"])):
        window = materialize_window(canonical, view, row, content_size=8, context=7)
        for position, kind in zip(window["canonical_token_positions"], window["target_kinds"], strict=True):
            if kind >= 0:
                seen.add((int(position), int(kind)))
    assert seen == {(j, 0) for j in range(len(ids))} | {(-1, 1)}
    assert view["final_window"].sum() == 1
