"""Prefix-only ownership of canonical content targets and true EOS."""

import numpy as np

from reticular_diploid.data.diploid_windows import event_boundaries


def evaluation_groups(canonical, tokenizer, context=256, stride=128, *, split=1):
    """Own fixed target blocks; event-align their starts using already known IDs.

    The first block owns context-3 targets, then every block owns stride targets.
    A later block's nominal source start precedes its first target. Snapping
    back by at most one token keeps complete events at the source boundary.
    The target's type and the sequence's future end cannot change its owner.
    """
    capacity = context - 3
    if context < 5 or not 1 <= stride < capacity or split not in {0, 1, 2}:
        raise ValueError("Invalid evaluation context, stride or partition")
    result = []
    ptr = canonical["target_indptr"]
    for person in np.flatnonzero(canonical["participant_split"] == split):
        a, b = map(int, ptr[person:person + 2])
        ids = canonical["target_ids"][a:b]
        boundaries = event_boundaries(ids, tokenizer.motif_entries, tokenizer.first,
                                      tokenizer.second, len(tokenizer.positions))
        begins = [0, *range(capacity, len(ids) + 1, stride)]
        for block, begin in enumerate(begins):
            stop = min(len(ids) + 1, begin + (capacity if block == 0 else stride))
            nominal = block * stride
            start = int(boundaries[np.searchsorted(boundaries, nominal, side="right") - 1])
            if not start <= begin < stop or 2 + stop - 1 - start > context - 1:
                raise ValueError("Prefix-only evaluation group exceeds its context")
            result.append((int(person), start, begin, stop))
    return np.asarray(result, dtype=np.int64).reshape(-1, 4)


def materialize_group(canonical, group, special, context=256):
    person, start, begin, stop = map(int, group)
    a, b = map(int, canonical["target_indptr"][person:person + 2])
    n = b - a
    if not 0 <= start <= begin < stop <= n + 1:
        raise ValueError("Evaluation group escapes its original person")
    # Last query predicts target stop-1, so that target is never an input.
    last = stop - 1
    length = 2 + last - start
    if length > context - 1:
        raise ValueError("Evaluation prefix exceeds its context")
    ids = np.full(context - 1, special["pad"], np.int32)
    ids[:2] = [special["bos"], special["forward"]]
    ids[2:length] = canonical["target_ids"][a + start:a + last]
    frontier = int(canonical["target_coordinate_ids"][a + start - 1]) if start else 0
    coords = np.full(context - 1, frontier, np.int64)
    coords[2:length] = canonical["target_coordinate_ids"][a + start:a + last]
    if length > 2:
        coords[length:] = coords[length - 1]
    positions = np.arange(begin, stop)
    queries = 1 + positions - start
    labels = np.full(context - 1, special["pad"], np.int32)
    content = positions < n
    labels[queries[content]] = canonical["target_ids"][a + positions[content]]
    labels[queries[~content]] = special["eos"]
    mask = np.zeros(context - 1, bool)
    mask[queries] = True
    return {"input_ids": ids, "coordinate_ids": coords, "labels": labels, "target_mask": mask,
            "canonical_positions": positions, "query_indices": queries, "input_length": length}
