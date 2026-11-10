"""Event-aligned canonical windows for a frozen tokenizer."""

import numpy as np

def event_boundaries(ids, motif_tokens, first, second, mutations):
    """Reject incomplete residuals and return boundaries after whole events."""
    ids = np.asarray(ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ValueError("Expected integer content tokens")
    boundaries = [0]
    cursor = 0
    while cursor < len(ids):
        token = int(ids[cursor])
        if 0 <= token < motif_tokens:
            cursor += 1
        else:
            code = token - motif_tokens
            if not 0 <= code < 2 * first or cursor + 1 >= len(ids):
                raise ValueError("Incomplete or invalid residual event")
            offset = int(ids[cursor + 1]) - motif_tokens - 2 * first
            if not 0 <= offset < second or (code % first) * second + offset >= mutations:
                raise ValueError("Residual address exceeds mutation universe")
            cursor += 2
        boundaries.append(cursor)
    return np.asarray(boundaries, np.int64)


def person_ranges(ids, tokenizer, context=256, stride=128):
    """Snap nominal content starts backward to event boundaries, preserving tails."""
    capacity = context - 3  # BOS + forward + room for the one real final EOS.
    if context < 5 or not 1 <= stride < capacity:
        raise ValueError("Context/stride cannot preserve whole events without gaps")
    boundaries = event_boundaries(ids, len(tokenizer.selected), tokenizer.first,
                                  tokenizer.second, len(tokenizer.positions))
    if len(boundaries) > 1 and np.diff(boundaries).max() > capacity:
        raise ValueError("An indivisible event cannot fit the context budget")
    if not len(ids):
        return [(0, 0)]
    rows, previous = [], -1
    for nominal in range(0, len(ids), stride):
        start = int(boundaries[np.searchsorted(boundaries, nominal, side="right") - 1])
        if start == previous:
            continue
        stop = int(boundaries[np.searchsorted(boundaries, start + capacity, side="right") - 1])
        if stop <= start:
            raise ValueError("Event boundary cannot advance")
        rows.append((start, stop))
        previous = start
        if stop == len(ids):
            break  # No redundant tail windows and no double-coverage requirement.
    return rows


def window_arrays(canonical, tokenizer, context=256, stride=128):
    """Store only window indices/state and canonical coverage, never duplicate IDs."""
    ptr = np.asarray(canonical["target_indptr"])
    ids = canonical["target_ids"]
    people = canonical["participant_indices"]
    if (len(ptr) != len(people) + 1 or ptr[0] != 0 or ptr[-1] != len(ids)
            or np.any(ptr[1:] < ptr[:-1])):
        raise ValueError("Canonical row pointers differ")
    if any(len(canonical[k]) != len(ids) for k in ["target_coordinate_ids", "target_genetic_cm", "genomic_positions"]):
        raise ValueError("Canonical sidecars differ")
    coverage = np.zeros(len(ids), np.uint16)
    starts, stops, owners, finals = [], [], [], []
    for local in range(len(people)):
        a, b = int(ptr[local]), int(ptr[local + 1])
        rows = person_ranges(ids[a:b], tokenizer, context, stride)
        for offset, (start, stop) in enumerate(rows):
            starts.append(a + start)
            stops.append(a + stop)
            owners.append(local)
            finals.append(offset == len(rows) - 1)
            coverage[a + start:a + stop] += 1
    if not owners or np.any(coverage == 0):
        raise ValueError("Canonical tokens are not all represented")
    owner = np.asarray(owners, np.int64)
    starts, stops = np.asarray(starts, np.int64), np.asarray(stops, np.int64)
    final = np.asarray(finals, bool)
    targets = stops - starts + final.astype(np.int64)
    lengths = targets + 2
    if np.any(lengths > context) or np.count_nonzero(final) != len(people):
        raise ValueError("Context budget or real person endings differ")
    first = starts == ptr[owner]
    frontier = np.zeros(len(starts), np.uint16)
    cm = np.zeros(len(starts), np.float32)
    frontier[~first] = canonical["target_coordinate_ids"][starts[~first] - 1]
    cm[~first] = canonical["target_genetic_cm"][starts[~first] - 1]
    counts = np.bincount(owner, minlength=len(people))
    effective = np.asarray([float(np.sum(1.0 / coverage[a:b])) + int(last)
                            for a, b, last in zip(starts, stops, final, strict=True)])
    if not np.isclose(effective.sum(), len(ids) + len(people), rtol=1e-12, atol=1e-9):
        raise ValueError("Inverse coverage target weights differ")
    return {
        "window_indptr": np.r_[np.int64(0), np.cumsum(lengths)],
        "window_lengths": lengths,
        "canonical_local_person": owner,
        "participant_indices": people[owner],
        "participant_split": canonical["participant_split"][owner],
        "canonical_prefix_end": starts,
        "canonical_content_end": stops,
        "canonical_coverage": coverage,
        "coordinate_frontier": frontier,
        "genetic_frontier": cm,
        "final_window": final,
        "person_window_count": counts[owner],
        "person_valid_target_count": (np.diff(ptr) + 1)[owner],
        "valid_targets_per_window": targets,
        "effective_target_weight": effective,
        "padding_slots": context - lengths,
    }


def materialize_window(canonical, view, row, content_size, context=256, *, pad=True):
    """Collate a view without encoding; target identity and weights are explicit.

    Canonical token indices and per-person positions are -1 for inserted
    specials/padding. The only scored special is the one real EOS (kind 1).
    Content is kind 0; prefixes/padding are kind -1. Validation dedup keys are
    (parent canonical identity, person index, position, kind).
    """
    local = int(view["canonical_local_person"][row])
    start, stop = (int(view[k][row]) for k in ["canonical_prefix_end", "canonical_content_end"])
    a, b = (int(v) for v in canonical["target_indptr"][local:local + 2])
    last = bool(view["final_window"][row])
    if not a <= start <= stop <= b or last != (stop == b):
        raise ValueError("View escapes its source person or mislabels a real end")
    bos, forward, eos = (content_size + i for i in (1, 4, 3))
    ids = np.asarray([bos, forward, *canonical["target_ids"][start:stop],
                      *([eos] if last else [])], np.int32)
    length = len(ids)
    if length != view["window_lengths"][row] or length > context:
        raise ValueError("Materialized row budget differs")
    padding = context - length if pad else 0
    indices = np.r_[[-1, -1], np.arange(start, stop), *([-1] if last else [])].astype(np.int64)
    positions = np.where(indices >= 0, indices - a, -1)
    kinds = np.r_[[-1, -1], np.zeros(stop - start, np.int8), *([1] if last else [])].astype(np.int8)
    weights = np.r_[[0.0, 0.0], 1.0 / view["canonical_coverage"][start:stop], *([1.0] if last else [])]
    frontier, cm = (view[k][row] for k in ["coordinate_frontier", "genetic_frontier"])
    end_frontier = canonical["target_coordinate_ids"][stop - 1] if stop > a else frontier
    end_cm = canonical["target_genetic_cm"][stop - 1] if stop > a else cm
    coordinates = np.r_[[frontier, frontier], canonical["target_coordinate_ids"][start:stop],
                        *([end_frontier] if last else [])].astype(canonical["target_coordinate_ids"].dtype)
    genetic = np.r_[[cm, cm], canonical["target_genetic_cm"][start:stop],
                    *([end_cm] if last else [])].astype(canonical["target_genetic_cm"].dtype)
    genomic = np.r_[[-1.0, -1.0], canonical["genomic_positions"][start:stop],
                    *([-1.0] if last else [])].astype(canonical["genomic_positions"].dtype)
    return {
        "input_ids": np.pad(ids, (0, padding), constant_values=content_size),
        "attention_mask": np.r_[np.ones(length, bool), np.zeros(padding, bool)],
        "target_mask": np.r_[kinds >= 0, np.zeros(padding, bool)],
        "target_weights": np.pad(weights, (0, padding)),
        "canonical_token_indices": np.pad(indices, (0, padding), constant_values=-1),
        "canonical_token_positions": np.pad(positions, (0, padding), constant_values=-1),
        "target_kinds": np.pad(kinds, (0, padding), constant_values=-1),
        "coordinate_ids": np.pad(coordinates, (0, padding), constant_values=end_frontier),
        "genetic_cm": np.pad(genetic, (0, padding), constant_values=end_cm),
        "genomic_positions": np.pad(genomic, (0, padding), constant_values=-1),
    }
