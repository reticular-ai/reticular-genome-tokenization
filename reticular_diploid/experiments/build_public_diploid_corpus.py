"""Reproduce development token streams from a frozen public dictionary only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reticular_diploid.data.diploid_windows import window_arrays
from reticular_diploid.data.frozen_diploid import FrozenDiploidTokenizer, sha256
from reticular_diploid.experiments.prepare_public_1000g import clean_commit, public_output


def causal_coordinates(codec, ids):
    """A coordinate at index t uses IDs <=t, never the next residual payload."""
    frontier, pending, result = 0, None, []
    for token in ids:
        token = int(token)
        if token < codec.motif_entries:
            rank = int(codec.motifs[token // 2][0])
            frontier = max(frontier, rank // codec.second)
        elif token < codec.motif_entries + 2 * codec.first:
            coarse = (token - codec.motif_entries) % codec.first
            frontier, pending = max(frontier, coarse), coarse
        elif pending is not None:
            pending = None
        else:
            raise ValueError("R2 has no consumed coarse address")
        result.append(frontier)
    return np.asarray(result, dtype=np.uint16)


def build(args):
    commit = clean_commit()
    public_output(args.output)
    if args.output.exists():
        raise ValueError("Public corpus attempt already exists")
    codec = FrozenDiploidTokenizer.load(args.tokenizer, args.tokenizer_sha256)
    source = json.loads((args.prepared / "preparation.json").read_text())
    cohort = args.prepared / "cohort.npz"
    if source.get("public_data_only") is not True or sha256(cohort) != source["cohort_sha256"]:
        raise ValueError("Public source provenance differs")
    with np.load(cohort, allow_pickle=False) as z:
        if not all(np.array_equal(z[name], getattr(codec, name)) for name in ["positions", "refs", "alts"]):
            raise ValueError("Variant universe or REF/ALT orientation differs")
        split = z["split"]
        people = np.flatnonzero(split != 2)
        rows = []
        for person in people:
            g = z["dosages"][person]
            ids = codec.encode_genotype(g)
            if not np.array_equal(codec.decode(ids), g):
                raise ValueError("Public corpus reconstruction failed")
            rows.append(ids)
    ptr = np.r_[0, np.cumsum([len(row) for row in rows])].astype(np.int64)
    ids = np.concatenate(rows)
    coords = np.concatenate([causal_coordinates(codec, row) for row in rows])
    canonical = {"target_indptr": ptr, "target_ids": ids, "target_coordinate_ids": coords,
                 "target_genetic_cm": np.zeros(len(ids), np.float32),
                 "genomic_positions": np.zeros(len(ids), np.float64),
                 "participant_indices": people, "participant_split": split[people]}
    view = window_arrays(canonical, codec, args.context, args.stride)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "canonical.npz", **canonical)
    np.savez_compressed(args.output / "windows.npz", **view)
    receipt = {"schema_version": "reticular-public-diploid-corpus/v1", "public_data_only": True,
               "tokenizer_sha256": args.tokenizer_sha256, "cohort_sha256": source["cohort_sha256"],
               "canonical_sha256": sha256(args.output / "canonical.npz"),
               "windows_sha256": sha256(args.output / "windows.npz"),
               "context": args.context, "stride": args.stride,
               "content_size": codec.content_size, "vocabulary_size": codec.content_size + 6,
               "special_tokens": codec.special_tokens, "coordinate_vocabulary_size": codec.first,
               "training_people": int(np.sum(split[people] == 0)),
               "validation_people": int(np.sum(split[people] == 1)), "locked_test_used": False,
               "genetic_distance_enabled": False, "graph_geometry_enabled": False,
               "code_commit": commit}
    (args.output / "corpus.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"development_people": len(people), "canonical_tokens": len(ids),
                      "windows": len(view["window_lengths"]), "locked_test_used": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["prepared", "tokenizer", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
