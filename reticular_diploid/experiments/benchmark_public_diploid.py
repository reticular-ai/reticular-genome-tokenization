"""Measure frozen-tokenizer reconstruction and lengths on development people."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reticular_diploid.data.frozen_diploid import FrozenDiploidTokenizer, sha256
from reticular_diploid.experiments.prepare_public_1000g import clean_commit, public_output


def summarize(lengths, motif_counts, residual_lengths):
    lengths = np.asarray(lengths)
    if not len(lengths) or np.any(lengths < 0):
        raise ValueError("Expected nonempty nonnegative token lengths")
    total = int(lengths.sum())
    motif_fraction = float(np.sum(motif_counts) / total) if total else None
    return {"people": len(lengths), "mean": float(lengths.mean()),
            "median": float(np.median(lengths)), "p95": float(np.percentile(lengths, 95)),
            "maximum": int(lengths.max()), "total_content_tokens": total,
            "motif_token_fraction": motif_fraction,
            "residual_token_fraction": 1 - motif_fraction if total else None,
            "residual_only_mean": float(np.mean(residual_lengths)),
            "framing_tokens_per_person": 3}


def benchmark(prepared, tokenizer, tokenizer_sha256, output):
    commit = clean_commit()
    public_output(output)
    if output.exists():
        raise ValueError("Benchmark receipt already exists; preserve immutable results")
    source = json.loads((prepared / "preparation.json").read_text())
    if source.get("public_data_only") is not True or sha256(prepared / "cohort.npz") != source["cohort_sha256"]:
        raise ValueError("Public cohort provenance differs")
    codec = FrozenDiploidTokenizer.load(tokenizer, tokenizer_sha256,
        expected_chromosome=source["interval"]["chromosome"])
    result = {"schema_version": "reticular-public-tokenization-benchmark/v1",
              "code_commit": commit, "public_data_only": True, "locked_test_used": False,
              "cohort_sha256": source["cohort_sha256"], "tokenizer_sha256": tokenizer_sha256,
              "variant_count": len(codec.positions), "motif_count": codec.motif_entries // 2,
              "vocabulary_size": codec.content_size + 6, "length_unit": "content tokens",
              "comparison_unit": "allele-specific dosage vector, not the full nucleotide string"}
    with np.load(prepared / "cohort.npz", allow_pickle=False) as z:
        for key in ["positions", "refs", "alts"]:
            if not np.array_equal(z[key], getattr(codec, key)):
                raise ValueError("Variant identity or orientation differs")
        dosages, partitions = z["dosages"], z["split"]
        for split, name in [(0, "training"), (1, "validation")]:
            lengths, motifs, residuals = [], [], []
            for row in np.flatnonzero(partitions == split):
                dosage = dosages[row]
                ids = codec.encode_genotype(dosage)
                if not np.array_equal(codec.decode(ids), dosage):
                    raise ValueError("Exact genotype reconstruction failed")
                lengths.append(len(ids))
                motifs.append(int(np.sum(ids < codec.motif_entries)))
                residuals.append(2 * np.count_nonzero(dosage))
            result[name] = summarize(lengths, motifs, residuals)
    result["exact_reconstruction_people"] = sum(result[k]["people"] for k in ["training", "validation"])
    result["exact_genotype_reconstruction_rate"] = 1.0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["prepared", "tokenizer", "output"]:
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    args = parser.parse_args()
    benchmark(args.prepared, args.tokenizer, args.tokenizer_sha256, args.output)


if __name__ == "__main__":
    main()
