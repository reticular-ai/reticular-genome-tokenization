"""Deterministic synthetic additive teachers; these are not SBayesRC estimates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from reticular_diploid.data.frozen_diploid import FrozenDiploidTokenizer, sha256
from reticular_diploid.experiments.prepare_public_1000g import clean_commit, public_output


def load_scores(root, canonical, corpus_receipt):
    metadata = json.loads((root / "teachers.json").read_text())
    if (metadata.get("public_data_only") is not True
            or metadata.get("locked_test_scores_generated") is not False
            or any(metadata[k] != corpus_receipt[k] for k in ["cohort_sha256", "tokenizer_sha256"])
            or sha256(root / "scores.npz") != metadata["scores_sha256"]
            or sha256(root / "weights.tsv") != metadata["weights_sha256"]):
        raise ValueError("Teacher provenance differs from the public development corpus")
    with np.load(root / "scores.npz", allow_pickle=False) as z:
        if (not np.array_equal(z["participant_indices"], canonical["participant_indices"])
                or not np.array_equal(z["participant_split"], canonical["participant_split"])):
            raise ValueError("Teacher rows or partitions differ from canonical people")
        scores = z["scores"]
    if (scores.shape != (len(canonical["participant_indices"]), len(metadata["teachers"]))
            or not np.isfinite(scores).all()):
        raise ValueError("Teacher scores are malformed or nonfinite")
    return metadata, scores


def load_frozen_effects(path, positions, refs, alts, metadata):
    lookup = {key: rank for rank, key in enumerate(zip(positions, refs, alts, strict=True))}
    if len(lookup) != len(positions):
        raise ValueError("Frozen teacher universe contains duplicate allele identities")
    selected = {t["effect_variants"]: [] for t in metadata["teachers"]}
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            teacher = int(row["teacher"])
            key = (int(row["position"]), row["REF"], row["ALT"])
            beta = float(row["beta"])
            if row["chromosome"] != "22" or teacher not in selected or key not in lookup or not np.isfinite(beta):
                raise ValueError("Frozen effect has an unknown teacher, allele or nonfinite weight")
            selected[teacher].append((lookup[key], beta))
    effects = []
    for teacher in metadata["teachers"]:
        size = teacher["effect_variants"]
        rows = selected[size]
        if len(rows) != size or len({rank for rank, _ in rows}) != size:
            raise ValueError("Frozen effect count or allele uniqueness differs")
        if (not np.isfinite(teacher["training_mean"]) or not np.isfinite(teacher["training_std"])
                or teacher["training_std"] <= 0):
            raise ValueError("Frozen teacher normalization is invalid")
        effects.append({"ranks": np.asarray([r for r, _ in rows], np.int64),
                        "beta": np.asarray([b for _, b in rows], np.float64),
                        "mean": teacher["training_mean"], "std": teacher["training_std"]})
    return effects


def score_frozen_effects(dosages, effects):
    g = np.asarray(dosages)
    if g.ndim != 2 or g.dtype.kind not in "iub" or np.any(g < 0) or np.any(g > 2):
        raise ValueError("Teacher requires callable allele-specific integer dosages")
    columns = []
    for effect in effects:
        raw = np.einsum("ij,j->i", g[:, effect["ranks"]].astype(np.float64), effect["beta"], optimize=False)
        columns.append((raw - effect["mean"]) / effect["std"])
    result = np.stack(columns, axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Frozen teacher scores are nonfinite")
    return result


def select_variants(training, positions, motif_ranks, count=1024, seed=20261023):
    """Round-robin region, training MAF and motif-membership strata.

    LD is retained deliberately and diagnosed separately. Number of effect
    variants therefore does not mean number of independent genetic effects.
    """
    g = np.asarray(training, dtype=np.float64)
    frequency = g.mean(axis=0) / 2
    maf = np.minimum(frequency, 1 - frequency)
    eligible = (maf >= .01) & (g.var(axis=0) > 0)
    if np.count_nonzero(eligible) < count:
        raise ValueError("Insufficient training-polymorphic eligible public alleles")
    region = np.minimum(3, 4 * (positions - positions.min()) // (positions.max() - positions.min() + 1))
    bins = np.searchsorted([.05, .2], maf, side="right")
    covered = np.zeros(len(positions), bool)
    covered[np.asarray(motif_ranks, dtype=int)] = True
    rng = np.random.default_rng(seed)
    cells = []
    for index in range(24):
        spatial, membership = index % 4, (index // 4) % 2
        frequency_bin = (index // 8 + spatial + membership) % 3
        ranks = np.flatnonzero(eligible & (region == spatial) & (covered == bool(membership))
                              & (bins == frequency_bin))
        cells.append(list(rng.permutation(ranks)))
    selected = []
    while len(selected) < count:
        before = len(selected)
        for cell in cells:
            if cell and len(selected) < count:
                selected.append(int(cell.pop()))
        if len(selected) == before:
            raise ValueError("Stratified teacher selection cannot advance")
    return np.asarray(selected, np.int64), frequency, covered


def make_teachers(training, development, positions, motif_ranks, *, seed=20261023, sizes=(16, 128, 1024)):
    selected, frequency, covered = select_variants(training, positions, motif_ranks, max(sizes), seed)
    effects = np.random.default_rng(seed + 1).normal(size=max(sizes))
    teachers, scores = [], []
    for size in sizes:
        ranks = selected[:size]
        beta = effects[:size] / np.sqrt(size)
        raw_train = np.einsum("ij,j->i", np.asarray(training[:, ranks], np.float64), beta,
                              optimize=False)
        mean, scale = float(raw_train.mean()), float(raw_train.std(ddof=0))
        if not np.isfinite(scale) or scale <= 1e-12:
            raise ValueError("Synthetic teacher has zero or nonfinite training variance")
        expected = (np.einsum("ij,j->i", np.asarray(development[:, ranks], np.float64), beta,
                              optimize=False) - mean) / scale
        # Independent summation oracle catches allele/order and score-normalization errors.
        oracle = np.zeros(len(development), np.float64)
        for rank, effect in zip(ranks, beta, strict=True):
            oracle += development[:, rank] * effect
        oracle = (oracle - mean) / scale
        error = float(np.max(np.abs(oracle - expected)))
        if not np.all(np.isfinite(expected)) or not np.isfinite(error) or error > 1e-12:
            raise ValueError("Direct dosage oracle differs from teacher scores")
        standardized = np.asarray(training[:, ranks], np.float64)
        standardized -= standardized.mean(axis=0)
        standardized /= np.linalg.norm(standardized, axis=0)
        pairs = np.triu_indices(size, 1)
        if len(pairs[0]) > 10000:
            chosen = np.random.default_rng(seed + size).choice(len(pairs[0]), 10000, replace=False)
            pairs = (pairs[0][chosen], pairs[1][chosen])
        correlations = np.sum(standardized[:, pairs[0]] * standardized[:, pairs[1]], axis=0)
        teachers.append({"effect_variants": size, "ranks": ranks, "beta": beta,
                         "training_mean": mean, "training_std": scale,
                         "training_frequency": frequency[ranks],
                         "motif_covered_variants": int(covered[ranks].sum()),
                         "residual_only_variants": int((~covered[ranks]).sum()),
                         "oracle_max_absolute_error": error,
                         "ld_pairs_measured": len(correlations),
                         "ld_mean_r_squared": float(np.mean(correlations ** 2)),
                         "ld_max_absolute_r": float(np.max(np.abs(correlations)))})
        scores.append(expected)
    return teachers, np.stack(scores, axis=1)


def build(args):
    commit = clean_commit()
    public_output(args.output)
    if args.output.exists():
        raise ValueError("Teacher attempt exists; never select a new teacher after seeing outcomes")
    codec = FrozenDiploidTokenizer.load(args.tokenizer, args.tokenizer_sha256)
    preparation = json.loads((args.prepared / "preparation.json").read_text())
    cohort_path = args.prepared / "cohort.npz"
    if preparation.get("public_data_only") is not True or sha256(cohort_path) != preparation["cohort_sha256"]:
        raise ValueError("Public cohort provenance differs")
    with np.load(cohort_path, allow_pickle=False) as z:
        if not all(np.array_equal(z[k], getattr(codec, k)) for k in ["positions", "refs", "alts"]):
            raise ValueError("Teacher allele universe differs from the tokenizer")
        split = z["split"]
        people = np.flatnonzero(split != 2)
        teachers, scores = make_teachers(z["dosages"][split == 0], z["dosages"][people],
                                         z["positions"], codec.motif_ranks, seed=args.seed)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "scores.npz", participant_indices=people,
                        participant_split=split[people], scores=scores)
    with (args.output / "weights.tsv").open("x") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["teacher", "chromosome", "position", "REF", "ALT", "beta", "training_alt_frequency"])
        for teacher in teachers:
            for rank, beta, frequency in zip(teacher["ranks"], teacher["beta"],
                                              teacher["training_frequency"], strict=True):
                writer.writerow([teacher["effect_variants"], "22", int(codec.positions[rank]),
                                 str(codec.refs[rank]), str(codec.alts[rank]),
                                 format(beta, ".17g"), format(frequency, ".17g")])
    metadata = {"schema_version": "reticular-synthetic-additive-teachers/v1", "public_data_only": True,
                "code_commit": commit, "cohort_sha256": preparation["cohort_sha256"],
                "tokenizer_sha256": args.tokenizer_sha256, "seed": args.seed,
                "selection": "nested region/MAF/motif strata, training MAF>=0.01; natural LD retained",
                "effect_distribution": "seed+1 PCG64 standard normal, divided by sqrt(effect_count)",
                "normalization": "training mean and population standard deviation only",
                "teachers": [{k: v for k, v in t.items() if k not in ["ranks", "beta", "training_frequency"]}
                             for t in teachers], "locked_test_scores_generated": False,
                "scores_sha256": sha256(args.output / "scores.npz"),
                "weights_sha256": sha256(args.output / "weights.tsv")}
    (args.output / "teachers.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"synthetic_teachers": [t["effect_variants"] for t in teachers],
                      "oracle_verified": True, "locked_test_used": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["prepared", "tokenizer", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--seed", type=int, default=20261023)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
