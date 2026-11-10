"""Matched public additive-teacher ridge baselines, using development people only."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from reticular_diploid.data.frozen_diploid import sha256
from reticular_diploid.data.public_diploid_corpus import load_corpus
from reticular_diploid.experiments.prepare_public_1000g import clean_commit, public_output
from reticular_diploid.experiments.public_additive_teachers import load_scores


def regression_metrics(truth, predicted):
    truth, predicted = np.asarray(truth, np.float64), np.asarray(predicted, np.float64)
    if truth.shape != predicted.shape or truth.ndim != 1 or not np.isfinite([truth, predicted]).all():
        raise ValueError("Regression metrics require aligned finite score vectors")
    centered = truth - truth.mean()
    denominator = float(centered @ centered)
    if denominator <= 0:
        raise ValueError("R-squared requires nonconstant targets")
    error = float(np.sum((truth - predicted) ** 2))
    p = predicted - predicted.mean()
    scale = np.linalg.norm(centered) * np.linalg.norm(p)
    return {"r_squared": 1 - error / denominator, "mse": error / len(truth),
            "pearson_r": float(centered @ p / scale) if scale > 0 else None,
            "people": len(truth)}


def training_subset(split, count, seed):
    training = np.flatnonzero(split == 0)
    if not 1 < count <= len(training):
        raise ValueError("Training subset size is outside the frozen training partition")
    return np.random.default_rng(seed).permutation(training)[:count]


def ridge_predictions(training, validation, targets, alphas=(.01, 1., 100.)):
    """Center on the supplied training subset, solving the small dual system."""
    x, v, y = (np.asarray(a, np.float64) for a in [training, validation, targets])
    mean, target_mean = x.mean(axis=0), y.mean(axis=0)
    x, v, y = x - mean, v - mean, y - target_mean
    gram, cross = x @ x.T, v @ x.T
    if not np.isfinite(gram).all() or not np.isfinite(cross).all() or not np.isfinite(y).all():
        raise ValueError("Ridge training arrays or kernel are nonfinite")
    eigenvalues, vectors = np.linalg.eigh(gram)
    projected = vectors.T @ y
    predictions = []
    for alpha in alphas:
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge penalties must be positive and finite")
        coefficients = vectors @ (projected / (np.maximum(eigenvalues, 0)[:, None] + alpha))
        predictions.append(cross @ coefficients + target_mean)
    result = np.stack(predictions)
    if not np.isfinite(result).all():
        raise ValueError("Ridge predictions are nonfinite")
    return result


def token_counts(canonical, vocabulary_size):
    result = np.zeros((len(canonical["participant_split"]), vocabulary_size), np.float32)
    ptr, ids = canonical["target_indptr"], canonical["target_ids"]
    for row in range(len(result)):
        result[row] = np.bincount(ids[ptr[row]:ptr[row + 1]], minlength=vocabulary_size)
    return result


def genotype_pca_features(features, training_rows, width=128):
    """Fit centered, unwhitened PCs using all unlabeled training genotypes only."""
    centered = np.array(features, dtype=np.float64, copy=True)
    training_rows = np.asarray(training_rows, dtype=int)
    if centered.ndim != 2 or len(training_rows) < 2 or width < 1 or not np.isfinite(centered).all():
        raise ValueError("PCA requires finite features, training people and a positive width")
    centered -= centered[training_rows].mean(axis=0)
    training = centered[training_rows]
    values, vectors = np.linalg.eigh(training @ training.T)
    threshold = np.finfo(np.float64).eps * max(training.shape) * max(float(values[-1]), 1.)
    chosen = np.flatnonzero(values > threshold)[::-1][:width]
    if not len(chosen):
        raise ValueError("Training genotypes have no positive-variance components")
    projected = (centered @ training.T) @ vectors[:, chosen] / np.sqrt(values[chosen])
    if not np.isfinite(projected).all():
        raise ValueError("Genotype PCA projection is nonfinite")
    return projected


def teacher_variant_ranks(path, positions, refs, alts):
    """Privileged positive control: supply effect-locus identities, learn weights."""
    lookup = {key: rank for rank, key in enumerate(zip(positions, refs, alts, strict=True))}
    result = {}
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (int(row["position"]), row["REF"], row["ALT"])
            if row["chromosome"] != "22" or key not in lookup:
                raise ValueError("Teacher effect allele is absent from the public universe")
            result.setdefault(int(row["teacher"]), []).append(lookup[key])
    if any(len(ranks) != size or len(set(ranks)) != size for size, ranks in result.items()):
        raise ValueError("Teacher effect-locus counts or identities are malformed")
    return result


def run(args):
    commit = clean_commit()
    public_output(args.output)
    if args.output.exists():
        raise ValueError("Baseline attempt already exists")
    receipt, canonical, _ = load_corpus(args.corpus)
    teachers, targets = load_scores(args.teachers, canonical, receipt)
    cohort = args.prepared / "cohort.npz"
    if sha256(cohort) != receipt["cohort_sha256"]:
        raise ValueError("Direct-genotype baseline cohort differs")
    with np.load(cohort, allow_pickle=False) as z:
        # Materialize only public development rows; do not score locked-test people.
        genotypes = z["dosages"][canonical["participant_indices"]]
        effect_ranks = teacher_variant_ranks(args.teachers / "weights.tsv", z["positions"], z["refs"], z["alts"])
    validation = np.flatnonzero(canonical["participant_split"] == 1)
    alphas = [.01, 1., 100., 10000., 1000000.]
    records, predictions = [], []
    started = time.monotonic()
    all_training = np.flatnonzero(canonical["participant_split"] == 0)
    pca = genotype_pca_features(genotypes, all_training, width=128)
    for name, features in [("direct_genotype_ridge", genotypes),
                           ("token_count_ridge", token_counts(canonical, receipt["content_size"])),
                           ("genotype_pca128_ridge", pca)]:
        for count in [32, 128, 400]:
            training = training_subset(canonical["participant_split"], count, args.seed)
            predicted = ridge_predictions(features[training], features[validation], targets[training], alphas)
            for trait, teacher in enumerate(teachers["teachers"]):
                metrics = [regression_metrics(targets[validation, trait], p[:, trait]) for p in predicted]
                best = int(np.argmin([m["mse"] for m in metrics]))
                records.append({"baseline": name, "training_people": count,
                                "effect_variants": teacher["effect_variants"],
                                "selected_alpha": alphas[best], "validation": metrics[best],
                                "alpha_validation": metrics})
                if name == "genotype_pca128_ridge":
                    records[-1].update({"unlabeled_training_people": len(all_training),
                                        "pca_components": pca.shape[1],
                                        "pca_transform": "training-only centered raw dosages; unwhitened"})
                predictions.append(predicted[best, :, trait])
    for trait, teacher in enumerate(teachers["teachers"]):
        features = genotypes[:, effect_ranks[teacher["effect_variants"]]]
        for count in [32, 128, 400]:
            training = training_subset(canonical["participant_split"], count, args.seed)
            predicted = ridge_predictions(features[training], features[validation], targets[training, trait:trait + 1], alphas)
            metrics = [regression_metrics(targets[validation, trait], p[:, 0]) for p in predicted]
            best = int(np.argmin([m["mse"] for m in metrics]))
            records.append({"baseline": "effect_locus_genotype_ridge", "training_people": count,
                            "effect_variants": teacher["effect_variants"], "selected_alpha": alphas[best],
                            "validation": metrics[best], "alpha_validation": metrics,
                            "privileged_information": "teacher effect-locus identities supplied; weights learned"})
            predictions.append(predicted[best, :, 0])
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "validation-predictions.npz", predictions=np.stack(predictions),
                        canonical_rows=validation)
    result = {"schema_version": "reticular-public-teacher-baselines/v1", "code_commit": commit,
              "corpus": receipt, "teachers": teachers, "seed": args.seed,
              "selection": "minimum validation MSE over predeclared alpha grid", "alphas": alphas,
              "feature_transform": "training-subset centering; no variance scaling",
              "sample_subsets": "nested seeded permutation of training people, shared with students",
              "protocol_revision": "v3 adds fixed PCA128 on all unlabeled training genotypes to match NTP access; v1/v2 retained",
              "elapsed_seconds": time.monotonic() - started, "locked_test_used": False,
              "records": records, "predictions_sha256": sha256(args.output / "validation-predictions.npz")}
    (args.output / "baselines.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"records": records, "elapsed_seconds": result["elapsed_seconds"], "locked_test_used": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["prepared", "corpus", "teachers", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261023)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
