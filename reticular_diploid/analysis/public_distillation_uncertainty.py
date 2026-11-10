"""Paired family-cluster percentile intervals for fixed public teacher models."""

from __future__ import annotations

import numpy as np

from reticular_diploid.experiments.public_teacher_baselines import regression_metrics


def cluster_resamples(groups, draws=2000, seed=20261025):
    groups = np.asarray(groups)
    if groups.ndim != 1 or not len(groups) or draws < 1:
        raise ValueError("Bootstrap requires nonempty family groups and positive draws")
    labels, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(len(labels))]
    rng = np.random.default_rng(seed)
    return [np.concatenate([members[index] for index in rng.integers(len(labels), size=len(labels))])
            for _ in range(draws)]


def interval(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return {"lower": float(np.percentile(finite, 2.5)) if len(finite) else None,
            "upper": float(np.percentile(finite, 97.5)) if len(finite) else None,
            "valid_draws": len(finite)}


def family_intervals(truth, predictions, groups, *, pairs=(), draws=2000, seed=20261025):
    """Resample whole families, applying the same draw to every compared model.

    Intervals condition on the frozen models/teachers; they do not estimate
    variability from vocabulary fitting, pretraining seeds or teacher sampling.
    `pairs` contains model-index pairs; differences are first minus second.
    """
    truth = np.asarray(truth, np.float64)
    predictions = np.asarray(predictions, np.float64)
    groups = np.asarray(groups)
    if (truth.ndim != 1 or predictions.ndim != 2 or predictions.shape[1] != len(truth)
            or groups.shape != truth.shape or not np.isfinite(truth).all()
            or not np.isfinite(predictions).all()):
        raise ValueError("Bootstrap requires aligned finite targets, predictions and family groups")
    points = [regression_metrics(truth, row) for row in predictions]
    names = ["r_squared", "mse", "pearson_r"]
    samples = np.full((draws, len(predictions), 3), np.nan)
    for draw, rows in enumerate(cluster_resamples(groups, draws, seed)):
        y, p = truth[rows], predictions[:, rows]
        centered = y - y.mean()
        error = np.mean((p - y) ** 2, axis=1)
        samples[draw, :, 1] = error
        variance = float(np.mean(centered ** 2))
        if variance > 0:
            samples[draw, :, 0] = 1 - error / variance
            p = p - p.mean(axis=1, keepdims=True)
            denominator = np.linalg.norm(centered) * np.linalg.norm(p, axis=1)
            np.divide(p @ centered, denominator, out=samples[draw, :, 2], where=denominator > 0)
    bounds = [{name: interval(samples[:, model, metric]) for metric, name in enumerate(names)}
              for model in range(len(predictions))]
    paired = []
    for first, second in pairs:
        if not 0 <= first < len(points) or not 0 <= second < len(points):
            raise ValueError("Paired comparison references an absent model")
        point = {name: (points[first][name] - points[second][name]
                        if points[first][name] is not None and points[second][name] is not None else None)
                 for name in names}
        paired.append({"first": first, "second": second, "point_difference": point,
                       "interval": {name: interval(samples[:, first, metric] - samples[:, second, metric])
                                    for metric, name in enumerate(names)}})
    return {"points": points, "intervals": bounds, "paired": paired,
            "draws": draws, "seed": seed, "people": len(truth),
            "family_groups": len(np.unique(groups)), "confidence_level": .95,
            "method": "paired whole-pedigree-component percentile bootstrap; conditional on frozen models"}
