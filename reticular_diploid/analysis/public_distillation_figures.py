"""Render complete public teacher comparisons from aggregate evaluation receipts."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from reticular_diploid.data.frozen_diploid import sha256
from reticular_diploid.experiments.prepare_public_1000g import clean_commit, public_output

METHODS = {
    "scratch": ("Scratch", "#777777", "-"),
    "pretrained": ("NTP → score", "#0072B2", "-"),
    "joint": ("NTP + joint", "#D55E00", "-"),
    "direct_genotype_ridge": ("Genotype ridge", "#222222", "--"),
    "genotype_pca128_ridge": ("Training PCA128 ridge", "#009E73", ":"),
    "token_count_ridge": ("Token-count ridge", "#CC79A7", "-."),
    "effect_locus_genotype_ridge": ("Causal-locus ridge (privileged)", "#8B5A2B", ":"),
}


def sample_efficiency(evaluation):
    """Keep all fixed methods and seeds, including negative R-squared."""
    if evaluation["schema_version"] != "reticular-public-final-evaluation/v1":
        raise ValueError("Expected the complete aggregate held-out evaluation")
    records = [r for r in evaluation["records"] if r["method"] != "supplied_beta_genotype_oracle"]
    sizes = sorted({r["effect_variants"] for r in records})
    counts = sorted({r["training_people"] for r in records})
    seeds = sorted({r["seed"] for r in records})
    methods = sorted({r["method"] for r in records})
    keyed = {(r["effect_variants"], r["training_people"], r["seed"], r["method"]): r for r in records}
    expected = set(itertools.product(sizes, counts, seeds, methods))
    if len(keyed) != len(records) or set(keyed) != expected or not set(methods) <= METHODS.keys():
        raise ValueError("Figure input omits or duplicates predeclared comparisons")
    summary = []
    for size, count, method in itertools.product(sizes, counts, methods):
        values = [keyed[size, count, seed, method]["metrics"]["r_squared"] for seed in seeds]
        if not np.isfinite(values).all():
            raise ValueError("Figure R-squared must be finite; report failed evaluations separately")
        summary.append({"effect_variants": size, "training_people": count, "method": method,
                        "seeds": seeds, "values": values, "mean": float(np.mean(values)),
                        "minimum": min(values), "maximum": max(values)})
    return summary


def save(fig, output, name):
    for suffix in ["png", "svg"]:
        fig.savefig(output / (name + "." + suffix), dpi=220,
                    metadata={"Date": None} if suffix == "svg" else None)
    plt.close(fig)


def collect_learning_curves(roots, evaluation):
    """Export aggregate validation curves, excluding minibatch losses and logs."""
    records = []
    fields = ["step", "person_draws", "region_targets_seen", "total_worker_seconds", "validation",
              "peak_allocated_bytes"]
    for model in evaluation["records"]:
        if "arm" not in model:
            continue
        name = model["id"]
        if Path(name).name != name:
            raise ValueError("Curve model ID must be a directory basename")
        paths = [root / name for root in roots if (root / name / "contract.json").is_file()]
        if len(paths) != 1 or paths[0].is_symlink():
            raise ValueError("Every selected student must have exactly one curve source")
        root = paths[0]
        selected = json.loads((root / "selected-checkpoint.json").read_text())
        if selected["sha256"] != model["checkpoint_sha256"]:
            raise ValueError("Learning curve belongs to another selected model")
        curve = root / "learning-curve.jsonl"
        points = [json.loads(line) for line in curve.read_text().splitlines()]
        records.append({k: model[k] for k in ["id", "effect_variants", "training_people", "seed", "arm",
                                              "checkpoint_sha256"]} | {
            "source_curve_sha256": sha256(curve),
            "points": [{k: row[k] for k in fields} for row in points]})
    return {"schema_version": "reticular-public-learning-curves/v1", "records": records,
            "partition": "validation only; used for checkpoint selection", "locked_test_used": False}


def learning_curve_figures(data, evaluation, output):
    students = {r["id"]: r for r in evaluation["records"] if "arm" in r}
    records = data["records"]
    if (data["schema_version"] != "reticular-public-learning-curves/v1" or data["locked_test_used"] is not False
            or len(records) != len(students) or {r["id"] for r in records} != set(students)):
        raise ValueError("Learning curves must include every predeclared student")
    allowed = {"step", "person_draws", "region_targets_seen", "total_worker_seconds", "validation", "peak_allocated_bytes"}
    for row in records:
        if any(row[k] != students[row["id"]][k] for k in ["effect_variants", "training_people", "seed", "arm",
                                                        "checkpoint_sha256"]):
            raise ValueError("Learning curve identity differs from the frozen evaluation")
        points = row["points"]
        if (not points or any(set(p) != allowed or set(p["validation"]) != {"r_squared", "mse", "pearson_r", "people"}
                              for p in points)
                or any(a["step"] >= b["step"] or a["total_worker_seconds"] > b["total_worker_seconds"]
                       for a, b in zip(points, points[1:]))):
            raise ValueError("Learning curves contain unexpected fields or nonmonotonic steps/time")
    sizes = sorted({r["effect_variants"] for r in records})
    counts = sorted({r["training_people"] for r in records})
    seeds = sorted({r["seed"] for r in records})
    for xname, divisor, label, name in [("step", 1, "Optimization steps", "teacher-learning-steps"),
                                       ("total_worker_seconds", 60, "Worker time (minutes)", "teacher-learning-time")]:
        fig, axes = plt.subplots(len(sizes), len(counts), figsize=(11.5, 8.5), squeeze=False, layout="constrained")
        for i, size in enumerate(sizes):
            for j, count in enumerate(counts):
                axis = axes[i, j]
                for row in records:
                    if row["effect_variants"] != size or row["training_people"] != count:
                        continue
                    display, color, _ = METHODS[row["arm"]]
                    points = row["points"]
                    style = "-" if row["seed"] == seeds[0] else "--"
                    axis.plot([p[xname] / divisor for p in points], [p["validation"]["r_squared"] for p in points],
                              color=color, linestyle=style, linewidth=1.3, label=f"{display}, seed {seeds.index(row['seed']) + 1}")
                axis.axhline(0, color="#aaaaaa", linewidth=.6)
                axis.set(title=f"{size:,} variants · {count} labeled people", xlabel=label if i == len(sizes) - 1 else "",
                         ylabel="Validation R²" if j == 0 else "")
                axis.grid(alpha=.15)
        axes[0, -1].legend(frameon=False, fontsize=7)
        fig.suptitle("Validation learning curves; worker time includes validation and checkpointing", fontsize=12)
        save(fig, output, name)


def build(args):
    commit = clean_commit()
    public_output(args.output)
    if args.output.exists():
        raise ValueError("Figure attempt exists; preserve it")
    evaluation = json.loads(args.evaluation.read_text())
    summary = sample_efficiency(evaluation)
    sizes = sorted({r["effect_variants"] for r in summary})
    partition = "Reserved test" if evaluation["locked_test_used"] else "Validation replay"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "reticular-public-distillation-v1"})
    args.output.mkdir(parents=True)
    fig, axes = plt.subplots(1, len(sizes), figsize=(12, 4.4), squeeze=False, layout="constrained")
    for size, axis in zip(sizes, axes[0], strict=True):
        for method, (label, color, style) in METHODS.items():
            rows = [r for r in summary if r["effect_variants"] == size and r["method"] == method]
            if not rows:
                continue
            x = np.array([r["training_people"] for r in rows])
            axis.plot(x, [r["mean"] for r in rows], color=color, linestyle=style, marker="o", label=label)
            axis.fill_between(x, [r["minimum"] for r in rows], [r["maximum"] for r in rows], color=color, alpha=.1)
            for row in rows:
                axis.scatter([row["training_people"]] * len(row["values"]), row["values"], color=color, s=12, alpha=.7)
        axis.axhline(0, color="#aaaaaa", linewidth=.7)
        axis.set(xscale="log", xlabel="Labeled training people", ylabel=f"{partition} R²",
                 title=f"{size:,} effect variants")
        axis.set_xticks(sorted({r["training_people"] for r in summary}),
                        labels=[str(v) for v in sorted({r["training_people"] for r in summary})])
        axis.grid(alpha=.15)
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{partition}, n={evaluation['people']}: fixed additive teachers", fontweight="bold")
    save(fig, args.output, "teacher-sample-efficiency")

    if "uncertainty" in evaluation:
        by_id = {r["id"]: r for r in evaluation["records"]}
        fig, axes = plt.subplots(1, len(sizes), figsize=(12, 4.1), squeeze=False, layout="constrained")
        for size, axis in zip(sizes, axes[0], strict=True):
            data = evaluation["uncertainty"][str(size)]
            seeds = sorted({r["seed"] for r in evaluation["records"] if "seed" in r})
            for pair in data["paired"]:
                row = by_id[data["model_ids"][pair["first"]]]
                label, color, _ = METHODS[row["method"]]
                offset = (.96 if row["method"] == "pretrained" else 1.04) * (1 + .015 * seeds.index(row["seed"]))
                x = row["training_people"] * offset
                bounds = pair["interval"]["r_squared"]
                if bounds["lower"] is not None:
                    axis.vlines(x, bounds["lower"], bounds["upper"], color=color, alpha=.6)
                axis.scatter(x, pair["point_difference"]["r_squared"], color=color,
                             marker=["o", "s"][seeds.index(row["seed"]) % 2], s=23)
            axis.axhline(0, color="#777777", linestyle="--", linewidth=1)
            counts = sorted({r["training_people"] for r in summary})
            axis.set(xscale="log", xlabel="Labeled training people", ylabel="R² difference versus scratch",
                     title=f"{size:,} effect variants", xticks=counts, xticklabels=[str(v) for v in counts])
            axis.grid(alpha=.15)
        fig.suptitle("Paired family-bootstrap 95% intervals; conditional on fixed teachers and models", fontsize=12)
        save(fig, args.output, "teacher-paired-differences")

    with (args.output / "teacher-metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "effect_variants", "training_people", "seed", "method",
                                                   "r_squared", "mse", "pearson_r", "people"])
        writer.writeheader()
        for row in evaluation["records"]:
            writer.writerow({k: row.get(k) for k in writer.fieldnames if k not in row["metrics"]} | row["metrics"])
    curves = None
    if getattr(args, "curves", None):
        curves = json.loads(args.curves.read_text())
    elif getattr(args, "student_roots", None):
        curves = collect_learning_curves(args.student_roots, evaluation)
    if curves is not None:
        learning_curve_figures(curves, evaluation, args.output)
        (args.output / "learning-curves.json").write_text(json.dumps(curves, indent=2) + "\n")
    (args.output / "figure-data.json").write_text(json.dumps({
        "schema_version": "reticular-public-teacher-figures/v1", "code_commit": commit,
        "evaluation_sha256": sha256(args.evaluation), "partition": evaluation["partition"],
        "sample_efficiency": summary, "locked_test_used": evaluation["locked_test_used"],
        "caption": "Lines show two optimization-seed means; shading shows their range, not a confidence interval. Both seeds use the same frozen NTP parent. NTP and PCA use all 400 training genotypes at every label count. Targets use frozen full-training normalization. All models use validation-only selection. Privileged causal-locus ridge receives the teacher's variant identities. The supplied-beta oracle is listed separately in the metric table. Synthetic recovery is not clinical prediction."}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    curves = parser.add_mutually_exclusive_group()
    curves.add_argument("--student-roots", type=Path, nargs="+", help="Collect aggregate validation curves from selected runs")
    curves.add_argument("--curves", type=Path, help="Previously exported aggregate learning-curves.json")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
