"""Run or resume the predeclared public teacher comparison on one existing GPU."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from reticular_diploid.data.frozen_diploid import sha256
from reticular_diploid.experiments.prepare_public_1000g import clean_commit, public_output


def trial_grid(recipe, counts, seeds, exclude_primary_full=False):
    if (not counts or not seeds or len(set(counts)) != len(counts) or len(set(seeds)) != len(seeds)
            or not set(counts) <= set(recipe["training_people"])
            or not set(seeds) <= set(recipe["seeds"])):
        raise ValueError("Requested comparison is outside the frozen recipe or contains duplicates")
    return [(count, seed, effects, arm) for count in counts for seed in seeds
            if not (exclude_primary_full and count == 400 and seed == recipe["seeds"][0])
            for effects in recipe["effect_variants"] for arm in recipe["arms"]]


def history(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def run(args):
    commit = clean_commit()
    public_output(args.output)
    recipe = json.loads(args.config.read_text())
    grid = trial_grid(recipe, args.training_people or sorted(recipe["training_people"], reverse=True),
                      args.seeds or recipe["seeds"], args.exclude_primary_full)
    selected_ntp = json.loads((args.ntp_run / "selected-checkpoint.json").read_text())
    name = selected_ntp["path"]
    if Path(name).name != name or sha256(args.ntp_run / name) != selected_ntp["sha256"]:
        raise ValueError("Selected NTP checkpoint path or checksum differs")
    config_hash = sha256(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    previous = sum(rows[-1]["total_worker_seconds"] for path in args.output.glob("A*/learning-curve.jsonl")
                   if (rows := history(path)))
    cap = min(recipe["resources"].get("campaign_additional_worker_hours", args.maximum_hours), args.maximum_hours) * 3600
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("Campaign runtime cap must be positive and finite")
    started = time.monotonic()
    records = []
    source_root = Path(__file__).resolve().parents[2]
    for count, seed, effects, arm in grid:
        tag = f"A{recipe['effect_variants'].index(effects) + 1}-{arm}-n{count}-s{seed}"
        output = args.output / tag
        rows = history(output / "learning-curve.jsonl")
        current = rows[-1]["step"] if rows else 0
        if output.exists():
            contract = json.loads((output / "contract.json").read_text())
            expected = {"code_commit": commit, "recipe_sha256": config_hash, "arm": arm,
                        "seed": seed, "training_people": count, "effect_variants": effects,
                        "pretrained_sha256": selected_ntp["sha256"]}
            if any(contract[k] != v for k, v in expected.items()):
                raise ValueError("Existing trial belongs to another immutable comparison")
            if not current:
                raise ValueError("Existing attempt has no resumable checkpoint; preserve it and use a new output root")
        common = [sys.executable, "-m", "reticular_diploid.experiments.distill_public_diploid",
                  "--corpus", str(args.corpus.resolve()), "--teachers", str(args.teachers.resolve()),
                  "--config", str(args.config.resolve()), "--pretrained", str((args.ntp_run / name).resolve()),
                  "--pretrained-sha256", selected_ntp["sha256"], "--arm", arm,
                  "--effect-variants", str(effects), "--training-people", str(count), "--seed", str(seed),
                  "--output", str(output.resolve()), "--device", "cuda"]
        while current < recipe["training"]["steps"]:
            remaining = cap - previous - (time.monotonic() - started)
            if remaining < 60:
                raise ValueError("Campaign runtime cap reached; all completed and partial trials are preserved")
            endpoint = 2 if not current else recipe["training"]["steps"]
            command = common + ["--stop-after", str(endpoint)]
            if current:
                checkpoint = ("latest.pt" if recipe.get("checkpoint_retention") == "latest_and_best"
                              else f"step-{current:06d}.pt")
                command += ["--resume", str((output / checkpoint).resolve())]
            attempt = len(list(args.output.glob(tag + f"-to-{endpoint}-attempt*.log"))) + 1
            log = args.output / f"{tag}-to-{endpoint}-attempt{attempt}.log"
            with log.open("x") as handle:
                result = subprocess.run(command, cwd=source_root, stdout=handle, stderr=subprocess.STDOUT,
                                        timeout=min(remaining, recipe["resources"]["maximum_worker_hours"] * 3600 + 120))
            if result.returncode:
                raise ValueError(f"Trial {tag} failed with exit {result.returncode}; inspect its preserved log")
            rows = history(output / "learning-curve.jsonl")
            current = rows[-1]["step"]
        selected = json.loads((output / "selected-checkpoint.json").read_text())
        if sha256(output / selected["path"]) != selected["sha256"]:
            raise ValueError("Selected student checksum differs")
        record = {"trial": tag, "training_people": count, "seed": seed,
                  "effect_variants": effects, "arm": arm, "selected": selected, "final": rows[-1]}
        records.append(record)
        print(json.dumps(record), flush=True)
        receipt = {"schema_version": "reticular-public-student-comparison/v1", "code_commit": commit,
                   "recipe_sha256": config_hash, "pretrained_sha256": selected_ntp["sha256"],
                   "records_in_this_invocation": records, "locked_test_used": False,
                   "prior_recorded_worker_seconds": previous,
                   "invocation_seconds": time.monotonic() - started}
        (args.output / "comparison.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["corpus", "teachers", "ntp-run", "config", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--training-people", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--exclude-primary-full", action="store_true",
                        help="The first seed's n400 arms were already run under the v1 recipe")
    parser.add_argument("--maximum-hours", type=float, default=3.5,
                        help="Campaign cap when the immutable recipe has no aggregate cap")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
