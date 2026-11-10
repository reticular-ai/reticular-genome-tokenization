"""Plot verified public token lengths and complete-stream NTP likelihoods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from reticular_diploid.data.frozen_diploid import sha256
from reticular_diploid.data.public_diploid_corpus import load_corpus, same_corpus
from reticular_diploid.experiments.evaluate_public_diploid import bigram_baseline, unigram_baseline
from reticular_diploid.experiments.prepare_public_1000g import clean_commit, public_output


def load_causal_curve(path, receipt):
    data = json.loads(path.read_text())
    if (data.get("schema_version") != "reticular-public-ntp-curve/v1"
            or data.get("evaluation_ownership") != "prefix_causal"
            or data.get("locked_test_used") is not False
            or not same_corpus(receipt, data["training_contract"]["corpus"])):
        raise ValueError("Publication figures require verified prefix-only likelihoods")
    rows = data["records"]
    if not rows or any(row["validation"]["people"] != receipt["validation_people"] for row in rows):
        raise ValueError("Expected complete validation likelihood records")
    return data["training_contract"], rows


def build(args):
    commit = clean_commit()
    public_output(args.output)
    if args.output.exists():
        raise ValueError("Figure attempt already exists")
    receipt, canonical, _ = load_corpus(args.corpus)
    if receipt["tokenizer_sha256"] != "692c032232df5667ab1323b30f10b1db9ffad105f42cef3a02280bc758e2f352":
        raise ValueError("This figure describes the accepted block589 tokenizer")
    curve_path = args.ntp_curve
    contract, curve = load_causal_curve(curve_path, receipt)
    unigram, bigram = unigram_baseline(receipt, canonical), bigram_baseline(receipt, canonical)
    lengths = np.diff(canonical["target_indptr"])
    if np.any(canonical["participant_split"] == 2):
        raise ValueError("Development figures must exclude the locked test")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "reticular-public-diploid-v1"})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), layout="constrained")
    colors = ["#0072B2", "#D55E00"]
    statistics = {}
    for code, name, color in [(0, "Training", colors[0]), (1, "Validation", colors[1])]:
        values = np.sort(lengths[canonical["participant_split"] == code])
        axes[0].step(values, np.arange(1, len(values) + 1) / len(values), where="post",
                     label=f"{name} (n={len(values)})", color=color, linewidth=1.9)
        statistics[name.lower()] = {"mean": float(values.mean()), "median": float(np.median(values)),
                                    "p95": float(np.quantile(values, .95)), "max": int(values.max())}
    axes[0].set(xlabel="Content tokens per person", ylabel="Cumulative fraction of people",
                title="A  Frozen diploid representation", ylim=(0, 1.02))
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].grid(alpha=.18)
    axes[1].plot([r["target_tokens_seen"] / 1e6 for r in curve],
                 [r["validation"]["bits_per_person"] / 1000 for r in curve],
                 "o-", color=colors[0], label=f"{contract['parameter_count'] / 1e6:.2f}M Kimi-derived model", markersize=4)
    axes[1].axhline(unigram["bits_per_person"] / 1000, color="#777777", linestyle=":", label="Training unigram")
    axes[1].axhline(bigram["bits_per_person"] / 1000, color=colors[1], linestyle="--", label="Training bigram")
    axes[1].set(xlabel="Training prediction targets processed (millions)",
                ylabel="Validation NLL (kbits per person)", title="B  Prefix-only next-token prediction")
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].grid(alpha=.18)
    fig.suptitle("Public 1000 Genomes: chr22 block 589", fontweight="bold", fontsize=13)
    args.output.mkdir(parents=True)
    for suffix in ["png", "svg"]:
        fig.savefig(args.output / ("tokenization-ntp." + suffix), dpi=220,
                    metadata={"Date": None} if suffix == "svg" else None)
    plt.close(fig)
    summary = {"schema_version": "reticular-public-figure/v1", "code_commit": commit,
               "corpus": receipt, "ntp_contract": contract, "curve_sha256": sha256(curve_path),
               "lengths": statistics, "training_unigram": unigram, "training_bigram": bigram,
               "locked_test_used": False,
               "caption": "82,401 allele-specific variants across 2,756,558 bp (GRCh38). Lengths exclude fixed BOS/direction/EOS framing (+3). Training exposures include overlapping windows; prefix-only validation scores every content target and true EOS once. Context selection depends only on the observed prefix. NLL is a likelihood statistic, not observed compressed-file size; shared dictionary, reference and model storage are additional. The 48 test people are excluded."}
    (args.output / "figure-data.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"figures_created": True, "locked_test_used": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["corpus", "ntp-curve", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
