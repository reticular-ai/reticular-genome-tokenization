"""Prepare publicly sourced, family-grouped autosomal allele-specific dosage data.

All genotype artifacts are written outside Git. This entry point never reads AoU
inputs. Sequence normalization is delegated to bcftools with a pinned reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np

from reticular_diploid.data.frozen_diploid import sha256

RELEASE = "20220422_3202_phased_SNV_INDEL_SV"
BASE = ("https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
        "1000G_2504_high_coverage/working/")
CHROMOSOME_LENGTHS = {"5": 181538259, "6": 170805979, "22": 50818468}


def vcf_url(chromosome):
    if chromosome not in CHROMOSOME_LENGTHS:
        raise ValueError("Unsupported public demonstration chromosome")
    return BASE + RELEASE + f"/1kGP_high_coverage_Illumina.chr{chromosome}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"


VCF_URL = vcf_url("22")
PEDIGREE_URL = BASE + "1kGP.3202_samples.pedigree_info.txt"
PANEL_URL = ("https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
             "integrated_call_samples_v3.20130502.ALL.panel")
PINS = {
    "pedigree": "0f9cfcbadf85fc10a39af88a969c1ae46c8c8efa15f7d5288ef59be82a7a20f9",
    "panel": "b4023dc6ee2d62ee89c8d4d347db4d348e65518d66d346574cdae7a4bbd76858",
    "reference": "ce3ee1ca39356238f7aee438a40a88b4f1b9d80b316b263e16fb12402212d10f",
}
RECIPE = Path(__file__).resolve().parents[2] / "configs/data.json"


def clean_commit():
    root = Path(__file__).resolve().parents[2]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True)
    if dirty.strip():
        raise ValueError("Commit coherent code and configuration before scientific measurement")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def public_output(path: Path):
    """Never place public participant files in any Git checkout or AoU mount."""
    absolute = Path(path).absolute()
    if any(p.is_symlink() for p in (absolute, *absolute.parents)):
        raise ValueError("Symlinked output paths are not accepted")
    for parent in (absolute, *absolute.parents):
        if (parent / ".git").exists() or parent.name == "workspace-bucket":
            raise ValueError("Public-data outputs must be outside Git and AoU storage")


def family_splits(samples, pedigree_rows, seed=20261023):
    """Group entire pedigree connected components, including unselected relatives.

    A fixed SHA-256 maps each component to 80/10/10 train/validation/test bins.
    It does not force exact sizes or inspect genotypes. Test membership is frozen
    here; downstream development entry points must never encode/score those rows.
    """
    samples = list(samples)
    if not samples or len(set(samples)) != len(samples):
        raise ValueError("Expected a nonempty unique sample roster")
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    listed = set()
    for row in pedigree_rows:
        sample = row["sampleID"]
        if sample in listed:
            raise ValueError("Duplicate pedigree sample")
        listed.add(sample)
        find(sample)
        for key in ("fatherID", "motherID"):
            relative = row[key]
            if relative not in {"0", ".", "-9", "NA"}:
                union(sample, relative)
    if not set(samples) <= listed:
        raise ValueError("Every selected public sample needs pedigree metadata")
    roots = [find(sample) for sample in samples]
    codes = []
    for root in roots:
        value = int.from_bytes(hashlib.sha256(f"{seed}:{root}".encode()).digest()[:8], "big")
        codes.append(0 if value < (1 << 64) * 8 // 10 else
                     1 if value < (1 << 64) * 9 // 10 else 2)
    return np.asarray(codes, dtype=np.uint8), np.asarray(roots)


def allele_dosages(gt, alternate_count):
    if len(gt) != 2 or any(a is None or not 0 <= a <= alternate_count for a in gt):
        raise ValueError("Non-diploid, missing or invalid GT; no reference imputation is allowed")
    return np.asarray([sum(a == index for a in gt) for index in range(1, alternate_count + 1)],
                      dtype=np.uint8)


def membership_sha256(samples, split, groups):
    records = [[str(s), int(r), str(g)] for s, r, g in zip(samples, split, groups, strict=True)]
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def read_dosages(vcf_path: Path, samples, start: int, end: int, chromosome="22"):
    import pysam

    variants, columns, genotype_columns, seen = [], [], [], set()
    counts = {"records_read": 0, "symbolic_alleles_excluded": 0, "outside_interval": 0}
    with pysam.VariantFile(str(vcf_path)) as source:
        source.subset_samples(samples)
        if set(source.header.samples) != set(samples):
            raise ValueError("VCF sample membership differs")
        for record in source:
            counts["records_read"] += 1
            if record.contig != "chr" + chromosome or not start <= record.pos <= end:
                counts["outside_interval"] += 1
                continue
            alleles = record.alleles
            if len(alleles) < 2 or any(not a or set(a) - set("ACGT") for a in alleles):
                counts["symbolic_alleles_excluded"] += 1
                continue
            gts = [record.samples[s]["GT"] for s in samples]
            dosage = np.stack([allele_dosages(gt, len(alleles) - 1) for gt in gts])
            for alt_index, alt in enumerate(alleles[1:], 1):
                key = (record.pos, record.ref, alt)
                if key in seen:
                    raise ValueError("Duplicate normalized allele identity; resolve explicitly")
                seen.add(key)
                variants.append(key)
                columns.append(dosage[:, alt_index - 1])
                genotype_columns.append(np.asarray([[a == alt_index for a in gt] for gt in gts],
                                                   dtype=np.uint8))
    if not variants:
        raise ValueError("No supported variants remain")
    order = sorted(range(len(variants)), key=lambda j: variants[j])
    variants = [variants[j] for j in order]
    return (variants, np.stack([columns[j] for j in order], axis=1),
            np.stack([genotype_columns[j] for j in order], axis=2), counts)


def prepare(args):
    commit = clean_commit()
    recipe = json.loads(args.config.read_text())
    chromosome = recipe["interval"]["chromosome"]
    source_url = vcf_url(chromosome)
    pins = {**PINS, "reference": recipe["reference_fasta_sha256"]}
    if args.vcf_sha256 != recipe["vcf_sha256"]:
        raise ValueError("VCF must match the frozen independently public release")
    public_output(args.output)
    if args.output.exists():
        raise ValueError("Preparation output exists; do not replace immutable attempts")
    for key, path in [("pedigree", args.pedigree), ("panel", args.panel),
                      ("reference", args.reference)]:
        if path.is_symlink() or sha256(path) != pins[key]:
            raise ValueError(f"Pinned public {key} checksum differs")
    if args.vcf.is_symlink() or sha256(args.vcf) != args.vcf_sha256:
        raise ValueError("Public VCF checksum differs")
    with args.panel.open() as handle:
        panel = list(csv.DictReader(handle, delimiter="\t"))
    samples = sorted(row["sample"] for row in panel if row["super_pop"] == "EUR")
    with args.pedigree.open() as handle:
        pedigree = list(csv.DictReader(handle, delimiter=" ", skipinitialspace=True))
    split, groups = family_splits(samples, pedigree, args.seed)
    if set(split) != {0, 1, 2}:
        raise ValueError("Frozen family split must include all three roles")
    membership = membership_sha256(samples, split, groups)
    if recipe.get("membership_sha256", membership) != membership:
        raise ValueError("Public sample/split membership differs from the frozen three-region selection")
    if not 1 <= args.start <= args.end <= CHROMOSOME_LENGTHS[chromosome]:
        raise ValueError("Invalid GRCh38 chromosome interval")
    args.output.mkdir(parents=True)
    roster = args.output / "public-eur-samples.txt"
    roster.write_text("\n".join(samples) + "\n")
    selected, normalized = args.output / "selected.vcf.gz", args.output / "normalized.vcf.gz"
    commands = [
        [args.bcftools, "view", "--regions-overlap", "0", "-r",
         f"chr{chromosome}:{args.start}-{args.end}", "-S", str(roster), "-f", "PASS,.",
         "-Oz", "-o", str(selected), str(args.vcf)],
        [args.bcftools, "norm", "-f", str(args.reference), "-c", "e", "-m", "-any",
         "--multi-overlaps", "0", "-Oz", "-o", str(normalized), str(selected)],
    ]
    for command in commands:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    variants, dosages, homologs, qc = read_dosages(normalized, samples, args.start, args.end, chromosome)
    positions, refs, alts = map(np.asarray, zip(*variants, strict=True))
    # No genotype-dependent variant filter: retain release-fixed allele identities,
    # including alleles monomorphic in training. Such alleles always use residuals.
    if not np.array_equal(homologs.sum(axis=1), dosages):
        raise ValueError("Allele-specific genotype projection failed")
    np.savez_compressed(args.output / "cohort.npz", positions=positions, refs=refs, alts=alts,
                        dosages=dosages, homologs=homologs, sample_ids=np.asarray(samples),
                        split=split, family_groups=groups)
    receipt = {"schema_version": "reticular-public-1000g/v1", "public_data_only": True,
               "release": RELEASE, "source_url": source_url,
               "input_sha256": {"vcf": args.vcf_sha256, **pins},
               "interval": {"chromosome": chromosome, "build": "GRCh38", "start": args.start,
                            "end": args.end, "coordinate_system": "1-based-inclusive"},
               "samples": len(samples), "variants": len(variants),
               "split_counts": {name: int(np.count_nonzero(split == i)) for i, name in
                                enumerate(["train", "validation", "test"])},
               "membership_sha256": membership, "split_seed": args.seed, "split_policy": "SHA256 whole-pedigree components 80/10/10",
               "relatedness_limit": "published pedigree only; no inferred genome-wide kinship audit",
               "variant_policy": "PASS or unfiltered; normalized split sequence alleles; no MAF filter",
               "missing_policy": "fail; never impute", "phase_used_for_encoding": False,
               "symbolic_variants_supported": False, "qc": qc,
               "bcftools_version": subprocess.check_output([args.bcftools, "--version"], text=True).splitlines()[0],
               "cohort_sha256": sha256(args.output / "cohort.npz"),
               "normalized_vcf_sha256": sha256(normalized),
               "code_commit": commit, "recipe_sha256": sha256(args.config),
               "runtime": {name: importlib.metadata.version(name) for name in ["numpy", "pysam"]}}
    if recipe.get("acquisition_mode") == "indexed_region":
        receipt["source_acquisition"] = {"mode": "indexed_region",
            "whole_source_publisher_md5": recipe["vcf_publisher_md5"],
            "whole_source_md5_verified": False, "source_index_sha256": recipe["index_sha256"],
            "extraction_backend": recipe.get("extraction_backend", "bcftools")}
    (args.output / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return {key: receipt[key] for key in ("samples", "variants", "split_counts", "qc")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vcf", "pedigree", "panel", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--vcf-sha256", required=True)
    parser.add_argument("--config", type=Path, default=RECIPE)
    parser.add_argument("--bcftools", default="bcftools")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.start, args.end = config["interval"]["start"], config["interval"]["end"]
    args.seed = config["split"]["seed"]
    print(json.dumps(prepare(args), sort_keys=True))


if __name__ == "__main__":
    main()
