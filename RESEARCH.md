# Reticular public diploid genome demonstration

Historical development record. This document preserves the original experimental results and their limitations; model checkpoints are not included in this release. The experiments below use public 1000 Genomes data or independently synthetic fixtures.

This repository applies a frozen, phase-independent tokenizer to allele-specific diploid dosages across GRCh38 chr22:43,317,956–46,074,513 inclusive (2,756,558 bases). It includes causal language-model training and deterministic synthetic additive-teacher experiments. Vocabulary discovery and fitting algorithms are proprietary and intentionally excluded. Reproduction starts from the supplied frozen dictionary.

## Measured development results

The public release provides 82,401 normalized sequence-allele identities. Published EUR panel membership and complete pedigree connected components give 400 training, 55 validation, and 48 reserved test people. Related people never cross these partitions. Cryptic relatedness has not been excluded by a genome-wide kinship analysis.

| Content tokens/person | Training | Validation |
|---|---:|---:|
| Mean | 1,208.805 | 1,451.909 |
| Median | 1,188.5 | 1,427 |
| p95 | 1,463.25 | 1,762.5 |
| Maximum | 1,985 | 1,875 |
| Residual-only mean | 10,650.59 | 10,705.236 |

All 455 development individuals reconstruct exactly. Add three framing tokens per complete model stream. The dictionary has 3,662 motifs and 8,192 total token IDs; 32.42% of validation content tokens are motifs. This is a compact representation of genetic differences relative to a shared reference and fixed allele catalog. It does not encode an entire nucleotide string, phase, excluded symbolic SVs, or VCF annotations.

The tested model is a 2,749,676-parameter, geometry-free Kimi-derived ablation: four width-128 layers, KDA/dense hybrid attention, latent MoE, Engram memory, and prefix-only genomic coordinates. The historical 729M configuration, not run on this public cohort, is preserved in `configs/historical-reference.json`; its graph geometry is not part of this public model.

| Validation next-token model | CE, nats/target | Perplexity | Legacy window-owned score, bits/person |
|---|---:|---:|---:|
| Training unigram | 7.682494 | 2,170.026 | 16,103.313 |
| Fixed training bigram | 4.577294 | 97.251 | 9,594.488 |
| Small Kimi-derived model | 3.252425 | 25.853 | 6,817.423 |

The model trained for 2,000 updates and 3,933,456 scored training-target exposures. Context-256/stride-128 evaluation owns each canonical validation content target and true EOS exactly once: 79,910 targets across 55 people. These legacy scores are reproducible window-owned diagnostics. A subsequent synthetic audit found that event-aligned ownership can depend on the next token type, so these values must not be interpreted as a normalized complete-stream likelihood. The prefix-only scorer is implemented and tested; corrected measurements are not included in this archived table. These are not measured entropy-coded file sizes. Dictionary, reference, and model storage are additional costs. The original run used 0.1544 worker GPU-hours on one existing A100, including cold evaluation compilation. Matched-runtime uninterrupted replay reproduced all compared training-state values exactly; cross-hardware bitwise identity is not claimed.

The first teacher comparison used 400 training people, seed 20261023, 1,000 updates, and 2,000 supervised person draws per arm. Each entry is validation-selected R² on 55 people.

| Effect variants | Scratch | NTP → distillation | NTP + joint distillation | Direct genotype ridge |
|---|---:|---:|---:|---:|
| 16 | 0.0350 | 0.2135 | 0.3233 | 0.2300 |
| 128 | 0.0773 | 0.0601 | 0.0609 | 0.4474 |
| 1,024 | 0.1749 | 0.1460 | 0.1196 | 0.6611 |

This seed supports an A1 improvement and provides negative controls against a general pretraining advantage. This historical table does not include replicated sample-efficiency comparisons or final reserved-test evaluation. Synthetic targets are fixed additive dosage scores, not actual GWAS or SBayesRC posterior estimates. Recovering them cannot establish clinical prediction or improvement over the teacher. The supplied-weight genotype oracle reproduces scores exactly; causal-locus ridge is a separate privileged control.

The primary distillation endpoint is held-out teacher-score recovery. A benefit from NTP initialization is a secondary empirical question; it is not required for this demonstration to succeed. Current recovery is moderate and must not be described as exact neural reconstruction of the teacher.

## Reproduce the data and tokenizer

Use Python 3.11 or 3.12 for CPU data work. Install [bcftools 1.24](https://github.com/samtools/bcftools/releases/tag/1.24) on `PATH`; its official release archive bundles HTSlib. The measured Linux training environment used Python 3.11.15, Torch 2.10.0+cu128, and flash-linear-attention 0.5.2. Commit any local modifications before measurements; entry points record and require clean source.

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
DEMO_DATA="$HOME/reticular-diploid-demo-data"
VOCAB_SHA=692c032232df5667ab1323b30f10b1db9ffad105f42cef3a02280bc758e2f352

python -m reticular_diploid.experiments.download_public_1000g --output "$DEMO_DATA/inputs"
python -m reticular_diploid.experiments.prepare_public_1000g \
  --vcf "$DEMO_DATA/inputs/chr22.vcf.gz" \
  --vcf-sha256 91aeaf0d1e805cf4223c5cbf12e22afc28db1709054c42bd668337219f8fddcc \
  --pedigree "$DEMO_DATA/inputs/pedigree.txt" --panel "$DEMO_DATA/inputs/panel.tsv" \
  --reference "$DEMO_DATA/inputs/chr22.fa" --output "$DEMO_DATA/prepared"
python -m reticular_diploid.experiments.benchmark_public_diploid \
  --prepared "$DEMO_DATA/prepared" --tokenizer tokenizer/vocab.npz \
  --tokenizer-sha256 "$VOCAB_SHA" --output "$DEMO_DATA/tokenization.json"
python -m reticular_diploid.experiments.build_public_diploid_corpus \
  --prepared "$DEMO_DATA/prepared" --tokenizer tokenizer/vocab.npz \
  --tokenizer-sha256 "$VOCAB_SHA" --output "$DEMO_DATA/corpus"
python -m reticular_diploid.experiments.public_additive_teachers \
  --prepared "$DEMO_DATA/prepared" --tokenizer tokenizer/vocab.npz \
  --tokenizer-sha256 "$VOCAB_SHA" --output "$DEMO_DATA/teachers"
python -m reticular_diploid.experiments.public_teacher_baselines \
  --prepared "$DEMO_DATA/prepared" --corpus "$DEMO_DATA/corpus" \
  --teachers "$DEMO_DATA/teachers" --output "$DEMO_DATA/baselines"
```

Outputs must be outside every Git checkout. Attempts are immutable: use a new output location or an explicit exact resume, rather than replacing failed results. The approximately 446 MB chromosome VCF is downloaded with publisher MD5 and SHA-256 checks. Input release, coordinate convention, normalization, symbolic-allele exclusion, missing-call failure, and split rules are frozen in `configs/data.json`. No cohort MAF filter changes the externally fixed allele universe.

The standalone pipeline has rebuilt byte-identical cohort, canonical streams, windows, teacher scores, and teacher weights from these public inputs on macOS; an independent Linux preparation also matched. Expected payload hashes are available in the frozen configurations and result receipts.

`tokenizer/vocab.npz` stores selected motif allele ranks, dosage states, and deterministic inference tie ordering. `variants.tsv.gz` maps normalized chromosome/position/REF/ALT identities to ranks. `config.json` specifies all IDs and residual grammar. Motifs combine by maximum dosage; adjacent R1/R2 pairs add remaining dosage and identify its allele rank. The encoder consumes dosages 0/1/2, never haplotype phase. `tokenizer/codec.py` also runs independently with NumPy alone. Exhaustive synthetic tests bind canonical sequence identity and reject malformed residuals or dosage overflow.

## Train and evaluate

For the tested Linux CUDA stack, install `requirements-linux-cuda.txt` in an isolated environment. It pins external dependencies; the repository commit pins project source. GPU kernels require compatible NVIDIA hardware. Cold compilation can take several minutes. The recipes impose time and memory limits; no new GPU allocation is requested by this code.

```bash
python -m pytest -q tests
python -m reticular_diploid.experiments.canary_public_diploid \
  --config configs/ntp.json --output "$DEMO_DATA/cuda-canary"
python -m reticular_diploid.experiments.train_public_diploid \
  --config configs/ntp.json --corpus "$DEMO_DATA/corpus" --output "$DEMO_DATA/pretrain"
python -m reticular_diploid.experiments.run_public_distillation \
  --config configs/distillation.json --corpus "$DEMO_DATA/corpus" \
  --teachers "$DEMO_DATA/teachers" --ntp-run "$DEMO_DATA/pretrain" \
  --training-people 400 --seeds 20261023 --output "$DEMO_DATA/students-v1"
python -m reticular_diploid.experiments.run_public_distillation \
  --config configs/distillation-v2.json --corpus "$DEMO_DATA/corpus" \
  --teachers "$DEMO_DATA/teachers" --ntp-run "$DEMO_DATA/pretrain" \
  --exclude-primary-full --output "$DEMO_DATA/students-v2"
```

The v2 recipe changes checkpoint storage only: `latest.pt` and `best.pt` are mutable training states, with complete learning curves and an append-only selection-hash history. The primary v1 batch contains nine trials; the remaining v2 grid contains 45 paired trials across two seeds and 32/128/400 training people. Allow roughly three worker GPU-hours for the full teacher grid, plus NTP and cold compilation. `--training-people` and `--seeds` select smaller predeclared batches. The v2 aggregate runtime cap is 2.5 hours for its remaining trials; a lower `--maximum-hours` is honored. A failed attempt and its logs are preserved.

Each run writes `selected-checkpoint.json` with the selected path, SHA-256, step, and validation metrics. Pass that path and SHA to `evaluate_public_diploid --checkpoint ... --checkpoint-sha256 ... --corpus ... --output ...`, or to `evaluate_public_distillation` with the additional `--teachers ...` argument. The NTP evaluator defaults to `--ownership prefix_causal` and also requires `--tokenizer tokenizer/vocab.npz`; use `--ownership legacy` only to reproduce the archived diagnostic. The latter independently reports supervised-training and validation fit. Public-trained checkpoints are not distributed in this release.

To regenerate the corrected learning curve, use `evaluate_public_diploid --checkpoint-directory "$DEMO_DATA/pretrain" --corpus "$DEMO_DATA/corpus" --tokenizer tokenizer/vocab.npz --output "$DEMO_DATA/causal-ntp-curve.json"`. This rescores preserved checkpoints with prefix-only ownership and retains the original checkpoint selection.

`configs/final-evaluation.json` freezes the complete 54-student and 72-baseline comparison before test scoring. First generate both baseline seeds with `public_teacher_baselines --seed 20261023` and `--seed 20261024`, using separate output directories. Then run `evaluate_public_test` with `--prepared`, `--corpus`, `--teachers`, `--tokenizer`, `--ntp-checkpoint`, `--protocol configs/final-evaluation.json`, both `--student-roots`, both `--baseline-roots`, a new `--output`, and `--canary-validation`. Reserved-test evaluation requires the same arguments, a separate output, and `--validation-canary` pointing to that successful canary instead of `--canary-validation`. The evaluator verifies every selected checkpoint and penalty, writes a selection certificate before loading test genotypes, preserves all people and alleles, and computes paired whole-family bootstrap intervals. It aborts if the selections differ or the canary is absent. Participant predictions stay outside Git.

For local weight review, `package_public_students --selection-certificate ... --protocol configs/final-evaluation.json --student-roots ... --output ...` creates inference checkpoints for every selected student. It preserves and fingerprints all model tensors, binds the completed learning curves and validation-only choices, and excludes optimizer/RNG state and minibatch logs. These weights support evaluation rather than exact optimizer resume. Retain the original NTP parent checkpoint identified in `bundle.json` for the complete-grid evaluator. Packaging writes no network destination and does not authorize publication.

Teacher selection uses only training MAF and fixed region/frequency/motif strata. Effects use frozen PCG64 seeds and a normal distribution; targets use training mean and population standard deviation. Natural LD is retained, so effect counts do not equal independent signal counts. For 32/128 labeled people, NTP still uses all 400 training genotypes; the PCA128 baseline matches this unlabeled access. Frozen target normalization uses the full training partition for every label count. All students see the entire region in context 2,048 without cropping. Scratch, pretrained, and joint arms share sample subsets, draws, head initialization, supervised updates, and validation cadence. Joint training adds auxiliary NTP with weight 0.1. Pretraining cost and extra joint FLOPs must be reported separately.

## Scope, provenance, and acknowledgements

No All of Us participant data, fitted dictionaries, graph features, or model weights enter this pipeline. The historical interval identifies a region, not a common mutation universe. Selected public motifs are intentionally visible; candidate generation, vocabulary objectives, fitting code, and candidate archives are absent.

This small, single-region EUR cohort limits statistical power and generalization. Validation has informed development, and the reserved test is kept outside selection. Ancestry structure and LD can explain part of synthetic-score recovery. No phenotype or clinical claim is made. The fixed source allele catalog reflects the full external release; vocabulary fitting and teacher selection are training-only.

Data come from [IGSR's high-coverage collection](https://www.internationalgenome.org/data-portal/data-collection/1000genomes_30x/). Cite [Byrska-Bishop et al., Cell 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9439720/), and acknowledge NYGC, NHGRI, Coriell, and the sample donors according to the [source reuse statement](https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/20200526_1000G_2504plus698_high_cov_data_reuse_README.txt). Dataset reuse terms remain those of its providers. Existing code copyright notices are retained in `LICENSE` and `NOTICE`; release scope and licensing are documented in NOTICE and RELEASE_REVIEW.md.
