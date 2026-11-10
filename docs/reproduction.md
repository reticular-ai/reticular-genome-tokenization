# Reproduce the block 589 benchmark

Run commands from the repository root. Tokenization and preprocessing run on CPU; no GPU, cloud account, private Reticular repository, or phenotype data is required. The source chromosome VCF is approximately 446 MB; allow additional space for the reference, download ranges, normalized calls and dosage arrays. Full preparation time depends on network speed and CPU. A synthetic smoke test is available in the main README before undertaking the download.

## 1. Download, prepare and encode


Use Python 3.11 or 3.12 for CPU data work. Install [bcftools 1.24](https://github.com/samtools/bcftools/releases/tag/1.24) on `PATH`; its official release archive bundles HTSlib. Commit any local modifications before measurements; entry points record and require clean source.

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
```

Outputs must be outside every Git checkout. Attempts are immutable: use a new output location or an explicit exact resume, rather than replacing failed results. The approximately 446 MB chromosome VCF is downloaded with publisher MD5 and SHA-256 checks. Input release, coordinate convention, normalization, symbolic-allele exclusion, missing-call failure, and split rules are frozen in `configs/data.json`. No cohort MAF filter changes the externally fixed allele universe.

The frozen input hashes are in `configs/data.json`. Compare your benchmark output with `results/tokenization.json`: scientific metrics and input hashes should agree; your code commit will differ. The published aggregate was rerun with the released encoder against the cached, checksum-verified public cohort. This is distinct from a fresh network download.

`tokenizer/vocab.npz` stores selected motif allele ranks, dosage states, and deterministic inference tie ordering. `variants.tsv.gz` maps normalized chromosome/position/REF/ALT identities to ranks. `config.json` specifies all IDs and residual grammar. Motifs combine by maximum dosage; adjacent R1/R2 pairs add remaining dosage and identify its allele rank. The encoder consumes dosages 0/1/2, never haplotype phase. `tokenizer/codec.py` also runs independently with NumPy alone. Exhaustive synthetic tests bind canonical sequence identity and reject malformed residuals or dosage overflow.



## 2. Reproduce motif frequencies

The corpus command above stores each development individual's complete content stream once. The frequency counter uses these canonical streams, not the overlapping language-model windows.

```bash
python -m reticular_diploid.analysis.motif_frequency \
  --corpus "$DEMO_DATA/corpus" --output "$DEMO_DATA/motif-frequency"
```

Compare the two output CSVs with `results/motif-frequency-training.csv` and `results/motif-frequency-validation.csv`. Token IDs are motif dosage states: D1 and D2 have separate counts. Residual and framing tokens are excluded; unobserved motif types remain as zero rows. No test streams are counted.

## 3. Regenerate the figures

The released figures can be regenerated directly from the small aggregate files in Git, without downloading genotypes:

```bash
python -m pip install '.[figures]'
python examples/plot_tokenization.py --results results --output /tmp/diploid-figures
```

## Common problems

| Symptom | Resolution |
|---|---|
| `bcftools` not found | Install the linked version and ensure its executable is on `PATH`. |
| Source tree is dirty | Commit your changes before a scientific run; output receipts bind the code commit. |
| Output already exists | Use a new directory; failed attempts are preserved rather than overwritten. |
| Hash or reference mismatch | Verify the source release and the reference against `configs/data.json`; do not bypass the check. |
| Output inside Git rejected | Keep `DEMO_DATA` outside the checkout; generated files include public participant-level data. |

The optional training recipes are in [RESEARCH.md](../RESEARCH.md). They require a separate GPU environment and are not needed to reproduce tokenization.
