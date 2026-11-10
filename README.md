# Reticular diploid tokenization demo

**Lossless, phase-independent tokenization of human genetic variation using a frozen motif vocabulary.**

This public research demo encodes allele-specific diploid dosages across a 2.76 Mb region of chromosome 22 using approximately **1,452 tokens per validation individual**. The supplied dictionary was fitted exclusively on public 1000 Genomes training individuals. You can download the source data, encode genotypes, and verify exact reconstruction without private infrastructure.

## What is demonstrated?

| Block 589 development benchmark | Result |
|---|---:|
| GRCh38 interval (1-based, inclusive) | chr22:43,317,956–46,074,513 |
| Normalized allele-specific variant universe | 82,401 |
| Training / validation / reserved test people | 400 / 55 / 48 |
| Vocabulary IDs / selected motifs | 8,192 / 3,662 |
| Validation tokens: mean / median | 1,451.9 / 1,427 |
| Validation tokens: p95 / maximum | 1,762.5 / 1,875 |
| Validation residual-only baseline, mean | 10,705.2 |
| Validation motif-token share | 32.42% |
| Exact reconstruction on development individuals | 455 / 455 |

Lengths count content tokens; model framing adds three tokens. The reduction is approximately **7.37× relative to this residual-only grammar**, whose entries use two tokens per nonreference allele. This is not compression of an entire 2.76 Mb nucleotide string. The reference, allele catalog, and dictionary are shared side information. Phase, excluded symbolic structural variants, and VCF annotations are not represented.

Known pedigree components stay within a partition. Vocabulary fitting uses training individuals only; validation informed development. The 48 reserved test individuals have not been scored. These are development results for a small European-ancestry cohort, not a genome-wide or population-general benchmark.

## Try the codec without downloading genomes

Requires Python 3.11 or 3.12. From a clone of this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python examples/roundtrip.py
```

The example uses a synthetic dosage vector and the real frozen vocabulary. It checks the dictionary hash, encodes, decodes, and asserts exact equality. It is a correctness example, not a compression benchmark on a person.

## Reproduce the public-data benchmark


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
```

Outputs must be outside every Git checkout. Attempts are immutable: use a new output location or an explicit exact resume, rather than replacing failed results. The approximately 446 MB chromosome VCF is downloaded with publisher MD5 and SHA-256 checks. Input release, coordinate convention, normalization, symbolic-allele exclusion, missing-call failure, and split rules are frozen in `configs/data.json`. No cohort MAF filter changes the externally fixed allele universe.

The standalone pipeline has rebuilt byte-identical cohort, canonical streams, windows, teacher scores, and teacher weights from these public inputs on macOS; an independent Linux preparation also matched. Expected payload hashes are available in the frozen configurations and result receipts.

`tokenizer/vocab.npz` stores selected motif allele ranks, dosage states, and deterministic inference tie ordering. `variants.tsv.gz` maps normalized chromosome/position/REF/ALT identities to ranks. `config.json` specifies all IDs and residual grammar. Motifs combine by maximum dosage; adjacent R1/R2 pairs add remaining dosage and identify its allele rank. The encoder consumes dosages 0/1/2, never haplotype phase. `tokenizer/codec.py` also runs independently with NumPy alone. Exhaustive synthetic tests bind canonical sequence identity and reject malformed residuals or dosage overflow.


## How the representation works

A selected motif identifies multiple allele-specific positions. D1 and D2 token states specify dosage coverage; overlapping motifs combine by maximum dosage. Adjacent R1/R2 residual pairs encode any remaining dosage and its allele rank. Decoding reconstructs every callable 0/1/2 dosage exactly. Because the input is dosage rather than phased homologs, changing haplotype phase cannot change the input representation.

The frozen motifs, token meanings, variant mapping, and deterministic inference code are public. **Vocabulary discovery, candidate generation, and fitting optimization remain proprietary.** Reproducing the published encoder does not require refitting the dictionary.

## Tests and experimental modeling

```bash
python -m pip install '.[test]'
python -m pytest -q tests/test_frozen_diploid.py tests/test_release_bundle.py
```

[Research notes and training recipes](RESEARCH.md) preserve the causal-model and synthetic-teacher experiments, including negative results. Those extensions are experimental: archived window-owned NTP scores are not normalized complete-stream likelihoods, corrected measurements are not packaged here, and no trained model weights are included. The historical large architecture is a reference configuration, not a public-data training result. Additional-region configurations are included for development; this release supplies only the block 589 dictionary.

See [release review](RELEASE_REVIEW.md) for provenance and verification scope, [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) for file hashes, and [NOTICE](NOTICE) / [LICENSE](LICENSE) for attribution and licensing.

## Data acknowledgements

The data are from [IGSR's high-coverage 1000 Genomes collection](https://www.internationalgenome.org/data-portal/data-collection/1000genomes_30x/). Cite [Byrska-Bishop et al., Cell 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9439720/). Samples originated from the NIGMS Human Genetic Cell Repository at the Coriell Institute for Medical Research; sequencing was generated at the New York Genome Center with NHGRI support (3UM1HG008901-03S1 and 3UM1HG008901-04S2). We acknowledge the sample donors. See the [provider acknowledgement statement](https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/20200526_1000G_2504plus698_high_cov_data_reuse_README.txt). Source data retain their providers' terms.
