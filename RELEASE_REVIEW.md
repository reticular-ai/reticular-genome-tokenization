# Public demo release review

Release scope: block 589 public-data frozen vocabulary, inference codec, preprocessing and reproduction code, synthetic tests, and experimental modeling recipes. Publication was explicitly requested by the repository owner on 2026-11-09. This is a research demo, not a finished foundation-model release.

The release is a clean snapshot of standalone source f35bcb3 plus release documentation, smoke example and CI. Private repository history is not published. The 8,192-ID vocabulary SHA-256 is `692c032232df5667ab1323b30f10b1db9ffad105f42cef3a02280bc758e2f352`. Its metadata asserts public-data-only provenance, training-only fitting, and no locked-test use. All published payload hashes are recorded in SOURCE_MANIFEST.json; the manifest excludes itself.

The source snapshot contains no fitting/candidate-generation implementation, raw participant arrays, participant predictions, trained checkpoints, credentials, or controlled bucket locators. The only genomic binary payloads are the selected public dictionary and public allele catalog. NPZ loading prohibits pickled objects. Selected motif identities and deterministic inference ordering are intentionally disclosed.

Historical development audits reported exact reconstruction and private/public sequence parity for all 455 development individuals. Release checks verify the shipped hashes, synthetic round-trips and source tests. A new full download, cohort rebuild, GPU retraining or reserved-test evaluation is not claimed by this publication review. Experimental NTP scoring limitations remain explicit in RESEARCH.md. The private discovery procedure cannot be independently rerun from this repository; frozen-dictionary inference can.

The retained MIT notice covers existing code; NOTICE states the release license for Reticular additions and the frozen artifact. External runtime dependencies and genomic source data retain their own terms. The provider acknowledgement statement was reviewed for the release. This is a repository content review, not certification by a data provider.

Release verification: 85 CPU tests passed on Python 3.12, including the shipped dictionary round-trip and manifest checks. Three macOS numerical-runtime warnings arose in an existing synthetic ridge-baseline test; its assertions passed. No warning was suppressed. The standalone example reconstructed 82,401 synthetic dosages exactly. CI runs the tokenizer, public-data preprocessing and acquisition tests on Python 3.11 and 3.12.
