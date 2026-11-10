# Contributing

Reproduction reports, portability fixes and small correctness improvements are welcome. Open an issue with your platform, Python and dependency versions, repository commit, command and the aggregate result or error. Do not include participant-level inputs, predictions, credentials or private data locations.

Keep generated genomic files outside the checkout. Tests should use synthetic fixtures and require no cloud account or GPU. Preserve the frozen dictionary, scientific split rules and existing result records; propose changes to those explicitly rather than silently replacing them.

Run the relevant focused tests before opening a pull request. CPU tests for the optional modeling modules additionally require PyTorch, SciPy and Matplotlib. A full CPU suite is distinct from a GPU training replication.

If changing a release file, update its SHA-256 in SOURCE_MANIFEST.json. Include the reason for the change and relevant verification in the PR. Published experimental numbers must remain tied to their original input and code provenance.

## Run the core checks

```bash
python -m pip install '.[test]'
python -m pytest -q tests/test_frozen_diploid.py tests/test_release_bundle.py tests/test_motif_frequency.py
```

CI additionally checks public preprocessing and acquisition on Python 3.11 and 3.12.
