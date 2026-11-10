import json

import numpy as np
import pytest

from reticular_diploid.data.frozen_diploid import sha256
from reticular_diploid.data.public_diploid_corpus import load_corpus, same_corpus


def test_independent_corpus_rebuild_accepts_only_identical_data_and_semantics():
    original = {"canonical_sha256": "a", "windows_sha256": "b", "context": 256,
                "special_tokens": {"eos": 11}, "code_commit": "original-source"}
    assert same_corpus(original, {**original, "code_commit": "independent-source"})
    assert not same_corpus(original, {**original, "canonical_sha256": "other-bytes"})
    assert not same_corpus(original, {**original, "context": 128})
    assert not same_corpus(original, {**original, "special_tokens": {"eos": 12}})


def test_loader_rejects_corruption_and_nonpublic_provenance(tmp_path):
    receipt = {"public_data_only": True, "locked_test_used": False}
    for name in ["canonical", "windows"]:
        path = tmp_path / (name + ".npz")
        np.savez_compressed(path, synthetic=np.arange(3))
        receipt[name + "_sha256"] = sha256(path)
    manifest = tmp_path / "corpus.json"
    manifest.write_text(json.dumps(receipt))
    actual, canonical, windows = load_corpus(tmp_path)
    assert actual == receipt
    np.testing.assert_array_equal(canonical["synthetic"], windows["synthetic"])
    (tmp_path / "canonical.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_corpus(tmp_path)
    manifest.write_text(json.dumps({**receipt, "locked_test_used": True}))
    with pytest.raises(ValueError, match="development-only"):
        load_corpus(tmp_path)
