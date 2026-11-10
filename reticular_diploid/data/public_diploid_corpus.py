"""Verify public development-corpus identity without importing a training stack."""

import json

import numpy as np

from reticular_diploid.data.frozen_diploid import sha256


def load_corpus(root):
    receipt = json.loads((root / "corpus.json").read_text())
    if receipt.get("public_data_only") is not True or receipt.get("locked_test_used") is not False:
        raise ValueError("Expected public development-only corpus")
    result = []
    for name in ["canonical", "windows"]:
        path = root / (name + ".npz")
        if sha256(path) != receipt[name + "_sha256"]:
            raise ValueError("Canonical corpus or window checksum differs")
        with np.load(path, allow_pickle=False) as z:
            result.append({k: z[k] for k in z.files})
    return receipt, *result


def same_corpus(left, right):
    """Independent source commits may reproduce identical bound data and semantics."""
    return ({k: v for k, v in left.items() if k != "code_commit"}
            == {k: v for k, v in right.items() if k != "code_commit"})
