"""Synthetic round-trip using the released dictionary; no participant data."""
from pathlib import Path
import runpy
import numpy as np

root = Path(__file__).resolve().parents[1]
codec = runpy.run_path(str(root / "tokenizer/codec.py"))
tokenizer = codec["FrozenDiploidTokenizer"].load(
    root / "tokenizer/vocab.npz",
    "692c032232df5667ab1323b30f10b1db9ffad105f42cef3a02280bc758e2f352",
)
genotype = np.zeros(len(tokenizer.positions), dtype=np.uint8)
genotype[tokenizer.motifs[0]] = 1
genotype[-1] = 2
tokens = tokenizer.encode_genotype(genotype)
np.testing.assert_array_equal(tokenizer.decode(tokens), genotype)
print(f"Exact synthetic round-trip: {len(genotype):,} allele dosages, {len(tokens)} tokens.")
