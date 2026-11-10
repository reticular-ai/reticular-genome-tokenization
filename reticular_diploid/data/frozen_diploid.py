"""Inference-only, phase-independent diploid motif codec. No vocabulary fitting.

This module is self-contained apart from NumPy and may be exported separately.
Content IDs exclude BOS, direction, EOS and padding. Motifs combine by maximum;
an adjacent R1/R2 pair adds the remaining dosage at one allele-specific rank.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = "reticular-frozen-diploid/v1"
SEMANTICS = "max_motif_plus_residual/v1"
ARRAY_FIELDS = ("positions", "refs", "alts", "motif_indptr", "motif_ranks",
                "motif_nodes", "motif_priorities")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class FrozenDiploidTokenizer:
    positions: np.ndarray
    refs: np.ndarray
    alts: np.ndarray
    motif_indptr: np.ndarray
    motif_ranks: np.ndarray
    motif_nodes: np.ndarray
    motif_priorities: np.ndarray
    chromosome: str = "22"

    def __post_init__(self):
        if self.chromosome not in {str(c) for c in range(1, 23)}:
            raise ValueError("Expected an autosomal chromosome string")
        integer_names = ("positions", "motif_indptr", "motif_ranks", "motif_nodes", "motif_priorities")
        for name in integer_names:
            value = np.asarray(getattr(self, name))
            if value.ndim != 1 or value.dtype.kind not in "iu":
                raise ValueError(f"Expected one-dimensional integer {name}")
            setattr(self, name, value.astype(np.int64, copy=True))
        self.refs, self.alts = np.asarray(self.refs), np.asarray(self.alts)
        n, k = len(self.positions), len(self.motif_nodes)
        if (not n or self.refs.shape != (n,) or self.alts.shape != (n,)
                or self.refs.dtype.kind != "U" or self.alts.dtype.kind != "U"
                or np.any(self.positions <= 0) or np.any(np.diff(self.positions) < 0)
                or len(set(zip(self.positions, self.refs, self.alts, strict=True))) != n
                or any(not r or not a or r == a or set(r + a) - set("ACGT")
                       for r, a in zip(self.refs, self.alts, strict=True))):
            raise ValueError("Invalid ordered allele-specific variant universe")
        if (self.motif_indptr.shape != (k + 1,) or self.motif_indptr[0] != 0
                or self.motif_indptr[-1] != len(self.motif_ranks)
                or np.any(np.diff(self.motif_indptr) < 2)
                or self.motif_priorities.shape != (k,)
                or len(set(self.motif_nodes)) != k or np.any(self.motif_nodes < 0)
                or np.any(self.motif_priorities < 0)
                or np.any(self.motif_ranks < 0) or np.any(self.motif_ranks >= n)):
            raise ValueError("Invalid selected motif CSR")
        self.motifs = [self.motif_ranks[a:b] for a, b in
                       zip(self.motif_indptr[:-1], self.motif_indptr[1:], strict=True)]
        if any(np.any(np.diff(ranks) <= 0) for ranks in self.motifs):
            raise ValueError("Motif ranks must be unique and sorted")
        choices = [(a, math.ceil(n / a)) for a in range(1, math.isqrt(n) + 2)]
        a, b = min(choices, key=lambda x: (sum(x), abs(x[0] - x[1]), max(x), x[0]))
        self.first, self.second = min(a, b), max(a, b)
        self.motif_entries = 2 * k
        self.content_size = self.motif_entries + 2 * self.first + self.second
        self.masks = [sum(1 << int(v) for v in ranks) for ranks in self.motifs]
        self.coordinates = [float(np.median(self.positions[ranks])) for ranks in self.motifs]

    @property
    def selected(self):
        """Inference layout adapter for the existing event-aligned window builder."""
        return range(self.motif_entries)

    @property
    def special_tokens(self):
        return {name: self.content_size + offset for offset, name in
                enumerate(["pad", "bos", "sep", "eos", "forward", "reverse"])}

    def encode_genotype(self, dosages):
        g = np.asarray(dosages)
        if (g.shape != self.positions.shape or g.dtype.kind not in "iub"
                or np.any(g < 0) or np.any(g > 2)):
            raise ValueError("Expected callable allele-specific integer dosages in [0, 2]")
        one = int.from_bytes(np.packbits(g >= 1, bitorder="little").tobytes(), "little")
        two = int.from_bytes(np.packbits(g == 2, bitorder="little").tobytes(), "little")
        heterozygous = one & ~two
        completed = covered_one = covered_two = 0
        heap, groups = [], []

        def gain(token):
            target = two if token % 2 else heterozygous
            return 2 * (self.masks[token // 2] & target & ~completed).bit_count() - 1

        def push(token):
            m = token // 2
            heapq.heappush(heap, (-gain(token), -int(self.motif_priorities[m]),
                                 int(self.motif_nodes[m]), token % 2, token))

        for token in range(self.motif_entries):
            eligible = two if token % 2 else one
            mask = self.masks[token // 2]
            if mask & eligible == mask:
                push(token)
        while heap:
            negative, _, _, _, token = heapq.heappop(heap)
            if negative >= 0:
                break
            if -negative != gain(token):
                push(token)
                continue
            mask = self.masks[token // 2]
            covered_one |= mask
            if token % 2:
                covered_two |= mask
            completed = (covered_one & heterozygous) | covered_two
            groups.append((self.coordinates[token // 2], 0,
                           int(self.motif_nodes[token // 2]), [token]))
        coverage = np.zeros(len(g), dtype=np.uint8)
        for layer in (covered_one, covered_two):
            raw = np.frombuffer(layer.to_bytes((len(g) + 7) // 8, "little"), dtype=np.uint8)
            coverage += np.unpackbits(raw, bitorder="little", count=len(g))
        for rank in np.flatnonzero(g - coverage):
            dosage = int(g[rank] - coverage[rank])
            groups.append((float(self.positions[rank]), 1, int(rank), [
                self.motif_entries + (dosage - 1) * self.first + int(rank) // self.second,
                self.motif_entries + 2 * self.first + int(rank) % self.second]))
        groups.sort()
        return np.asarray([token for *_, tokens in groups for token in tokens], dtype=np.int32)

    def decode(self, ids):
        ids = np.asarray(ids)
        if ids.ndim != 1 or ids.dtype.kind not in "iu":
            raise ValueError("Expected one-dimensional integer content tokens")
        coverage = np.zeros(len(self.positions), dtype=np.uint8)
        residual = np.zeros_like(coverage)
        seen, j = set(), 0
        while j < len(ids):
            token = int(ids[j])
            if not 0 <= token < self.content_size:
                raise ValueError("Token outside content vocabulary")
            if token < self.motif_entries:
                motif = token // 2
                if motif in seen:
                    raise ValueError("Duplicate motif identity")
                seen.add(motif)
                ranks = self.motifs[motif]
                coverage[ranks] = np.maximum(coverage[ranks], token % 2 + 1)
                j += 1
            else:
                code = token - self.motif_entries
                if code >= 2 * self.first or j + 1 >= len(ids):
                    raise ValueError("Malformed residual pair")
                fine = int(ids[j + 1]) - self.motif_entries - 2 * self.first
                rank = (code % self.first) * self.second + fine
                if not 0 <= fine < self.second or rank >= len(residual) or residual[rank]:
                    raise ValueError("Invalid or duplicate residual address")
                residual[rank] = code // self.first + 1
                j += 2
        dosage = coverage.astype(np.int16) + residual
        if np.any(dosage > 2):
            raise ValueError("Decoded dosage exceeds two")
        return dosage.astype(np.uint8)

    def save(self, path: Path, provenance: dict):
        path = Path(path)
        metadata = {"schema_version": SCHEMA, "coverage_semantics": SEMANTICS,
                    "genome_build": "GRCh38", "chromosome": self.chromosome,
                    "residual_dosage": "remaining", "motif_anchor": "median",
                    "residual_order": "genomic", "provenance": provenance}
        with path.open("xb") as handle:
            np.savez_compressed(handle, metadata_json=json.dumps(metadata, sort_keys=True),
                                **{name: getattr(self, name) for name in ARRAY_FIELDS})

    @classmethod
    def load(cls, path: Path, expected_sha256: str, *, expected_chromosome=None):
        path = Path(path)
        if path.is_symlink() or sha256(path) != expected_sha256:
            raise ValueError("Frozen tokenizer checksum differs or input is symlinked")
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"]))
            expected = {"schema_version": SCHEMA, "coverage_semantics": SEMANTICS,
                        "genome_build": "GRCh38",
                        "residual_dosage": "remaining", "motif_anchor": "median",
                        "residual_order": "genomic"}
            if any(metadata.get(k) != v for k, v in expected.items()):
                raise ValueError("Unsupported frozen tokenizer semantics")
            if expected_chromosome is not None and metadata.get("chromosome") != expected_chromosome:
                raise ValueError("Frozen tokenizer chromosome differs")
            if set(archive.files) != {*ARRAY_FIELDS, "metadata_json"}:
                raise ValueError("Unexpected frozen tokenizer payload fields")
            return cls(**{name: archive[name] for name in ARRAY_FIELDS},
                       chromosome=metadata["chromosome"])
