import hashlib
import itertools

import numpy as np
import pytest

from reticular_diploid.data.frozen_diploid import FrozenDiploidTokenizer, sha256


def public_fixture():
    positions = np.array([100, 100, 200, 300, 400])
    return FrozenDiploidTokenizer(
        positions, np.array(["A", "A", "C", "T", "G"]),
        np.array(["C", "G", "T", "A", "C"]),
        np.array([0, 2, 5, 7]), np.array([0, 1, 0, 2, 3, 2, 4]),
        np.array([13, 7, 8]), np.array([40, 50, 50]))




def test_public_roundtrip_and_canonical_sequence_digest():
    public = public_fixture()
    digest = hashlib.sha256()
    for values in itertools.product(range(3), repeat=5):
        g = np.asarray(values, np.uint8)
        ids = public.encode_genotype(g)
        np.testing.assert_array_equal(public.decode(ids), g)
        digest.update(g.tobytes())
        digest.update(np.array([len(ids)], dtype="<i4").tobytes())
        digest.update(ids.astype("<i4").tobytes())
        # Reassign each heterozygous site's copy independently between homologs.
        first = np.minimum(g, 1)
        second = g - first
        swap = np.arange(len(g)) % 2 == 0
        left, right = np.where(swap, second, first), np.where(swap, first, second)
        np.testing.assert_array_equal(public.encode_genotype(left + right), ids)
    assert digest.hexdigest() == "afe867e8c2a6fbbdea49acf59f632e33b1d8b2b3d51981f9adc56d393f14e278"


def test_public_payload_checksums_and_immutable_save(tmp_path):
    public = public_fixture()
    path = tmp_path / "vocab.npz"
    public.save(path, {"source": "synthetic"})
    restored = FrozenDiploidTokenizer.load(path, sha256(path))
    g = np.array([1, 0, 2, 1, 0])
    np.testing.assert_array_equal(restored.encode_genotype(g), public.encode_genotype(g))
    with pytest.raises(ValueError, match="checksum"):
        FrozenDiploidTokenizer.load(path, "0" * 64)
    with pytest.raises(FileExistsError):
        public.save(path, {})




def test_malformed_tokens_and_uncallable_input_fail():
    public = public_fixture()
    r1, r2 = public.motif_entries, public.motif_entries + 2 * public.first
    for ids in [[0, 1], [r1], [r2], [-1], [public.content_size],
                [r1, r2, r1, r2], [1, r1, r2]]:
        with pytest.raises(ValueError):
            public.decode(np.array(ids))
    for g in [[0.0] * 5, [0, 0, 0, -1, 0], [0, 0, 3, 0, 0], [0] * 4]:
        with pytest.raises(ValueError):
            public.encode_genotype(np.array(g))


def test_reference_only_and_multiallelic_residuals():
    t = FrozenDiploidTokenizer(np.array([100, 100]), np.array(["A", "A"]),
                              np.array(["C", "G"]), np.array([0]),
                              np.array([], dtype=int), np.array([], dtype=int),
                              np.array([], dtype=int))
    for g in [np.array([0, 0]), np.array([1, 1]), np.array([0, 2])]:
        ids = t.encode_genotype(g)
        assert len(ids) == 2 * np.count_nonzero(g)
        np.testing.assert_array_equal(t.decode(ids), g)


def test_payload_validation():
    public = public_fixture()
    arrays = {name: getattr(public, name) for name in public.__dataclass_fields__}
    for name, value in [("motif_indptr", np.array([0, 1, 5, 7])),
                        ("motif_ranks", np.array([0, 0, 0, 2, 3, 2, 4])),
                        ("positions", np.array([100, 99, 200, 300, 400])),
                        ("motif_nodes", np.array([1, 1, 2]))]:
        with pytest.raises(ValueError):
            FrozenDiploidTokenizer(**{**arrays, name: value})


@pytest.mark.parametrize("chromosome", ["5", "6", "22"])
def test_chromosome_identity_survives_frozen_roundtrip(tmp_path, chromosome):
    codec = public_fixture()
    codec.chromosome = chromosome
    path = tmp_path / "synthetic.npz"
    codec.save(path, {"source": "synthetic"})
    restored = FrozenDiploidTokenizer.load(path, sha256(path), expected_chromosome=chromosome)
    assert restored.chromosome == chromosome
    with pytest.raises(ValueError, match="chromosome"):
        FrozenDiploidTokenizer.load(path, sha256(path), expected_chromosome="1")
    assert restored.special_tokens == codec.special_tokens
    np.testing.assert_array_equal(restored.encode_genotype(np.array([1, 0, 2, 1, 0])),
                                  codec.encode_genotype(np.array([1, 0, 2, 1, 0])))
