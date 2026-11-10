import numpy as np
import pytest

from reticular_diploid.experiments.prepare_public_1000g import (
    allele_dosages,
    family_splits,
    membership_sha256,
    public_output,
    read_dosages,
    vcf_url,
)


def test_unselected_relative_connects_siblings_and_order_is_irrelevant():
    rows = [{"sampleID": s, "fatherID": father, "motherID": "0"}
            for s, father in [("a", "parent"), ("b", "parent"), ("c", "0")]]
    codes, groups = family_splits(["a", "b", "c"], rows)
    reverse, reverse_groups = family_splits(["c", "b", "a"], rows[::-1])
    assert codes[0] == codes[1] and groups[0] == groups[1]
    np.testing.assert_array_equal(codes, reverse[::-1])
    np.testing.assert_array_equal(groups, reverse_groups[::-1])
    with pytest.raises(ValueError, match="pedigree"):
        family_splits(["missing"], rows)


def test_multiallelic_dosages_and_missing_calls():
    np.testing.assert_array_equal(allele_dosages((1, 2), 2), [1, 1])
    np.testing.assert_array_equal(allele_dosages((2, 2), 2), [0, 2])
    np.testing.assert_array_equal(allele_dosages((0, 0), 2), [0, 0])
    for gt in [(None, 0), (1,), (0, 3)]:
        with pytest.raises(ValueError):
            allele_dosages(gt, 2)


@pytest.mark.parametrize("chromosome", ["5", "6", "22"])
def test_allele_identity_sorting_and_phase_independence(tmp_path, chromosome):
    path = tmp_path / "synthetic.vcf"
    path.write_text(
        f'##fileformat=VCFv4.2\n##contig=<ID=chr{chromosome},length=181538259>\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\tb\n'
        f'chr{chromosome}\t100\t.\tA\tG,C\t.\tPASS\t.\tGT\t1|2\t2/1\n'
        f'chr{chromosome}\t200\t.\tG\tT\t.\tPASS\t.\tGT\t0|1\t1/0\n'
        f'chr{chromosome}\t300\t.\tG\t<DEL>\t.\tPASS\t.\tGT\t0|1\t1/0\n')
    variants, g, homologs, qc = read_dosages(path, ["a", "b"], 100, 300, chromosome)
    assert variants == [(100, "A", "C"), (100, "A", "G"), (200, "G", "T")]
    np.testing.assert_array_equal(g, [[1, 1, 1], [1, 1, 1]])
    np.testing.assert_array_equal(homologs.sum(axis=1), g)
    assert qc["symbolic_alleles_excluded"] == 1


def test_artifacts_cannot_be_written_in_git_or_aou(tmp_path):
    (tmp_path / ".git").mkdir()
    with pytest.raises(ValueError, match="outside Git"):
        public_output(tmp_path / "result")
    with pytest.raises(ValueError, match="AoU"):
        public_output(tmp_path.parent / "workspace-bucket" / "public")


def test_region_download_urls_and_frozen_membership():
    for chrom in ["5", "6", "22"]:
        assert f".chr{chrom}.filtered." in vcf_url(chrom)
    with pytest.raises(ValueError, match="chromosome"):
        vcf_url("chr6")
    baseline = membership_sha256(["a", "b"], [0, 1], ["a", "b"])
    assert baseline != membership_sha256(["a", "b"], [0, 2], ["a", "b"])
    assert baseline != membership_sha256(["b", "a"], [0, 1], ["b", "a"])
