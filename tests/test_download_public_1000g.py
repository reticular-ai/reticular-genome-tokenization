import hashlib
import io

import pytest

from reticular_diploid.experiments.download_public_1000g import checked_asset, checked_region, ranged_vcf


class Response(io.BytesIO):
    status = 206

    def __init__(self, data):
        super().__init__(data)
        self.headers = {"Content-Range": f"bytes 0-{len(data)-1}/{len(data)}"}


def test_exact_ranges_assemble_and_verify_without_network(tmp_path, monkeypatch):
    data = b"clearly synthetic VCF fixture bytes"
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(data))
    path = tmp_path / "synthetic.vcf.gz"
    digest = hashlib.sha256(data).hexdigest()
    ranged_vcf(path, size=len(data), expected_sha256=digest, workers=1)
    assert path.read_bytes() == data
    ranged_vcf(path, size=len(data), expected_sha256=digest, workers=1)
    with pytest.raises(ValueError, match="checksum"):
        ranged_vcf(path, size=len(data), expected_sha256="0" * 64)


def test_server_ignoring_range_is_rejected(tmp_path, monkeypatch):
    response = Response(b"synthetic")
    response.status = 200
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: response)
    with pytest.raises(ValueError, match="exact byte range"):
        ranged_vcf(tmp_path / "synthetic.vcf.gz", size=9, expected_sha256="0" * 64, workers=1)


def test_failed_asset_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(b"synthetic"))
    path = tmp_path / "metadata.tsv"
    with pytest.raises(ValueError, match="checksum"):
        checked_asset("https://example.invalid/synthetic", path, "0" * 64)
    assert path.with_suffix(".tsv.partial").read_bytes() == b"synthetic"
    assert not path.exists()


def test_range_request_uses_requested_chromosome_url(tmp_path, monkeypatch):
    data = b"synthetic region source"
    url = "https://example.invalid/chr6.vcf.gz"
    def open_requested(request, **kwargs):
        assert request.full_url == url
        return Response(data)
    monkeypatch.setattr("urllib.request.urlopen", open_requested)
    ranged_vcf(tmp_path / "synthetic.vcf.gz", size=len(data),
               expected_sha256=hashlib.sha256(data).hexdigest(), workers=1, url=url)


def test_regional_extraction_checks_hash_and_uses_verified_local_index(tmp_path, monkeypatch):
    from pathlib import Path

    payload = b"synthetic regional VCF"
    source = "https://example.invalid/chr5.vcf.gz"
    index = tmp_path / "verified.tbi"
    index.write_bytes(b"synthetic index")
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if command[1] == "view":
            assert command[-1] == source + "##idx##" + str(index)
            assert command[command.index("-r") + 1] == "chr5:100-200"
            assert "--no-version" in command
            Path(command[command.index("-o") + 1]).write_bytes(payload)
        else:
            Path(command[-1] + ".tbi").write_bytes(b"synthetic local index")
    monkeypatch.setattr("subprocess.run", run)
    path = tmp_path / "regional.vcf.gz"
    kwargs = dict(source_url=source, index=index,
                  interval={"chromosome": "5", "start": 100, "end": 200},
                  expected_sha256=hashlib.sha256(payload).hexdigest(), bcftools="bcftools")
    checked_region(path, **kwargs)
    checked_region(path, **kwargs)
    assert len(commands) == 2 and path.read_bytes() == payload
    with pytest.raises(ValueError, match="checksum"):
        checked_region(path, **{**kwargs, "expected_sha256": "0" * 64})
    bad = tmp_path / "bad.vcf.gz"
    with pytest.raises(ValueError, match="checksum"):
        checked_region(bad, **{**kwargs, "expected_sha256": "0" * 64})
    assert not bad.exists() and bad.with_suffix(".gz.partial").exists()


def test_pysam_backend_keeps_only_anchor_positions_and_preserves_gt(tmp_path):
    import pysam

    header = pysam.VariantHeader()
    header.contigs.add("chr6", length=1000)
    header.formats.add("GT", 1, "String", "Synthetic GT")
    header.add_sample("synthetic")
    source = tmp_path / "source.vcf.gz"
    with pysam.VariantFile(str(source), "wz", header=header) as out:
        for position in [99, 100, 200, 201]:
            record = out.new_record(contig="chr6", start=position - 1, alleles=("A", "C"))
            record.samples["synthetic"]["GT"] = (1, 0)
            record.samples["synthetic"].phased = True
            out.write(record)
    pysam.tabix_index(str(source), preset="vcf")
    path = tmp_path / "region.vcf.gz"
    kwargs = dict(source_url=str(source), index=str(source) + ".tbi",
                  interval={"chromosome": "6", "start": 100, "end": 200},
                  expected_sha256="0" * 64, bcftools="unused", backend="pysam")
    with pytest.raises(ValueError, match="checksum"):
        checked_region(path, **kwargs)
    partial = path.with_suffix(".gz.partial")
    with pysam.VariantFile(str(partial)) as result:
        records = list(result)
        assert [r.pos for r in records] == [100, 200]
        assert all(r.samples["synthetic"]["GT"] == (1, 0) for r in records)
        assert all(r.samples["synthetic"].phased for r in records)


def test_tabix_ranges_preserve_physical_offsets_and_index_identity(tmp_path):
    import gzip
    import struct

    from reticular_diploid.experiments.download_public_1000g import tabix_byte_ranges

    size = 50 * 1024 * 1024
    first, last = 2 * 1024 * 1024, 3 * 1024 * 1024
    header = b"TBI\x01" + struct.pack("<8i", 1, 2, 1, 2, 0, 35, 0, 5) + b"chr6\0"
    bins = struct.pack("<iIiQQ", 1, 4681, 1, first << 16, last << 16)
    linear = struct.pack("<iQ", 1, first << 16)
    index = tmp_path / "synthetic.tbi"
    index.write_bytes(gzip.compress(header + bins + linear))
    interval = {"chromosome": "6", "start": 100, "end": 200}
    assert tabix_byte_ranges(index, interval, size) == [
        (0, 1 << 20), (first, last + 65536), (size - 65536, size)]
    with pytest.raises(ValueError, match="chromosome"):
        tabix_byte_ranges(index, {**interval, "chromosome": "5"}, size)


def test_bgzf_response_crc_rejects_corrupt_payload_and_handles_partial_edges(tmp_path):
    import pysam

    from reticular_diploid.experiments.download_public_1000g import verified_bgzf_blocks

    p = tmp_path / "synthetic.bgz"
    with pysam.BGZFile(str(p), "wb") as f:
        f.write(b"clearly synthetic public fixture\n" * 100)
    payload = p.read_bytes()
    assert verified_bgzf_blocks(payload) == (0, len(payload))
    assert verified_bgzf_blocks(b"edge" + payload) == (4, len(payload) + 4)
    bad = bytearray(payload)
    bad[30] ^= 8
    with pytest.raises(ValueError, match="CRC|decompression"):
        verified_bgzf_blocks(bytes(bad))
