"""Download the exact public demo inputs with checksums and resumable ranges."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import shutil
import struct
import subprocess
import time
import urllib.request
import zlib
from pathlib import Path

from reticular_diploid.data.frozen_diploid import sha256
from reticular_diploid.experiments.prepare_public_1000g import (
    PANEL_URL,
    PEDIGREE_URL,
    PINS,
    RECIPE,
    VCF_URL,
    public_output,
    vcf_url,
)

REFERENCE_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz"
REFERENCE_GZIP_SHA256 = "05f9d97d6fbfd08a44ca45b50837ca2ae9c471f35ba79dffec04d2cb5eaaf695"
INDEX_SHA256 = "016cbb66398337629e4b06de238703a17bbfbc25f48f7f660876e8968e2da4de"


def checked_asset(url, destination, expected_sha256):
    if destination.exists():
        if destination.is_symlink() or sha256(destination) != expected_sha256:
            raise ValueError("Existing public asset differs; preserve and diagnose it")
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        raise ValueError("Partial asset exists; preserve it and use a new attempt directory")
    with urllib.request.urlopen(url, timeout=120) as source, partial.open("xb") as out:
        shutil.copyfileobj(source, out)
    if sha256(partial) != expected_sha256:
        raise ValueError("Downloaded public asset checksum differs")
    partial.rename(destination)


def ranged_vcf(destination, *, size, expected_sha256, workers=8, url=VCF_URL):
    if not 1 <= workers <= 8:
        raise ValueError("Use between one and eight download workers")
    if destination.exists():
        if destination.is_symlink() or sha256(destination) != expected_sha256:
            raise ValueError("Existing VCF checksum differs")
        return
    chunks = destination.parent / "vcf-byte-ranges"
    chunks.mkdir(exist_ok=True)
    chunk_size = 8 * 1024 * 1024
    number = (size + chunk_size - 1) // chunk_size

    def fetch(index):
        first, last = index * chunk_size, min(size, (index + 1) * chunk_size) - 1
        path = chunks / f"{index:04d}"
        if path.exists():
            if path.is_symlink() or path.stat().st_size != last - first + 1:
                raise ValueError("Existing byte range differs")
            return index
        request = urllib.request.Request(url, headers={"Range": f"bytes={first}-{last}"})
        with urllib.request.urlopen(request, timeout=120) as source:
            if (source.status != 206
                    or source.headers.get("Content-Range") != f"bytes {first}-{last}/{size}"):
                raise ValueError("Server did not honor the exact byte range")
            payload = source.read()
        if len(payload) != last - first + 1:
            raise ValueError("Truncated public byte range")
        with path.open("xb") as handle:
            handle.write(payload)
        return index

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, j) for j in range(number)]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            print(json.dumps({"download_ranges_complete": completed, "total": number}), flush=True)
    partial = destination.with_suffix(destination.suffix + ".assembled")
    if partial.exists():
        if sha256(partial) != expected_sha256:
            raise ValueError("Existing assembled VCF differs")
    else:
        with partial.open("xb") as out:
            for j in range(number):
                with (chunks / f"{j:04d}").open("rb") as source:
                    shutil.copyfileobj(source, out)
    if sha256(partial) != expected_sha256:
        raise ValueError("Complete public VCF checksum differs")
    partial.rename(destination)


def tabix_byte_ranges(index, interval, size):
    """Physical BGZF coverage for a single-contig VCF TBI query, plus header/EOF."""
    raw = gzip.decompress(Path(index).read_bytes())
    if raw[:4] != b"TBI\x01":
        raise ValueError("Expected a public TBI index")
    header = struct.unpack_from("<8i", raw, 4)
    if header[:2] != (1, 2):
        raise ValueError("Expected a single-contig VCF TBI")
    names = raw[36:36 + header[-1]].rstrip(b"\0").decode()
    if names != "chr" + interval["chromosome"]:
        raise ValueError("Source TBI chromosome differs")
    offset = 36 + header[-1]
    number = struct.unpack_from("<i", raw, offset)[0]
    offset += 4
    bins = {}
    for _ in range(number):
        bin_id, count = struct.unpack_from("<Ii", raw, offset)
        offset += 8
        bins[bin_id] = [struct.unpack_from("<QQ", raw, offset + 16 * j) for j in range(count)]
        offset += 16 * count
    count = struct.unpack_from("<i", raw, offset)[0]
    offset += 4
    linear = struct.unpack_from("<" + "Q" * count, raw, offset)
    beg, end = interval["start"] - 1, interval["end"] - 1
    if not 0 <= beg <= end < 1 << 29 or beg >> 14 >= len(linear):
        raise ValueError("Query outside indexed chromosome")
    selected = [0]
    for shift, base in [(26, 1), (23, 9), (20, 73), (17, 585), (14, 4681)]:
        selected.extend(range(base + (beg >> shift), base + (end >> shift) + 1))
    minimum = linear[beg >> 14]
    ranges = [(0, min(size, 1 << 20)), (max(0, size - 65536), size)]
    for bin_id in selected:
        for first, last in bins.get(bin_id, []):
            if last > minimum:
                ranges.append((max(first, minimum) >> 16, min(size, (last >> 16) + 65536)))
    merged = []
    for first, last in sorted(ranges):
        if not 0 <= first < last <= size:
            raise ValueError("Invalid physical source range")
        if merged and first <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], last))
        else:
            merged.append((first, last))
    return merged


def verified_bgzf_blocks(payload):
    """Return only complete CRC-verified BGZF blocks, excluding edge fragments."""
    first = payload.find(b"\x1f\x8b\x08\x04")
    if first < 0:
        raise ValueError("No complete BGZF header in response")
    offset, count = first, 0
    while offset + 18 <= len(payload):
        if payload[offset:offset + 4] != b"\x1f\x8b\x08\x04" or payload[offset + 12:offset + 16] != b"BC\x02\x00":
            raise ValueError("Invalid BGZF header in response")
        length = struct.unpack_from("<H", payload, offset + 16)[0] + 1
        if not 26 <= length <= 65536:
            raise ValueError("Invalid BGZF block size")
        if offset + length > len(payload):
            break
        try:
            zlib.decompress(payload[offset:offset + length], 31)
        except zlib.error as error:
            raise ValueError("Public BGZF response failed CRC/decompression") from error
        offset += length
        count += 1
    if not count:
        raise ValueError("No complete verified BGZF block in response")
    return first, offset


def cached_bgzf_source(directory, *, source_url, index, interval, size, workers=8):
    """Bounded verified HTTP ranges; preserve original offsets in a sparse cache.

    The cache is explicitly incomplete and must only be queried with this index
    and interval. It is never certified as a complete source chromosome.
    """
    ranges = tabix_byte_ranges(index, interval, size)
    directory.mkdir(parents=True, exist_ok=True)
    contract = {"source_url": source_url, "index_sha256": sha256(index),
                "interval": interval, "source_size": size, "ranges": [list(r) for r in ranges],
                "bgzf_crc_verified": True, "read_ahead_bytes": 65536}
    receipt = directory / "coverage.json"
    if receipt.exists() and json.loads(receipt.read_text()) != contract:
        raise ValueError("Existing sparse-cache contract differs")
    if not receipt.exists():
        receipt.write_text(json.dumps(contract, indent=2) + "\n")
    requests = [(first, min(size, min(last, first + 2 * 1024 * 1024) + 65536)) for a, last in ranges
                for first in range(a, last, 2 * 1024 * 1024)]
    started = time.time()

    def fetch(pair):
        first, stop = pair
        path = directory / f"{first}-{stop}.bgzf-range"
        if path.exists():
            if path.is_symlink() or path.stat().st_size != stop - first:
                raise ValueError("Existing cached byte range differs")
            verified_bgzf_blocks(path.read_bytes())
            return path
        for attempt in range(3):
            if time.time() - started > 900:
                raise ValueError("Public range acquisition exceeded wall budget")
            url = source_url if not attempt else source_url + f"?reticular_range_retry={attempt}"
            request = urllib.request.Request(url, headers={"Range": f"bytes={first}-{stop - 1}",
                                                          "Accept-Encoding": "identity", "Cache-Control": "no-cache"})
            try:
                with urllib.request.urlopen(request, timeout=60) as source:
                    if source.status != 206 or source.headers.get("Content-Range") != f"bytes {first}-{stop - 1}/{size}":
                        raise ValueError("Exact public byte range required")
                    payload = source.read()
                if len(payload) != stop - first:
                    raise ValueError("Truncated public byte range")
                try:
                    verified_bgzf_blocks(payload)
                except ValueError:
                    bad = directory / (path.name + ".bad-" + hashlib.sha256(payload).hexdigest())
                    if not bad.exists():
                        bad.write_bytes(payload)
                    raise
                with path.open("xb") as out:
                    out.write(payload)
                return path
            except (OSError, TimeoutError, ValueError):
                if attempt == 2:
                    raise
        raise ValueError("Public range acquisition failed")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, pair) for pair in requests]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            print(json.dumps({"verified_ranges": completed, "total": len(requests)}), flush=True)
    sparse = directory / "incomplete-source.bgzf"
    if sparse.exists():
        raise ValueError("Sparse cache already assembled; preserve the previous attempt")
    with sparse.open("xb") as out:
        out.truncate(size)
        for first, stop in requests:
            payload = (directory / f"{first}-{stop}.bgzf-range").read_bytes()
            begin, end = verified_bgzf_blocks(payload)
            out.seek(first + begin)
            out.write(payload[begin:end])
    return sparse


def checked_region(destination, *, source_url, index, interval, expected_sha256, bcftools, backend="bcftools", source_size=None):
    """Reproduce a frozen regional VCF without downloading the whole chromosome."""
    if backend not in {"bcftools", "pysam", "cached_bgzf"}:
        raise ValueError("Unsupported public regional extraction backend")
    if destination.exists():
        if destination.is_symlink() or sha256(destination) != expected_sha256:
            raise ValueError("Existing regional VCF checksum differs")
    else:
        partial = destination.with_suffix(destination.suffix + ".partial")
        if partial.exists():
            raise ValueError("Regional extraction partial exists; preserve the failed attempt")
        region = f"chr{interval['chromosome']}:{interval['start']}-{interval['end']}"
        if backend == "cached_bgzf":
            source_url = str(cached_bgzf_source(destination.parent / "source-byte-cache",
                source_url=source_url, index=index, interval=interval, size=source_size))
        if backend == "pysam":
            import pysam

            with pysam.VariantFile(source_url, index_filename=str(index)) as source:
                with pysam.VariantFile(str(partial), "wz", header=source.header) as out:
                    for record in source.fetch("chr" + interval["chromosome"],
                                               interval["start"] - 1, interval["end"]):
                        if interval["start"] <= record.pos <= interval["end"]:
                            out.write(record)
        else:
            subprocess.run([bcftools, "view", "--no-version", "--regions-overlap", "0",
                            "-r", region, "-Oz", "-o", str(partial),
                            source_url + "##idx##" + str(index)], check=True, timeout=1200)
        if sha256(partial) != expected_sha256:
            raise ValueError("Extracted regional VCF checksum differs")
        partial.rename(destination)
    local_index = Path(str(destination) + ".tbi")
    if local_index.is_symlink():
        raise ValueError("Regional VCF index cannot be symlinked")
    if not local_index.exists():
        subprocess.run([bcftools, "index", "-t", str(destination)], check=True)


def download(output, config, workers=8, bcftools="bcftools"):
    public_output(output)
    output.mkdir(parents=True, exist_ok=True)
    recipe = json.loads(config.read_text())
    chromosome = recipe["interval"]["chromosome"]
    source_url = vcf_url(chromosome)
    name = "chr" + chromosome
    mode = recipe.get("acquisition_mode", "whole_chromosome")
    if mode not in {"whole_chromosome", "indexed_region"}:
        raise ValueError("Unsupported public acquisition mode")
    vcf = output / (name + (".region.vcf.gz" if mode == "indexed_region" else ".vcf.gz"))
    if mode == "whole_chromosome":
        ranged_vcf(vcf, size=recipe["vcf_size_bytes"],
                   expected_sha256=recipe["vcf_sha256"], workers=workers, url=source_url)
    assets = [
        (source_url + ".tbi", name + ".vcf.gz.tbi", recipe.get("index_sha256", INDEX_SHA256)),
        (PANEL_URL, "panel.tsv", PINS["panel"]),
        (PEDIGREE_URL, "pedigree.txt", PINS["pedigree"]),
        (REFERENCE_URL.replace("chr22", name), name + ".fa.gz",
         recipe.get("reference_gzip_sha256", REFERENCE_GZIP_SHA256)),
    ]
    for url, filename, checksum in assets:
        checked_asset(url, output / filename, checksum)
    reference = output / (name + ".fa")
    if not reference.exists():
        with gzip.open(output / (name + ".fa.gz"), "rb") as source, reference.open("xb") as out:
            shutil.copyfileobj(source, out)
    if reference.is_symlink() or sha256(reference) != recipe["reference_fasta_sha256"]:
        raise ValueError("Decompressed reference checksum differs")
    if mode == "indexed_region":
        checked_region(vcf, source_url=source_url, index=output / (name + ".vcf.gz.tbi"),
                       interval=recipe["interval"], expected_sha256=recipe["vcf_sha256"], bcftools=bcftools,
                       backend=recipe.get("extraction_backend", "bcftools"),
                       source_size=recipe.get("whole_source_size_bytes"))
    else:
        # Publisher MD5 checks the entire source only in whole-chromosome mode.
        digest = hashlib.md5()
        with vcf.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != recipe["vcf_publisher_md5"]:
            raise ValueError("Publisher VCF MD5 differs")
    receipt = {"public_data_only": True, "recipe_sha256": sha256(config),
               "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                         for p in [vcf, reference, *[output / a[1] for a in assets]]}}
    if mode == "indexed_region":
        receipt.update(acquisition_mode=mode, whole_source_md5_verified=False,
                       source_url=source_url, publisher_vcf_md5=recipe["vcf_publisher_md5"],
                       extraction_backend=recipe.get("extraction_backend", "bcftools"),
                       source_size=recipe.get("whole_source_size_bytes"))
        receipt["files"][vcf.name + ".tbi"] = {
            "sha256": sha256(Path(str(vcf) + ".tbi")), "bytes": Path(str(vcf) + ".tbi").stat().st_size}
    path = output / "download.json"
    if not path.exists():
        path.write_text(json.dumps(receipt, indent=2) + "\n")
    elif json.loads(path.read_text()) != receipt:
        raise ValueError("Existing download provenance differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=RECIPE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bcftools", default="bcftools")
    args = parser.parse_args()
    download(args.output, args.config, args.workers, args.bcftools)


if __name__ == "__main__":
    main()
