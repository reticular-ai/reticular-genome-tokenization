"""Verify shipped artifacts independently of external data or services."""
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]


def test_release_manifest():
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    for name, record in manifest["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == record["sha256"], name


def test_shipped_codec_roundtrip():
    runpy.run_path(str(ROOT / "examples/roundtrip.py"), run_name="__main__")
