"""Download the external evaluation data into data/external/ (plan step 0.4).

Fetches 3 files from their official sources, checks each against a pinned
SHA-256 fingerprint, and unpacks them:

- BIRD dev (the original package): questions, gold SQL, and the 11 SQLite
  databases, from the BIRD site.
- BIRD dev 2025-11-06: the BIRD team's cleaned questions, hints, and gold SQL
  for the same databases, from Hugging Face.
- TrustSQL: questions and SQLite databases for ATIS, Advising, EHRSQL, and
  Spider, from the authors' Google Drive link. Its EHRSQL portion runs on a
  94-patient MIMIC-IV demo database, which is open access (ODbL v1.0), so no
  PhysioNet credentialed account is needed.

data/external/ is gitignored, so none of this data is republished from the
repo. A fingerprint mismatch stops the script: it means the upstream file
changed, and results would no longer be comparable.

Run from the repo root (about 470 MB):

    uv run python data/download_external.py
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "external"
DOWNLOADS = ROOT / "_downloads"

SOURCES = [
    {
        "name": "BIRD dev",
        "url": "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip",
        "file": "bird_dev.zip",
        "sha256": "cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630",
        "unpack_to": "bird",
    },
    {
        "name": "BIRD dev 2025-11-06 (cleaned)",
        "url": "https://huggingface.co/datasets/birdsql/bird_sql_dev_20251106"
        "/resolve/main/data/dev_20251106-00000-of-00001.json",
        "file": "bird_dev_20251106.json",
        "sha256": "ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb",
        "unpack_to": "bird",
    },
    {
        "name": "TrustSQL",
        "url": "https://drive.usercontent.google.com/download"
        "?id=19IpLSc2QncO2273E8z-lvU2ME9wEHIan&export=download&confirm=t",
        "file": "trustsql.zip",
        "sha256": "d5ab772244bba13247669d19039258b0e68d6870ea6996c11ccf6463627d2aff",
        "unpack_to": "trustsql",
    },
]

CHUNK = 1 << 20  # 1 MiB


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    """Stream a URL to disk, writing to a temp file first so a failed download leaves nothing behind."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "abstain-sql-downloader"})
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        next_report = 0
        while block := resp.read(CHUNK):
            out.write(block)
            done += len(block)
            if done >= next_report:
                pct = f" ({100 * done / total:.0f}%)" if total else ""
                print(f"    {done / 1e6:.0f} MB{pct}", flush=True)
                next_report += 50 * 1_000_000
    tmp.replace(dest)


def safe_extract(zip_path: Path, dest: Path) -> None:
    """Unzip, refusing any entry that would land outside dest, and skipping macOS metadata."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            name = member.filename
            if name.startswith("__MACOSX/") or "/__MACOSX/" in name or name.endswith(".DS_Store"):
                continue
            target = (dest / name).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"{zip_path.name}: entry escapes target folder: {name}")
            z.extract(member, dest)


def unpack_nested(dest: Path) -> None:
    """Some packages ship their databases as a zip inside the zip; unpack those beside themselves."""
    for inner in sorted(dest.rglob("*.zip")):
        marker = inner.with_suffix(".unpacked")
        if marker.exists():
            continue
        print(f"  unpacking nested {inner.relative_to(ROOT)}")
        safe_extract(inner, inner.parent)
        marker.touch()


def main() -> int:
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    unpinned = []
    for src in SOURCES:
        path = DOWNLOADS / src["file"]
        print(f"{src['name']}: {src['url']}")
        if path.exists() and src["sha256"] and sha256_of(path) == src["sha256"]:
            print("  already downloaded, fingerprint matches")
        else:
            fetch(src["url"], path)

        digest = sha256_of(path)
        if src["sha256"] is None:
            unpinned.append((src["name"], digest))
            print(f"  sha256 {digest} (NOT PINNED YET)")
        elif digest != src["sha256"]:
            print(f"  FINGERPRINT MISMATCH: expected {src['sha256']}, got {digest}", file=sys.stderr)
            print("  The upstream file changed. Stopping.", file=sys.stderr)
            return 1
        else:
            print(f"  sha256 {digest} (matches)")

        dest = ROOT / src["unpack_to"]
        if path.suffix == ".zip":
            print(f"  unpacking to {dest.relative_to(ROOT.parent)}")
            safe_extract(path, dest)
            unpack_nested(dest)
        else:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest / path.name)

    if unpinned:
        print("\nPin these fingerprints in SOURCES:", file=sys.stderr)
        for name, digest in unpinned:
            print(f"  {name}: {digest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
