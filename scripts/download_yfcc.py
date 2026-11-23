"""Download YFCC (YFCC100M-GEO100) eval images from the multimedia-commons S3 mirror.

    python scripts/download_yfcc.py [--db downloads/yfcc/yfcc100m_dataset.sqlite] [--out data/images/yfcc]

The original YFCC100M-GEO100 site needs a login, but every YFCC100M photo is mirrored at
s3://multimedia-commons/data/images/<h[:3]>/<h[3:6]>/<h>.jpg, where h is the MD5 of the
photo's Flickr download URL written byte by byte in hex *without* zero padding (so usually
<32 chars). Download URLs come from the YFCC100M metadata SQLite in the same bucket:
    aws s3 cp --no-sign-request s3://multimedia-commons/tools/etc/yfcc100m_dataset.sql \
        downloads/yfcc/yfcc100m_dataset.sqlite      (65 GB; delete after this script)
Images land at <out>/<class>/<photo_id>.jpg, matching LocBench's image refs.
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_locbench  # noqa: E402
from src.utils.progress import progress  # noqa: E402

MIRROR = "https://multimedia-commons.s3.amazonaws.com/data/images"


def mirror_hash(download_url: str) -> str:
    return "".join(f"{b:x}" for b in hashlib.md5(download_url.encode()).digest())


def fetch(url: str, out: Path) -> str | None:
    if out.exists():
        return None
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        return repr(e)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.rename(out)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="downloads/yfcc/yfcc100m_dataset.sqlite")
    ap.add_argument("--out", default="data/images/yfcc")
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    refs = [str(r) for r in load_locbench("yfcc").image_refs]
    ids = {int(Path(r).stem) for r in refs}
    db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    urls = {}
    for pid in progress(sorted(ids), desc="lookup", unit="id"):
        row = db.execute("SELECT downloadurl FROM yfcc100m_dataset WHERE photoid = ?", (pid,)).fetchone()
        if row:
            urls[pid] = row[0]
    print(f"{len(urls)}/{len(ids)} photo ids found in YFCC100M metadata")

    jobs = []
    for r in refs:
        pid = int(Path(r).stem)
        if pid in urls:
            h = mirror_hash(urls[pid])
            jobs.append((f"{MIRROR}/{h[:3]}/{h[3:6]}/{h}.jpg", Path(args.out) / r))
    errors = {}
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [(out, pool.submit(fetch, url, out)) for url, out in jobs]
        for out, fut in progress(futures, desc="download yfcc", unit="img"):
            if err := fut.result():
                errors[str(out)] = err
    print(f"{len(jobs) - len(errors)}/{len(refs)} images ok; {len(errors)} failed")
    for k, v in list(errors.items())[:10]:
        print("  ", k, v)


if __name__ == "__main__":
    main()
