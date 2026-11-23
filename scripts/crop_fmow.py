"""Crop fMoW-rgb val images to their bounding boxes, producing LocBench's image refs.

    python scripts/crop_fmow.py [--root data/images/fmow] [--workers 32] [--delete-raw]

Reads <root>/raw/val/.../<name>_rgb.{jpg,json} (scripts/download_images.sh fmow) and writes
<root>/val/.../<name>_rgb_crop_<k>.jpg for bounding box k. Context padding follows the fMoW
baseline (fMoW-baseline/code/data_ml_functions/dataFunctions.py): smaller boxes get more
surrounding context. Crops are downscaled to the model's 1.6 MP cap (the decider resizes to
it anyway), which keeps the kept set to a few GB.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_locbench  # noqa: E402
from src.utils.progress import progress  # noqa: E402

Image.MAX_IMAGE_PIXELS = None  # fMoW scenes reach ~75 MP
MAX_PIXELS = 1_638_400  # pplx-decider max_pixels


def _context(ratio: float) -> float:
    for lo, mult in ((0.5, 0.15), (0.4, 0.2), (0.3, 0.3), (0.2, 0.5), (0.1, 1.0)):
        if ratio >= lo:
            return mult
    return 2.0


def crop_box(width: int, height: int, box: list[int]) -> tuple[int, int, int, int]:
    """(left, upper, right, lower) for an fMoW box [x, y, w, h], padded and clipped."""
    x, y, w, h = box
    # The baseline compares box width to image rows and box height to image columns; kept as is.
    pad_w = int(w * _context(w / height) / 2)
    pad_h = int(h * _context(h / width) / 2)
    return max(0, x - pad_w), max(0, y - pad_h), min(width, x + w + pad_w), min(height, y + h + pad_h)


def crop_one(ref: str, root: str) -> str | None:
    out = Path(root) / ref
    if out.exists():
        return None
    stem, k = ref.rsplit("_crop_", 1)
    src = Path(root) / "raw" / f"{stem}.jpg"
    try:
        meta = json.loads(src.with_suffix(".json").read_text())
        box = meta["bounding_boxes"][int(k.removesuffix(".jpg"))]["box"]
        with Image.open(src) as im:
            im = im.convert("RGB").crop(crop_box(im.width, im.height, box))
            scale = (MAX_PIXELS / (im.width * im.height)) ** 0.5
            if scale < 1:
                im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.BICUBIC)
            out.parent.mkdir(parents=True, exist_ok=True)
            im.save(out, quality=95)
    except Exception as e:  # noqa: BLE001  report and continue
        return f"{ref}: {e!r}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/images/fmow")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--delete-raw", action="store_true")
    args = ap.parse_args()

    refs = [str(r) for r in load_locbench("fmow").image_refs]
    errors = []
    with ProcessPoolExecutor(args.workers) as pool:
        for err in progress(pool.map(crop_one, refs, [args.root] * len(refs), chunksize=16),
                            total=len(refs), desc="crop fmow", unit="img"):
            if err:
                errors.append(err)
    print(f"{len(refs) - len(errors)}/{len(refs)} crops ok")
    for e in errors[:20]:
        print("  ", e)
    if args.delete_raw and not errors:
        shutil.rmtree(Path(args.root) / "raw")
        print("deleted raw/")


if __name__ == "__main__":
    main()
