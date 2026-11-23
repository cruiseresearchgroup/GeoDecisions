#!/usr/bin/env bash
# Download raw eval images for the LocBench image conditions (image_only, image_location).
# Only the images LocBench evaluates on are kept; see configs/paths.yaml image_roots.
#
#   iNat2017/2018  The competition archives are single .tar.gz files (no per-image URLs),
#                  so each archive is streamed through tar and only val images are written.
#                  Transfer: 199 GB (2017) / 128 GB (2018); kept: ~28 GB / ~7 GB.
#   fMoW           Individual *_rgb.jpg/json files from the public spacenet S3 bucket
#                  (val split, <= 23 GB), then cropped by scripts/crop_fmow.py.
#
# Usage: scripts/download_images.sh {inat_2017|inat_2018|fmow} [OUT_DIR]
#   OUT_DIR defaults to <repo>/data/images/<dataset>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DS="$1"
OUT="${2:-$ROOT/data/images/$DS}"
INAT_S3='s3://ml-inat-competition-datasets'
FMOW_S3='s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb'
mkdir -p "$OUT"

# More parallel requests than the AWS CLI default (10), without touching ~/.aws/config.
export AWS_CONFIG_FILE="$OUT/.aws_config"
printf '[default]\ns3 =\n  max_concurrent_requests = 64\n  max_queue_size = 10000\n' > "$AWS_CONFIG_FILE"

# Image refs of the LocBench eval split, one per line (relative to the image root).
list_refs() {
  (cd "$ROOT" && python -c "
import sys
from src.data import load_locbench
for r in load_locbench(sys.argv[1]).image_refs: print(r)
" "$1")
}

stream_inat() {  # stream_inat <archive url>
  local list="$OUT/.members.txt"
  list_refs "$DS" > "$list"
  echo "[stream] $1 -> $OUT ($(wc -l < "$list") members)"
  # The AWS CLI streams with parallel ranged GETs (~4x a single curl stream). tar reads names
  # verbatim from the list (no wildcards); everything else is skipped, as are files kept from
  # an earlier, interrupted run (a restart still re-streams the whole archive).
  aws s3 cp --no-sign-request --only-show-errors "$1" - \
    | tar -xz -C "$OUT" --no-wildcards --skip-old-files -T "$list"
}

case "$DS" in
  inat_2017) stream_inat "$INAT_S3/2017/train_val_images.tar.gz" ;;
  inat_2018) stream_inat "$INAT_S3/2018/train_val2018.tar.gz" ;;
  fmow)
    # Refs look like val/<cat>/<cat>_<id>/<cat>_<id>_<n>_rgb_crop_<k>.jpg; fetch the source
    # <..>_rgb.jpg and its metadata JSON (bounding boxes) into $OUT/raw.
    list_refs fmow | sed -E 's/_crop_[0-9]+\.jpg$//' | sort -u > "$OUT/.sources.txt"
    echo "[s3] $(wc -l < "$OUT/.sources.txt") fMoW source images -> $OUT/raw"
    # One call per file would be ~50k requests; sync the val split with filters instead
    # (val holds exactly LocBench's fMoW eval images plus msrgb variants, excluded here).
    aws s3 sync --no-sign-request --only-show-errors "$FMOW_S3/val/" "$OUT/raw/val/" \
      --exclude '*' --include '*_rgb.jpg' --include '*_rgb.json'
    ;;
  *) echo "unknown dataset: $DS" >&2; exit 1 ;;
esac

echo "[done] $DS: $(find "$OUT" -name '*.jpg' | wc -l) images, $(du -sh "$OUT" | cut -f1)"
