#!/usr/bin/env bash
# Download iNat2017/iNat2018 for LocBench and place them next to fmow/ and yfcc/
# so TorchSpatial's paths.py finds them.
#
# LocBench's figshare record points to a Dropbox folder for iNat, but that folder
# only mirrors the figshare archives. The iNat files come from the original
# geo_prior release (Mac Aodha et al., ICCV 2019), which TorchSpatial builds on:
#   - locations, categories, Inception features and image-only predictions
#     (*_preds_sparse.npz) from geo_prior_data.zip on CaltechDATA
#   - train/val annotation JSONs from the iNat competition S3 bucket
#
# Usage: scripts/download_inat.sh [DATA_DIR]
#   DATA_DIR defaults to <repo>/data/locbench
#   Raw archives are kept in $ARCHIVE_DIR (default <repo>/downloads)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${1:-$ROOT/data/locbench}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$ROOT/downloads}"
STAGING="$DATA_DIR/_inat_staging"

GEO_PRIOR_URL='https://data.caltech.edu/records/s2fbb-qb406/files/geo_prior_data.zip?download=1'
# md5 published on https://homepages.inf.ed.ac.uk/omacaod/projects/geopriors/index.html
GEO_PRIOR_MD5=08b0db89332e551b9338d88b8c86782f
INAT_S3='https://ml-inat-competition-datasets.s3.amazonaws.com'

mkdir -p "$ARCHIVE_DIR" "$DATA_DIR"

fetch() {  # fetch <url> <out>
  [[ -s "$2" ]] || rm -f "$2"
  # -C - resumes; CaltechDATA redirects to a short-lived signed URL, re-issued on retry.
  curl -L --fail --retry 5 --retry-delay 10 -C - -o "$2" "$1"
}

# 1. geo_prior bundle (22.3 GB): extract only the iNat folders.
zip_out="$ARCHIVE_DIR/geo_prior_data.zip"
if [[ -f "$DATA_DIR/.inat_geo_prior.extracted" ]]; then
  echo "[skip] geo_prior iNat files already extracted"
else
  echo "[download] geo_prior_data.zip"
  fetch "$GEO_PRIOR_URL" "$zip_out"

  echo "[verify] geo_prior_data.zip"
  echo "$GEO_PRIOR_MD5  $zip_out" | md5sum -c -

  rm -rf "$STAGING"
  mkdir -p "$STAGING"
  for d in inat_2017 inat_2018; do
    if [[ -e "$DATA_DIR/$d" ]]; then
      echo "ERROR: $DATA_DIR/$d already exists; move it aside and rerun" >&2
      exit 1
    fi
    echo "[extract] $d"
    unzip -q "$zip_out" "geo_prior_data/data/$d/*" -d "$STAGING"
    mv "$STAGING/geo_prior_data/data/$d" "$DATA_DIR/$d"
  done
  rm -rf "$STAGING"
  touch "$DATA_DIR/.inat_geo_prior.extracted"
fi

# 2. Competition annotations (train20xx.json / val20xx.json), read by
#    TorchSpatial's load_inat_data alongside the *_locations.json files.
if [[ -f "$DATA_DIR/.inat_annotations.extracted" ]]; then
  echo "[skip] iNat annotations already extracted"
else
  echo "[download] iNat 2017 annotations"
  fetch "$INAT_S3/2017/train_val2017.zip" "$ARCHIVE_DIR/inat2017_train_val2017.zip"
  unzip -tq "$ARCHIVE_DIR/inat2017_train_val2017.zip"
  unzip -oqj "$ARCHIVE_DIR/inat2017_train_val2017.zip" '*.json' -d "$DATA_DIR/inat_2017"

  for s in train val; do
    echo "[download] iNat 2018 $s annotations"
    fetch "$INAT_S3/2018/${s}2018.json.tar.gz" "$ARCHIVE_DIR/inat2018_${s}2018.json.tar.gz"
    gzip -t "$ARCHIVE_DIR/inat2018_${s}2018.json.tar.gz"
    tar -xzf "$ARCHIVE_DIR/inat2018_${s}2018.json.tar.gz" -C "$DATA_DIR/inat_2018"
  done
  touch "$DATA_DIR/.inat_annotations.extracted"
fi

echo
for d in inat_2017 inat_2018; do
  echo "== $d"
  ls -la "$DATA_DIR/$d"
done
for f in inat_2017/train2017.json inat_2017/val2017.json inat_2018/train2018.json inat_2018/val2018.json; do
  [[ -f "$DATA_DIR/$f" ]] || echo "WARNING: missing $DATA_DIR/$f" >&2
done
