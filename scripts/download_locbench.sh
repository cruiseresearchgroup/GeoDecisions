#!/usr/bin/env bash
# Download LocBench fMoW and YFCC from figshare (DOI 10.6084/m9.figshare.26026798)
#
# Usage: scripts/download_locbench.sh [DATA_DIR]
#   DATA_DIR defaults to <repo>/data/locbench
#   Raw archives are kept in $ARCHIVE_DIR (default <repo>/downloads)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${1:-$ROOT/data/locbench}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$ROOT/downloads}"

declare -A files=(
  [fmow.tar.gz]=47020948
  [yfcc.tar.gz]=47021080
)
# computed_md5 from https://api.figshare.com/v2/articles/26026798
declare -A md5s=(
  [fmow.tar.gz]=fa655fa405dca08fb370866d1c3e1db1
  [yfcc.tar.gz]=9110b8b1f071b6751cf536a9b65b7e06
)

mkdir -p "$ARCHIVE_DIR"

for name in "${!files[@]}"; do
  id="${files[$name]}"
  out="$ARCHIVE_DIR/$name"
  done_marker="$DATA_DIR/.${name%.tar.gz}.extracted"

  if [[ -f "$done_marker" ]]; then
    echo "[skip] $name already extracted"
    continue
  fi

  echo "[download] $name (figshare file $id)"
  # Use ndownloader.figshare.com: figshare.com/ndownloader sits behind an AWS WAF
  # bot challenge that returns an empty 202. -C - resumes partial downloads.
  [[ -s "$out" ]] || rm -f "$out"
  curl -L --fail --retry 5 --retry-delay 10 -C - \
    -o "$out" "https://ndownloader.figshare.com/files/$id"

  echo "[verify] $name"
  echo "${md5s[$name]}  $out" | md5sum -c -

  echo "[extract] $name -> $DATA_DIR"
  tar -xzf "$out" -C "$DATA_DIR"
  # yfcc.tar.gz nests its folder under data/sphere2vec/geo_prior_data/data/.
  nested="$DATA_DIR/data/sphere2vec/geo_prior_data/data/${name%.tar.gz}"
  if [[ -d "$nested" ]]; then
    mv "$nested" "$DATA_DIR/"
    rmdir -p "$(dirname "$nested")" 2>/dev/null || true
  fi
  touch "$done_marker"
done

echo
echo "Contents of $DATA_DIR:"
ls -la "$DATA_DIR"
for d in fmow yfcc; do
  if [[ ! -d "$DATA_DIR/$d" ]]; then
    echo "WARNING: expected $DATA_DIR/$d/ not found; check archive layout" >&2
  fi
done
echo "Expect fmow/category.json and yfcc/class_names.csv for label names."
