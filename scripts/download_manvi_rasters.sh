#!/usr/bin/env bash
# Ground-truth rasters for the Manvi et al. (ICML 2024) objective topics.
#
#   topic                          source (paper)                 product used here
#   Infant Mortality Rate          CIESIN 2021, v2.01             shipped with GeoLLM (see README)
#   Population Density             WorldPop (Tatem 2017)          WorldPop 2020 1km, not UN-adjusted
#                                                                 (GeoLLM loads ppp_2020_1km_Aggregated.tif; the
#                                                                 global mosaic is gone, so we mosaic the
#                                                                 per-country files into a VRT)
#   Built-Up to Non Built-Up Ratio GHSL Data Package 2019         GHS_BUILT_LDS2014_GLOBE_R2018A 1km (Mollweide)
#   Average Temperature            CHELSA (Karger et al. 2018)    CHELSA v1.2 bio10_01 (annual mean, degC*10)
#   Annual Precipitation           CHELSA (Karger et al. 2018)    CHELSA v1.2 bio10_12 (annual sum, mm)
#   Nighttime Light Intensity      VIIRS (Elvidge et al. 2017)    VNL v2 2016 average_masked; EOG requires a login,
#                                                                 so it is converted here once downloaded by hand
#
# Usage: scripts/download_manvi_rasters.sh
#   Rasters used by src/data/manvi.py go to <repo>/data/manvi/rasters as real files (no
#   symlinks, so data/ can be copied to another machine). $ARCHIVE_DIR (default
#   <repo>/downloads/manvi) only holds intermediates and is not needed afterwards.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_DIR="${ARCHIVE_DIR:-$ROOT/downloads/manvi}"
OUT="$ROOT/data/manvi/rasters"
WP_DIR="$ROOT/data/manvi/worldpop_2020_1km"
mkdir -p "$ARCHIVE_DIR" "$OUT" "$WP_DIR"

fetch() {  # fetch <url> <out>
  if [[ -s "$2" ]]; then echo "[skip] $(basename "$2")"; return; fi
  curl -L --fail --retry 8 --retry-delay 15 --retry-all-errors -C - -o "$2.part" "$1"
  mv "$2.part" "$2"
}

# CHELSA v1.2 bioclim.
CHELSA='https://os.zhdk.cloud.switch.ch/chelsav1/climatologies/bio'
fetch "$CHELSA/CHELSA_bio10_01.tif" "$OUT/temperature_chelsa_v12_bio01.tif"
fetch "$CHELSA/CHELSA_bio10_12.tif" "$OUT/precipitation_chelsa_v12_bio12.tif"

# GHSL built-up fraction (percent of each 1km cell built up), 2014 epoch.
GHSL='https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_LDSMT_GLOBE_R2018A/GHS_BUILT_LDS2014_GLOBE_R2018A_54009_1K/V2-0/GHS_BUILT_LDS2014_GLOBE_R2018A_54009_1K_V2_0.zip'
fetch "$GHSL" "$ARCHIVE_DIR/GHS_BUILT_LDS2014_GLOBE_R2018A_54009_1K_V2_0.zip"
if [[ ! -e "$OUT/builtup_ghsl_r2018a_2014_1k.tif" ]]; then
  unzip -oq "$ARCHIVE_DIR/GHS_BUILT_LDS2014_GLOBE_R2018A_54009_1K_V2_0.zip" -d "$ROOT/data/manvi/ghsl"
  tif=$(find "$ROOT/data/manvi/ghsl" -name '*.tif' -print -quit)
  for ext in "" .ovr .aux.xml; do  # overviews/metadata follow the tif's name
    [[ -e "$tif$ext" ]] && mv "$tif$ext" "$OUT/builtup_ghsl_r2018a_2014_1k.tif$ext"
  done
fi

# WorldPop 2020 1km per-country rasters -> one VRT.
WP='https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020'
listing="$ARCHIVE_DIR/worldpop_2020_listing.html"
[[ -s "$listing" ]] || curl -sL --fail --retry 8 --retry-delay 15 --retry-all-errors -o "$listing" "$WP/"
grep -oE 'href="[A-Z]{3}/"' "$listing" | sed -E 's/href="([A-Z]{3})\/"/\1/' > "$ARCHIVE_DIR/worldpop_iso3.txt"
echo "[worldpop] $(wc -l < "$ARCHIVE_DIR/worldpop_iso3.txt") countries"
export WP WP_DIR
xargs -P 6 -I{} bash -c '
  iso={}; lower=$(echo "$iso" | tr A-Z a-z); f="$WP_DIR/${lower}_ppp_2020_1km_Aggregated.tif"
  [[ -s "$f" ]] && exit 0
  curl -sL --fail --retry 8 --retry-delay 15 --retry-all-errors -o "$f.part" "$WP/$iso/${lower}_ppp_2020_1km_Aggregated.tif" \
    && mv "$f.part" "$f" || echo "WARNING: worldpop $iso failed" >&2
' < "$ARCHIVE_DIR/worldpop_iso3.txt"
n_wp=$(find "$WP_DIR" -name '*_ppp_2020_1km_Aggregated.tif' | wc -l)
echo "[worldpop] $n_wp country rasters"
# Relative source paths, so the VRT survives moving data/.
(cd "$OUT" && gdalbuildvrt -q -overwrite population_worldpop_2020_1km.vrt ../worldpop_2020_1km/*_ppp_2020_1km_Aggregated.tif)

# Nighttime lights: convert the hand-downloaded file to a losslessly compressed copy
# (11.6 GB -> ~0.3 GB, pixel-identical).
VNL="$ARCHIVE_DIR/VNL_v2_npp_2016_global_vcmslcfg_c202102150000.average_masked.tif"
[[ -s "$VNL" || ! -s "$VNL.gz" ]] || gunzip -k "$VNL.gz"
if [[ -s "$VNL" && ! -s "$OUT/nightlights_viirs_vnl2_2016.tif" ]]; then
  gdal_translate -q -co COMPRESS=DEFLATE -co PREDICTOR=3 -co TILED=YES -co BIGTIFF=YES -co NUM_THREADS=ALL_CPUS \
    "$VNL" "$OUT/nightlights_viirs_vnl2_2016.tif"
fi

echo
echo "Rasters in $OUT:"
ls -la "$OUT"
echo
[[ -s "$OUT/nightlights_viirs_vnl2_2016.tif" ]] || {
  echo "Nighttime lights: register at https://eogdata.mines.edu/products/vnl/, download"
  echo "VNL_v2_npp_2016_global_vcmslcfg_c202102150000.average_masked.tif.gz into $ARCHIVE_DIR and rerun."
}
