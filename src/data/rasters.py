"""Point lookups in ground-truth rasters (Manvi et al. objective topics, covariates)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np

# Manvi et al. (GeoLLM utils.extract_data) aggregate a (2*12+1)^2 pixel window.
MANVI_WINDOW = 12


def sample_raster(
    path: str | Path,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    window: int = MANVI_WINDOW,
    reduce: Literal["sum", "mean"] = "mean",
    nonnegative_only: bool = False,
    fill_values: Sequence[float] = (),
) -> np.ndarray:
    """Aggregate valid raster values in a (2*window+1)^2 pixel window around each point.

    Valid = finite, not the raster's nodata value, and not in ``fill_values``
    (undeclared fill codes, e.g. -9999 and -7777 in the CIESIN infant-mortality
    raster). ``nonnegative_only=True`` instead keeps values >= 0, which is what
    GeoLLM's ``extract_data`` does; use it only for parity checks, since it drops
    real negative values (CHELSA stores temperature as degC*10, so every
    sub-zero location would vanish).

    ``reduce="sum"`` matches GeoLLM and suits count rasters (population). Use
    ``"mean"`` for rates and fractions: a sum shrinks near coasts and raster
    edges, where fewer valid pixels fall in the window.

    Points are reprojected when the raster is not in EPSG:4326 (e.g. GHSL's
    Mollweide). Returns NaN outside the raster or with no valid pixels.
    """
    import rasterio
    from rasterio.warp import transform as warp_transform

    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    out = np.full(lat.shape, np.nan)
    ok = np.isfinite(lat) & np.isfinite(lon)
    with rasterio.open(path) as src:
        xs, ys = lon[ok], lat[ok]
        if src.crs is not None and src.crs.to_epsg() != 4326:
            xs, ys = (np.asarray(v) for v in warp_transform("EPSG:4326", src.crs, xs, ys))
        inv = ~src.transform
        nodata = src.nodata
        for n, x, y in zip(np.flatnonzero(ok), xs, ys):
            px, py = inv * (x, y)
            px, py = round(px), round(py)
            if not (0 <= px < src.width and 0 <= py < src.height):
                continue
            win = (
                (max(py - window, 0), min(py + window + 1, src.height)),
                (max(px - window, 0), min(px + window + 1, src.width)),
            )
            vals = src.read(1, window=win).astype(np.float64)
            if nonnegative_only:
                vals = vals[vals >= 0]
            else:
                valid = np.isfinite(vals)
                for fill in (nodata, *fill_values):
                    if fill is not None:
                        valid &= ~np.isclose(vals, fill)
                vals = vals[valid]
            if vals.size:
                out[n] = vals.sum() if reduce == "sum" else vals.mean()
    return out
