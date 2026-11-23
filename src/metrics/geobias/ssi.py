"""Spatial Self-Information (SSI) geo-bias scores (TorchSpatial / GeoBS).

For each center point, take observed points within ``radius`` (capped at the
``max_neighbors`` nearest), fill the disk with a regular grid of background
points (value 0), and score how surprising the spatial autocorrelation
(Moran's I under a kNN weight matrix) of the values is:
    unmarked: observed = 1, background = 0  -> data (sampling) bias
    marked:   observed = performance mark (nonzero), background = 0 -> performance bias
SSI = -log(two-sided tail probability). Higher = more geo-biased.

Ported from GeoBS (MIT) ``gbs/ssi/ssi_scores.py`` and ``ssi_example_code.py``.
TorchSpatial reports both at a 100 km scale with a 4-NN weight matrix.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import BallTree

from src.metrics.geobias.geometry import (
    auto_density,
    generate_background_points,
    km_to_rad,
    knn_weight_matrix,
)
from src.metrics.geobias.surprisal import AnalyticalSurprisal

#: SSI ceiling: GeoBS floors the tail probability at 1e-256, so SSI saturates at -ln(1e-256).
SSI_CAP = float(-np.log(1e-256))


def _ssi(points: np.ndarray, values: np.ndarray, center: np.ndarray, radius: float, density, k: int) -> float:
    if density == "auto":
        density = auto_density(radius, points.shape[0])
    bg = generate_background_points(center, radius, density)
    pts = np.concatenate([points, bg])
    vals = np.concatenate([values, np.zeros(len(bg))])

    cs, ns = np.unique(vals, return_counts=True)
    ignores = np.ones_like(cs)
    ignores[np.argmax(ns)] = 0
    w = knn_weight_matrix(pts, k)

    s = AnalyticalSurprisal()
    s.fit(cs, ns, w, ignores)
    return float(-np.log(s.get_probability(vals, w) + 1e-256))


def unmarked_ssi(presence_points, center, radius, density="auto", k=4) -> float:
    """SSI of observed-vs-background arrangement around ``center``."""
    return _ssi(presence_points, np.ones(len(presence_points)), center, radius, density, k)


def marked_ssi(presence_points, presence_values, center, radius, density="auto", k=4) -> float:
    """SSI of performance marks (must be nonzero; 0 is reserved for background)."""
    presence_values = np.asarray(presence_values, dtype=np.float64)
    if np.any(presence_values == 0):
        raise ValueError("marks must be nonzero; encode with binary_marks/binned_marks")
    return _ssi(presence_points, presence_values, center, radius, density, k)


def binary_marks(good: np.ndarray) -> np.ndarray:
    """GeoBS convention for binary performance: good -> 1, bad -> -1."""
    return np.where(np.asarray(good).astype(bool), 1.0, -1.0)


def binned_marks(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Continuous marks (rating error, calibration error) -> bin ids 1..len(edges)-1."""
    return np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2).astype(np.float64) + 1.0


def ssi_scores(
    lat: np.ndarray,
    lon: np.ndarray,
    marks: np.ndarray | None = None,
    *,
    radius_km: float = 100.0,
    k: int = 4,
    max_neighbors: int = 400,
    centers: np.ndarray | None = None,
    density="auto",
) -> np.ndarray:
    """Per-center SSI; unmarked if ``marks`` is None, else marked.

    Args:
        lat, lon: degrees. Rows without coordinates get NaN and are excluded
            from every neighborhood.
        marks: per-point nonzero performance marks.
        max_neighbors: neighborhood cap (GeoBS's partitioner ``k``).
        centers: row indices to score (default: all rows with coordinates).
    """
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    ok = np.isfinite(lat) & np.isfinite(lon)
    idx_ok = np.flatnonzero(ok)
    coords = np.deg2rad(np.stack([lat[idx_ok], lon[idx_ok]], axis=1))
    vals = None if marks is None else np.asarray(marks, float)[idx_ok]
    radius = km_to_rad(radius_km)

    pos = np.full(len(lat), -1)
    pos[idx_ok] = np.arange(len(idx_ok))
    centers = idx_ok if centers is None else np.asarray(centers)
    centers = centers[pos[centers] >= 0]

    # Neighbors are only needed for the scored centers.
    tree = BallTree(coords, leaf_size=30, metric="haversine")
    dists, nbrs = tree.query(coords[pos[centers]], k=min(max_neighbors, len(coords)))

    out = np.full(len(lat), np.nan)
    for c, d, nb in zip(centers, dists, nbrs):
        p = pos[c]
        nb = nb[d <= radius]
        if vals is None:
            out[c] = unmarked_ssi(coords[nb], coords[p], radius, density, k)
        else:
            out[c] = marked_ssi(coords[nb], vals[nb], coords[p], radius, density, k)
    return out
