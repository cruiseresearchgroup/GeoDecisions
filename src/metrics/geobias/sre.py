"""Spatial Relative Entropy (SRE) geo-bias scores (GeoBS; named SRI in the GeoBS code).

Around each center, the neighborhood within ``radius`` is split into parts
(Scale-Grid cells, Distance-Lag rings, or Direction-Sector wedges). Each part's
histogram of performance marks is compared to the whole neighborhood's with
KL(neighborhood || part); the center's score is the part-size-weighted mean.
Higher = performance differs more across parts = more geo-biased.

Ported from GeoBS (MIT) ``partition/partitioner.py``, ``gbs/sri/sri_scores.py``
and ``sri_example_code.py``, with fixes:
  * ``SRIPartitioner.get_neighborhood`` in GeoBS returns positions within the
    center's kNN row instead of dataset indices, so every part indexes the wrong
    points. Here neighbors are dataset indices (as in GeoBS's
    ``baselines/gbsloss.py`` version).
  * GeoBS histograms each part over its own value range (``np.histogram(x,
    bins=int)``), so bin edges differ between part and neighborhood. Here every
    histogram uses the neighborhood's edges.
  * Direction sectors use the initial great-circle bearing (see geometry.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.special
from sklearn.neighbors import BallTree

from src.metrics.geobias.geometry import initial_bearing, km_to_rad

Partition = Literal["scale_grid", "distance_lag", "direction_sector"]
PARTITIONS: tuple[Partition, ...] = ("scale_grid", "distance_lag", "direction_sector")


def kl_sre(
    part_values: list[np.ndarray],
    neighborhood_values: np.ndarray,
    *,
    bins: int = 2,
    min_count: int = 20,
    smooth: float = 1e-8,
) -> float:
    """Part-size-weighted KL(neighborhood || part); parts below ``min_count`` are skipped."""
    n = len(neighborhood_values)
    if n == 0:
        return np.nan
    lo = np.min(neighborhood_values)
    if 0 < np.ptp(neighborhood_values) <= 1e-12 * max(1.0, abs(lo)):
        # Values equal up to float noise (e.g. renormalized 2-decimal API probabilities) cannot be split into
        # finite bins; score them as the constant they are (parts are subsets of the neighborhood).
        neighborhood_values = np.full(n, lo)
        part_values = [np.full(len(v), lo) for v in part_values]
    edges = np.histogram_bin_edges(neighborhood_values, bins=bins)
    p_global = np.histogram(neighborhood_values, bins=edges)[0] / n
    kls, weights = [], []
    for v in part_values:
        if len(v) < min_count:
            continue
        p_local = np.histogram(v, bins=edges)[0] / len(v) + smooth
        kls.append(np.sum(scipy.special.rel_entr(p_global, p_local)))
        weights.append(len(v))
    if not kls:
        return np.nan
    return float(np.average(kls, weights=weights))


@dataclass
class SREResult:
    scores: np.ndarray  # per row, NaN where not scored
    weights: np.ndarray  # neighborhood size / N (GeoBS's aggregation weight)

    def aggregate(self) -> float:
        ok = np.isfinite(self.scores) & (self.weights > 0)
        return float(np.average(self.scores[ok], weights=self.weights[ok])) if ok.any() else np.nan


def _parts(
    kind: Partition,
    coords: np.ndarray,
    center: int,
    nb: np.ndarray,
    nb_dist: np.ndarray,
    radius: float,
    scale: float,
    lag: float,
    n_sectors: int,
) -> list[np.ndarray]:
    if kind == "scale_grid":
        # Square lat/lon cells of side ``scale`` radians anchored at the center (GeoBS).
        rel = coords[nb] - coords[center]
        rel[:, 1] = (rel[:, 1] + np.pi) % (2 * np.pi) - np.pi  # wrap across the dateline
        cell = np.floor(rel / scale).astype(np.int64)
        _, inv = np.unique(cell, axis=0, return_inverse=True)
        return [nb[inv.ravel() == g] for g in range(inv.max() + 1)]
    if kind == "distance_lag":
        ring = np.floor(nb_dist / lag).astype(np.int64)
        return [nb[ring == r] for r in range(int(np.ceil(radius / lag)))]
    if kind == "direction_sector":
        bearing = initial_bearing(coords[nb], coords[center])
        sector = np.clip(((bearing + np.pi) / (2 * np.pi) * n_sectors).astype(np.int64), 0, n_sectors - 1)
        return [nb[sector == s] for s in range(n_sectors)]
    raise ValueError(f"unknown partition {kind!r}")


def sre_scores(
    lat: np.ndarray,
    lon: np.ndarray,
    marks: np.ndarray,
    kind: Partition,
    *,
    radius_km: float = 300.0,
    scale_km: float = 60.0,
    lag_km: float = 60.0,
    n_sectors: int = 8,
    bins: int = 2,
    min_count: int = 20,
    max_neighbors: int = 400,
    centers: np.ndarray | None = None,
) -> SREResult:
    """Per-center SRE for one partition type. Coordinates in degrees; NaN rows skipped.

    Defaults follow ``sri_example_code.py`` (radius 0.05 rad ~ 320 km, scale and
    lag 0.01 rad ~ 64 km, 8 sectors, threshold 20, k 400), rounded to km.
    """
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    ok = np.isfinite(lat) & np.isfinite(lon)
    idx_ok = np.flatnonzero(ok)
    coords = np.deg2rad(np.stack([lat[idx_ok], lon[idx_ok]], axis=1))
    vals = np.asarray(marks, float)[idx_ok]
    radius, scale, lag = km_to_rad(radius_km), km_to_rad(scale_km), km_to_rad(lag_km)

    pos = np.full(len(lat), -1)
    pos[idx_ok] = np.arange(len(idx_ok))
    centers = idx_ok if centers is None else np.asarray(centers)
    centers = centers[pos[centers] >= 0]

    # Neighbors are only needed for the scored centers.
    tree = BallTree(coords, leaf_size=30, metric="haversine")
    dists, nbrs = tree.query(coords[pos[centers]], k=min(max_neighbors, len(coords)))

    scores = np.full(len(lat), np.nan)
    weights = np.zeros(len(lat))
    for c, d, nb in zip(centers, dists, nbrs):
        keep = d <= radius
        nb, nb_dist = nb[keep], d[keep]
        parts = _parts(kind, coords, pos[c], nb, nb_dist, radius, scale, lag, n_sectors)
        scores[c] = kl_sre([vals[x] for x in parts], vals[nb], bins=bins, min_count=min_count)
        weights[c] = len(nb) / len(coords)
    return SREResult(scores, weights)
