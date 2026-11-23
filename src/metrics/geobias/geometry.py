"""Spherical geometry helpers for geo-bias scores.

Ported from GeoBS (MIT, Spatially Explicit AI Lab): ``utils/geometry.py`` and
``gbs/ssi/utils.py``. All coordinates are (lat, lon) in radians.

Deviation: GeoBS's ``_latlon_to_xyz`` computes y = sign(lat)*sin(lon)*cos(lat),
mirroring longitudes south of the equator. It only receives northern points
when generating SSI background grids (identical results there), but it also
feeds GeoBS's Direction-Sector SRE, which is replaced here by
:func:`initial_bearing`.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0088


def km_to_rad(km: float) -> float:
    return km / EARTH_RADIUS_KM


def latlon_to_xyz(points: np.ndarray) -> np.ndarray:
    lat, lon = points[:, 0], points[:, 1]
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)


def xyz_to_latlon(xyz: np.ndarray) -> np.ndarray:
    return np.stack([np.arcsin(np.clip(xyz[:, 2], -1, 1)), np.arctan2(xyz[:, 1], xyz[:, 0])], axis=1)


def _polar_rings(radius: float, spacing: float) -> np.ndarray:
    """Concentric rings around the north pole, ring i at polar distance i*spacing."""
    pts = []
    for i in range(1, int(np.ceil(radius / spacing)) + 1):
        n = int(np.ceil(2 * np.pi * i))
        pts.append(np.stack([np.full(n, np.pi / 2 - spacing * i), -np.pi + 2 * np.pi / n * np.arange(n)], axis=1))
    return np.concatenate(pts) if pts else np.empty((0, 2))


def generate_background_points(
    center: np.ndarray, radius: float, density: float | None = None, n_points: int | None = None
) -> np.ndarray:
    """Regular grid of 'non-presence' points within ``radius`` of ``center``.

    ``density`` is points per square radian; alternatively ``n_points`` sets
    the spacing to sqrt(pi / n_points) * radius (GeoBS conventions).
    """
    if density is not None:
        spacing = 1 / np.sqrt(density)
    elif n_points is not None:
        spacing = np.sqrt(np.pi / n_points) * radius
    else:
        raise ValueError("specify density or n_points")
    xyz = latlon_to_xyz(_polar_rings(radius, spacing))
    rot = Rotation.from_euler("yz", [0.5 * np.pi - center[0], center[1]])
    return xyz_to_latlon(rot.apply(xyz))


def auto_density(radius: float, n_neighbors: int) -> int:
    """GeoBS default: about 5 background points per observed point."""
    return int(np.ceil((5 * n_neighbors) / (np.pi * radius ** 2)))


def knn_weight_matrix(points: np.ndarray, k: int) -> np.ndarray:
    """Dense 0/1 k-nearest-neighbor weight matrix (row i marks i's k neighbors)."""
    k = min(k, len(points) - 1)
    w = np.zeros((len(points), len(points)))
    if k <= 0:
        return w
    nbrs = BallTree(points, leaf_size=30, metric="haversine").query(points, k=k + 1, return_distance=False)
    rows = np.repeat(np.arange(len(points)), k + 1)
    w[rows, nbrs.ravel()] = 1.0
    np.fill_diagonal(w, 0.0)
    return w


def initial_bearing(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Bearing from ``center`` to each point, radians in [-pi, pi], 0 = north."""
    lat1, lon1 = center
    lat2, dlon = points[:, 0], points[:, 1] - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x)
