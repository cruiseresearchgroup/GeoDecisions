"""Spatial grids and fixed evaluation subsamples.

Paid API evaluation cannot cover full LocBench splits (fMoW val alone has 53k
images), so every model, including the TorchSpatial baselines, is scored on
the same seeded subsample.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Strategy = Literal["uniform", "proportional", "balanced"]


def grid_cell_ids(lat: np.ndarray, lon: np.ndarray, cell_deg: float) -> np.ndarray:
    """Integer id of the ``cell_deg`` x ``cell_deg`` lat/lon cell; -1 for missing coords."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    n_lon = int(np.ceil(360.0 / cell_deg))
    n_lat = int(np.ceil(180.0 / cell_deg))
    ok = np.isfinite(lat) & np.isfinite(lon)
    i = np.clip(np.floor((lat[ok] + 90.0) / cell_deg), 0, n_lat - 1)
    j = np.clip(np.floor((lon[ok] + 180.0) / cell_deg), 0, n_lon - 1)
    ids = np.full(lat.shape, -1, dtype=np.int64)
    ids[ok] = (i * n_lon + j).astype(np.int64)
    return ids


def spatial_subsample(
    lat: np.ndarray,
    lon: np.ndarray,
    n: int,
    *,
    strategy: Strategy = "proportional",
    cell_deg: float = 10.0,
    seed: int = 0,
) -> np.ndarray:
    """Sorted row indices of a seeded subsample of size ``n`` (rows with coords only).

    Strategies:
        uniform: simple random sample.
        proportional: stratified by grid cell with allocation proportional to cell
            size (largest remainder). Keeps the data's spatial distribution, so
            data-bias scores (unmarked SSI) stay comparable to the full split.
        balanced: equal allocation per cell, capped at cell size, leftovers
            redistributed. Flattens the spatial distribution; use for per-region
            reporting, not for data-bias scores.
    """
    rng = np.random.default_rng(seed)
    cells = grid_cell_ids(lat, lon, cell_deg)
    valid = np.flatnonzero(cells >= 0)
    if n >= len(valid):
        return valid
    if strategy == "uniform":
        return np.sort(rng.choice(valid, size=n, replace=False))

    uniq, inverse, counts = np.unique(cells[valid], return_inverse=True, return_counts=True)
    if strategy == "proportional":
        quota = counts * (n / len(valid))
        alloc = np.floor(quota).astype(int)
        leftover = n - alloc.sum()
        alloc[np.argsort(-(quota - alloc), kind="stable")[:leftover]] += 1
    elif strategy == "balanced":
        alloc = np.zeros_like(counts)
        remaining = n
        open_cells = counts > 0
        while remaining > 0 and open_cells.any():
            share = max(remaining // open_cells.sum(), 1)
            for c in np.flatnonzero(open_cells):
                take = min(share, counts[c] - alloc[c], remaining)
                alloc[c] += take
                remaining -= take
                if remaining == 0:
                    break
            open_cells = alloc < counts
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    picked = [
        rng.choice(valid[inverse == c], size=a, replace=False)
        for c, a in enumerate(alloc)
        if a > 0
    ]
    return np.sort(np.concatenate(picked))
