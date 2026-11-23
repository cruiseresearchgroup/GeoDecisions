"""TorchSpatial location-encoder baselines from their per-sample eval CSVs.

Files are ``eval_<dataset>__<split>_<encoder>.csv`` with columns
``lon, lat, true_class_prob, reciprocal_rank, hit@1, hit@3``. The ``no_prior``
(image-only) file has no lon/lat. Rows follow TorchSpatial's loader order, which
:func:`src.data.locbench.load_locbench` reproduces, so rows are joined by position
and checked against our coordinates.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.locbench import EVAL_SPLIT, LocBenchSplit
from src.paths import load_paths

METRIC_COLUMNS = ("true_class_prob", "reciprocal_rank", "hit@1", "hit@3")
IMAGE_ONLY = "no_prior"

# TorchSpatial writes the first two columns of the *encoded* location input as
# "lon"/"lat". For most encoders that is raw degrees; for "wrap" it is
# (sin(lon), sin(2*lat)). Maps (lat, lon) in degrees -> expected (lon, lat) columns.
_CSV_COORDS = {
    "wrap": lambda lat, lon: (np.sin(np.deg2rad(lon)), np.sin(2 * np.deg2rad(lat))),
}


def _has_rows(path: Path) -> bool:
    """Some shipped CSVs exist but hold only whitespace (e.g. fMoW spherical_harmonics)."""
    with open(path) as f:
        return bool(f.read(256).strip())


def _eval_dir(eval_dir: str | Path | None) -> Path:
    return Path(eval_dir) if eval_dir is not None else load_paths().torchspatial_eval_dir


def available_encoders(dataset: str, split: str | None = None, eval_dir=None) -> list[str]:
    """Encoder names with a non-empty eval CSV for this dataset/split."""
    split = split or EVAL_SPLIT[dataset]
    pat = re.compile(rf"^eval_{re.escape(dataset)}__{re.escape(split)}_(.+)\.csv$")
    found = []
    for p in sorted(_eval_dir(eval_dir).glob(f"eval_{dataset}__{split}_*.csv")):
        m = pat.match(p.name)
        if m and _has_rows(p):
            found.append(m.group(1))
    return found


def load_baseline(
    reference: LocBenchSplit,
    encoder: str,
    *,
    eval_dir: str | Path | None = None,
    coord_atol: float = 1e-3,
) -> pd.DataFrame:
    """Per-sample TorchSpatial results aligned to ``reference`` rows.

    ``reference`` may be a subset; rows are selected by ``reference.rows``.
    Returns columns ``item_id, lat, lon`` plus :data:`METRIC_COLUMNS`, with
    lat/lon taken from ``reference`` after checking they match the CSV.
    """
    path = _eval_dir(eval_dir) / f"eval_{reference.dataset}__{reference.split}_{encoder}.csv"
    if not path.exists() or not _has_rows(path):
        raise FileNotFoundError(f"no TorchSpatial results for {encoder!r}: {path}")
    df = pd.read_csv(path, index_col=0)

    n_full = int(reference.rows.max()) + 1 if len(reference) else 0
    if len(df) < n_full:
        raise ValueError(f"{path.name} has {len(df)} rows; reference needs >= {n_full}")
    df = df.iloc[reference.rows].reset_index(drop=True)

    if {"lat", "lon"} <= set(df.columns):
        both = reference.has_location & df["lat"].notna().to_numpy()
        if (reference.has_location != df["lat"].notna().to_numpy()).any():
            raise ValueError(f"{path.name}: missing-location rows differ from reference")
        to_csv = _CSV_COORDS.get(encoder, lambda lat, lon: (lon, lat))
        exp_lon, exp_lat = to_csv(reference.lat[both], reference.lon[both])
        dlat = np.abs(df["lat"].to_numpy()[both] - exp_lat)
        dlon = np.abs(df["lon"].to_numpy()[both] - exp_lon)
        if both.any() and max(dlat.max(), dlon.max()) > coord_atol:
            raise ValueError(f"{path.name}: coordinates do not match reference rows")

    out = pd.DataFrame({"item_id": reference.item_ids, "lat": reference.lat, "lon": reference.lon})
    for col in METRIC_COLUMNS:
        out[col] = df[col].to_numpy()
    return out
