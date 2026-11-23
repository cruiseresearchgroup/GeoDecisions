"""TorchSpatial-style LocBench scoring shared by baselines and decision models.

Everything reduces to a per-sample table (TorchSpatial eval-CSV columns plus
lat/lon). TorchSpatial baselines come from :func:`src.data.load_baseline`;
decision models go through :func:`results_to_table`. :func:`report` then
computes the same accuracy and geo-bias numbers for both.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.data.datasets import LocBenchDataset
from src.metrics.calibration import brier_per_sample, calibration_error_per_sample
from src.metrics.classification import per_sample_table, summarize
from src.metrics.geobias import PARTITIONS, SSI_CAP, binary_marks, sre_scores, ssi_scores
from src.modules.base import DecisionResult


def results_to_table(results: Sequence[DecisionResult], dataset: LocBenchDataset) -> pd.DataFrame:
    """Per-sample table for decision-model results on ``dataset`` items.

    Results must have the dataset's canonical option order (as returned by
    :func:`src.modules.decide_batch`). Adds Brier and per-sample calibration
    error columns, available here because decision models return full
    probability vectors.
    """
    by_id = {r.item_id: r for r in results}
    ids = dataset.split.item_ids
    rows = [i for i, item_id in enumerate(ids) if item_id in by_id]
    if not rows:
        raise ValueError("no results match dataset items")
    matched = [by_id[ids[i]] for i in rows]
    if len({len(r.probs) for r in matched}) != 1:
        raise ValueError("results have differing option counts")
    probs = np.stack([r.probs for r in matched])
    label_index = dataset.label_indices[rows]

    table = per_sample_table(probs, label_index)
    table.insert(0, "item_id", [ids[i] for i in rows])
    table.insert(1, "lat", dataset.split.lat[rows])
    table.insert(2, "lon", dataset.split.lon[rows])
    table["brier"] = brier_per_sample(probs, label_index)
    table["calibration_error"] = calibration_error_per_sample(probs, label_index)
    table["selected"] = [r.selected for r in matched]
    return table


def _marked_scores(lat, lon, ssi_marks, sre_values, centers, ssi_radius_km, ssi_k, sre_kwargs, include_sre):
    out: dict[str, float] = {}
    m = ssi_scores(lat, lon, ssi_marks, radius_km=ssi_radius_km, k=ssi_k, centers=centers)
    out["ssi_marked"] = float(np.nanmean(m))
    out["ssi_marked_saturated"] = float(np.mean(m[np.isfinite(m)] >= SSI_CAP - 1e-6))
    if include_sre:
        for kind in PARTITIONS:
            out[f"sre_{kind}"] = sre_scores(lat, lon, sre_values, kind, centers=centers, **(sre_kwargs or {})).aggregate()
    return out


def geobias_scores(
    lat: np.ndarray,
    lon: np.ndarray,
    ssi_marks: np.ndarray,
    sre_values: np.ndarray,
    *,
    centers: np.ndarray | None = None,
    ssi_radius_km: float = 100.0,
    ssi_k: int = 4,
    sre_kwargs: dict | None = None,
    include_unmarked: bool = True,
    include_sre: bool = True,
    shuffle_seed: int | None = 0,
) -> dict[str, float]:
    """Mean SSI (unmarked, marked) and aggregated SRE over ``centers``.

    ``ssi_marks`` must be nonzero (see ``binary_marks``/``binned_marks``);
    ``sre_values`` are histogrammed directly. Also reports the fraction of
    centers whose SSI hit the GeoBS ceiling, where scores stop discriminating.

    Marked SSI and SRE both depend on the overall mark distribution (e.g.
    accuracy), not only on its spatial arrangement: a model that misses almost
    everywhere scores high marked SSI, and rare hits inflate SRE through the KL
    smoothing floor. With ``shuffle_seed`` set, every marked score is recomputed
    after permuting marks across located points (same distribution, no spatial
    structure) and ``<score>_excess = score - shuffled`` is reported; compare
    models on the excess.
    """
    out: dict[str, float] = {}
    if include_unmarked:
        u = ssi_scores(lat, lon, radius_km=ssi_radius_km, k=ssi_k, centers=centers)
        out["ssi_unmarked"] = float(np.nanmean(u))
    args = (centers, ssi_radius_km, ssi_k, sre_kwargs, include_sre)
    observed = _marked_scores(lat, lon, ssi_marks, sre_values, *args)
    out.update(observed)
    if shuffle_seed is not None:
        located = np.flatnonzero(np.isfinite(np.asarray(lat, float)) & np.isfinite(np.asarray(lon, float)))
        perm = located[np.random.default_rng(shuffle_seed).permutation(len(located))]
        sm, sv = np.array(ssi_marks, dtype=float), np.array(sre_values, dtype=float)
        sm[located], sv[located] = sm[perm], sv[perm]
        shuffled = _marked_scores(lat, lon, sm, sv, *args)
        for key, value in shuffled.items():
            if key.endswith("_saturated"):
                continue
            out[f"{key}_shuffled"] = value
            out[f"{key}_excess"] = observed[key] - value
    return out


def report(
    table: pd.DataFrame,
    *,
    ssi_radius_km: float = 100.0,
    ssi_k: int = 4,
    sre_kwargs: dict | None = None,
    mark: str = "hit@1",
    centers: np.ndarray | None = None,
    include_sre: bool = True,
    include_unmarked: bool = True,
) -> dict[str, float]:
    """Accuracy plus SSI/SRE geo-bias for one per-sample table.

    SSI defaults match TorchSpatial's Table 2 (100 km, 4-NN). Marks are the
    binary ``mark`` column (GeoBS convention: hit -> 1, miss -> -1). ``centers``
    restricts which rows are scored (positions in ``table``), since every center
    needs its own neighborhood fit.
    """
    out = summarize(table[[c for c in table.columns if c == "reciprocal_rank" or c.startswith("hit@")]])
    if "brier" in table:
        out["brier"] = float(table["brier"].mean())
    good = table[mark].to_numpy()
    out.update(geobias_scores(
        table["lat"].to_numpy(), table["lon"].to_numpy(), binary_marks(good), good.astype(float),
        centers=centers, ssi_radius_km=ssi_radius_km, ssi_k=ssi_k, sre_kwargs=sre_kwargs,
        include_unmarked=include_unmarked, include_sre=include_sre,
    ))
    return out
