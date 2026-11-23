"""Calibration metrics: Brier score and top-label ECE, overall and per region.

All functions take canonical-order probabilities (N, K) and ``label_index``
(position of the true option, -1 if absent, as in :mod:`src.metrics.classification`).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def _onehot(label_index: np.ndarray, k: int) -> np.ndarray:
    y = np.zeros((len(label_index), k))
    ok = label_index >= 0
    y[np.flatnonzero(ok), label_index[ok]] = 1.0
    return y


def brier_per_sample(probs: np.ndarray, label_index: np.ndarray) -> np.ndarray:
    """Multi-class Brier score per sample: sum_k (p_k - y_k)^2, in [0, 2]."""
    probs = np.asarray(probs, dtype=np.float64)
    return ((probs - _onehot(np.asarray(label_index), probs.shape[1])) ** 2).sum(axis=1)


def brier_score(probs: np.ndarray, label_index: np.ndarray) -> float:
    return float(brier_per_sample(probs, label_index).mean())


def top_label_confidence(probs: np.ndarray, label_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Confidence of the argmax option and whether it is correct."""
    probs = np.asarray(probs)
    pred = probs.argmax(axis=1)
    return probs.max(axis=1), (pred == np.asarray(label_index)).astype(float)


def reliability_table(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> pd.DataFrame:
    """Equal-width confidence bins: count, mean confidence, accuracy per bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    df = pd.DataFrame({"bin": bins, "confidence": confidence, "correct": correct})
    agg = df.groupby("bin").agg(n=("correct", "size"), confidence=("confidence", "mean"), accuracy=("correct", "mean"))
    return agg.reindex(range(n_bins)).assign(lower=edges[:-1], upper=edges[1:])


def expected_calibration_error(
    probs: np.ndarray, label_index: np.ndarray, n_bins: int = 15
) -> float:
    """Top-label ECE with equal-width bins (Guo et al., 2017)."""
    conf, correct = top_label_confidence(probs, label_index)
    t = reliability_table(conf, correct, n_bins).dropna()
    return float((t["n"] / len(conf) * (t["confidence"] - t["accuracy"]).abs()).sum())


def calibration_error_per_sample(probs: np.ndarray, label_index: np.ndarray) -> np.ndarray:
    """|confidence - correct| per sample: a per-point mark for spatial analysis.

    Its mean is not ECE (which bins before taking the gap), but it localizes
    over/under-confidence so it can be tested for spatial clustering.
    """
    conf, correct = top_label_confidence(probs, label_index)
    return np.abs(conf - correct)


def by_group(
    groups: np.ndarray,
    probs: np.ndarray,
    label_index: np.ndarray,
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]] | None = None,
    min_count: int = 1,
) -> pd.DataFrame:
    """Evaluate metrics within each group (continent, grid cell, ...)."""
    metrics = metrics or {"brier": brier_score, "ece": expected_calibration_error}
    groups = np.asarray(groups)
    probs, label_index = np.asarray(probs), np.asarray(label_index)
    rows = []
    for g in pd.unique(groups):
        m = groups == g
        if m.sum() < min_count:
            continue
        row = {"group": g, "n": int(m.sum())}
        row.update({name: fn(probs[m], label_index[m]) for name, fn in metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False, ignore_index=True)
