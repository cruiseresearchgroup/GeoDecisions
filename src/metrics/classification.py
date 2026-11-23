"""Per-sample classification metrics in TorchSpatial's eval-CSV format.

TorchSpatial writes, per sample: ``true_class_prob``, ``reciprocal_rank``,
``hit@1``, ``hit@3``; dataset-level Top-1 and MRR are their means. The same
table is computed here for decision-model outputs, so both feed the geo-bias
metrics identically.

Labels are given as an index into each row's options. ``-1`` means the true
class is not among the options (iNat shortlist miss): rank = inf, so
reciprocal rank and every hit are 0, and true_class_prob is 0.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse

Ties = Literal["optimistic", "pessimistic"]


def label_ranks(
    probs: np.ndarray | sparse.spmatrix,
    label_index: np.ndarray,
    *,
    ties: Ties = "optimistic",
    chunk_size: int = 4096,
) -> np.ndarray:
    """1-based rank of the true option per row (inf when absent).

    ``optimistic`` counts only strictly larger probabilities (ties favor the
    label); ``pessimistic`` also counts ties. TorchSpatial sorts with numpy's
    default (unstable) argsort, so tie order there is arbitrary; ties are rare
    except among exact zeros in sparse iNat predictions.
    """
    label_index = np.asarray(label_index)
    n = len(label_index)
    ranks = np.full(n, np.inf)
    for start in range(0, n, chunk_size):
        block = probs[start:start + chunk_size]
        block = np.asarray(block.todense() if sparse.issparse(block) else block, dtype=np.float64)
        li = label_index[start:start + len(block)]
        ok = li >= 0
        rows = np.flatnonzero(ok)
        p_true = block[rows, li[ok]]
        greater = (block[rows] > p_true[:, None]).sum(axis=1)
        if ties == "pessimistic":
            greater += (block[rows] == p_true[:, None]).sum(axis=1) - 1
        ranks[start + rows] = 1 + greater
    return ranks


def true_class_probs(probs: np.ndarray | sparse.spmatrix, label_index: np.ndarray) -> np.ndarray:
    label_index = np.asarray(label_index)
    out = np.zeros(len(label_index))
    ok = label_index >= 0
    rows = np.flatnonzero(ok)
    vals = probs[rows, label_index[ok]]
    out[rows] = np.asarray(vals).ravel()
    return out


def per_sample_table(
    probs: np.ndarray | sparse.spmatrix,
    label_index: np.ndarray,
    *,
    ks: tuple[int, ...] = (1, 3),
    ties: Ties = "optimistic",
) -> pd.DataFrame:
    """TorchSpatial-format per-sample results (columns true_class_prob, reciprocal_rank, hit@k)."""
    ranks = label_ranks(probs, label_index, ties=ties)
    table = pd.DataFrame(
        {
            "true_class_prob": true_class_probs(probs, label_index),
            "reciprocal_rank": 1.0 / ranks,
        }
    )
    for k in ks:
        table[f"hit@{k}"] = (ranks <= k).astype(int)
    return table


def summarize(table: pd.DataFrame) -> dict[str, float]:
    """Top-k accuracy and MRR (as fractions) from a per-sample table."""
    out = {"n": float(len(table)), "mrr": float(table["reciprocal_rank"].mean())}
    for col in table.columns:
        if col.startswith("hit@"):
            out[f"top{col[4:]}"] = float(table[col].mean())
    return out


def location_gain(with_location: pd.DataFrame, without_location: pd.DataFrame) -> dict[str, float]:
    """Gain from adding location (image + location minus image only), per summary metric."""
    a, b = summarize(with_location), summarize(without_location)
    return {k: a[k] - b[k] for k in a if k != "n"}
