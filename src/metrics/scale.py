"""Label/scale utilization (cf. Gao et al., 2026, ordinal compression).

Effective number of labels = exp(Shannon entropy) of how often each option is
used. Compare the model's value with the gold distribution's: a model that
compresses onto fewer labels scores lower than gold.
"""

from __future__ import annotations

import numpy as np


def effective_num_labels(labels: np.ndarray, num_options: int | None = None) -> float:
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_options or 0).astype(float)
    p = counts[counts > 0] / counts.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def expected_value(probs: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Probability-weighted rating per row; ``values`` are the numeric option values."""
    return np.asarray(probs, dtype=np.float64) @ np.asarray(values, dtype=np.float64)


def utilization_ratio(pred_labels: np.ndarray, gold_labels: np.ndarray, num_options: int) -> float:
    """Model effective labels / gold effective labels (1 = matches gold usage)."""
    return effective_num_labels(pred_labels, num_options) / effective_num_labels(gold_labels, num_options)
