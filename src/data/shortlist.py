"""Fixed top-k candidate shortlists from the image-only classifier.

iNat has thousands of classes, too many to pass as options to a decision model.
Each item gets the k classes the image-only model ranks highest; the decision
model then chooses among those. Shortlist recall (label in shortlist) is the
accuracy ceiling for any model evaluated on the shortlist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class Shortlist:
    """Per-item candidate class ids, ordered by image-only probability (desc)."""

    candidates: np.ndarray  # (N, k) int64

    @property
    def k(self) -> int:
        return self.candidates.shape[1]

    def label_position(self, labels: np.ndarray) -> np.ndarray:
        """Position of each label in its row's shortlist, or -1 if absent."""
        hit = self.candidates == np.asarray(labels)[:, None]
        return np.where(hit.any(axis=1), hit.argmax(axis=1), -1)

    def recall(self, labels: np.ndarray) -> float:
        return float((self.label_position(labels) >= 0).mean())

    def subset(self, idx: np.ndarray) -> "Shortlist":
        return Shortlist(self.candidates[np.asarray(idx)])


def topk_shortlist(
    probs: np.ndarray | sparse.spmatrix, k: int, *, chunk_size: int = 4096
) -> Shortlist:
    """Top-k classes per row; ties broken by lower class id.

    Works on dense arrays or sparse rows. Sparse rows with fewer than ``k``
    stored entries are padded with the lowest-id unseen classes (all probability 0).
    """
    n, c = probs.shape
    if not 0 < k <= c:
        raise ValueError(f"k must be in [1, {c}], got {k}")
    out = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, chunk_size):
        block = probs[start:start + chunk_size]
        block = np.asarray(block.todense() if sparse.issparse(block) else block)
        # Stable sort on -p keeps lower class ids first among ties.
        out[start:start + len(block)] = np.argsort(-block, axis=1, kind="stable")[:, :k]
    return Shortlist(out)
