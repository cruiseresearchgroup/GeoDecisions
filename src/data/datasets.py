"""PyTorch Dataset over a LocBench split, the single entry point for evaluation loops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from src.data.locbench import LocBenchSplit, load_locbench
from src.data.sampling import Strategy, spatial_subsample
from src.data.shortlist import Shortlist, topk_shortlist
from src.paths import Paths, load_paths


def humanize_class_name(name: str) -> str:
    """'airport_hangar' -> 'airport hangar' (fMoW); other datasets unchanged."""
    return name.replace("_", " ")


class LocBenchDataset(Dataset):
    """Items for a decision-model run on LocBench, one per image.

    Each item is a dict:
        item_id, row, dataset, lat, lon, has_location, label, image_ref,
        image_path (str or None if images are not available locally),
        candidates (class ids offered as options, in canonical order),
        options (display names of ``candidates``),
        label_index (position of the label in ``candidates``, -1 if absent).

    Without a shortlist, ``candidates`` is every class (fMoW, YFCC). With one,
    it is that item's top-k image-only classes (iNat).
    """

    def __init__(
        self,
        split: LocBenchSplit,
        *,
        shortlist: Shortlist | None = None,
        image_root: str | Path | None = None,
    ) -> None:
        if shortlist is not None and len(shortlist.candidates) != len(split):
            raise ValueError("shortlist rows do not match split rows")
        self.split = split
        self.shortlist = shortlist
        self.image_root = Path(image_root) if image_root else None
        self._display = [humanize_class_name(n) for n in split.class_names]
        if shortlist is not None:
            self._label_index = shortlist.label_position(split.labels)
        else:
            self._label_index = split.labels

    @classmethod
    def from_config(
        cls,
        dataset: str,
        split: str | None = None,
        *,
        paths: Paths | None = None,
        shortlist_k: int | None = None,
        subsample: int | None = None,
        subsample_strategy: Strategy = "proportional",
        subsample_cell_deg: float = 10.0,
        seed: int = 0,
    ) -> "LocBenchDataset":
        """Load, optionally shortlist (top-k from the full split), and subsample."""
        paths = paths or load_paths()
        data = load_locbench(dataset, split, paths=paths, load_image_probs=shortlist_k is not None)
        sl = topk_shortlist(data.image_probs, shortlist_k) if shortlist_k else None
        if subsample is not None:
            idx = spatial_subsample(
                data.lat, data.lon, subsample,
                strategy=subsample_strategy, cell_deg=subsample_cell_deg, seed=seed,
            )
            data = data.subset(idx)
            sl = sl.subset(idx) if sl is not None else None
        return cls(data, shortlist=sl, image_root=paths.image_root(dataset))

    def __len__(self) -> int:
        return len(self.split)

    @property
    def label_indices(self) -> np.ndarray:
        """Per-item position of the true class among its options (-1 if absent)."""
        return np.asarray(self._label_index)

    def candidates(self, i: int) -> np.ndarray:
        if self.shortlist is not None:
            return self.shortlist.candidates[i]
        return np.arange(self.split.num_classes)

    def image_path(self, i: int) -> str | None:
        if self.image_root is None:
            return None
        p = self.image_root / str(self.split.image_refs[i])
        return str(p) if p.exists() else None

    def __getitem__(self, i: int) -> dict[str, Any]:
        s = self.split
        cands = self.candidates(i)
        return {
            "item_id": f"{s.dataset}:{s.split}:{s.rows[i]}",
            "row": int(s.rows[i]),
            "dataset": s.dataset,
            "lat": float(s.lat[i]),
            "lon": float(s.lon[i]),
            "has_location": bool(s.has_location[i]),
            "label": int(s.labels[i]),
            "image_ref": str(s.image_refs[i]),
            "image_path": self.image_path(i),
            "candidates": cands,
            "options": [self._display[c] for c in cands],
            "label_index": int(self._label_index[i]),
        }


def collate_items(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep items as a list: option lists differ in length across items."""
    return batch
