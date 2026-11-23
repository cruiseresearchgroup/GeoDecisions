"""Unified loader for the LocBench geo-aware image classification datasets.

Reads the TorchSpatial/LocBench release (fMoW, YFCC) and the geo_prior release
(iNat2017, iNat2018) into one :class:`LocBenchSplit` structure.

Row order and filtering follow TorchSpatial's ``datasets.load_dataset`` so that
row ``i`` here is row ``i`` in TorchSpatial's per-sample eval CSVs
(``eval_results/classification/eval_<dataset>__<split>_<encoder>.csv``).
Those CSVs keep rows without a location (NaN lon/lat), so by default we do too
and expose :attr:`LocBenchSplit.has_location` instead of dropping them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from src.paths import Paths, load_paths

DATASETS = ("fmow", "yfcc", "inat_2017", "inat_2018")

# Split that TorchSpatial reports results on (and ships eval CSVs for).
EVAL_SPLIT = {"fmow": "val", "yfcc": "test", "inat_2017": "val", "inat_2018": "val"}

SPLITS = {
    "fmow": ("train", "val"),
    "yfcc": ("train", "val", "test"),
    "inat_2017": ("train", "val"),
    "inat_2018": ("train", "val"),
}

NUM_CLASSES = {"fmow": 62, "yfcc": 100, "inat_2017": 5089, "inat_2018": 8142}


@dataclass
class LocBenchSplit:
    """One split of one LocBench dataset, row-aligned with TorchSpatial outputs.

    Attributes:
        dataset, split: identifiers, e.g. ``("fmow", "val")``.
        image_ids: source image ids (iNat/fMoW annotation ids; YFCC photo ids).
        lat, lon: degrees, float64; NaN where the source has no location.
        labels: int64 class ids, indexing into ``class_names``.
        class_names: class id -> raw class name.
        image_refs: image path relative to the dataset's raw-image root.
        image_probs: image-only classifier probabilities, (N, C) dense ndarray or
            CSR matrix (iNat). ``None`` if not loaded.
        features: image-only classifier penultimate features (N, D), memory-mapped.
            ``None`` if not loaded.
        meta: per-row extras (user_id, date_c, ...), row-aligned.
        rows: row positions in the full TorchSpatial-ordered split. Lets
            subsets map back to TorchSpatial eval CSV rows.
    """

    dataset: str
    split: str
    image_ids: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    labels: np.ndarray
    class_names: list[str]
    image_refs: np.ndarray
    image_probs: np.ndarray | sparse.csr_matrix | None = None
    features: np.ndarray | None = None
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    rows: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.rows is None:
            self.rows = np.arange(len(self.labels))

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def has_location(self) -> np.ndarray:
        return ~(np.isnan(self.lat) | np.isnan(self.lon))

    @property
    def item_ids(self) -> list[str]:
        """Globally unique, stable ids: ``<dataset>:<split>:<row>``."""
        return [f"{self.dataset}:{self.split}:{r}" for r in self.rows]

    def coords_rad(self) -> np.ndarray:
        """(N, 2) array of (lat, lon) in radians, the GeoBS convention."""
        return np.deg2rad(np.stack([self.lat, self.lon], axis=1))

    def image_probs_dense(self, idx: int | np.ndarray | slice) -> np.ndarray:
        """Dense image-only probabilities for the given rows."""
        if self.image_probs is None:
            raise ValueError("image_probs not loaded; pass load_image_probs=True")
        p = self.image_probs[idx]
        out = np.asarray(p.todense() if sparse.issparse(p) else p, dtype=np.float32)
        # A sparse matrix row stays 2-D (1, C); a single index returns one 1-D row, as for dense arrays.
        return out.reshape(-1) if isinstance(idx, (int, np.integer)) else out

    def subset(self, idx: np.ndarray) -> "LocBenchSplit":
        """Row subset that keeps every field aligned (and remembers source rows)."""
        idx = np.asarray(idx)
        return replace(
            self,
            image_ids=self.image_ids[idx],
            lat=self.lat[idx],
            lon=self.lon[idx],
            labels=self.labels[idx],
            image_refs=self.image_refs[idx],
            image_probs=None if self.image_probs is None else self.image_probs[idx],
            features=None if self.features is None else self.features[idx],
            meta=self.meta.iloc[idx].reset_index(drop=True),
            rows=self.rows[idx],
        )

    def to_frame(self) -> pd.DataFrame:
        """Tabular view of the per-row metadata (no probabilities/features)."""
        df = pd.DataFrame(
            {
                "item_id": self.item_ids,
                "row": self.rows,
                "image_id": self.image_ids,
                "lat": self.lat,
                "lon": self.lon,
                "label": self.labels,
                "class_name": [self.class_names[c] for c in self.labels],
                "image_ref": self.image_refs,
            }
        )
        return pd.concat([df, self.meta.reset_index(drop=True)], axis=1)


# ---------------------------------------------------------------------------
# Per-dataset readers. Each returns the raw fields in TorchSpatial row order.
# ---------------------------------------------------------------------------


def _read_coco_style(
    data_dir: Path, loc_file: str, ann_file: str, remove_invalid: bool
) -> dict[str, Any]:
    """Port of TorchSpatial ``datasets.load_inat_data`` (used for iNat and fMoW).

    Row order is ``ann["images"]`` order. With ``remove_invalid``, rows whose
    location or user id is missing are dropped (TorchSpatial's ``remove_empty``).
    """
    with open(data_dir / loc_file) as f:
        loc_by_id = {ll["id"]: ll for ll in json.load(f)}
    with open(data_dir / ann_file) as f:
        ann = json.load(f)
    label_by_id = {a["image_id"]: a["category_id"] for a in ann["annotations"]}

    keep, lat, lon, labels, ids, refs, users, dates = [], [], [], [], [], [], [], []
    for ii, img in enumerate(ann["images"]):
        loc = loc_by_id[img["id"]]
        if remove_invalid and (loc["lon"] is None or loc["user_id"] is None):
            continue
        keep.append(ii)
        lat.append(np.nan if loc["lat"] is None else loc["lat"])
        lon.append(np.nan if loc["lon"] is None else loc["lon"])
        labels.append(label_by_id[int(img["id"])])
        ids.append(img["id"])
        refs.append(img["file_name"])
        users.append(-1 if loc["user_id"] is None else loc["user_id"])
        dates.append(loc.get("date_c", np.nan))

    return {
        "keep": np.asarray(keep, dtype=np.int64),
        "lat": np.asarray(lat, dtype=np.float64),
        "lon": np.asarray(lon, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.int64),
        "image_ids": np.asarray(ids),
        "image_refs": np.asarray(refs, dtype=object),
        "meta": pd.DataFrame({"user_id": users, "date_c": dates}),
    }


def _fmow_ref(file_name: str) -> str:
    # '/atlas/u/buzkent/fmow-rgb/val/<cat>/<inst>/<img>.jpg' -> 'val/<cat>/<inst>/<img>.jpg'
    marker = "fmow-rgb/"
    pos = file_name.find(marker)
    return file_name[pos + len(marker):] if pos >= 0 else file_name


def _read_fmow(d: Path, split: str, remove_invalid: bool) -> dict[str, Any]:
    if split == "test":
        raise ValueError("fMoW test split is not available in LocBench")
    out = _read_coco_style(d, f"{split}_location.json", f"{split}.json", remove_invalid)
    out["image_refs"] = np.asarray([_fmow_ref(r) for r in out["image_refs"]], dtype=object)
    with open(d / "category.json") as f:
        name_to_id = json.load(f)
    out["class_names"] = _names_by_id({v: k for k, v in name_to_id.items()})
    out["probs_file"] = d / "feature_moco" / f"fmow_{split}_preds.npy"
    out["feats_file"] = d / "features_inception_pretrain" / f"fmow_{split}_pretrain_net_feats.npy"
    return out


def _read_yfcc(d: Path, split: str, remove_invalid: bool) -> dict[str, Any]:
    # Port of TorchSpatial ``load_yfcc_data``: rows of train_test_split.csv in file order.
    df = pd.read_csv(d / "train_test_split.csv")
    df = df[df["split"] == split].reset_index(drop=True)
    names = pd.read_csv(d / "class_names.csv")
    return {
        "keep": np.arange(len(df)),
        "lat": df["lat"].to_numpy(np.float64),
        "lon": df["lon"].to_numpy(np.float64),
        "labels": df["class"].to_numpy(np.int64),
        "image_ids": df["path"].map(lambda p: Path(p).stem).to_numpy(),
        "image_refs": df["path"].to_numpy(object),
        "meta": pd.DataFrame(index=range(len(df))),
        "class_names": _names_by_id(dict(zip(names["id"], names["name"]))),
        "probs_file": d / "features_inception" / f"YFCC_{split}_preds.npy",
        "feats_file": d / "features_inception" / f"YFCC_{split}_net_feats.npy",
    }


def _read_inat(year: str):
    def reader(d: Path, split: str, remove_invalid: bool) -> dict[str, Any]:
        if split == "test":
            raise ValueError(f"iNat{year} test labels are not public")
        out = _read_coco_style(
            d, f"{split}{year}_locations.json", f"{split}{year}.json", remove_invalid
        )
        with open(d / f"categories{year}.json") as f:
            cats = json.load(f)
        out["class_names"] = _names_by_id({c["id"]: c["name"] for c in cats})
        out["probs_file"] = d / "features_inception" / f"inat{year}_{split}_preds_sparse.npz"
        out["feats_file"] = d / "features_inception" / f"inat{year}_{split}_net_feats.npy"
        return out

    return reader


_READERS = {
    "fmow": _read_fmow,
    "yfcc": _read_yfcc,
    "inat_2017": _read_inat("2017"),
    "inat_2018": _read_inat("2018"),
}


def _names_by_id(id_to_name: dict[int, str]) -> list[str]:
    ids = sorted(id_to_name)
    if ids != list(range(len(ids))):
        raise ValueError("class ids are not contiguous from 0")
    return [id_to_name[i] for i in ids]


def _load_probs(path: Path) -> np.ndarray | sparse.csr_matrix:
    if path.suffix == ".npz":
        return sparse.load_npz(path).tocsr().astype(np.float32)
    return np.load(path).astype(np.float32)


def load_locbench(
    dataset: str,
    split: str | None = None,
    *,
    paths: Paths | None = None,
    load_image_probs: bool = True,
    load_features: bool = False,
    remove_invalid: bool = False,
) -> LocBenchSplit:
    """Load one LocBench split.

    Args:
        dataset: one of :data:`DATASETS`.
        split: defaults to TorchSpatial's evaluation split (:data:`EVAL_SPLIT`).
        load_image_probs: load the image-only classifier's class probabilities
            (TorchSpatial's ``no_prior`` model; source of the iNat shortlist).
        load_features: memory-map the image classifier features.
        remove_invalid: drop rows without location/user, like TorchSpatial's
            ``remove_invalid``. Off by default because TorchSpatial's eval CSVs
            keep those rows.
    """
    if dataset not in _READERS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {DATASETS}")
    split = split or EVAL_SPLIT[dataset]
    if split not in SPLITS[dataset]:
        raise ValueError(f"{dataset} has splits {SPLITS[dataset]}, not {split!r}")

    paths = paths or load_paths()
    d = paths.dataset_dir(dataset)
    raw = _READERS[dataset](d, split, remove_invalid)
    keep = raw["keep"]

    probs = feats = None
    if load_image_probs:
        probs = _load_probs(raw["probs_file"])
        if probs.shape[0] != len(keep):
            probs = probs[keep]
    if load_features:
        feats = np.load(raw["feats_file"], mmap_mode="r")
        if feats.shape[0] != len(keep):
            feats = feats[keep]

    out = LocBenchSplit(
        dataset=dataset,
        split=split,
        image_ids=raw["image_ids"],
        lat=raw["lat"],
        lon=raw["lon"],
        labels=raw["labels"],
        class_names=raw["class_names"],
        image_refs=raw["image_refs"],
        image_probs=probs,
        features=feats,
        meta=raw["meta"],
    )
    if len(out.class_names) != NUM_CLASSES[dataset]:
        raise ValueError(
            f"{dataset}: expected {NUM_CLASSES[dataset]} classes, got {len(out.class_names)}"
        )
    for name, arr in (("image_probs", probs), ("features", feats)):
        if arr is not None and arr.shape[0] != len(out):
            raise ValueError(f"{dataset}/{split}: {name} has {arr.shape[0]} rows, expected {len(out)}")
    return out
