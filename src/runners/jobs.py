"""Expand a run config into jobs for Track 1 (LocBench) and Track 2 (Manvi et al.).

Output layout (under ``run.out_dir``/<model name>):
    locbench/<dataset>/<condition>-<location_format>-seed<k>.jsonl
    locbench/<dataset>/null-seed<k>.jsonl
    locbench/<dataset>/<condition>-<location_format>-orig.jsonl   (``original_order``: unshuffled labels)
    manvi/<locations>/<scale>/<location_format>-seed<k>.jsonl   (all topics)
    manvi/<locations>/<scale>/null-seed<k>.jsonl
    manvi/<locations>/<scale>/<location_format>-<variant>-seed<k>.jsonl   (``manvi.variants``: template rewordings)
    quiz/quiz.jsonl   (``quiz``: data contamination quiz, unshuffled)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.data.datasets import LocBenchDataset
from src.data.manvi import TOPICS, load_manvi_locations
from src.data.sampling import spatial_subsample
from src.modules.base import DecisionRequest
from src.data.contamination import quiz_requests
from src.modules.prompts import MANVI_VARIANTS, build_locbench_request, build_manvi_request, build_null_request
from src.runners.core import Job
from src.utils.progress import progress

log = logging.getLogger(__name__)

# Datasets whose label set exceeds what fits as options get a top-k shortlist.
SHORTLIST_DATASETS = ("inat_2017", "inat_2018")


def model_dir_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _locbench_dataset(dataset: str, cfg: dict[str, Any]) -> LocBenchDataset:
    return LocBenchDataset.from_config(
        dataset,
        shortlist_k=cfg.get("shortlist_k", 20) if dataset in SHORTLIST_DATASETS else None,
        subsample=cfg.get("subsample"),
        seed=cfg.get("subsample_seed", 0),
    )


def locbench_requests(
    dataset: str, condition: str, location_format: str, cfg: dict[str, Any], *, max_items: int | None = None
) -> list[DecisionRequest]:
    """Requests for every item usable in ``condition`` (location conditions skip items without coords).

    ``max_items`` keeps a fixed spatial subsample of those items (used for null-input runs).
    """
    ds = _locbench_dataset(dataset, cfg)
    rows = np.arange(len(ds))
    if condition != "image_only":
        rows = rows[ds.split.has_location]
    if condition != "location_only":
        # Some source images are unavailable (e.g. deleted Flickr photos); those items are skipped.
        has_image = np.array([ds.image_path(i) is not None for i in rows], dtype=bool)
        if not has_image.any():
            raise FileNotFoundError(f"{dataset}: no local images; see scripts/download_images.sh")
        if not has_image.all():
            log.warning("%s/%s: %d of %d items have no local image, skipped", dataset, condition,
                        (~has_image).sum(), len(rows))
        rows = rows[has_image]
    if max_items is not None and max_items < len(rows):
        keep = spatial_subsample(ds.split.lat[rows], ds.split.lon[rows], max_items, seed=cfg.get("subsample_seed", 0))
        rows = rows[keep]
    return [
        build_locbench_request(ds[i], condition, location_format=location_format)
        for i in progress(rows, desc=f"build {dataset}/{condition}", unit="req", leave=False)
    ]


def locbench_null_requests(dataset: str, cfg: dict[str, Any]) -> list[DecisionRequest]:
    """Null-input twins of location-only requests (same options), on a fixed spatial subsample."""
    n = (cfg.get("null_input") or {}).get("subsample")
    return [build_null_request(r) for r in locbench_requests(dataset, "location_only", "coords", cfg, max_items=n)]


def manvi_requests(location_format: str, scale: str, cfg: dict[str, Any], variant: str | None = None
                   ) -> list[DecisionRequest]:
    """``subsample: N`` keeps a fixed spatial subsample of N locations (same ids as the full set, so a
    later full run resumes from a pilot instead of re-sending it). ``variant`` selects a template rewording
    (:data:`src.modules.prompts.MANVI_VARIANTS`), which also fixes the scale."""
    df = load_manvi_locations(cfg.get("locations", "world"))
    if cfg.get("subsample"):
        keep = spatial_subsample(df["lat"].to_numpy(), df["lon"].to_numpy(), cfg["subsample"],
                                 seed=cfg.get("subsample_seed", 0))
        df = df.iloc[keep]
    locs = df.to_dict("records")
    topics = list(TOPICS) if cfg.get("topics", "all") == "all" else cfg["topics"]
    return [build_manvi_request(r, t, location_format=location_format, scale=scale, variant=variant)
            for t in topics for r in locs]


def build_jobs(config: dict[str, Any]) -> list[Job]:
    root = Path(config["run"]["out_dir"]) / model_dir_name(config["model"]["name"])
    seeds = config["run"].get("seeds", [0])
    jobs: list[Job] = []

    lb = config.get("locbench")
    if lb:
        for ds in lb["datasets"]:
            for seed in seeds:
                for cond in lb["conditions"]:
                    for fmt in lb["location_formats"]:
                        jobs.append(Job(
                            name=f"locbench/{ds}/{cond}-{fmt}-seed{seed}",
                            out_path=root / "locbench" / ds / f"{cond}-{fmt}-seed{seed}.jsonl",
                            build=lambda ds=ds, cond=cond, fmt=fmt: locbench_requests(ds, cond, fmt, lb),
                            seed=seed,
                        ))
                if lb.get("null_input"):
                    jobs.append(Job(
                        name=f"locbench/{ds}/null-seed{seed}",
                        out_path=root / "locbench" / ds / f"null-seed{seed}.jsonl",
                        build=lambda ds=ds: locbench_null_requests(ds, lb),
                        seed=seed,
                    ))
            if ds in lb.get("original_order", []):
                if ds in SHORTLIST_DATASETS:
                    # Shortlists are ranked by the TorchSpatial image model, so their canonical order
                    # puts likely answers first: an unshuffled run would leak the label.
                    raise ValueError(f"original_order is not allowed for shortlist dataset {ds!r}")
                for cond in lb["conditions"]:
                    for fmt in lb["location_formats"]:
                        jobs.append(Job(
                            name=f"locbench/{ds}/{cond}-{fmt}-orig",
                            out_path=root / "locbench" / ds / f"{cond}-{fmt}-orig.jsonl",
                            build=lambda ds=ds, cond=cond, fmt=fmt: locbench_requests(ds, cond, fmt, lb),
                            seed=None,  # item_order(seed=None) keeps the dataset's class order
                        ))

    mv = config.get("manvi")
    if mv and mv.get("variants"):
        # Template-rewording check: one job per variant and format, unshuffled score questions as usual.
        loc_name = mv.get("locations", "world")
        for variant in mv["variants"]:
            scale = MANVI_VARIANTS[variant][3]
            for seed in seeds:
                for fmt in mv["location_formats"]:
                    jobs.append(Job(
                        name=f"manvi/{loc_name}/{scale}/{fmt}-{variant}-seed{seed}",
                        out_path=root / "manvi" / loc_name / scale / f"{fmt}-{variant}-seed{seed}.jsonl",
                        build=lambda fmt=fmt, scale=scale, variant=variant: manvi_requests(fmt, scale, mv, variant),
                        seed=seed,
                    ))
    elif mv:
        loc_name = mv.get("locations", "world")
        for scale in mv.get("scales", ["0.0-9.9"]):
            for seed in seeds:
                for fmt in mv["location_formats"]:
                    jobs.append(Job(
                        name=f"manvi/{loc_name}/{scale}/{fmt}-seed{seed}",
                        out_path=root / "manvi" / loc_name / scale / f"{fmt}-seed{seed}.jsonl",
                        build=lambda fmt=fmt, scale=scale: manvi_requests(fmt, scale, mv),
                        seed=seed,
                    ))
                if mv.get("null_input"):
                    jobs.append(Job(
                        name=f"manvi/{loc_name}/{scale}/null-seed{seed}",
                        out_path=root / "manvi" / loc_name / scale / f"null-seed{seed}.jsonl",
                        # The prior depends on topic and scale, not location: one null per topic.
                        build=lambda scale=scale: _manvi_null_requests(scale, mv),
                        seed=seed,
                    ))

    quiz = config.get("quiz")
    if quiz:
        # Contamination quiz: options carry their own rotation, so they must not be shuffled (seed None).
        jobs.append(Job(
            name="quiz/quiz",
            out_path=root / "quiz" / "quiz.jsonl",
            build=lambda: quiz_requests(load_manvi_locations(quiz.get("locations", "world")),
                                        n_instances=quiz.get("n_instances", 200), seed=quiz.get("seed", 0)),
            seed=None,
        ))
    return jobs


def _manvi_null_requests(scale: str, cfg: dict[str, Any]) -> list[DecisionRequest]:
    """One null-input request per topic: the option prior depends on topic and scale, not location."""
    by_topic: dict[str, DecisionRequest] = {}
    for r in manvi_requests("full", scale, cfg):
        by_topic.setdefault(r.metadata["topic"], r)
    return [build_null_request(r) for r in by_topic.values()]
