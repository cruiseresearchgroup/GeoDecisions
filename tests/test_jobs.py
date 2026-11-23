"""build_jobs: option-order variants for the order-sensitivity test (configs/order/)."""

import pytest

from src.runners.jobs import build_jobs


def _config(**locbench):
    lb = {
        "datasets": ["fmow", "yfcc", "inat_2017"],
        "conditions": ["image_only", "image_location"],
        "location_formats": ["coords"],
        "subsample": 2000,
    }
    lb.update(locbench)
    return {"model": {"name": "m"}, "run": {"out_dir": "out", "seeds": [1, 2]}, "locbench": lb}


def test_seeds_only_by_default():
    jobs = build_jobs(_config())
    assert len(jobs) == 3 * 2 * 2  # datasets x seeds x conditions
    assert {j.seed for j in jobs} == {1, 2}
    assert not any(j.name.endswith("-orig") for j in jobs)


def test_original_order_adds_unshuffled_jobs():
    jobs = build_jobs(_config(original_order=["fmow", "yfcc"]))
    orig = [j for j in jobs if j.name.endswith("-orig")]
    assert sorted(j.name for j in orig) == [
        "locbench/fmow/image_location-coords-orig", "locbench/fmow/image_only-coords-orig",
        "locbench/yfcc/image_location-coords-orig", "locbench/yfcc/image_only-coords-orig",
    ]
    assert all(j.seed is None for j in orig)
    assert str(orig[0].out_path).endswith("-orig.jsonl")


def test_original_order_rejects_shortlist_datasets():
    with pytest.raises(ValueError, match="shortlist"):
        build_jobs(_config(original_order=["inat_2017"]))
