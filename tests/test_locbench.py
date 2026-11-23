"""Checks against the real LocBench data and TorchSpatial results (skipped if absent).

The key regression test: our loader + metrics applied to the shipped image-only
probabilities reproduce TorchSpatial's ``no_prior`` eval CSV row by row, which
pins down row order, filtering, labels and the metric definitions.
"""

import numpy as np
import pandas as pd
import pytest

from src.data import (
    DATASETS,
    LocBenchDataset,
    available_encoders,
    load_baseline,
    load_locbench,
)
from src.data.locbench import NUM_CLASSES
from src.metrics import per_sample_table
from src.paths import load_paths

PATHS = load_paths()
pytestmark = pytest.mark.data

EXPECTED_ROWS = {"fmow": 53041, "yfcc": 17798, "inat_2017": 95986, "inat_2018": 24426}


def _need(dataset):
    if not PATHS.dataset_dir(dataset).exists():
        pytest.skip(f"{dataset} not downloaded")
    if not PATHS.torchspatial_eval_dir.exists():
        pytest.skip("TorchSpatial eval results not present")


@pytest.mark.parametrize("dataset", DATASETS)
def test_image_only_reproduces_torchspatial(dataset):
    _need(dataset)
    data = load_locbench(dataset)
    assert len(data) == EXPECTED_ROWS[dataset]
    assert data.num_classes == NUM_CLASSES[dataset]

    ours = per_sample_table(data.image_probs, data.labels)
    ref = load_baseline(data, "no_prior")
    np.testing.assert_array_equal(ours["hit@1"].to_numpy(), ref["hit@1"].to_numpy())
    np.testing.assert_allclose(ours["true_class_prob"], ref["true_class_prob"], rtol=1e-4, atol=1e-6)
    # Ranks can differ only where probabilities tie (TorchSpatial's argsort is unstable).
    assert abs(ours["reciprocal_rank"].mean() - ref["reciprocal_rank"].mean()) < 1e-3


@pytest.mark.parametrize("dataset", DATASETS)
def test_encoder_baselines_align(dataset):
    _need(dataset)
    data = load_locbench(dataset, load_image_probs=False)
    encoders = [e for e in available_encoders(dataset) if e != "no_prior"]
    assert encoders
    for enc in encoders:
        df = load_baseline(data, enc)  # raises on coordinate mismatch
        assert len(df) == len(data)


def test_subset_baseline_alignment():
    _need("fmow")
    data = load_locbench("fmow", load_image_probs=False)
    sub = data.subset(np.array([5, 17, 40000]))
    df = load_baseline(sub, "rbf")
    full = load_baseline(data, "rbf")
    pd.testing.assert_frame_equal(df.drop(columns="item_id"), full.iloc[[5, 17, 40000]].reset_index(drop=True).drop(columns="item_id"))


def test_dataset_items_with_shortlist():
    _need("inat_2018")
    ds = LocBenchDataset.from_config("inat_2018", shortlist_k=20, subsample=200, seed=0)
    assert len(ds) == 200
    item = ds[0]
    assert len(item["options"]) == 20 and len(item["candidates"]) == 20
    if item["label_index"] >= 0:
        assert item["candidates"][item["label_index"]] == item["label"]
    assert 0.5 < ds.shortlist.recall(ds.split.labels) <= 1.0


def test_fmow_items_full_label_set():
    _need("fmow")
    ds = LocBenchDataset.from_config("fmow", subsample=50)
    item = ds[3]
    assert len(item["options"]) == 62 and item["label_index"] == item["label"]
    assert item["image_ref"].startswith("val/")


def test_report_pipeline_with_fake_model():
    """End-to-end wiring (dataset -> requests -> decide -> table -> report) with a stub model."""
    _need("fmow")
    from src.evaluation import report, results_to_table
    from src.modules import DecisionModel, ModelOutput, build_locbench_request, decide_batch

    class Uniformish(DecisionModel):
        name = "stub"

        def predict(self, request):
            p = np.linspace(1, 2, len(request.options))
            return ModelOutput(probs=p / p.sum())

    ds = LocBenchDataset.from_config("fmow", subsample=300, seed=0)
    reqs = [build_locbench_request(ds[i], "location_only") for i in range(len(ds))]
    results = decide_batch(Uniformish(), reqs, rng=np.random.default_rng(0))
    table = results_to_table(results, ds)
    assert len(table) == 300 and table["brier"].between(0, 2).all()
    out = report(table, centers=np.arange(10), include_sre=True)
    for key in ("top1", "mrr", "brier", "ssi_unmarked", "ssi_marked", "sre_scale_grid"):
        assert key in out

    baseline = load_baseline(ds.split, "rbf")
    assert np.isfinite(report(baseline, centers=np.arange(10), include_sre=False)["ssi_marked"])
