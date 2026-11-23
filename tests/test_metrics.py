import json
from pathlib import Path

import numpy as np
import pytest

from src.metrics import (
    brier_score,
    effective_num_labels,
    expected_calibration_error,
    label_ranks,
    manvi_bias_score,
    per_sample_table,
    summarize,
)
from src.metrics.geobias import binary_marks, marked_ssi, sre_scores, ssi_scores, unmarked_ssi
from src.metrics.geobias.geometry import initial_bearing
from tests.reference_inputs import cluster, geobs_ssi_inputs

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_ranks_and_table():
    probs = np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    label_index = np.array([1, 2, -1])
    np.testing.assert_array_equal(label_ranks(probs, label_index), [1, 3, np.inf])
    t = per_sample_table(probs, label_index)
    np.testing.assert_allclose(t["reciprocal_rank"], [1, 1 / 3, 0])
    np.testing.assert_array_equal(t["hit@1"], [1, 0, 0])
    np.testing.assert_array_equal(t["hit@3"], [1, 1, 0])
    np.testing.assert_allclose(t["true_class_prob"], [0.7, 0.2, 0.0])
    s = summarize(t)
    assert s["top1"] == pytest.approx(1 / 3) and s["mrr"] == pytest.approx(4 / 9)


def test_tie_modes():
    probs = np.array([[0.4, 0.4, 0.2]])
    assert label_ranks(probs, np.array([1]))[0] == 1
    assert label_ranks(probs, np.array([1]), ties="pessimistic")[0] == 2


def test_calibration_perfect_and_overconfident():
    labels = np.array([0, 1, 2, 0])
    onehot = np.eye(3)[labels]
    assert brier_score(onehot, labels) == 0
    assert expected_calibration_error(onehot, labels) == 0
    wrong = np.eye(3)[(labels + 1) % 3]
    assert brier_score(wrong, labels) == pytest.approx(2.0)
    assert expected_calibration_error(wrong, labels) == pytest.approx(1.0)


def test_effective_num_labels():
    assert effective_num_labels(np.array([0, 1, 2, 3])) == pytest.approx(4)
    assert effective_num_labels(np.array([2, 2, 2])) == pytest.approx(1)


def test_manvi_flat_map_is_zero():
    anchor = np.arange(10.0)
    assert manvi_bias_score(np.full(10, 5.0), anchor) == 0
    assert manvi_bias_score(anchor, anchor) == pytest.approx(2.5)


def test_ssi_matches_geobs_reference():
    """Our port reproduces GeoBS's own SSI functions (tests/fixtures/geobs_ssi.json)."""
    ref = json.loads((FIXTURES / "geobs_ssi.json").read_text())
    pts, marks, radius = geobs_ssi_inputs()
    ours = [unmarked_ssi(pts, pts[0], radius), marked_ssi(pts, marks, pts[0], radius)]
    np.testing.assert_allclose(ours, [ref["unmarked"], ref["marked"]], rtol=1e-6)


def test_marked_ssi_higher_for_clustered_marks():
    lat, lon = cluster()
    rng = np.random.default_rng(1)
    clustered = ssi_scores(lat, lon, binary_marks(lon > -100), radius_km=60, centers=np.arange(20))
    shuffled = ssi_scores(lat, lon, binary_marks(rng.permutation(lon > -100)), radius_km=60, centers=np.arange(20))
    assert np.nanmean(clustered) > np.nanmean(shuffled)


def test_ssi_skips_missing_coords():
    lat, lon = cluster(n=50)
    lat[3] = np.nan
    out = ssi_scores(lat, lon, radius_km=60, centers=np.array([2, 3]))
    assert np.isfinite(out[2]) and np.isnan(out[3])


@pytest.mark.parametrize("kind", ["scale_grid", "distance_lag", "direction_sector"])
def test_sre_segregated_beats_shuffled(kind):
    lat, lon = cluster(n=600)
    rng = np.random.default_rng(2)
    good = (lon > -100) if kind != "distance_lag" else (np.hypot(lat - 40, lon + 100) < 0.4)
    kw = dict(radius_km=120, scale_km=40, lag_km=30, centers=np.arange(30))
    seg = sre_scores(lat, lon, good.astype(float), kind, **kw).aggregate()
    shuf = sre_scores(lat, lon, rng.permutation(good).astype(float), kind, **kw).aggregate()
    assert seg > shuf


def test_initial_bearing_cardinal_directions():
    c = np.deg2rad([-30.0, 20.0])  # southern hemisphere on purpose
    pts = np.deg2rad([[-29.0, 20.0], [-30.0, 21.0], [-31.0, 20.0], [-30.0, 19.0]])
    np.testing.assert_allclose(np.rad2deg(initial_bearing(pts, c)), [0, 90, 180, -90], atol=0.6)


def test_shuffle_excess_separates_clustered_from_random_marks():
    """Excess (score - shuffled score) ~0 for spatially random marks, large when clustered,
    regardless of the overall hit rate."""
    from src.evaluation import geobias_scores

    rng = np.random.default_rng(0)
    n = 2000
    lat, lon = 40 + rng.normal(0, 1.5, n), -100 + rng.normal(0, 1.5, n)
    centers = rng.choice(n, 60, replace=False)

    def excess(good):
        r = geobias_scores(lat, lon, binary_marks(good), good.astype(float), centers=centers, include_unmarked=False)
        return r["ssi_marked_excess"], r["sre_scale_grid_excess"]

    rand_ssi, rand_sre = excess(rng.random(n) < 0.06)
    clus_ssi, clus_sre = excess(lon > -100)
    assert abs(rand_ssi) < 3 and abs(rand_sre) < 0.1
    assert clus_ssi > 10 and clus_sre > 1


def test_kl_sre_float_noise_range_is_constant():
    from src.metrics.geobias.sre import kl_sre

    v = np.array([0.07] * 30 + [0.07000000000000001] * 30)  # 2-decimal API probabilities after renormalizing
    assert kl_sre([v[:30], v[30:]], v, bins=4, min_count=5) == kl_sre([v[:30], v[:30]], np.full(60, 0.07), bins=4, min_count=5)


def test_manvi_bias_scales_with_squared_answer_rate():
    rng = np.random.default_rng(0)
    anchor = rng.normal(size=500)
    ratings = 5 - anchor + rng.normal(scale=0.1, size=500)
    full = manvi_bias_score(ratings, anchor)
    assert full < 0
    assert np.isclose(manvi_bias_score(ratings, anchor, answer_rate=0.5), 0.25 * full)
