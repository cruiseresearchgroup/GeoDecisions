import numpy as np
from scipy import sparse

from src.data.sampling import grid_cell_ids, spatial_subsample
from src.data.shortlist import topk_shortlist


def test_grid_cells_handle_missing():
    ids = grid_cell_ids(np.array([0.0, np.nan, 89.9]), np.array([0.0, 1.0, 179.9]), 10.0)
    assert ids[1] == -1 and ids[0] >= 0 and ids[2] == 17 * 36 + 35


def test_subsample_sizes_and_proportions():
    rng = np.random.default_rng(0)
    lat = np.concatenate([rng.uniform(0, 9, 900), rng.uniform(50, 59, 100), [np.nan]])
    lon = np.concatenate([rng.uniform(0, 9, 900), rng.uniform(50, 59, 100), [0.0]])
    for strategy in ("uniform", "proportional", "balanced"):
        idx = spatial_subsample(lat, lon, 100, strategy=strategy, seed=1)
        assert len(idx) == 100 and len(np.unique(idx)) == 100 and 1000 not in idx
    prop = spatial_subsample(lat, lon, 100, strategy="proportional")
    assert (prop >= 900).sum() == 10
    bal = spatial_subsample(lat, lon, 100, strategy="balanced")
    assert (bal >= 900).sum() == 50
    np.testing.assert_array_equal(spatial_subsample(lat, lon, 100, seed=3), spatial_subsample(lat, lon, 100, seed=3))


def test_topk_shortlist_dense_sparse_agree():
    probs = np.array([[0.1, 0.5, 0.0, 0.4], [0.0, 0.0, 1.0, 0.0]])
    dense = topk_shortlist(probs, 3)
    sp = topk_shortlist(sparse.csr_matrix(probs), 3)
    np.testing.assert_array_equal(dense.candidates, sp.candidates)
    np.testing.assert_array_equal(dense.candidates, [[1, 3, 0], [2, 0, 1]])
    np.testing.assert_array_equal(dense.label_position(np.array([0, 3])), [2, -1])
    assert dense.recall(np.array([0, 3])) == 0.5
