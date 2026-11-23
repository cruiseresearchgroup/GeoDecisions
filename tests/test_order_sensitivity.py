"""order_metrics in scripts/order_sensitivity.py on hand-built per-item results."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "order_sensitivity", Path(__file__).resolve().parents[1] / "scripts" / "order_sensitivity.py")
order_sensitivity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(order_sensitivity)


def _s(values, ids=("a", "b", "c", "d")):
    return pd.Series(values, index=list(ids))


def test_shuffle_spread_and_agreement():
    hits = {"seed0": _s([1, 1, 0, 0]), "seed1": _s([1, 0, 0, 0]), "seed2": _s([1, 1, 1, 0])}
    sel = {"seed0": _s([3, 3, 5, 7]), "seed1": _s([3, 1, 5, 7]), "seed2": _s([3, 3, 2, 7])}
    m = order_sensitivity.order_metrics(hits, sel)
    assert m["n"] == 4
    assert m["acc_shuf_mean"] == pytest.approx(0.5)
    assert m["acc_shuf_range"] == pytest.approx(0.5)  # 0.75 - 0.25
    assert m["pred_agree"] == pytest.approx(0.5)  # items a and d
    assert "acc_orig" not in m


def test_original_order_gain_on_common_items():
    hits = {"seed0": _s([0, 1, 0, 0]), "seed1": _s([0, 1, 0, 0]),
            "orig": _s([1, 1, 1, 0, 1], ids=("a", "b", "c", "d", "extra"))}
    sel = {k: v * 0 for k, v in hits.items()}
    m = order_sensitivity.order_metrics(hits, sel)
    assert m["n"] == 4  # "extra" is not in every variant
    assert m["acc_orig"] == pytest.approx(0.75)
    assert m["orig_gain"] == pytest.approx(0.5)
    assert m["orig_only"] == pytest.approx(0.5)  # a, c
    assert m["shuf_only"] == pytest.approx(0.0)
