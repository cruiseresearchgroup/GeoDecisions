"""Manvi et al. Track 2: prompt parsing, request building, ground truth, GeoLLM parity."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.data.manvi import (
    OBJECTIVE_TOPICS,
    SUBJECTIVE_TOPICS,
    TOPICS,
    ground_truth,
    load_geollm_results,
    load_manvi_locations,
)
from src.metrics import expected_value, manvi_bias_score, spearman_with_truth
from src.modules import (
    MANVI_FORMATS,
    build_manvi_request,
    build_null_request,
    manvi_original_prompt,
    manvi_scale_values,
)
from src.paths import load_paths
from tests.reference_inputs import prompt_list_digest

PATHS = load_paths()
FIXTURES = Path(__file__).resolve().parent / "fixtures"
GEOLLM_RESULTS = FIXTURES / "geollm"  # GPT-3.5 predictions shipped with GeoLLM
pytestmark = pytest.mark.skipif(not (Path(PATHS.manvi_prompt_dir) / "world_prompts.jsonl").exists(),
                                reason="Manvi prompts missing (see README: Data Preparation)")


@pytest.fixture(scope="module")
def locs():
    return load_manvi_locations("world")


@pytest.fixture(scope="module")
def geollm_ref():
    """Outputs of GeoLLM's own code (GeoLLM e5bdd45, saved to tests/fixtures)."""
    return json.loads((FIXTURES / "geollm_parity.json").read_text())


def test_topics():
    assert len(OBJECTIVE_TOPICS) == 6 and len(SUBJECTIVE_TOPICS) == 5
    assert all(TOPICS[t].raster for t in OBJECTIVE_TOPICS)


def test_world_locations_parse(locs):
    assert len(locs) == 2000 and locs["location_id"].is_unique
    assert locs["lat"].between(-90, 90).all() and locs["lon"].between(-180, 180).all()
    assert (locs["address"].str.len() > 0).all()


def test_original_prompt_matches_geollm(locs, geollm_ref):
    ours = [manvi_original_prompt(r, "Population Density") for r in locs.to_dict("records")]
    assert prompt_list_digest(ours) == geollm_ref["population_density_prompts_sha256"]


def test_full_format_reproduces_location_block(locs):
    for r in locs.head(200).to_dict("records"):
        req = build_manvi_request(r, "Average Temperature", location_format="full")
        assert req.context == r["prompt_text"].split("\n\n<TASK>")[0]


def test_request_formats_and_scales(locs):
    r = locs.iloc[0].to_dict()
    reqs = {f: build_manvi_request(r, "Infant Mortality Rate", location_format=f) for f in MANVI_FORMATS}
    assert "Address" not in reqs["coords"].context and "Coordinates" not in reqs["place_name"].context
    assert len({q.item_id for q in reqs.values()}) == 3
    assert reqs["full"].kind == "score" and len(reqs["full"].options) == 100
    five = build_manvi_request(r, "Infant Mortality Rate", scale="1-5")
    assert five.options == ("1", "2", "3", "4", "5")
    assert build_null_request(reqs["full"]).context == "Location: N/A"
    np.testing.assert_allclose(expected_value(np.full((1, 100), 0.01), manvi_scale_values("0.0-9.9")), [4.95])


def test_infant_mortality_ground_truth_and_geollm_parity(locs, geollm_ref):
    """GeoLLM-compatible extraction reproduces their ground truth; GPT-3.5's
    shipped expected-value predictions correlate positively with it."""
    if not (Path(PATHS.manvi_raster_dir) / TOPICS["Infant Mortality Rate"].raster).exists():
        pytest.skip("rasters not downloaded")
    sub = locs.head(300)
    want = np.array(geollm_ref["infant_mortality_extract_first300"], dtype=float)  # None -> nan
    got = ground_truth(sub, "Infant Mortality Rate", geollm_compat=True)
    np.testing.assert_allclose(got, want, rtol=1e-9, equal_nan=True)

    pred = load_geollm_results(GEOLLM_RESULTS / "gpt_3_5_turbo_0613_Infant_Mortality_Rate_world_prompts_expected_value.csv", locs)
    truth = ground_truth(locs, "Infant Mortality Rate")
    assert spearman_with_truth(pred.to_numpy(), truth) > 0.8  # 0.836 with masked mean extraction


def test_bias_score_on_geollm_attractiveness(locs):
    """Bias score runs on GPT-3.5's shipped attractiveness ratings (sign per the paper's anchor)."""
    if not (Path(PATHS.manvi_raster_dir) / TOPICS["Infant Mortality Rate"].raster).exists():
        pytest.skip("rasters not downloaded")
    pred = load_geollm_results(GEOLLM_RESULTS / "gpt_3_5_turbo_0613_Average_Attractiveness_of_Residents_world_prompts_expected_value.csv", locs)
    anchor = ground_truth(locs, "Infant Mortality Rate")
    assert np.isfinite(manvi_bias_score(pred.to_numpy(), anchor))


@pytest.mark.parametrize("topic", [t for t in OBJECTIVE_TOPICS if t != "Infant Mortality Rate"])
def test_other_ground_truth_rasters(locs, topic):
    if not (Path(PATHS.manvi_raster_dir) / TOPICS[topic].raster).exists():
        pytest.skip(f"{topic} raster not downloaded")
    vals = ground_truth(locs.head(200), topic)
    assert np.isfinite(vals).mean() > 0.8


def test_infant_mortality_fill_codes_masked(locs):
    if not (Path(PATHS.manvi_raster_dir) / TOPICS["Infant Mortality Rate"].raster).exists():
        pytest.skip("rasters not downloaded")
    vals = ground_truth(locs, "Infant Mortality Rate")
    assert np.nanmin(vals) >= 0
