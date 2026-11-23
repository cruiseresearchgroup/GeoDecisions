"""Contamination checks 2 (quiz) and 3 (template rewording): request builders, jobs, quiz scoring."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.contamination import GEOLLM_PREFIX, TEMPLATE_REWORDINGS, quiz_requests
from src.modules.prompts import MANVI_VARIANTS, build_manvi_request, manvi_location_text
from src.runners.jobs import build_jobs


def _locations(n=6):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        nearby = "\n".join(f"{d:.1f} km North: Place {i}-{j}" for j, d in enumerate(np.sort(rng.uniform(0.5, 9, 6))))
        rows.append({"location_id": f"loc{i}", "lat": round(float(rng.uniform(-60, 60)), 5),
                     "lon": round(float(rng.uniform(-170, 170)), 5), "address": f"Town {i}, Country",
                     "nearby": nearby, "prompt_text": ""})
    return pd.DataFrame(rows)


def test_quiz_counts_rotations_and_originals():
    locs = _locations()
    reqs = quiz_requests(locs, n_instances=4, seed=0)
    assert len(reqs) == 4 + 8 * 4
    by_kind = {k: [r for r in reqs if r.metadata["quiz_kind"] == k] for k in ("template", "instance", "control")}
    assert [len(v) for v in by_kind.values()] == [4, 16, 16]
    for r in reqs:
        assert r.kind == "choice" and len(set(r.options)) == 4
    tmpl = by_kind["template"]
    assert {r.options[r.metadata["original_index"]] for r in tmpl} == {GEOLLM_PREFIX}
    assert sorted(r.metadata["original_index"] for r in tmpl) == [0, 1, 2, 3]
    assert set(tmpl[0].options) == {GEOLLM_PREFIX, *TEMPLATE_REWORDINGS}
    inst = by_kind["instance"][0]
    loc = locs.set_index("location_id").loc[inst.metadata["location_id"]].to_dict()
    loc["location_id"] = inst.metadata["location_id"]
    assert inst.options[inst.metadata["original_index"]] == manvi_location_text(loc, "full")


def test_quiz_copies_keep_distance_order_and_are_deterministic():
    reqs = quiz_requests(_locations(), n_instances=4, seed=0)
    for r in reqs:
        if r.metadata["quiz_kind"] == "template":
            continue
        for text in r.options:
            block = text.split('Nearby Places:\n"\n')[1].rstrip('\n"')
            dists = [float(line.split(" km ")[0]) for line in block.split("\n")]
            assert all(d > 0 for d in dists) and dists == sorted(dists)
    again = quiz_requests(_locations(), n_instances=4, seed=0)
    assert [r.options for r in reqs] == [r.options for r in again]


def test_manvi_variants_keep_default_and_metadata():
    loc = _locations().iloc[0].to_dict()
    base = build_manvi_request(loc, "Average Intelligence of Residents", location_format="coords")
    assert base.item_id == "loc0|Average Intelligence of Residents|coords|0.0-9.9" and "variant" not in base.metadata
    orig = build_manvi_request(loc, "Average Intelligence of Residents", location_format="coords", variant="original")
    assert orig.question == base.question and orig.item_id.endswith("|original")
    syn = build_manvi_request(loc, "Average Intelligence of Residents", location_format="coords", variant="synonyms")
    assert "How smart" in syn.question and syn.metadata["topic"] == "Average Intelligence of Residents"
    five = build_manvi_request(loc, "Average Intelligence of Residents", location_format="coords", variant="scale5")
    assert five.options == ("1", "2", "3", "4", "5") and five.metadata["scale"] == "1-5"
    rev = build_manvi_request(loc, "Average Intelligence of Residents", location_format="coords", variant="reversed")
    assert rev.metadata["reverse_scale"] is True and "0.0 is the highest" in rev.question
    assert set(MANVI_VARIANTS) == {"original", "reworded", "synonyms", "scale5", "reversed"}


def test_build_jobs_variants_and_quiz():
    cfg = {"model": {"name": "m"}, "run": {"out_dir": "out", "seeds": [0]},
           "manvi": {"location_formats": ["coords"], "variants": ["original", "scale5"]},
           "quiz": {"n_instances": 3}}
    jobs = {j.name: j for j in build_jobs(cfg)}
    assert set(jobs) == {"manvi/world/0.0-9.9/coords-original-seed0", "manvi/world/1-5/coords-scale5-seed0", "quiz/quiz"}
    assert jobs["quiz/quiz"].seed is None


def _load_report():
    spec = importlib.util.spec_from_file_location(
        "contamination_report", Path(__file__).resolve().parents[1] / "scripts" / "contamination_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_quiz_report_scores_original_picks(tmp_path):
    reqs = quiz_requests(_locations(), n_instances=2, seed=0)
    (tmp_path / "quiz").mkdir()
    with open(tmp_path / "quiz" / "quiz.jsonl", "w") as f:
        for r in reqs:
            oi = r.metadata["original_index"]
            # template: always picks the original; instance/control: always picks position 0
            pick = oi if r.metadata["quiz_kind"] == "template" else 0
            probs = [0.7 if i == pick else 0.1 for i in range(4)]
            f.write(json.dumps({"item_id": r.item_id, "options": list(r.options), "probs": probs,
                                "selected": pick, "metadata": r.metadata}) + "\n")
    df = _load_report().quiz_report(tmp_path).set_index("kind")
    assert df.loc["template", "pick_original"] == 1.0 and df.loc["template", "item_original_top"] == 1.0
    assert df.loc["instance", "pick_original"] == pytest.approx(0.25)  # position bias only
    assert df.loc["instance", "item_original_top"] == 0.0  # equal mean probabilities: no strict win
