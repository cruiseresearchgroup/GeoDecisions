"""JevClient request mapping and answer parsing, offline (fake System One client)."""

import io

import numpy as np
import pytest

pytest.importorskip("typesafe_sdk")
from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse

from src.modules import DecisionRequest, decide
from src.modules.jev import IMAGES_HEADER, JevClient, load_image
from src.modules.registry import build_model


class FakeSystemOne:
    """Answers each question with a fixed distribution favouring the *last* presented option."""

    def __init__(self):
        self.calls = []

    def system_one(self, state, questions, *, model, extra_headers=None):
        self.calls.append({"state": state, "questions": questions, "headers": extra_headers})
        q = questions["q"]
        if q.type == "noul":
            ans = NoulAnswer(type="noul", noul=0.8)
        else:
            n = len(q.criteria)
            p = np.full(n, 0.2 / (n - 1))
            p[-1] = 0.8
            if q.type == "score":
                ans = ScoreAnswer(type="score", score=float(p @ np.arange(n)), confidence=0.5,
                                  legend=dict(enumerate(q.criteria)), probabilities=dict(enumerate(p)))
            else:
                keys = list(q.criteria)
                ans = ChoiceAnswer(type="choice", choice=keys[-1], confidence=0.5,
                                   probabilities=dict(zip(keys, p)))
        return SystemOneResponse(model="jev-1.14.0", answers={"q": ans},
                                 usage={"input_tokens": 10, "output_tokens": 1})


def jev():
    return JevClient(client=FakeSystemOne(), max_concurrency=1)


def test_choice_maps_back_to_canonical_after_shuffle():
    model = jev()
    req = DecisionRequest("x", "choice", "Which?", ("a", "b", "c", "d"), context="Paris")
    r = decide(model, req, rng=np.random.default_rng(3))
    last = r.presented_order[-1]
    assert r.selected == last and r.probs[last] == pytest.approx(0.8)
    call = model.client.calls[0]
    assert call["state"] == {"location": "Paris"} and call["headers"] is None
    assert list(call["questions"]["q"].criteria) == [req.options[i] for i in r.presented_order]


def test_score_primitive_up_to_ten_levels_else_choice():
    model = jev()
    five = DecisionRequest("x", "score", "Rate", ("1", "2", "3", "4", "5"), context="c")
    r = decide(model, five)
    assert model.client.calls[-1]["questions"]["q"].type == "score"
    assert r.raw["primitive"] == "score" and r.selected == 4
    hundred = DecisionRequest("y", "score", "Rate", tuple(f"{i / 10:.1f}" for i in range(100)), context="c")
    r = decide(model, hundred)
    assert model.client.calls[-1]["questions"]["q"].type == "choice" and r.selected == 99


def test_binary_is_noul():
    r = decide(jev(), DecisionRequest("x", "binary", "Is it?", ("no", "yes"), context="c"))
    np.testing.assert_allclose(r.probs, [0.2, 0.8])


def test_empty_state_and_duplicate_options():
    model = jev()
    decide(model, DecisionRequest("x", "choice", "q", ("a", "b")))
    assert model.client.calls[-1]["state"] == "N/A"
    with pytest.raises(ValueError):
        decide(model, DecisionRequest("x", "choice", "q", ("a", "a")))


def test_image_request_sends_header_and_resized_image(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    small = tmp_path / "small.png"
    PIL.new("RGB", (100, 50), "red").save(small)
    big = tmp_path / "big.jpg"
    PIL.new("RGB", (4000, 2000), "blue").save(big)
    for path, want in [(small, (560, 280)), (big, (1568, 784))]:
        with PIL.open(io.BytesIO(bytes(load_image(path).data))) as im:
            assert im.size == want

    model = jev()
    decide(model, DecisionRequest("x", "choice", "Which?", ("a", "b"), context="c", image_path=str(big)))
    call = model.client.calls[-1]
    assert call["headers"] == IMAGES_HEADER and set(call["state"]) == {"location", "image"}


def test_registry_builds_jev_sdk():
    m = build_model({"kind": "jev_sdk", "model": "jev-1.14.0", "client": FakeSystemOne()})
    assert isinstance(m, JevClient) and m.name == "typesafe/jev-1.14.0"
