"""Closed-API clients against the documented request/response shapes (mock transport, no network)."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest
from PIL import Image

from src.modules.api import APIError
from src.modules.openai_decision import OpenAIDecisionsClient
from src.modules.typesafe_jev import JevClient
from src.modules.base import DecisionRequest

CHOICE = DecisionRequest("i1", "choice", "Which class?", ("cat", "dog", "bird"), context="Coordinates: (1.0, 2.0)")
SCORE100 = DecisionRequest("i2", "score", "Rate it.", tuple(f"{i / 10:.1f}" for i in range(100)), context="x")
SCORE3 = DecisionRequest("i3", "score", "Rate it.", ("low", "mid", "high"), context="x")
BINARY = DecisionRequest("i4", "binary", "Is it?", ("no", "yes"), context="x")


def mock(client, respond):
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append((body, dict(request.headers)))
        return respond(body)

    client._http = httpx.Client(transport=httpx.MockTransport(handler))
    return sent


def jev_answer(body):
    q = body["questions"]["q"]
    if q["type"] == "choice":
        keys = list(q["criteria"])
        probs = dict(zip(keys, np.linspace(1, 2, len(keys)) / np.linspace(1, 2, len(keys)).sum()))
        return httpx.Response(200, json={"model": "jev-1.14.0", "answers": {"q": {
            "type": "choice", "choice": keys[-1], "confidence": 0.1, "probabilities": probs}},
            "usage": {"input_tokens": 100, "output_tokens": 0}})
    if q["type"] == "score":
        n = len(q["criteria"])
        return httpx.Response(200, json={"answers": {"q": {"type": "score", "score": 1.0, "confidence": 0.5,
            "legend": {str(i): c for i, c in enumerate(q["criteria"])},
            "probabilities": {str(i): 1 / n for i in range(n)}}}, "usage": {"input_tokens": 5}})
    return httpx.Response(200, json={"answers": {"q": {"type": "noul", "noul": 0.8}}, "usage": {"input_tokens": 5}})


def luna_answer(body):
    q = body["questions"][0]
    if q["type"] == "choice":
        vals = [c["value"] for c in q["choices"]]
        ps = [{"value": v, "probability": (i + 1) / sum(range(1, len(vals) + 1))} for i, v in enumerate(vals)]
        ans = {"type": "choice", "name": "q", "choice": vals[-1], "probabilities": ps, "confidence": 0.2}
    elif q["type"] == "score":
        ans = {"type": "score", "name": "q", "score": 1.0, "confidence": 0.5, "probabilities": [
            {"value": i, "label": lv["label"], "probability": 1 / len(q["levels"])} for i, lv in enumerate(q["levels"])]}
    else:
        ans = {"type": "predicate", "name": "q", "probability": 0.8}
    return httpx.Response(200, json={"answers": [ans], "model": "gpt-6-luna", "usage": {"input_tokens": 7}})


@pytest.fixture
def jev():
    return JevClient("jev-1.14.0", api_key="k", score_as_choice=True)


@pytest.fixture
def luna():
    return OpenAIDecisionsClient("gpt-6-luna", api_key="k", score_as_choice=True)


def test_jev_choice_payload_and_parse(jev):
    sent = mock(jev, jev_answer)
    out = jev.predict(CHOICE)
    body, headers = sent[0]
    assert body == {"model": "jev-1.14.0", "state": CHOICE.context, "questions": {"q": {
        "type": "choice", "criteria": {"cat": None, "dog": None, "bird": None}, "instructions": "Which class?"}}}
    assert headers["authorization"] == "Bearer k" and "typesafe-danger-zone-images" not in headers
    np.testing.assert_allclose(out.probs, np.array([1, 1.5, 2]) / 4.5)
    assert out.selected == 2 and out.raw["usage"]["input_tokens"] == 100


def test_score_levels_cap(jev, luna):
    for client in (jev, luna):
        client.score_as_choice = False
        with pytest.raises(ValueError, match="at most 10 levels"):
            client.predict(SCORE100)


def test_jev_score_and_noul(jev):
    sent = mock(jev, jev_answer)
    np.testing.assert_allclose(jev.predict(SCORE3).probs, [1 / 3] * 3)
    assert sent[0][0]["questions"]["q"]["criteria"] == ["low", "mid", "high"]
    out = jev.predict(SCORE100)  # sent as choice
    assert sent[1][0]["questions"]["q"]["type"] == "choice" and out.raw["sent_as"] == "choice"
    assert len(out.probs) == 100
    np.testing.assert_allclose(jev.predict(BINARY).probs, [0.2, 0.8])
    assert sent[2][0]["questions"]["q"]["criteria"] == {"false": "no", "true": "yes"}


def test_jev_image(jev, tmp_path):
    img = tmp_path / "a.jpg"
    Image.new("RGB", (300, 300)).save(img)
    small = tmp_path / "b.jpg"
    Image.new("RGB", (8, 8)).save(small)
    sent = mock(jev, jev_answer)
    jev.predict(DecisionRequest("i5", "choice", "Which?", ("a", "b"), context="Coordinates: (1, 2)", image_path=str(img)))
    body, headers = sent[0]
    (ph, asset), = body["assets"].items()
    assert asset["type"] == "image" and asset["format"] == "jpeg" and asset["data_base64"]  # in limits: unchanged
    jev.predict(DecisionRequest("i7", "choice", "Which?", ("a", "b"), image_path=str(small)))
    (asset2,) = sent[1][0]["assets"].values()
    assert asset2["format"] == "png"  # below Jev's 280 px minimum: resized
    assert body["state"] == f"{ph}\n\nCoordinates: (1, 2)"
    assert headers["typesafe-danger-zone-images"] == "true"


def test_luna_choice_score_predicate(luna):
    sent = mock(luna, luna_answer)
    out = luna.predict(CHOICE)
    np.testing.assert_allclose(out.probs, np.array([1, 2, 3]) / 6)
    assert out.selected == 2
    body = sent[0][0]
    assert body["model"] == "gpt-6-luna" and body["input"] == CHOICE.context
    assert body["questions"] == [{"name": "q", "instructions": "Which class?", "type": "choice",
                                  "choices": [{"value": o, "description": o} for o in CHOICE.options]}]
    np.testing.assert_allclose(luna.predict(SCORE3).probs, [1 / 3] * 3)
    assert sent[1][0]["questions"][0]["levels"][0] == {"label": "low", "description": "low"}
    assert luna.predict(SCORE100).raw["sent_as"] == "choice"
    np.testing.assert_allclose(luna.predict(BINARY).probs, [0.2, 0.8])


def test_luna_image_and_refusal(luna, tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(img)
    sent = mock(luna, luna_answer)
    luna.predict(DecisionRequest("i6", "choice", "Which?", ("a", "b"), image_path=str(img)))
    (msg,) = sent[0][0]["input"]
    assert msg["role"] == "user" and msg["content"][0]["image_url"].startswith("data:image/png;base64,")
    assert len(msg["content"]) == 1  # no empty text part for image-only

    mock(luna, lambda b: httpx.Response(200, json={"answers": [{"type": "refusal", "name": "q"}]}))
    out = luna.predict(CHOICE)
    assert out.refused and out.raw["response"]["answers"][0]["type"] == "refusal"


def test_retry_then_fail(luna, monkeypatch):
    monkeypatch.setattr("src.modules.api.time.sleep", lambda s: None)
    calls = []

    def flaky(body):
        calls.append(1)
        return httpx.Response(429) if len(calls) < 3 else luna_answer(body)

    mock(luna, flaky)
    luna.predict(CHOICE)
    assert len(calls) == 3
    mock(luna, lambda b: httpx.Response(400, json={"error": "bad"}))
    with pytest.raises(APIError, match="HTTP 400"):
        luna.predict(CHOICE)


def test_luna_choice_breaks_rounded_ties(luna):
    """Rounded probabilities tie; the API's own choice (unrounded) decides the selected option."""
    mock(luna, lambda b: httpx.Response(200, json={"answers": [{"type": "choice", "name": "q", "choice": "bird",
        "confidence": 0.0, "probabilities": [{"value": v, "probability": p} for v, p in
                                             (("cat", 0.4), ("dog", 0.2), ("bird", 0.4))]}]}))
    out = luna.predict(CHOICE)
    assert out.selected == 2 and np.argmax(out.probs) == 0


def test_runner_records_refusals_and_skips_them_on_resume(tmp_path):
    from src.modules.base import DecisionModel, ModelOutput
    from src.runners.core import Job, completed_ids, load_results, refusals_path, run_job

    class Picky(DecisionModel):
        name = "picky"
        calls: list[str] = []

        def predict(self, request):
            self.calls.append(request.item_id)
            if request.item_id == "b":
                return ModelOutput(probs=np.empty(0), refused=True, raw={"why": "refusal"})
            return ModelOutput(probs=np.array([0.25, 0.75]))

    reqs = [DecisionRequest(i, "choice", "Q?", ("x", "y"), context="c") for i in "abc"]
    job = Job("t", tmp_path / "t.jsonl", lambda: reqs, seed=0)
    model = Picky()
    stats = run_job(model, job)
    assert stats["refused"] == 1 and stats["ran"] == 3
    assert [r.item_id for r in load_results(job.out_path)] == ["a", "c"]
    assert completed_ids(refusals_path(job.out_path)) == {"b"}
    assert load_results(refusals_path(job.out_path))[0].refused
    model.calls.clear()
    assert run_job(model, job)["ran"] == 0 and model.calls == []
