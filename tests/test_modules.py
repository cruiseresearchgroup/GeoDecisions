import numpy as np
import pytest

from src.modules import (
    DecisionCache,
    DecisionModel,
    DecisionRequest,
    ModelOutput,
    build_locbench_request,
    build_null_request,
    cached_decide_batch,
    decide,
    decide_batch,
)


class FirstOptionModel(DecisionModel):
    """Puts 0.7 on whatever option is presented first: exposes order handling."""

    name = "first-option"

    def __init__(self):
        self.calls = 0

    def predict(self, request):
        self.calls += 1
        n = len(request.options)
        p = np.full(n, 0.3 / (n - 1))
        p[0] = 0.7
        return ModelOutput(probs=p)


REQ = DecisionRequest(item_id="x", kind="choice", question="q?", options=("a", "b", "c", "d"))


def test_identity_order():
    r = decide(FirstOptionModel(), REQ)
    assert r.selected == 0 and r.presented_order == [0, 1, 2, 3]
    assert r.probs.sum() == pytest.approx(1)


def test_shuffle_maps_back_to_canonical():
    r = decide(FirstOptionModel(), REQ, rng=np.random.default_rng(5))
    first = r.presented_order[0]
    assert r.selected == first and r.probs[first] == pytest.approx(0.7)
    assert r.options == REQ.options


def test_image_rejected_for_text_model():
    with pytest.raises(ValueError):
        decide(FirstOptionModel(), DecisionRequest("x", "choice", "q", ("a", "b"), image_path="img.jpg"))


def test_cache_avoids_repeat_calls(tmp_path):
    model = FirstOptionModel()
    cache = DecisionCache(tmp_path / "c.jsonl")
    a = cached_decide_batch(model, [REQ], cache, rng=np.random.default_rng(1))
    b = cached_decide_batch(model, [REQ], DecisionCache(tmp_path / "c.jsonl"), rng=np.random.default_rng(1))
    assert model.calls == 1
    np.testing.assert_allclose(a[0].probs, b[0].probs)
    assert a[0].presented_order == b[0].presented_order
    np.testing.assert_allclose(decide_batch(model, [REQ], rng=np.random.default_rng(1))[0].probs, a[0].probs)


ITEM = {
    "item_id": "fmow:val:0", "dataset": "fmow", "lat": 39.5, "lon": -77.1, "has_location": True,
    "label_index": 2, "candidates": np.arange(3), "options": ["airport", "dam", "park"], "image_path": None,
}


def test_build_requests():
    r = build_locbench_request(ITEM, "location_only")
    assert "latitude 39.5000" in r.context and r.image_path is None and r.options == ("airport", "dam", "park")
    with pytest.raises(ValueError):
        build_locbench_request(ITEM, "image_location")  # no local image
    with pytest.raises(ValueError):
        build_locbench_request(ITEM, "location_only", location_format="place_name")
    null = build_null_request(r)
    assert null.context == "Location: N/A" and null.options == r.options and null.metadata["null_input"]


def test_hf_scorer_matches_naive_scoring():
    """KV-cached batched option scoring == scoring each full sequence separately.

    Uses a tiny randomly initialised GPT-2 built locally (no download, CPU)."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from src.modules.hf_readout import HFOptionScorer

    class CharTok:
        chat_template = None
        pad_token_id = 0

        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": [1 + (ord(ch) % 60) for ch in text]}

    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=64, n_positions=512, n_embd=32, n_layer=2, n_head=2)
    model = transformers.GPT2LMHeadModel(cfg)
    scorer = HFOptionScorer(model, CharTok(), name="tiny", length_norm="sum", option_batch_size=2)
    req = DecisionRequest("x", "choice", "Which?", ("ab", "abcd", "x", "hello"))
    got = scorer.option_logprobs(req)

    prompt = scorer._prompt_ids(req)
    want = []
    for o in req.options:
        opt = CharTok()(" " + o)["input_ids"]
        ids = torch.tensor([prompt + opt])
        lp = torch.log_softmax(model(ids).logits[0].float(), -1)
        want.append(sum(lp[len(prompt) - 1 + j, t].item() for j, t in enumerate(opt)))
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_item_order_deterministic_and_choice_only():
    from src.modules.base import item_order

    a = item_order(REQ, seed=0)
    assert a == item_order(REQ, seed=0) and sorted(a) == [0, 1, 2, 3]
    assert item_order(REQ, seed=None) == [0, 1, 2, 3]
    others = {tuple(item_order(DecisionRequest(f"x{i}", "choice", "q", REQ.options), 0)) for i in range(20)}
    assert len(others) > 1  # varies across items
    score = DecisionRequest("s", "score", "q", ("0", "1", "2"))
    assert item_order(score, seed=3) == [0, 1, 2]


def test_pplx_native_request_shapes():
    from src.modules.pplx_decider import to_native

    choice = to_native(DecisionRequest("c", "choice", "Pick one.", ("dam", "park"), context="Location: x"))
    assert choice == {
        "state": "Location: x",
        "questions": {"q": {"type": "choice", "criteria": {"dam": None, "park": None}, "instructions": "Pick one."}},
    }
    score = to_native(DecisionRequest("s", "score", "Rate.", ("0.0", "0.1"), image_path="/img.jpg"))
    assert score["questions"]["q"]["criteria"] == ["0.0", "0.1"] and score["images"] == ["/img.jpg"]
    with pytest.raises(ValueError):
        to_native(DecisionRequest("d", "choice", "q", ("a", "a")))
