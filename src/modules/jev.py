"""TypeSafe Jev as a DecisionModel, via the System One API (docs.typesafe.ai).

One question per API call, as for pplx-decider and GLiDE: benchmark items are
independent decisions. Calls go through the official ``typesafe-sdk`` (retries
with backoff on 429/529) and run concurrently in a thread pool.

Request mapping:
    state      ``{"location": context, "image": <image>}``, either field omitted
               when absent; ``"N/A"`` when both are (Jev requires a state).
    choice     Choice with ``criteria = {option: None, ...}`` in presented order.
    score      Score with ``criteria = [option, ...]`` up to Jev's 10-level limit;
               longer scales (Manvi's 100-point 0.0-9.9) become a Choice over the
               levels, kept in ordinal order (``item_order`` never shuffles scores).
    binary     Noul with ``criteria = {"false": options[0], "true": options[1]}``.

Jev returns probabilities rounded to two decimals; ``to_canonical`` renormalizes.
``model`` defaults to a versioned ID, not ``jev-latest``: an alias moves between
releases. Images use TypeSafe's opt-in ("dreamer-ready") image support.
"""

from __future__ import annotations

import io
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import numpy as np

from src.modules.base import DecisionModel, DecisionRequest, ModelOutput

QUESTION_KEY = "q"
MAX_CHOICES = 255
MAX_SCORE_LEVELS = 10
IMAGE_MIN_SIDE, IMAGE_MAX_SIDE = 280, 1568
IMAGES_HEADER = {"TypeSafe-Danger-Zone-Images": "true"}


def to_question(request: DecisionRequest):
    """DecisionRequest -> SDK question object."""
    from typesafe_sdk import Choice, Noul, Score

    opts = list(request.options)
    if request.kind == "binary":
        if len(opts) != 2:
            raise ValueError("binary requests need options (false, true)")
        return Noul(instructions=request.question, criteria={"false": opts[0], "true": opts[1]})
    if len(set(opts)) != len(opts):
        raise ValueError(f"{request.item_id}: options must be unique")
    if request.kind == "score" and len(opts) <= MAX_SCORE_LEVELS:
        return Score(instructions=request.question, criteria=opts)
    if request.kind in ("choice", "score"):
        return Choice(instructions=request.question, criteria={o: None for o in opts})
    raise ValueError(f"unsupported kind {request.kind!r}")


def to_state(request: DecisionRequest) -> dict[str, Any] | str:
    state: dict[str, Any] = {}
    if request.context:
        state["location"] = request.context
    if request.image_path is not None:
        state["image"] = load_image(request.image_path)
    return state or "N/A"


def load_image(path: str | os.PathLike):
    """SDK Image within Jev's limits: upright, sides in [280, 1568], baseline JPEG."""
    from PIL import Image as PILImage
    from PIL import ImageOps
    from typesafe_sdk import Image

    with PILImage.open(path) as src:
        img = ImageOps.exif_transpose(src).convert("RGB")
    w, h = img.size
    scale = min(1.0, IMAGE_MAX_SIDE / max(w, h))
    if min(w, h) * scale < IMAGE_MIN_SIDE:
        scale = IMAGE_MIN_SIDE / min(w, h)
    if scale != 1.0:
        size = tuple(min(IMAGE_MAX_SIDE, max(IMAGE_MIN_SIDE, round(s * scale))) for s in (w, h))
        img = img.resize(size, PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return Image(buf.getvalue())


def parse_answer(request: DecisionRequest, answer: Any, response: Any = None) -> ModelOutput:
    """Probabilities over ``request.options`` (presented order) from one System One answer."""
    opts = request.options
    raw: dict[str, Any] = {"primitive": answer.type, "confidence": getattr(answer, "confidence", None)}
    if response is not None:
        raw["model_version"] = response.model
        raw["input_tokens"] = response.usage.input_tokens
        try:
            raw["request_id"] = response.request_id
        except Exception:  # the SDK raises when the response header is missing
            raw["request_id"] = None
    if answer.type == "noul":
        p = float(answer.noul)
        return ModelOutput(probs=np.array([1.0 - p, p]), raw=raw)
    if answer.type == "score":
        probs = np.array([answer.probabilities.get(i, 0.0) for i in range(len(opts))], dtype=np.float64)
        return ModelOutput(probs=probs, raw={**raw, "score": float(answer.score)})
    if answer.type == "choice":
        probs = np.array([answer.probabilities.get(o, 0.0) for o in opts], dtype=np.float64)
        return ModelOutput(probs=probs, selected=opts.index(answer.choice), raw=raw)
    raise ValueError(f"unexpected answer type {answer.type!r}")


class JevClient(DecisionModel):
    provider = "typesafe"
    supports_images = True
    max_options = MAX_CHOICES

    def __init__(
        self,
        model: str = "jev-1.14.0",
        *,
        name: str | None = None,
        api_key: str | None = None,
        max_concurrency: int = 8,
        timeout: float = 60.0,
        max_retries: int = 6,
        client: Any = None,
    ) -> None:
        self.model = model
        self.name = name or f"{self.provider}/{model}"
        self.max_concurrency = max_concurrency
        if client is None:
            from typesafe_sdk import RetryPolicy, TypeSafeClient

            client = TypeSafeClient(
                api_key=api_key or os.environ.get("TYPESAFE_API_KEY"),
                model=model,
                timeout=timeout,
                retry=RetryPolicy(max_retries=max_retries, backoff_max=30.0, timeout=None),
            )
        self.client = client

    def predict(self, request: DecisionRequest) -> ModelOutput:
        self.check(request)
        resp = self.client.system_one(
            state=to_state(request),
            questions={QUESTION_KEY: to_question(request)},
            model=self.model,
            extra_headers=IMAGES_HEADER if request.image_path is not None else None,
        )
        return parse_answer(request, resp.answers[QUESTION_KEY], resp)

    def predict_batch(self, requests: Sequence[DecisionRequest]) -> list[ModelOutput]:
        if self.max_concurrency <= 1 or len(requests) <= 1:
            return [self.predict(r) for r in requests]
        with ThreadPoolExecutor(self.max_concurrency) as pool:
            return list(pool.map(self.predict, requests))
