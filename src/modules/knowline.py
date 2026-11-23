"""PelaAI KnowLine-31B-Gen1 (Gemma-4-31B decision model) as a DecisionModel.

The model is served exactly as released (``serve_knowline.sh`` from the HF repo: SGLang bf16 engine plus the
``knowline_server.py`` /v1/systemone front end, read twice on, per-type temperatures from ``temperatures.json``);
this client sends one question per request to that local front end, in TypeSafe's System One format.

- choice: ``criteria = {option: null}`` (shown by name); score: ``criteria = [levels]`` (up to 255, so Manvi's
  100-level scale is read as a score, as for pplx/GLiDE); binary: noul with (true, false) descriptions.
- Images go in as Jev ``assets``, referenced from a chat-form state (one user turn: the image part, then the text),
  which the front end renders as ``Image `image`:\\n<image>\\n\\n<text>``. The front end enforces Jev 1.14's image
  limits (280-1568 px per side, 78,400-2,458,624 px) without resizing; images outside them are resized here (aspect
  kept, lossless PNG; padded only when the aspect ratio cannot fit) and sent unchanged otherwise.
- The front end returns probabilities (temperature folded into the weights); ``raw["logits"]`` stores their logs
  (logits up to a constant, presented order).
"""

from __future__ import annotations

import base64
import io
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.modules.base import DecisionModel, DecisionRequest, ModelOutput

QUESTION_KEY = "q"
ASSET = "image"
MIN_SIDE, MAX_SIDE = 280, 1568
MIN_PIXELS, MAX_PIXELS = 78_400, 2_458_624


def jev_image(path: str | Path) -> tuple[str, str, bool]:
    """Local image -> (format, base64, resized) within Jev 1.14's limits."""
    from PIL import Image

    data = Path(path).read_bytes()
    img = Image.open(io.BytesIO(data))
    fmt = {"JPEG": "jpeg", "PNG": "png"}.get(img.format)
    w, h = img.size
    in_limits = MIN_SIDE <= min(w, h) and max(w, h) <= MAX_SIDE and MIN_PIXELS <= w * h <= MAX_PIXELS
    # Non-RGB images (grayscale "L", CMYK) are re-encoded as RGB: SGLang's Gemma 4 vision tower assumes 3 channels and a
    # single grayscale image crashed the engine (36 such images in our data; pplx/GLiDE convert to RGB themselves).
    if fmt and in_limits and img.mode == "RGB":
        return fmt, base64.b64encode(data).decode(), False
    s_min = max(MIN_SIDE / min(w, h), math.sqrt(MIN_PIXELS / (w * h)))  # smallest scale meeting the minimums
    s_max = min(MAX_SIDE / max(w, h), math.sqrt(MAX_PIXELS / (w * h)))  # largest scale meeting the maximums
    s = min(max(1.0, s_min), s_max)  # s_min > s_max only for extreme aspect ratios: fit the long side, pad below
    nw, nh = (min(MAX_SIDE, max(1, round(x * s))) for x in (w, h))
    if s_min <= s_max:  # rounding can leave a side one pixel short of the minimum
        nw, nh = max(nw, MIN_SIDE), max(nh, MIN_SIDE)
    img = img.convert("RGB").resize((nw, nh), Image.BICUBIC)
    if min(nw, nh) < MIN_SIDE:
        canvas = Image.new("RGB", (max(nw, MIN_SIDE), max(nh, MIN_SIDE)), (0, 0, 0))
        canvas.paste(img, ((canvas.width - nw) // 2, (canvas.height - nh) // 2))
        img = canvas
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "png", base64.b64encode(buf.getvalue()).decode(), True


def to_native(request: DecisionRequest) -> dict[str, Any]:
    if request.kind == "choice":
        if len(set(request.options)) != len(request.options):
            raise ValueError(f"{request.item_id}: choice options must be unique")
        q = {"type": "choice", "criteria": {o: None for o in request.options}}
    elif request.kind == "score":
        q = {"type": "score", "criteria": list(request.options)}
    elif request.kind == "binary":
        if len(request.options) != 2:
            raise ValueError("binary requests need options (false, true)")
        q = {"type": "noul", "criteria": {"false": request.options[0], "true": request.options[1]}}
    else:
        raise ValueError(f"unsupported kind {request.kind!r}")
    q["instructions"] = request.question
    body: dict[str, Any] = {"model": "m", "state": request.context, "questions": {QUESTION_KEY: q}}
    if request.image_path is not None:
        fmt, data, resized = jev_image(request.image_path)
        body["assets"] = {ASSET: {"format": fmt, "data_base64": data}}
        content: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": ASSET}}]
        if request.context:
            content.append({"type": "text", "text": request.context})
        body["state"] = [{"role": "user", "content": content}]
        body["_resized"] = resized
    return body


class KnowLineClient(DecisionModel):
    supports_images = True
    max_options = 255

    def __init__(
        self,
        model: str = "KnowLine-31B-Gen1",
        *,
        name: str | None = None,
        url: str = "http://127.0.0.1:8080/v1/systemone",
        max_concurrency: int = 32,
        timeout_s: float = 600.0,
        max_retries: int = 5,
    ) -> None:
        import httpx

        logging.getLogger("httpx").setLevel(logging.WARNING)  # one INFO line per request otherwise
        self.name = name or model
        self.url = url
        self.max_retries = max_retries
        self._http = httpx.Client(timeout=timeout_s, limits=httpx.Limits(max_connections=max_concurrency))
        self._pool = ThreadPoolExecutor(max_concurrency)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        import time

        import httpx

        for attempt in range(self.max_retries + 1):
            try:
                r = self._http.post(self.url, json=body)
            except httpx.TransportError:
                if attempt == self.max_retries:
                    raise
            else:
                if r.status_code == 200:
                    return r.json()
                if r.status_code < 500 or attempt == self.max_retries:
                    raise RuntimeError(f"KnowLine HTTP {r.status_code}: {r.text[:500]}")
            time.sleep(min(30.0, 2.0 ** attempt))
        raise AssertionError("unreachable")

    def predict(self, request: DecisionRequest) -> ModelOutput:
        body = to_native(request)
        resized = body.pop("_resized", None)
        response = self._post(body)
        ans = response["answers"][QUESTION_KEY]
        if request.kind == "choice":
            p = [ans["probabilities"][o] for o in request.options]
        elif request.kind == "score":
            p = [ans["probabilities"][str(i)] for i in range(len(request.options))]
        else:
            p = [1.0 - ans["noul"], ans["noul"]]
        probs = np.asarray(p, dtype=np.float64)
        probs = probs / probs.sum()
        raw: dict[str, Any] = {"logits": np.log(np.clip(probs, 1e-300, None)).tolist(), "usage": response.get("usage")}
        if resized is not None:
            raw["image_resized"] = resized
        return ModelOutput(probs=probs, raw=raw)

    def predict_batch(self, requests: Sequence[DecisionRequest]) -> list[ModelOutput]:
        return list(self._pool.map(self.predict, requests))
