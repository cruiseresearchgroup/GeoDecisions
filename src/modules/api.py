"""Shared base for closed-API decision models.

    src.modules.typesafe_jev.JevClient                   TypeSafe Jev (/v1/systemone)
    src.modules.jev.JevClient                            TypeSafe Jev via the typesafe-sdk (kind jev_sdk; Manvi results)
    src.modules.openai_decision.OpenAIDecisionsClient    OpenAI Decisions API (/v1/decisions, gpt-6-luna)
    OpenAIChatReadoutClient (here)                       GPT-6 Luna chat read out from logprobs (placeholder)

One question per request, as for the open models. A chunk of requests is sent
concurrently (``max_concurrency`` threads); 429 / 5xx / timeouts are retried
with exponential backoff (honouring Retry-After). Other errors and refusals
raise, so the chunk is not written and is re-sent on resume.

Neither API returns logits: ``raw`` keeps the answer and ``usage`` (input
tokens, for cost accounting). Both APIs cap score questions at 10 levels; with
``score_as_choice`` a longer scale (Manvi 0.0-9.9, 100 levels) is sent as a
choice over the level labels in ascending order (``raw["sent_as"] = "choice"``).

API keys: ``api_key`` argument, else the environment variable ``key_env``,
else that variable in ``<repo>/.env`` (git-ignored).
"""

from __future__ import annotations

import base64
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.modules.base import DecisionModel, DecisionRequest, ModelOutput

ROOT = Path(__file__).resolve().parents[2]
QUESTION_KEY = "q"
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
IMAGE_FORMATS = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png"}


class APIError(RuntimeError):
    """A request the provider rejected (not retried)."""


class Refusal(APIError):
    """The provider declined to answer; recorded as a refusal, not an error."""


def _api_key(key_env: str) -> str:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # does not override variables already set
    if not os.environ.get(key_env):
        raise RuntimeError(f"set {key_env} (environment or {ROOT / '.env'})")
    return os.environ[key_env]


def _image_part(path: str) -> tuple[str, str]:
    """-> (format, base64 data) for a local image file."""
    fmt = IMAGE_FORMATS.get(Path(path).suffix.lower())
    if fmt is None:
        raise ValueError(f"unsupported image type: {path}")
    return fmt, base64.b64encode(Path(path).read_bytes()).decode()


def _probs(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(p)) or p.sum() <= 0:
        raise APIError(f"invalid probabilities {values!r}")
    return p / p.sum()


class _HTTPDecisionClient(DecisionModel):
    provider = ""
    url = ""
    key_env = ""
    max_score_levels = 10

    def __init__(
        self,
        model: str,
        *,
        name: str | None = None,
        api_key: str | None = None,
        url: str | None = None,
        max_concurrency: int = 16,
        timeout_s: float = 120.0,
        max_retries: int = 8,
        score_as_choice: bool = False,
    ) -> None:
        import httpx

        self.model = model
        self.name = name or f"{self.provider}/{model}"
        self.url = url or self.url
        self.score_as_choice = score_as_choice
        self.max_retries = max_retries
        self._key = api_key or _api_key(self.key_env)
        self._http = httpx.Client(timeout=timeout_s, limits=httpx.Limits(max_connections=max_concurrency))
        self._pool = ThreadPoolExecutor(max_concurrency)

    # -------------------------------------------------------------- provider-specific
    def _payload(self, request: DecisionRequest, kind: str) -> tuple[dict[str, Any], dict[str, str]]:
        """-> (JSON body, extra headers)."""
        raise NotImplementedError

    def _parse(self, response: dict[str, Any], request: DecisionRequest, kind: str) -> tuple[np.ndarray, int | None]:
        """Provider response -> (probabilities over request.options in presented order, presented index of the
        provider's choice or None). Providers round probabilities (OpenAI: 2 decimals), so their own choice,
        made on unrounded values, breaks ties that the rounded vector cannot."""
        raise NotImplementedError

    # -------------------------------------------------------------- shared
    def _kind(self, request: DecisionRequest) -> str:
        if request.kind == "score" and len(request.options) > self.max_score_levels:
            if not self.score_as_choice:
                raise ValueError(
                    f"{self.name}: score questions allow at most {self.max_score_levels} levels, got "
                    f"{len(request.options)}; set score_as_choice: true to send them as a choice"
                )
            return "choice"
        return request.kind

    def _post(self, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        import httpx

        headers = {"Authorization": f"Bearer {self._key}", **headers}
        for attempt in range(self.max_retries + 1):
            try:
                r = self._http.post(self.url, json=body, headers=headers)
            except httpx.TransportError as e:
                if attempt == self.max_retries:
                    raise
                wait = min(60.0, 2.0 ** attempt)
            else:
                if r.status_code == 200:
                    return r.json()
                if r.status_code not in RETRY_STATUS or attempt == self.max_retries:
                    raise APIError(f"{self.provider} HTTP {r.status_code}: {r.text[:500]}")
                retry_after = r.headers.get("retry-after")
                wait = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else min(60.0, 2.0 ** attempt)
            time.sleep(wait * (1 + 0.25 * random.random()))
        raise AssertionError("unreachable")

    def predict(self, request: DecisionRequest) -> ModelOutput:
        if request.kind == "choice" and len(set(request.options)) != len(request.options):
            raise ValueError(f"{request.item_id}: choice options must be unique")
        kind = self._kind(request)
        body, headers = self._payload(request, kind)
        response = self._post(body, headers)
        try:
            probs, selected = self._parse(response, request, kind)
        except Refusal:
            return ModelOutput(probs=np.empty(0), refused=True, raw={"response": response, "usage": response.get("usage")})
        raw = {"response": response, "usage": response.get("usage")}
        if kind != request.kind:
            raw["sent_as"] = kind
        return ModelOutput(probs=probs, selected=selected, raw=raw)

    def predict_batch(self, requests: Sequence[DecisionRequest]) -> list[ModelOutput]:
        return list(self._pool.map(self.predict, requests))


class OpenAIChatReadoutClient(DecisionModel):
    """GPT-6 Luna in chat mode, read out as a decision model from answer-token
    logprobs. Requires the API to expose logprobs; otherwise it gives answer-level
    results only and cannot enter the calibration comparisons. Not implemented."""

    provider = "openai-chat"
    supports_images = True

    def __init__(self, model: str, **_: Any) -> None:
        self.name = f"{self.provider}/{model}"

    def predict(self, request: DecisionRequest) -> ModelOutput:
        raise NotImplementedError("OpenAIChatReadoutClient: not implemented")
