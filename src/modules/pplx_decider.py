"""Perplexity pplx-decider (protocol v3) as a DecisionModel.

Wraps the checkpoint's own ``pplx_decider`` package (``source/`` in the HF repo,
installed into the Python 3.12 env), which builds the joint prompt, runs the
noncausal backbone and applies the readout head. One question per request:
benchmark items are independent decisions, and the model card advises separate
requests for independent candidates.

Probabilities are softmax(logits / T) with the checkpoint's calibrated T, as in
``Decider.decide``; raw logits (presented order) are kept for recalibration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.modules.base import DecisionModel, DecisionRequest, ModelOutput

QUESTION_KEY = "q"


def to_native(request: DecisionRequest) -> dict[str, Any]:
    """DecisionRequest -> pplx_decider Request (one question)."""
    if request.kind == "choice":
        if len(set(request.options)) != len(request.options):
            raise ValueError(f"{request.item_id}: choice options must be unique")
        question = {"type": "choice", "criteria": {o: None for o in request.options}}
    elif request.kind == "score":
        question = {"type": "score", "criteria": list(request.options)}
    elif request.kind == "binary":
        if len(request.options) != 2:
            raise ValueError("binary requests need options (false, true)")
        question = {"type": "noul", "criteria": {"false": request.options[0], "true": request.options[1]}}
    else:
        raise ValueError(f"unsupported kind {request.kind!r}")
    question["instructions"] = request.question
    native: dict[str, Any] = {"state": request.context, "questions": {QUESTION_KEY: question}}
    if request.image_path is not None:
        native["images"] = [request.image_path]
    return native


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


class PplxDecider(DecisionModel):
    supports_images = True
    max_options = 255

    def __init__(
        self,
        repo: str | Path = "perplexity-ai/pplx-decider-v1.2",
        *,
        revision: str | None = None,
        name: str | None = None,
        device: str = "cuda",
        fast: bool | dict[str, Any] = True,
        max_length: int = 32768,
        batch_tokens: int = 32768,
        warmup: bool = True,
    ) -> None:
        from pplx_decider import Decider, FastOptions

        fast_opt = FastOptions(**{k: tuple(v) if k == "buckets" else v for k, v in fast.items()}) if isinstance(fast, dict) else fast
        self.decider = Decider.from_pretrained(
            str(repo), revision=revision, device=device, fast=fast_opt, max_length=max_length, batch_tokens=batch_tokens
        )
        self.temperature = float(self.decider.temperature)
        self.name = name or f"pplx/{Path(str(repo)).name}"
        if warmup:
            self.decider.warmup()

    def count_tokens(self, requests: Sequence[DecisionRequest]) -> list[int]:
        """Sequence length of each request (tokenization only, no forward pass)."""
        return [sum(c.length for c in self.decider.encoder.plan(to_native(r), self.decider.max_length)) for r in requests]

    def predict_batch(self, requests: Sequence[DecisionRequest]) -> list[ModelOutput]:
        # Decider.logits packs (fast path) or length-sorts and batches (reference path) internally.
        logits = self.decider.logits([to_native(r) for r in requests])
        outputs = []
        for row in logits:
            z = np.asarray(row[QUESTION_KEY], dtype=np.float64)
            outputs.append(ModelOutput(probs=_softmax(z / self.temperature), raw={"logits": z.tolist()}))
        return outputs

    def predict(self, request: DecisionRequest) -> ModelOutput:
        return self.predict_batch([request])[0]
