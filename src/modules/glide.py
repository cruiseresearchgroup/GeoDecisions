"""Fastino GLiDE as a DecisionModel.

Wraps the checkpoint's own ``glide`` package (``glide/`` at the HF repo root,
imported from the snapshot directory): it builds the prompt, runs the
noncausal Qwen3.8-27B backbone (+ vision encoder) and applies the 255-way readout.
One question per request, as for pplx-decider.

Mirrors ``Glide.decide_batch`` but keeps the raw readout logits (presented
order) for recalibration; probabilities are softmax(logits / T) with the
checkpoint's temperature, as in ``Glide``. Image decoding/resizing (CPU) runs
in a thread pool; GLiDE resizes every image to at most GLIDE_MAX_PIXELS
(default 1024 x 1024).
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.modules.base import DecisionModel, DecisionRequest, ModelOutput

QUESTION_KEY = "q"


def to_native(request: DecisionRequest) -> dict[str, Any]:
    """DecisionRequest -> GLiDE request (one question; images as file paths, read later)."""
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


class GlideModel(DecisionModel):
    supports_images = True
    max_options = 255

    def __init__(
        self,
        repo: str | Path = "fastino/GLiDE",
        *,
        revision: str | None = None,
        name: str | None = None,
        device: str = "cuda",
        max_len: int = 40000,
        token_budget: int = 16384,
        prepare_workers: int = 16,
    ) -> None:
        path = Path(str(repo))
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(str(repo), revision=revision))
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        from glide import Glide

        self.glide = Glide.from_pretrained(str(path), device=device, max_len=max_len, token_budget=token_budget)
        self.temperature = float(self.glide.temperature)
        self.name = name or f"fastino/{Path(str(repo)).name}"
        self._pool = ThreadPoolExecutor(prepare_workers)

    def _prepare(self, native: dict[str, Any]):
        if native.get("images"):
            native = {**native, "images": [Path(p).read_bytes() for p in native["images"]]}
        entries, vis = self.glide._prepare(native)
        (_, qtype, keys, text), = entries
        return qtype, keys, text, vis

    def count_tokens(self, requests: Sequence[DecisionRequest]) -> list[int]:
        """Prompt length of each request, image tokens included (no forward pass)."""
        prepared = list(self._pool.map(self._prepare, [to_native(r) for r in requests]))
        ids = self.glide.tokenizer([p[2] for p in prepared], add_special_tokens=False)["input_ids"]
        return [len(x) for x in ids]

    def predict_batch(self, requests: Sequence[DecisionRequest]) -> list[ModelOutput]:
        prepared = list(self._pool.map(self._prepare, [to_native(r) for r in requests]))
        ids = self.glide.tokenizer([p[2] for p in prepared], add_special_tokens=False)["input_ids"]
        for r, x in zip(requests, ids):
            if len(x) + 8 > self.glide.max_len:
                raise ValueError(f"{r.item_id}: prompt exceeds GLiDE's max context ({len(x)} tokens)")
        # Glide._logits length-sorts and packs into left-padded micro-batches under token_budget.
        logits = self.glide._logits(ids, [p[3] for p in prepared])
        outputs = []
        for (qtype, keys, _, _), row in zip(prepared, logits):
            z = np.asarray(row[: len(keys)], dtype=np.float64)
            if qtype == "noul":  # GLiDE presents (true, false); requests are (false, true)
                z = z[::-1].copy()
            outputs.append(ModelOutput(probs=_softmax(z / self.temperature), raw={"logits": z.tolist()}))
        return outputs

    def predict(self, request: DecisionRequest) -> ModelOutput:
        return self.predict_batch([request])[0]
