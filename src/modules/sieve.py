"""sthanika-ai Sieve-31B-v3 (Gemma-4-31B decision model) as a DecisionModel.

Runs the release's own engines in process (``sieve-decisions`` 0.4.4 and the repo's ``decision_index_vision.py``):
the letter readout at ``Answer: (``, bidirectional global attention, the release's two option orders for choice
questions (as given and rotated by n // 2, log-probabilities averaged), and its per-category temperatures.

- Text requests go through the letters engine in batches (``logits_many``), with the temperatures applied exactly as
  ``LettersEngine.__call__`` does; image requests go through ``SieveVision.__call__`` one at a time (Gemma 4's vision
  tower; images capped at 1.6 MP by the Decision Index kit, then Gemma 4's processor at 560 soft tokens, RGB).
- choice: ``criteria = {option: null}`` (shown by name); score: ``criteria = [levels]``. The engine reads scores with
  the levels as options in order (2-255; Sieve was trained on choice/noul only), so Manvi's 100-level scale is read
  as a score, as for the other models; binary: noul with (false, true) descriptions.
- ``raw["logits"]`` is the engine's pre-temperature readout (for two-order choices, the averaged log-probabilities),
  in presented order; ``raw["temperature"]`` is the temperature applied.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.modules.base import DecisionModel, DecisionRequest, ModelOutput

QUESTION_KEY = "q"


def to_native(request: DecisionRequest) -> dict[str, Any]:
    """DecisionRequest -> (System One) questions for one request."""
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
    return {QUESTION_KEY: q}


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


class SieveDecider(DecisionModel):
    supports_images = True
    max_options = 255

    def __init__(
        self,
        repo: str | Path = "sthanika-ai/Sieve-31B-v3",
        *,
        revision: str | None = None,
        name: str | None = None,
        device: str = "cuda",
        batch_size: int = 64,
        max_soft_tokens: int = 560,
    ) -> None:
        path = Path(str(repo))
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(str(repo), revision=revision))
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))  # decision_index_vision.py ships in the model repo
        from decision_index_vision import SieveVision

        self.engine = SieveVision(model=str(path), device=device, max_soft_tokens=max_soft_tokens)
        self.m = self.engine.m
        self.batch_size = batch_size
        self.name = name or Path(str(repo)).name

    def _text_batch(self, requests: Sequence[DecisionRequest]) -> list[ModelOutput]:
        """LettersEngine.__call__ for several one-question requests at once (logits_many)."""
        import torch
        from sieve.api import to_record
        from sieve.calibration import question_temperatures

        pairs = [(r.context, to_native(r)) for r in requests]
        zs = self.m.logits_many(pairs, budget=self.engine.inner.budget)
        out = []
        for r, (state, questions), z in zip(requests, pairs, zs):
            if not isinstance(z, list):  # TooLong, or another per-request failure
                raise ValueError(f"{r.item_id}: {z}")
            _, _, meta = to_record(state, questions)
            t = (question_temperatures(self.m.temperatures, meta, self.m.temperature, state) or [self.m.temperature])[0]
            zi = z[0].float()
            probs = torch.softmax(zi / t, -1).double().cpu().numpy()
            out.append(ModelOutput(probs=probs, raw={"logits": zi.cpu().tolist(), "temperature": float(t)}))
        return out

    def _image(self, request: DecisionRequest) -> ModelOutput:
        res, extra = self.engine(request.context, to_native(request), images=[str(request.image_path)])
        p = res["answers"][QUESTION_KEY]["probabilities"]
        probs = np.asarray([p[o] for o in request.options], dtype=np.float64)
        return ModelOutput(probs=probs / probs.sum(), raw={"logits": np.log(np.clip(probs, 1e-300, None)).tolist(), **(extra or {})})

    def predict_batch(self, requests: Sequence[DecisionRequest]) -> list[ModelOutput]:
        outputs: list[ModelOutput | None] = [None] * len(requests)
        text = [i for i, r in enumerate(requests) if r.image_path is None]
        for s in range(0, len(text), self.batch_size):
            idx = text[s:s + self.batch_size]
            for i, o in zip(idx, self._text_batch([requests[i] for i in idx])):
                outputs[i] = o
        for i, r in enumerate(requests):
            if r.image_path is not None:
                outputs[i] = self._image(r)
        return outputs  # type: ignore[return-value]

    def predict(self, request: DecisionRequest) -> ModelOutput:
        return self.predict_batch([request])[0]
