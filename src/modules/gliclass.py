"""Knowledgator GLiClass (encoder, text only) as a DecisionModel.

Follows the Jev-style mapping on the model card (knowledgator/gliclass-v4-edge,
"Serving a Jev-style decision API"): one pass per question, the state is the
text, the question's instructions are the prompt, every option is a label, and
probabilities are a softmax over the option logits. choice/score labels are the
option texts; binary labels are (true, false) descriptions. Inputs longer than
``max_length`` tokens are refused, never truncated.

``logits`` are the checkpoint's calibrated logits (per-pair temperature and
bias); ``raw`` keeps both those and ``uncalibrated_logits``, in presented order.
Requests are length-sorted into batches of ``batch_size`` to limit padding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.modules.base import DecisionModel, DecisionRequest, ModelOutput


def labels_for(request: DecisionRequest) -> list[str]:
    """GLiClass labels in presented order (binary: true first, as on the model card)."""
    if request.kind in ("choice", "score"):
        if request.kind == "choice" and len(set(request.options)) != len(request.options):
            raise ValueError(f"{request.item_id}: choice options must be unique")
        return list(request.options)
    if request.kind == "binary":
        if len(request.options) != 2:
            raise ValueError("binary requests need options (false, true)")
        return [request.options[1], request.options[0]]
    raise ValueError(f"unsupported kind {request.kind!r}")


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


class GLiClassDecider(DecisionModel):
    supports_images = False

    def __init__(
        self,
        repo: str | Path = "knowledgator/gliclass-v4-edge",
        *,
        revision: str | None = None,
        name: str | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_length: int = 16384,
        batch_size: int = 16,
    ) -> None:
        import torch
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline

        path = Path(str(repo))
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(str(repo), revision=revision))
        self._torch = torch
        self.model = GLiClassModel.from_pretrained(str(path), torch_dtype=getattr(torch, dtype)).to(device).eval()
        # max_length=10**9 disables the tokenizer's truncation; over-long inputs are refused below.
        self.pipe = ZeroShotClassificationPipeline(
            self.model, str(path), max_length=10**9, classification_type="single-label", device=device, progress_bar=False
        ).pipe
        self.max_length = max_length
        self.batch_size = batch_size
        self.name = name or f"knowledgator/{Path(str(repo)).name}"

    def _inputs(self, requests: Sequence[DecisionRequest]):
        labels = [labels_for(r) for r in requests]
        prompts = [r.question or None for r in requests]
        return labels, self.pipe.prepare_inputs([r.context for r in requests], labels, same_labels=False, prompt=prompts)

    def count_tokens(self, requests: Sequence[DecisionRequest]) -> list[int]:
        _, inputs = self._inputs(requests)
        return inputs["attention_mask"].sum(-1).tolist()

    def predict_batch(self, requests: Sequence[DecisionRequest]) -> list[ModelOutput]:
        # Length-sort by a cheap proxy (characters) so each batch pads little; restore order at the end.
        order = sorted(range(len(requests)), key=lambda i: len(requests[i].context) + len(requests[i].question)
                       + sum(len(o) for o in requests[i].options))
        outputs: list[ModelOutput | None] = [None] * len(requests)
        with self._torch.inference_mode():
            for s in range(0, len(order), self.batch_size):
                idx = order[s:s + self.batch_size]
                batch = [requests[i] for i in idx]
                labels, inputs = self._inputs(batch)
                lengths = inputs["attention_mask"].sum(-1).tolist()
                for r, n in zip(batch, lengths):
                    if n > self.max_length:
                        raise ValueError(f"{r.item_id}: {n} tokens exceeds max_length {self.max_length}")
                out = self.model(**inputs, max_num_classes=max(len(l) for l in labels))
                cal = out.logits.float().cpu().numpy()
                unc = out.uncalibrated_logits.float().cpu().numpy() if out.uncalibrated_logits is not None else None
                for j, (i, r, lab) in enumerate(zip(idx, batch, labels)):
                    z = cal[j, : len(lab)].astype(np.float64)
                    u = None if unc is None else unc[j, : len(lab)].astype(np.float64)
                    if r.kind == "binary":  # labels were (true, false); requests are (false, true)
                        z = z[::-1].copy()
                        u = None if u is None else u[::-1].copy()
                    raw: dict[str, Any] = {"logits": z.tolist(), "input_tokens": int(lengths[j])}
                    if u is not None:
                        raw["uncalibrated_logits"] = u.tolist()
                    outputs[i] = ModelOutput(probs=_softmax(z), raw=raw)
        return outputs  # type: ignore[return-value]

    def predict(self, request: DecisionRequest) -> ModelOutput:
        return self.predict_batch([request])[0]
