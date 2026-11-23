"""Base (non-decision) chat models read out as decision models through one shared prompt.

Used to test whether the decision models' biases come from their backbones: google/gemma-4-31B-it (base of KnowLine
and Sieve) and Qwen/Qwen3.8-27B (base of pplx-decider and GLiDE). Both models get the same prompt text:

    user turn:   [image]                       (image conditions only; placed before the text)
                 Context:
                 <context>                     (omitted when empty)

                 Question: <question>

                 Options:
                 (A) <option>
                 (B) <option>
                 ...

                 Answer with the letter of one option.
    assistant:   Answer: (                     (the answer is read here; nothing is generated)

The chat-template wrapper is each model's own (thinking off). Labels come from configs/base_readout_labels.json:
the same list for both models, each label one token right after "Answer: (" in both tokenizers (checked at load).
Probabilities are a softmax over the option labels' next-token log-probabilities at temperature 1 (no calibration);
``raw["logits"]`` keeps those log-probabilities (presented order).

Serving: an SGLang server (scripts/run_label_readout.sh) scored with /generate, max_new_tokens=1 and
token_ids_logprob (one prefill per request), as KnowLine's release front end does. Images: converted to RGB and
downscaled to at most 1.6 MP (aspect kept, LANCZOS; the pplx / Sieve cap), sent as PNG data URLs; each model's
processor then applies its own vision preprocessing.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.modules.base import DecisionModel, DecisionRequest, ModelOutput

ROOT = Path(__file__).resolve().parents[2]
LABELS_FILE = ROOT / "configs" / "base_readout_labels.json"
ANSWER_CUE = "Answer: ("
INSTRUCTION = "Answer with the letter of one option."
MAX_PIXELS = 1600 * 32 * 32  # 1.6 MP, the pplx-decider / Decision Index kit cap


def prompt_text(request: DecisionRequest, labels: Sequence[str]) -> str:
    """The shared user-turn text (without the image)."""
    parts = []
    if request.context:
        parts.append(f"Context:\n{request.context}")
    parts.append(f"Question: {request.question}")
    parts.append("Options:\n" + "\n".join(f"({c}) {o}" for c, o in zip(labels, request.options)))
    parts.append(INSTRUCTION)
    return "\n\n".join(parts)


def image_data_url(path: str | Path) -> str:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w * h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (w * h))
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class LabelReadoutClient(DecisionModel):
    supports_images = True

    def __init__(
        self,
        repo: str,
        *,
        revision: str | None = None,
        name: str | None = None,
        url: str = "http://127.0.0.1:9180",
        max_concurrency: int = 32,
        timeout_s: float = 600.0,
        max_retries: int = 5,
    ) -> None:
        import httpx
        from transformers import AutoProcessor

        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.name = name or repo
        self.url = url.rstrip("/")
        self.max_retries = max_retries
        self.processor = AutoProcessor.from_pretrained(repo, revision=revision)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.labels: list[str] = json.loads(LABELS_FILE.read_text())["labels"]
        self.max_options = len(self.labels)
        self.label_ids = [self._label_id(c) for c in self.labels]
        self._http = httpx.Client(timeout=timeout_s, limits=httpx.Limits(max_connections=max_concurrency))
        self._pool = ThreadPoolExecutor(max_concurrency)

    def _label_id(self, code: str) -> int:
        a = self.tokenizer(ANSWER_CUE, add_special_tokens=False)["input_ids"]
        b = self.tokenizer(ANSWER_CUE + code, add_special_tokens=False)["input_ids"]
        if b[:len(a)] != a or len(b) != len(a) + 1:
            raise ValueError(f"{self.name}: label {code!r} is not one token after {ANSWER_CUE!r}")
        return b[-1]

    def render(self, request: DecisionRequest) -> str:
        """Full prompt text: the model's chat template (thinking off) + the answer cue."""
        content: list[dict[str, Any]] = []
        if request.image_path is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt_text(request, self.labels)})
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return text + ANSWER_CUE

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        import time

        import httpx

        for attempt in range(self.max_retries + 1):
            try:
                r = self._http.post(f"{self.url}/generate", json=body)
            except httpx.TransportError:
                if attempt == self.max_retries:
                    raise
            else:
                if r.status_code == 200:
                    return r.json()
                if r.status_code < 500 or attempt == self.max_retries:
                    raise RuntimeError(f"SGLang HTTP {r.status_code}: {r.text[:500]}")
            time.sleep(min(30.0, 2.0 ** attempt))
        raise AssertionError("unreachable")

    def predict(self, request: DecisionRequest) -> ModelOutput:
        if request.kind == "binary":
            raise ValueError("binary requests are not used in this study")
        if len(set(request.options)) != len(request.options):
            raise ValueError(f"{request.item_id}: options must be unique")
        ids = self.label_ids[: len(request.options)]
        body: dict[str, Any] = {
            "text": self.render(request),
            "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
            "return_logprob": True,
            "logprob_start_len": -1,
            "token_ids_logprob": ids,
        }
        if request.image_path is not None:
            body["image_data"] = [image_data_url(request.image_path)]
        meta = self._post(body)["meta_info"]
        got = {int(t[1]): t[0] for t in (meta.get("output_token_ids_logprobs") or [[]])[0]}
        z = np.array([-np.inf if got.get(i) is None else got[i] for i in ids], dtype=np.float64)
        if not np.isfinite(z).any():
            raise RuntimeError(f"{request.item_id}: no label log-probabilities returned")
        p = np.exp(z - z[np.isfinite(z)].max())
        return ModelOutput(probs=p / p.sum(), raw={"logits": np.where(np.isfinite(z), z, -1e30).tolist(),
                                                   "prompt_tokens": meta.get("prompt_tokens")})

    def predict_batch(self, requests: Sequence[DecisionRequest]) -> list[ModelOutput]:
        return list(self._pool.map(self.predict, requests))
