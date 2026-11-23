"""OpenAI Decisions API (gpt-6-luna) as a DecisionModel: POST https://api.openai.com/v1/decisions.

Schema per https://developers.openai.com/api/docs/guides/decisions (2026-10-27).
Shared HTTP handling: src/modules/api.py.
"""

from __future__ import annotations

from typing import Any

from src.modules.api import QUESTION_KEY, Refusal, _HTTPDecisionClient, _image_part, _probs


class OpenAIDecisionsClient(_HTTPDecisionClient):
    """OpenAI Decisions API (gpt-6-luna): choice / score / predicate, text + image input.
    Images go into ``input`` as a user message with an ``input_image`` data URL before the text."""

    provider = "openai-decisions"
    url = "https://api.openai.com/v1/decisions"
    key_env = "OPENAI_API_KEY"
    supports_images = True

    def _payload(self, request, kind):
        q: dict[str, Any] = {"name": QUESTION_KEY, "instructions": request.question}
        if kind == "choice":
            q |= {"type": "choice", "choices": [{"value": o, "description": o} for o in request.options]}
        elif kind == "score":
            q |= {"type": "score", "levels": [{"label": o, "description": o} for o in request.options]}
        elif kind == "binary":
            q |= {"type": "predicate"}  # P(true); option texts are not sent
        else:
            raise ValueError(f"unsupported kind {kind!r}")
        if request.image_path is not None:
            fmt, data = _image_part(request.image_path)
            content = [{"type": "input_image", "image_url": f"data:image/{fmt};base64,{data}"}]
            if request.context:
                content.append({"type": "input_text", "text": request.context})
            input_: Any = [{"role": "user", "content": content}]
        else:
            input_ = request.context
        return {"model": self.model, "input": input_, "questions": [q]}, {}

    def _parse(self, response, request, kind):
        (ans,) = [a for a in response["answers"] if a.get("name") == QUESTION_KEY]
        if ans["type"] == "refusal":
            raise Refusal(f"{request.item_id}: refused")
        if kind == "choice":
            p = {d["value"]: d["probability"] for d in ans["probabilities"]}
            return _probs([p[o] for o in request.options]), request.options.index(ans["choice"])
        if kind == "score":  # "score" is a weighted mean of level indices, not a level
            p = {d["label"]: d["probability"] for d in ans["probabilities"]}
            return _probs([p[o] for o in request.options]), None
        pt = float(ans["probability"])
        return _probs([1.0 - pt, pt]), None
