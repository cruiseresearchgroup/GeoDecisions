"""TypeSafe Jev as a DecisionModel: POST https://api.typesafe.ai/v1/systemone.

Schema per https://docs.typesafe.ai/api (2026-10-27). Shared HTTP handling: src/modules/api.py.
The API rejects images outside 280-1568 px per side / 78,400-2,458,624 px (docs.typesafe.ai/images,
2026-11-11); those are resized as for KnowLine (``src.modules.knowline.jev_image``), others sent unchanged.
"""

from __future__ import annotations

import uuid
from typing import Any

from src.modules.api import QUESTION_KEY, _HTTPDecisionClient, _probs
from src.modules.knowline import jev_image


class JevClient(_HTTPDecisionClient):
    """TypeSafe Jev: choice / score / noul. Images are an opt-in ("Danger Zone") feature:
    sent as ``assets`` with header ``TypeSafe-Danger-Zone-Images: true``; the asset
    placeholder is placed before the text in ``state``."""

    provider = "typesafe"
    url = "https://api.typesafe.ai/v1/systemone"
    key_env = "TYPESAFE_API_KEY"
    supports_images = True
    max_options = 255

    def _payload(self, request, kind):
        if kind == "choice":
            question = {"type": "choice", "criteria": {o: None for o in request.options}}  # null: shown by name
        elif kind == "score":
            question = {"type": "score", "criteria": list(request.options)}
        elif kind == "binary":
            if len(request.options) != 2:
                raise ValueError("binary requests need options (false, true)")
            question = {"type": "noul", "criteria": {"false": request.options[0], "true": request.options[1]}}
        else:
            raise ValueError(f"unsupported kind {kind!r}")
        question["instructions"] = request.question
        body: dict[str, Any] = {"model": self.model, "state": request.context, "questions": {QUESTION_KEY: question}}
        headers = {}
        if request.image_path is not None:
            fmt, data, _ = jev_image(request.image_path)
            placeholder = str(uuid.uuid5(uuid.NAMESPACE_URL, request.item_id))  # docs: one UUID per image
            body["assets"] = {placeholder: {"type": "image", "format": fmt, "data_base64": data}}
            body["state"] = f"{placeholder}\n\n{request.context}" if request.context else placeholder
            headers["TypeSafe-Danger-Zone-Images"] = "true"
        return body, headers

    def _parse(self, response, request, kind):
        ans = response["answers"][QUESTION_KEY]
        if kind == "choice":
            return _probs([ans["probabilities"][o] for o in request.options]), request.options.index(ans["choice"])
        if kind == "score":  # "score" is a weighted mean of level indices, not a level
            return _probs([ans["probabilities"][str(i)] for i in range(len(request.options))]), None
        p = float(ans["noul"])
        return _probs([1.0 - p, p]), None
