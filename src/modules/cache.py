"""Append-only JSONL cache of decision results so no paid call runs twice.

Key = sha256 of (model name, request as presented). The presented option order
is part of the key, so each shuffle seed is cached separately.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from src.modules.base import (
    DecisionModel,
    DecisionRequest,
    DecisionResult,
    ModelOutput,
    assemble_results,
    present,
    presentation_order,
)


def request_key(model_name: str, presented: DecisionRequest) -> str:
    payload = {"model": model_name, **presented.to_dict()}
    payload.pop("metadata", None)
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class DecisionCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    rec = json.loads(line)
                    self._mem[rec["key"]] = rec["output"]

    def __contains__(self, key: str) -> bool:
        return key in self._mem

    def get(self, key: str) -> ModelOutput | None:
        rec = self._mem.get(key)
        if rec is None:
            return None
        return ModelOutput(probs=np.asarray(rec["probs"]), selected=rec["selected"], raw=rec["raw"])

    def put(self, key: str, output: ModelOutput) -> None:
        rec = {"probs": np.asarray(output.probs).tolist(), "selected": output.selected, "raw": output.raw}
        self._mem[key] = rec
        with open(self.path, "a") as f:
            f.write(json.dumps({"key": key, "output": rec}, ensure_ascii=False) + "\n")


def cached_decide_batch(
    model: DecisionModel,
    requests: Sequence[DecisionRequest],
    cache: DecisionCache,
    *,
    rng: np.random.Generator | None = None,
) -> list[DecisionResult]:
    """Like :func:`src.modules.base.decide_batch`, but only uncached requests hit the model.

    Presentation orders are drawn for every request before any lookup, so the
    rng stream (and therefore every order) is the same with or without hits.
    """
    orders = [presentation_order(len(r.options), rng) for r in requests]
    shown = [present(r, o) for r, o in zip(requests, orders)]
    keys = [request_key(model.name, s) for s in shown]

    todo = [i for i, k in enumerate(keys) if k not in cache]
    for i in todo:
        model.check(shown[i])
    for i, out in zip(todo, model.predict_batch([shown[i] for i in todo])):
        cache.put(keys[i], out)

    return assemble_results(model.name, requests, orders, [cache.get(k) for k in keys])
