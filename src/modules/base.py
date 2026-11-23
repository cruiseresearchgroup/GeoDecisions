"""One API for every model under test: closed decision APIs (Jev, GPT-6 Luna
Decisions), generative chat models read out as decision models, and open
Jev-style / backbone models scored locally.

A model receives a :class:`DecisionRequest` whose options are already in the
order to *present*, and returns probabilities over those presented options.
:func:`decide` handles option-order randomization: it permutes the canonical
options, calls the model, and maps probabilities back to canonical order, so
everything downstream (metrics, caching keys, analysis) sees canonical order.
"""

from __future__ import annotations

import abc
import hashlib
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, Sequence

import numpy as np

DecisionKind = Literal["choice", "score", "binary"]


@dataclass(frozen=True)
class DecisionRequest:
    """A single decision.

    Attributes:
        item_id: stable id of the underlying benchmark item.
        kind: "choice" (pick one option), "score" (ordinal rating; options are
            the rating values in increasing order), or "binary" (yes/no).
        question: what is being decided.
        context: text input (location description, etc.); empty for null-input runs.
        options: option labels, in the order they will be presented.
        image_path: local image file for multimodal conditions.
        metadata: free-form fields carried through to the result.
    """

    item_id: str
    kind: DecisionKind
    question: str
    options: tuple[str, ...]
    context: str = ""
    image_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["options"] = list(self.options)
        return d


@dataclass
class DecisionResult:
    """Model output, in canonical option order.

    Attributes:
        probs: probability per canonical option (sums to 1).
        presented_order: ``presented_order[j]`` is the canonical index of the option
            shown in position ``j``.
        selected: canonical index of the model's chosen option (argmax if the
            model returns no explicit choice).
        raw: provider response or readout details, for auditing. Per-option
            values in ``raw`` (e.g. logits) are in *presented* order.
        refused: the provider declined to answer (probs are NaN, selected is -1).
    """

    item_id: str
    model: str
    options: tuple[str, ...]
    probs: np.ndarray
    presented_order: list[int]
    selected: int
    raw: dict[str, Any] = field(default_factory=dict)
    refused: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = {
            "item_id": self.item_id,
            "model": self.model,
            "options": list(self.options),
            "probs": self.probs.tolist(),
            "presented_order": self.presented_order,
            "selected": self.selected,
            "raw": self.raw,
        }
        if self.refused:
            d["refused"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DecisionResult":
        return cls(
            item_id=d["item_id"],
            model=d["model"],
            options=tuple(d["options"]),
            probs=np.asarray(d["probs"], dtype=np.float64),
            presented_order=list(d["presented_order"]),
            selected=int(d["selected"]),
            raw=d.get("raw", {}),
            refused=bool(d.get("refused", False)),
        )


@dataclass
class ModelOutput:
    """What a model implementation returns: probabilities over *presented* options."""

    probs: np.ndarray
    selected: int | None = None  # presented index, if the API reports a choice
    raw: dict[str, Any] = field(default_factory=dict)
    refused: bool = False  # provider declined to answer; probs are ignored


class DecisionModel(abc.ABC):
    """Base class for all models. Implement :meth:`predict`."""

    #: Identifier recorded in results and cache keys; include the model version.
    name: str = "decision-model"
    supports_images: bool = False
    max_options: int | None = None

    @abc.abstractmethod
    def predict(self, request: DecisionRequest) -> ModelOutput:
        """Probabilities over ``request.options`` in the order given."""

    def predict_batch(self, requests: Sequence[DecisionRequest]) -> list[ModelOutput]:
        """Override for batched/parallel backends."""
        return [self.predict(r) for r in requests]

    def check(self, request: DecisionRequest) -> None:
        if request.image_path is not None and not self.supports_images:
            raise ValueError(f"{self.name} does not accept images")
        if self.max_options is not None and len(request.options) > self.max_options:
            raise ValueError(
                f"{self.name} accepts at most {self.max_options} options, got {len(request.options)}"
            )


def presentation_order(n: int, rng: np.random.Generator | None) -> list[int]:
    """Canonical indices in presentation order; identity when ``rng`` is None."""
    return list(range(n)) if rng is None else rng.permutation(n).tolist()


def item_order(request: DecisionRequest, seed: int | None) -> list[int]:
    """Deterministic per-item presentation order for shuffle ``seed``.

    Depends only on (seed, item_id), so a resumed or re-chunked run presents
    every item exactly as before. Only "choice" options are shuffled: score
    levels are ordinal and binary options are fixed (false, true).
    """
    n = len(request.options)
    if seed is None or request.kind != "choice":
        return list(range(n))
    digest = hashlib.sha256(request.item_id.encode()).digest()
    rng = np.random.default_rng([seed, int.from_bytes(digest[:8], "little")])
    return rng.permutation(n).tolist()


def present(request: DecisionRequest, order: Sequence[int]) -> DecisionRequest:
    """Request with options reordered so position j shows canonical ``order[j]``."""
    return replace(request, options=tuple(request.options[i] for i in order))


def to_canonical(output: ModelOutput, order: Sequence[int]) -> tuple[np.ndarray, int | None]:
    probs = np.zeros(len(order), dtype=np.float64)
    probs[np.asarray(order)] = np.asarray(output.probs, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("model returned invalid probabilities")
    selected = None if output.selected is None else int(order[output.selected])
    return probs / total, selected


def decide_batch(
    model: DecisionModel,
    requests: Sequence[DecisionRequest],
    *,
    rng: np.random.Generator | None = None,
    orders: Sequence[Sequence[int]] | None = None,
) -> list[DecisionResult]:
    """Run requests (canonical option order) with optional option shuffling.

    ``orders`` gives explicit presentation orders (e.g. from :func:`item_order`);
    otherwise they are drawn from ``rng`` (identity when ``rng`` is None).
    """
    if orders is None:
        orders = [presentation_order(len(r.options), rng) for r in requests]
    shown = [present(r, o) for r, o in zip(requests, orders)]
    for r in shown:
        model.check(r)
    return assemble_results(model.name, requests, orders, model.predict_batch(shown))


def assemble_results(
    model_name: str,
    requests: Sequence[DecisionRequest],
    orders: Sequence[Sequence[int]],
    outputs: Sequence[ModelOutput],
) -> list[DecisionResult]:
    """Map presented-order outputs back to canonical order."""
    results = []
    for req, order, out in zip(requests, orders, outputs):
        if out.refused:
            results.append(DecisionResult(item_id=req.item_id, model=model_name, options=req.options,
                                          probs=np.full(len(order), np.nan), presented_order=list(order),
                                          selected=-1, raw=out.raw, refused=True))
            continue
        probs, selected = to_canonical(out, order)
        results.append(
            DecisionResult(
                item_id=req.item_id,
                model=model_name,
                options=req.options,
                probs=probs,
                presented_order=list(order),
                selected=int(np.argmax(probs)) if selected is None else selected,
                raw=out.raw,
            )
        )
    return results


def decide(
    model: DecisionModel, request: DecisionRequest, *, rng: np.random.Generator | None = None
) -> DecisionResult:
    return decide_batch(model, [request], rng=rng)[0]
