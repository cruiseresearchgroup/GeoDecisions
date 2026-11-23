"""Build a DecisionModel from a config dict (``model:`` block of configs/runs/*.yaml)."""

from __future__ import annotations

from typing import Any

from src.modules.base import DecisionModel


def build_model(cfg: dict[str, Any]) -> DecisionModel:
    cfg = dict(cfg)
    kind = cfg.pop("kind")
    if kind == "pplx_decider":
        from src.modules.pplx_decider import PplxDecider

        return PplxDecider(**cfg)
    if kind == "glide":
        from src.modules.glide import GlideModel

        return GlideModel(**cfg)
    if kind == "label_readout":
        from src.modules.label_readout import LabelReadoutClient

        return LabelReadoutClient(**cfg)
    if kind == "sieve":
        from src.modules.sieve import SieveDecider

        return SieveDecider(**cfg)
    if kind == "knowline":
        from src.modules.knowline import KnowLineClient

        return KnowLineClient(**cfg)
    if kind == "gliclass":
        from src.modules.gliclass import GLiClassDecider

        return GLiClassDecider(**cfg)
    if kind == "hf_option_scorer":
        from src.modules.hf_readout import HFOptionScorer

        return HFOptionScorer.from_pretrained(cfg.pop("repo"), **cfg)
    if kind == "jev":
        from src.modules.typesafe_jev import JevClient

        return JevClient(**cfg)
    if kind == "jev_sdk":  # same model through the official typesafe-sdk (produced the Manvi results)
        from src.modules.jev import JevClient as JevSDKClient

        return JevSDKClient(**cfg)
    if kind == "openai_decisions":
        from src.modules.openai_decision import OpenAIDecisionsClient

        return OpenAIDecisionsClient(**cfg)
    if kind == "openai_chat":
        from src.modules.api import OpenAIChatReadoutClient

        return OpenAIChatReadoutClient(**cfg)
    raise ValueError(f"unknown model kind {kind!r}")
