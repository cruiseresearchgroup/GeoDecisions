"""Inputs for the parity tests; tests/fixtures/ holds the original GeoBS/GeoLLM code's outputs on them."""

from __future__ import annotations

import hashlib

import numpy as np

from src.metrics.geobias import binary_marks


def cluster(seed: int = 0, n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return 40 + rng.normal(0, 0.4, n), -100 + rng.normal(0, 0.4, n)


def geobs_ssi_inputs() -> tuple[np.ndarray, np.ndarray, float]:
    """Points (radians, [lat, lon]), +-1 marks and radius for the GeoBS SSI parity check."""
    lat, lon = cluster()
    return np.deg2rad(np.stack([lat, lon], 1)), binary_marks(lon > -100), 0.02


def prompt_list_digest(prompts: list[str]) -> str:
    return hashlib.sha256("\x00".join(prompts).encode()).hexdigest()
