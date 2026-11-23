"""Data contamination quiz (check 2 in _papers/DATA_CONTAMINATION.md).

Following the Data Contamination Quiz of Golchin & Surdeanu (arXiv 2311.06233), each quiz item is a
multiple-choice question whose options are one original text from the public Manvi/GeoLLM release and
three near-copies that differ only in small, meaning-preserving details. A model that has not seen the
release has no reason to prefer the original, so it should pick it 25% of the time; picking it far more
often suggests the text was in its training data.

Three item kinds:
    template  GeoLLM's published prompt prefix vs three word-level rewordings (one item).
    instance  a location's published text (coordinates, address, nearby places) vs three copies with
              the coordinates' last digit and the nearby-place distances jittered by 0.1 km, keeping the
              distance order, so every option is equally plausible.
    control   the same construction on a composite text that was never published (a location's
              coordinates and address with another location's nearby places). Any preference for the
              "original" here comes from the jitter itself (e.g. rounder numbers), not from memory.

Every item is asked once per position of the original (four rotations) to cancel position bias, so
requests must be sent unshuffled (job seed None).
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.sampling import spatial_subsample
from src.modules.base import DecisionRequest
from src.modules.prompts import MANVI_PREAMBLE, manvi_location_text

QUIZ_QUESTION = "Which of the following texts appears word for word in a publicly released dataset?"

# GeoLLM's published prompt prefix (as in manvi_original_prompt), and three word-level rewordings.
GEOLLM_PREFIX = (
    MANVI_PREAMBLE
    + "\nYou provide ONLY your answer in the exact format \"My answer is X.X.\" "
    "where 'X.X' represents your rating for the given topic."
)
TEMPLATE_REWORDINGS = (
    "You will be given information about a particular location randomly sampled from all human-populated "
    "places on Earth.\n"
    "You give your rating keeping in mind that it is relative to all other human-populated places on Earth "
    "(from all continents, countries, etc.).\n"
    "You provide ONLY your answer in the exact format \"My answer is X.X.\" where 'X.X' represents your "
    "rating for the given topic.",
    "You will be given data about a specific place randomly chosen from all human-populated locations on "
    "Earth.\n"
    "You provide your rating keeping in mind that it is relative to all other human-populated locations on "
    "Earth (from all continents, nations, etc.).\n"
    "You give ONLY your answer in the exact format \"My answer is X.X.\" where 'X.X' stands for your rating "
    "for the given topic.",
    "You will be shown data about a specific location randomly sampled from all inhabited locations on "
    "Earth.\n"
    "You give your rating bearing in mind that it is relative to all other inhabited locations on Earth "
    "(from all continents, countries, etc.).\n"
    "You provide ONLY your answer in the exact format \"My answer is X.X.\" where 'X.X' is your rating for "
    "the given topic.",
)

_NEARBY_RE = re.compile(r"^(\d+(?:\.\d+)?) km ([A-Za-z-]+): (.+)$")


def _jitter_coord(value: float, rng: np.random.Generator) -> float:
    """Shift the 5th decimal (about 1 m) by 1-3 units."""
    return round(value + rng.choice([-3, -2, -1, 1, 2, 3]) * 1e-5, 5)


def _jitter_nearby(nearby: str, rng: np.random.Generator) -> str:
    """Change each distance by -0.1/0/+0.1 km, keeping it positive and the list in distance order."""
    rows = [_NEARBY_RE.match(line).groups() for line in nearby.split("\n")]
    dists = [float(d) for d, _, _ in rows]
    for _ in range(100):
        new = [round(max(0.1, d + rng.choice([-0.1, 0.0, 0.1])), 1) for d in dists]
        if all(a <= b for a, b in zip(new, new[1:])) and new != dists:
            return "\n".join(f"{n:.1f} km {direction}: {name}" for n, (_, direction, name) in zip(new, rows))
    return nearby  # cannot jitter without breaking the order; coordinates still differ


def _near_copies(location: Mapping[str, Any], seed: Sequence[int], n: int = 3) -> list[str]:
    original = manvi_location_text(location, "full")
    rng = np.random.default_rng(seed)
    out: list[str] = []
    while len(out) < n:
        copy = {**location, "lat": _jitter_coord(location["lat"], rng), "lon": _jitter_coord(location["lon"], rng),
                "nearby": _jitter_nearby(location["nearby"], rng)}
        text = manvi_location_text(copy, "full")
        if text != original and text not in out:
            out.append(text)
    return out


def _rotations(item: str, kind: str, original: str, others: Sequence[str], meta: dict) -> list[DecisionRequest]:
    base = [original, *others]
    if len(set(base)) != len(base):
        raise ValueError(f"{item}: quiz options are not distinct")
    reqs = []
    for r in range(len(base)):
        options = tuple(base[-r:] + base[:-r]) if r else tuple(base)  # original at position r
        reqs.append(DecisionRequest(
            item_id=f"quiz|{kind}|{item}|pos{r}",
            kind="choice",
            question=QUIZ_QUESTION,
            options=options,
            metadata={**meta, "quiz_kind": kind, "item": item, "original_index": r},
        ))
    return reqs


def quiz_requests(locations: pd.DataFrame, n_instances: int = 200, seed: int = 0) -> list[DecisionRequest]:
    """Template item plus ``n_instances`` instance and control items, each in four rotations."""
    reqs = _rotations("template", "template", GEOLLM_PREFIX, TEMPLATE_REWORDINGS, {})
    keep = spatial_subsample(locations["lat"].to_numpy(), locations["lon"].to_numpy(), n_instances, seed=seed)
    locs = locations.iloc[np.sort(keep)].to_dict("records")
    for i, loc in enumerate(locs):
        lid = loc["location_id"]
        reqs += _rotations(str(lid), "instance", manvi_location_text(loc, "full"),
                           _near_copies(loc, seed=[seed, 0, i]), {"location_id": lid})
        donor = locs[(i + 1) % len(locs)]  # another location's nearby places: a text never published
        composite = {**loc, "nearby": donor["nearby"]}
        reqs += _rotations(str(lid), "control", manvi_location_text(composite, "full"),
                           _near_copies(composite, seed=[seed, 1, i]),
                           {"location_id": lid, "donor_location_id": donor["location_id"]})
    return reqs
