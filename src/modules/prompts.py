"""Build decision requests for LocBench items (Track 1) and Manvi et al.
locations (Track 2), and render them as text.

Conditions:
    image_only      image, no location
    image_location  image + location text
    location_only   location text, no image
Location formats:
    coords      "Latitude 39.5425, Longitude -77.1073"
    place_name  reverse-geocoded name, supplied by the caller (cached upstream)

The question wording below is a first draft; freeze it before any paid runs,
since changing it invalidates cached results.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

import numpy as np

from src.modules.base import DecisionRequest

Condition = Literal["image_only", "image_location", "location_only"]
LocationFormat = Literal["coords", "place_name"]

CONDITIONS: tuple[Condition, ...] = ("image_only", "image_location", "location_only")

_QUESTIONS: dict[str, dict[str, str]] = {
    "fmow": {
        "image": "Which land use or building category does this satellite image show?",
        "location": "Which land use or building category is most likely to be found at this location?",
    },
    "yfcc": {
        "image": "Which of these concepts best describes this photo?",
        "location": "Which of these concepts best describes a photo taken at this location?",
    },
    "inat": {
        "image": "Which species is shown in this photo?",
        "location": "Which species is most likely to be observed at this location?",
    },
}


def _questions(dataset: str) -> dict[str, str]:
    return _QUESTIONS["inat" if dataset.startswith("inat") else dataset]


def format_location(
    lat: float, lon: float, fmt: LocationFormat, place_name: str | None = None
) -> str:
    if fmt == "coords":
        return f"Location: latitude {lat:.4f}, longitude {lon:.4f}"
    if fmt == "place_name":
        if not place_name:
            raise ValueError("place_name format needs a reverse-geocoded name")
        return f"Location: {place_name}"
    raise ValueError(f"unknown location format {fmt!r}")


def build_locbench_request(
    item: Mapping[str, Any],
    condition: Condition,
    *,
    location_format: LocationFormat = "coords",
    place_name: str | None = None,
) -> DecisionRequest:
    """Choice request from a :class:`~src.data.datasets.LocBenchDataset` item.

    Options are the item's candidates in canonical (class-id or shortlist) order;
    :func:`src.modules.base.decide` handles shuffling.
    """
    uses_image = condition in ("image_only", "image_location")
    uses_location = condition in ("image_location", "location_only")
    if uses_location and not item["has_location"]:
        raise ValueError(f"{item['item_id']} has no location")
    if uses_image and item.get("image_path") is None:
        raise ValueError(f"{item['item_id']} has no local image (set image_roots in configs/paths.yaml)")

    q = _questions(item["dataset"])
    context = (
        format_location(item["lat"], item["lon"], location_format, place_name)
        if uses_location
        else ""
    )
    return DecisionRequest(
        item_id=item["item_id"],
        kind="choice",
        question=q["image" if uses_image else "location"],
        options=tuple(item["options"]),
        context=context,
        image_path=item["image_path"] if uses_image else None,
        metadata={
            "condition": condition,
            "location_format": location_format if uses_location else None,
            "label_index": item["label_index"],
            "candidates": [int(c) for c in item["candidates"]],
        },
    )


def build_null_request(request: DecisionRequest) -> DecisionRequest:
    """Same question and options with no input, to estimate the model's option prior
    (contextual calibration, Zhao et al. 2021)."""
    return DecisionRequest(
        item_id=f"{request.item_id}#null",
        kind=request.kind,
        question=request.question,
        options=request.options,
        context="Location: N/A" if request.context else "",
        image_path=None,
        metadata={**request.metadata, "null_input": True},
    )


def render_text_prompt(request: DecisionRequest) -> str:
    """Plain-text rendering for models without a native decision API."""
    lines = []
    if request.context:
        lines += [request.context, ""]
    lines.append(request.question)
    lines.append("Options:")
    lines += [f"- {opt}" for opt in request.options]
    lines += ["", "Answer with exactly one option."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Track 2: Manvi et al. location ratings
# ---------------------------------------------------------------------------

ManviFormat = Literal["full", "coords", "place_name"]
MANVI_FORMATS: tuple[ManviFormat, ...] = ("full", "coords", "place_name")

# Rating scales: option labels in increasing order. "0.0-9.9" is Manvi's scale;
# "1-5" is the coarser robustness rerun.
MANVI_SCALES: dict[str, tuple[str, ...]] = {
    "0.0-9.9": tuple(f"{i / 10:.1f}" for i in range(100)),
    "1-5": ("1", "2", "3", "4", "5"),
}

# GeoLLM's PREFIX minus its output-format sentence, which a decision API replaces.
MANVI_PREAMBLE = (
    "You will be given data about a specific location randomly sampled from all "
    "human-populated locations on Earth.\n"
    "You give your rating keeping in mind that it is relative to all other "
    "human-populated locations on Earth (from all continents, countries, etc.)."
)


def manvi_scale_values(scale: str) -> np.ndarray:
    """Numeric value of each option on ``scale`` (for expected-value readout)."""
    return np.array([float(v) for v in MANVI_SCALES[scale]])


def manvi_location_text(location: Mapping[str, Any], fmt: ManviFormat) -> str:
    """Location block in one of three forms.

    full: GeoLLM's original (coordinates, address, nearby places);
    coords: coordinates only; place_name: address only.
    """
    coords = f"Coordinates: ({location['lat']}, {location['lon']})"
    address = f'Address: "{location["address"]}"'
    if fmt == "coords":
        return coords
    if fmt == "place_name":
        return address
    if fmt == "full":
        return f'{coords}\n\n{address}\n\nNearby Places:\n"\n{location["nearby"]}\n"'
    raise ValueError(f"unknown Manvi location format {fmt!r}")


# Template-rewording variants (data contamination check 3, _papers/DATA_CONTAMINATION.md). Each variant
# changes one surface aspect of Manvi's template while keeping the task: if a model's ratings (e.g. their
# compression) change only under the original wording, the behaviour is specific to the published template.
MANVI_PREAMBLE_REWORDED = (
    "The text below describes a place where people live, chosen at random from all inhabited places in the "
    "world.\n"
    "Rate it relative to every other inhabited place worldwide, across all continents and countries."
)

# Paraphrases of each topic name (canonical names stay in the metadata, so scoring is unchanged).
MANVI_TOPIC_SYNONYMS: dict[str, str] = {
    "Infant Mortality Rate": "How many babies here die before their first birthday",
    "Population Density": "How many people live per square kilometre here",
    "Built-Up to Non Built-Up Area Ratio": "How much of the land here is covered by buildings and roads",
    "Nighttime Light Intensity": "How brightly lit this area is at night",
    "Average Temperature": "How warm it is here over the year",
    "Annual Precipitation": "How much rain and snow falls here per year",
    "Average Likability of Residents": "How likeable the people who live here are",
    "Average Attractiveness of Residents": "How good-looking the people who live here are",
    "Average Morality of Residents": "How moral the people who live here are",
    "Average Intelligence of Residents": "How smart the people who live here are",
    "Average Work Ethic of Residents": "How hard-working the people who live here are",
}

# name -> (preamble, use topic synonyms, reversed scale, scale)
MANVI_VARIANTS: dict[str, tuple[str, bool, bool, str]] = {
    "original": (MANVI_PREAMBLE, False, False, "0.0-9.9"),
    "reworded": (MANVI_PREAMBLE_REWORDED, False, False, "0.0-9.9"),
    "synonyms": (MANVI_PREAMBLE, True, False, "0.0-9.9"),
    "scale5": (MANVI_PREAMBLE, False, False, "1-5"),
    "reversed": (MANVI_PREAMBLE, False, True, "0.0-9.9"),
}


def build_manvi_request(
    location: Mapping[str, Any],
    topic: str,
    *,
    location_format: ManviFormat = "full",
    scale: str = "0.0-9.9",
    variant: str | None = None,
) -> DecisionRequest:
    """Score request for one location and topic; options are the scale values.

    ``variant`` (a key of :data:`MANVI_VARIANTS`) rewrites the template for the rewording check; it sets
    the scale itself and is recorded in ``item_id`` and ``metadata``. With ``reverse_scale`` the prompt
    says the lowest value means the highest level; scoring maps ratings back (``metadata["reverse_scale"]``).
    """
    preamble, synonyms, reverse = MANVI_PREAMBLE, False, False
    if variant is not None:
        preamble, synonyms, reverse, scale = MANVI_VARIANTS[variant]
    options = MANVI_SCALES[scale]
    shown = MANVI_TOPIC_SYNONYMS[topic] if synonyms else topic
    direction = f", where {options[0]} is the highest and {options[-1]} is the lowest" if reverse else ""
    item_id = f"{location['location_id']}|{topic}|{location_format}|{scale}"
    metadata = {
        "location_id": location["location_id"],
        "topic": topic,
        "location_format": location_format,
        "scale": scale,
    }
    if variant is not None:
        item_id += f"|{variant}"
        metadata.update(variant=variant, reverse_scale=reverse)
    return DecisionRequest(
        item_id=item_id,
        kind="score",
        question=f"{preamble}\n\nRate: {shown} (on a scale from {options[0]} to {options[-1]}{direction}).",
        options=options,
        context=manvi_location_text(location, location_format),
        metadata=metadata,
    )


def manvi_original_prompt(location: Mapping[str, Any], topic: str) -> str:
    """GeoLLM's exact chat prompt (PREFIX + text), for generative-model comparisons."""
    prefix = (
        MANVI_PREAMBLE
        + "\nYou provide ONLY your answer in the exact format \"My answer is X.X.\" "
        "where 'X.X' represents your rating for the given topic.\n\n\n"
    )
    return prefix + location["prompt_text"].strip().replace("<TASK>", topic)
