"""Project path resolution from configs/paths.yaml with environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "paths.yaml"


def _resolve(p: str | os.PathLike | None) -> Path | None:
    if p is None:
        return None
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class Paths:
    locbench_root: Path
    torchspatial_eval_dir: Path
    cache_dir: Path
    manvi_prompt_dir: Path
    manvi_raster_dir: Path
    image_roots: dict[str, Path | None] = field(default_factory=dict)

    def dataset_dir(self, dataset: str) -> Path:
        return self.locbench_root / dataset

    def image_root(self, dataset: str) -> Path | None:
        return self.image_roots.get(dataset)


def load_paths(config: str | os.PathLike | None = None) -> Paths:
    """Load paths from YAML; ``JEVGEO_<KEY>`` env vars override top-level keys."""
    with open(config or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f) or {}

    def get(key: str) -> str | None:
        return os.environ.get(f"JEVGEO_{key.upper()}", cfg.get(key))

    image_roots = {
        name: _resolve(os.environ.get(f"JEVGEO_IMAGE_ROOT_{name.upper()}", root))
        for name, root in (cfg.get("image_roots") or {}).items()
    }
    return Paths(
        locbench_root=_resolve(get("locbench_root")),
        torchspatial_eval_dir=_resolve(get("torchspatial_eval_dir")),
        cache_dir=_resolve(get("cache_dir")),
        manvi_prompt_dir=_resolve(get("manvi_prompt_dir")),
        manvi_raster_dir=_resolve(get("manvi_raster_dir")),
        image_roots=image_roots,
    )
