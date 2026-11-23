"""tqdm progress bars that stay readable when stderr is redirected to a log file."""

from __future__ import annotations

import sys
from typing import Any

from tqdm.auto import tqdm


def progress(iterable: Any = None, **kwargs: Any) -> tqdm:
    """``tqdm`` with defaults: live redraw on a terminal; one line every 30 s in log files."""
    interactive = sys.stderr.isatty()
    kwargs.setdefault("dynamic_ncols", interactive)
    kwargs.setdefault("mininterval", 0.5 if interactive else 30.0)
    kwargs.setdefault("smoothing", 0.1)
    return tqdm(iterable, **kwargs)
