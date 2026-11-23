"""Resumable execution of a list of decision requests against one model.

A job = one output JSONL file of DecisionResults. Results are appended chunk by
chunk; on restart, item_ids already in the file are skipped. Option orders come
from :func:`src.modules.base.item_order`, so resumed items are presented exactly
as they would have been in an uninterrupted run.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from src.modules.base import DecisionModel, DecisionRequest, DecisionResult, decide_batch, item_order
from src.utils.progress import progress

log = logging.getLogger(__name__)


@dataclass
class Job:
    """``build`` is called lazily so listing jobs (dry run) stays cheap."""

    name: str
    out_path: Path
    build: Callable[[], list[DecisionRequest]]
    seed: int | None = None


def completed_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["item_id"])
            except (json.JSONDecodeError, KeyError):
                continue  # truncated final line from an interrupted write
    return done


def refusals_path(out_path: Path) -> Path:
    """Refused items of a job: ``<dir>/refusals/<job file>`` (kept out of the results the reports glob)."""
    return out_path.parent / "refusals" / out_path.name


def load_results(path: Path) -> list[DecisionResult]:
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(DecisionResult.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def run_job(
    model: DecisionModel,
    job: Job,
    *,
    chunk_size: int = 1024,
    limit: int | None = None,
    requests: Sequence[DecisionRequest] | None = None,
) -> dict[str, float]:
    """Run a job to completion (or ``limit`` new items). Returns timing stats.

    Each chunk is one ``predict_batch`` call; results are flushed after it, so a
    smaller ``chunk_size`` loses less on interruption and updates progress more
    often, while a larger one gives the model more requests to pack.
    """
    requests = job.build() if requests is None else requests
    refused_path = refusals_path(job.out_path)
    done = completed_ids(job.out_path) | completed_ids(refused_path)  # refusals are not re-sent
    todo = [r for r in requests if r.item_id not in done]
    if limit is not None:
        todo = todo[:limit]
    job.out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("%s: %d requests, %d done, %d to run", job.name, len(requests), len(done), len(todo))

    start, n, n_refused = time.time(), 0, 0
    bar = progress(total=len(todo), desc=job.name, unit="req")
    for i in range(0, len(todo), chunk_size):
        chunk = todo[i:i + chunk_size]
        results = decide_batch(model, chunk, orders=[item_order(r, job.seed) for r in chunk])
        refused = [(req, res) for req, res in zip(chunk, results) if res.refused]
        if refused:
            refused_path.parent.mkdir(parents=True, exist_ok=True)
            n_refused += len(refused)
            log.warning("%s: %d refused in this chunk (%s)", job.name, len(refused), refused_path)
        for path, pairs in ((job.out_path, [(q, r) for q, r in zip(chunk, results) if not r.refused]), (refused_path, refused)):
            if not pairs:
                continue
            with open(path, "a") as f:
                for req, res in pairs:
                    rec = res.to_dict()
                    rec["seed"] = job.seed
                    rec["metadata"] = req.metadata
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += len(chunk)
        bar.update(len(chunk))
    bar.close()
    elapsed = time.time() - start
    return {"total": len(requests), "ran": n, "refused": n_refused, "seconds": elapsed,
            "req_per_s": n / elapsed if elapsed else 0.0}
