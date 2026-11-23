"""Run decision-model jobs from a config.

    python scripts/run.py CONFIG [--only PATTERN] [--limit N] [--dry-run] [--tokens N]

--only     run jobs whose name contains PATTERN (repeatable)
--limit    run at most N new items per job (smoke test); results append to the same files
--dry-run  list jobs with request counts; no model is loaded
--tokens   with --dry-run: tokenize N sampled requests per job and report length stats
           (loads only the tokenizer/processor)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.registry import build_model  # noqa: E402
from src.runners import build_jobs, completed_ids, run_job  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tokens", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    jobs = [j for j in build_jobs(config) if not args.only or any(p in j.name for p in args.only)]

    if args.dry_run:
        tokenizer = _token_counter(config) if args.tokens else None
        total = 0
        for job in jobs:
            reqs = job.build()
            total += len(reqs)
            line = f"{job.name:55s} {len(reqs):7d} requests  {len(completed_ids(job.out_path)):7d} done"
            if tokenizer:
                idx = np.random.default_rng(0).choice(len(reqs), min(args.tokens, len(reqs)), replace=False)
                lens = np.array(tokenizer([reqs[i] for i in idx]))
                line += f"  tokens mean {lens.mean():.0f} p95 {np.percentile(lens, 95):.0f} max {lens.max()}"
            tqdm.write(line)
        print(f"{'TOTAL':55s} {total:7d} requests")
        return

    model = build_model(config["model"])
    stats = {}
    for i, job in enumerate(jobs, 1):
        logging.info("job %d/%d: %s", i, len(jobs), job.name)
        stats[job.name] = run_job(model, job, chunk_size=config["run"].get("chunk_size", 1024), limit=args.limit)
    print(json.dumps(stats, indent=1))


def _token_counter(config):
    """Request -> token length without loading weights (pplx_decider protocol encoder)."""
    if config["model"]["kind"] != "pplx_decider":
        raise SystemExit("--tokens is implemented for pplx_decider only")
    from huggingface_hub import snapshot_download
    from pplx_decider.model import _resolve  # noqa: F401  (ensures package importable)
    from pplx_decider.protocol import Encoder, load_json
    from transformers import AutoProcessor

    from src.modules.pplx_decider import to_native

    ckpt = Path(snapshot_download(config["model"]["repo"]))
    processor = AutoProcessor.from_pretrained(str(ckpt))
    encoder = Encoder(processor, load_json(ckpt / "decision_config.json")["codes"])
    return lambda reqs: [encoder.encode(to_native(r)).length for r in reqs]


if __name__ == "__main__":
    main()
