"""Option-order sensitivity of Task 1 accuracy (DATA_CONTAMINATION.md, check 1).

    python scripts/order_sensitivity.py [CONFIG ...] [--main-root outputs/runs] [--out outputs/reports/order_sensitivity.csv]

CONFIG defaults to every configs/order/*.yaml. For each model, dataset, and image setting, compares
top-1 accuracy on the same items under:
  * seed 0 (the main runs in --main-root, restricted to the config's items),
  * the extra shuffle seeds of the config (in its run.out_dir), and
  * the original (unshuffled) label order, where the config sets ``original_order``.

Columns:
  acc_seed<k>          top-1 accuracy for shuffle seed k
  acc_shuf_mean/sd     mean and sample SD over shuffle seeds; acc_shuf_range = max - min
  pred_agree           fraction of items whose predicted label is the same under every shuffle seed
  acc_orig             accuracy in the original label order
  orig_gain            acc_orig - acc_shuf_mean (MM-Detect's correct-rate reduction, sign flipped)
  orig_gain_z          orig_gain / acc_shuf_sd
  orig_only            fraction correct in the original order but wrong under every shuffle seed
  shuf_only            fraction wrong in the original order but correct under every shuffle seed

A contamination signal is orig_gain consistently above the shuffle spread (orig_only >> shuf_only);
a small acc_shuf_sd with orig_gain near 0 is evidence against memorized items.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.locbench import results_to_table  # noqa: E402
from src.runners.core import load_results  # noqa: E402
from src.runners.jobs import _locbench_dataset, model_dir_name  # noqa: E402

log = logging.getLogger("order_sensitivity")


def _hits(path: Path, dataset) -> tuple[pd.Series, pd.Series] | None:
    """hit@1 and predicted canonical label per item_id for one results file, or None if missing."""
    if not path.exists():
        return None
    table = results_to_table(load_results(path), dataset)
    return table.set_index("item_id")["hit@1"], table.set_index("item_id")["selected"]


def order_metrics(hits: dict[str, pd.Series], sel: dict[str, pd.Series]) -> dict[str, float]:
    """Accuracy spread over shuffle seeds ("seed<k>" keys) and, if present, the original order ("orig").

    ``hits``/``sel`` map variant -> per-item hit@1 / predicted canonical label (indexed by item_id);
    only items present in every variant are compared.
    """
    shuf = [k for k in hits if k.startswith("seed")]
    common = sorted(set.intersection(*(set(h.index) for h in hits.values())))
    H = pd.DataFrame({k: h.loc[common] for k, h in hits.items()})
    S = pd.DataFrame({k: s.loc[common] for k, s in sel.items()})
    accs = H[shuf].mean()
    out = {"n": len(common)}
    out.update({f"acc_{k}": float(v) for k, v in accs.items()})
    out["acc_shuf_mean"] = float(accs.mean())
    out["acc_shuf_sd"] = float(accs.std(ddof=1))
    out["acc_shuf_range"] = float(accs.max() - accs.min())
    out["pred_agree"] = float((S[shuf].nunique(axis=1) == 1).mean())
    if "orig" in H:
        out["acc_orig"] = float(H["orig"].mean())
        out["orig_gain"] = out["acc_orig"] - out["acc_shuf_mean"]
        out["orig_gain_z"] = out["orig_gain"] / out["acc_shuf_sd"] if out["acc_shuf_sd"] > 0 else np.nan
        out["orig_only"] = float(((H["orig"] == 1) & ~H[shuf].any(axis=1)).mean())
        out["shuf_only"] = float(((H["orig"] == 0) & H[shuf].all(axis=1)).mean())
    return out


def analyse(config_path: Path, main_root: Path) -> list[dict]:
    config = yaml.safe_load(open(config_path))
    name = model_dir_name(config["model"]["name"])
    lb = config["locbench"]
    seeds = config["run"].get("seeds", [])
    order_root = Path(config["run"]["out_dir"]) / name / "locbench"
    rows = []
    for ds_name in lb["datasets"]:
        dataset = _locbench_dataset(ds_name, lb)
        for cond in lb["conditions"]:
            for fmt in lb["location_formats"]:
                stem = f"{cond}-{fmt}"
                variants = {"seed0": main_root / name / "locbench" / ds_name / f"{stem}-seed0.jsonl"}
                variants.update({f"seed{s}": order_root / ds_name / f"{stem}-seed{s}.jsonl" for s in seeds})
                if ds_name in lb.get("original_order", []):
                    variants["orig"] = order_root / ds_name / f"{stem}-orig.jsonl"

                hits, sel = {}, {}
                for key, path in variants.items():
                    out = _hits(path, dataset)
                    if out is None:
                        log.warning("%s %s %s: missing %s", name, ds_name, stem, path)
                        continue
                    hits[key], sel[key] = out
                shuf = [k for k in hits if k.startswith("seed")]
                if len(shuf) < 2:
                    log.warning("%s %s %s: fewer than two shuffle seeds, skipped", name, ds_name, stem)
                    continue

                row = {"model": config["model"]["name"], "dataset": ds_name, "setting": stem}
                row.update(order_metrics(hits, sel))
                rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", type=Path)
    ap.add_argument("--main-root", type=Path, default=Path("outputs/runs"))
    ap.add_argument("--out", type=Path, default=Path("outputs/reports/order_sensitivity.csv"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    configs = args.configs or sorted(Path("configs/order").glob("*.yaml"))
    rows = [r for c in configs for r in analyse(c, args.main_root)]
    if not rows:
        sys.exit("no results found")
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    cols = ["model", "dataset", "setting", "n", "acc_shuf_mean", "acc_shuf_sd", "acc_shuf_range", "pred_agree"]
    cols += [c for c in ("acc_orig", "orig_gain", "orig_gain_z", "orig_only", "shuf_only") if c in df]
    try:
        print(df[cols].to_markdown(index=False, floatfmt=".4f"))
    except ImportError:  # tabulate not installed
        print(df[cols].to_string(index=False, float_format="%.4f"))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
