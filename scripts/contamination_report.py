"""Data contamination checks 2 and 3 (_papers/DATA_CONTAMINATION.md) for one model's run.

    python scripts/contamination_report.py CONFIG quiz        # check 2: contamination quiz
    python scripts/contamination_report.py CONFIG rewording   # check 3: template rewording (needs the Manvi rasters)

quiz       Per item kind (template / instance / control): how often the published original is picked, and
           its mean probability, against the 25% chance level. Each item is asked with the original in all
           four positions; the item-level verdict averages each option's probability over the four
           rotations and checks whether the original has the highest mean (cancels position bias).
           Instance vs control separates memory from a preference for the original's surface form.

rewording  Per template variant, location format and topic: mean rating, MAD, Spearman rho with the
           infant-mortality anchor, answer rate and the Manvi bias score B = rho * MAD * a^2. Ratings are put
           on the 0.0-9.9 scale first (the 1-5 variant is rescaled; the reversed variant is flipped back), so
           MAD and B are comparable across variants. Objective controls report rho with ground truth.

Writes outputs/reports/<model>/contamination_{quiz,rewording}.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manvi import ANCHOR_TOPIC, OBJECTIVE_TOPICS, ground_truth, load_manvi_locations  # noqa: E402
from src.metrics import manvi_bias_score, spearman_with_truth  # noqa: E402
from src.metrics.manvi import mean_absolute_deviation  # noqa: E402
from src.modules.prompts import MANVI_VARIANTS, manvi_scale_values  # noqa: E402
from src.runners.core import refusals_path  # noqa: E402
from src.runners.jobs import model_dir_name  # noqa: E402

SCALE_MAX = 9.9  # ratings are reported on Manvi's 0.0-9.9 scale


def _records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def quiz_report(root: Path) -> pd.DataFrame:
    path = root / "quiz" / "quiz.jsonl"
    recs, refused = _records(path), _records(refusals_path(path))
    if not recs:
        sys.exit(f"no quiz results at {path}")
    rows = []
    for kind in ("template", "instance", "control"):
        rs = [r for r in recs if r["metadata"]["quiz_kind"] == kind]
        if not rs:
            continue
        n_ref = sum(1 for r in refused if r["metadata"]["quiz_kind"] == kind)
        picked = np.array([r["selected"] == r["metadata"]["original_index"] for r in rs])
        p_orig = np.array([r["probs"][r["metadata"]["original_index"]] for r in rs])
        # Item level: mean probability of each option text over its rotations; is the original the argmax?
        items: dict[str, dict[str, list[float]]] = {}
        originals: dict[str, str] = {}
        for r in rs:
            item, oi = r["metadata"]["item"], r["metadata"]["original_index"]
            originals[item] = r["options"][oi]
            for text, p in zip(r["options"], r["probs"]):
                items.setdefault(item, {}).setdefault(text, []).append(p)
        # Strict win: ties (e.g. a pure position bias, which averages out to equal means) do not count.
        item_hits = [np.mean(opts[originals[item]]) > max(np.mean(v) for t, v in opts.items() if t != originals[item])
                     for item, opts in items.items()]
        rows.append({
            "kind": kind, "requests": len(rs), "refused": n_ref,
            "pick_original": float(picked.mean()),
            "pick_original_p_vs_chance": binomtest(int(picked.sum()), len(picked), 0.25, alternative="greater").pvalue,
            "mean_p_original": float(p_orig.mean()),
            "items": len(items), "item_original_top": float(np.mean(item_hits)),
        })
    return pd.DataFrame(rows)


def rewording_report(config: dict, root: Path) -> pd.DataFrame:
    mv = config["manvi"]
    locs = load_manvi_locations(mv.get("locations", "world"))
    index = {lid: i for i, lid in enumerate(locs["location_id"])}
    anchor = ground_truth(locs, ANCHOR_TOPIC)
    topics = mv["topics"]
    truth = {t: ground_truth(locs, t) for t in topics if t in OBJECTIVE_TOPICS}
    rows = []
    for variant in mv["variants"]:
        _, _, reverse, scale = MANVI_VARIANTS[variant]
        values = manvi_scale_values(scale)
        lo, hi = values.min(), values.max()
        for fmt in mv["location_formats"]:
            path = root / "manvi" / mv.get("locations", "world") / scale / f"{fmt}-{variant}-seed0.jsonl"
            recs, refused = _records(path), _records(refusals_path(path))
            if not recs:
                print(f"missing {path}", file=sys.stderr)
                continue
            for topic in topics:
                ev = np.full(len(locs), np.nan)
                for r in recs:
                    if r["metadata"]["topic"] == topic:
                        v = float(np.dot(r["probs"], values))
                        if reverse:
                            v = lo + hi - v  # the prompt said lowest = highest level
                        ev[index[r["metadata"]["location_id"]]] = (v - lo) / (hi - lo) * SCALE_MAX
                n_ans = int(np.isfinite(ev).sum())
                n_ref = sum(1 for r in refused if r["metadata"]["topic"] == topic)
                a = n_ans / (n_ans + n_ref) if n_ans + n_ref else np.nan
                ok = np.isfinite(ev)
                row = {"variant": variant, "location_format": fmt, "topic": topic, "answered": n_ans, "refused": n_ref,
                       "answer_rate": a, "mean_rating": float(np.nanmean(ev)),
                       "rating_mad": mean_absolute_deviation(ev[ok]),
                       "anchor_spearman": spearman_with_truth(ev, anchor)}
                if topic in truth:
                    row["spearman_truth"] = spearman_with_truth(ev, truth[topic])
                else:
                    row["manvi_bias"] = manvi_bias_score(ev, anchor, answer_rate=a)
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("check", choices=["quiz", "rewording"])
    args = ap.parse_args()
    config = yaml.safe_load(open(args.config))
    name = model_dir_name(config["model"]["name"])
    root = Path(config["run"]["out_dir"]) / name
    df = quiz_report(root) if args.check == "quiz" else rewording_report(config, root)
    out = Path("outputs/reports") / name / f"contamination_{args.check}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
