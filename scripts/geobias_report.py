"""Geo-bias report (SSI / SRE) for one model's run, with TorchSpatial baselines.

    python scripts/geobias_report.py CONFIG [--n-centers 2000] [--workers 24] [--skip-manvi] [--skip-baselines]
        [--datasets D ...] [--only PATTERN ...] [--suffix S]

--datasets/--only restrict Track 1 to some datasets / result files (stem contains PATTERN);
--suffix names the outputs geobias_<track><S>.csv so partial reports don't overwrite the full one.

Track 1 (LocBench): every results file under locbench/<dataset>/ (except null
runs) and every TorchSpatial encoder, scored on the same spatial sample of
center items. Marks: hit@1 (GeoBS binary convention). For decision-model
results, also per-sample calibration error (binned) as marks.

Track 2 (Manvi): per location format and topic, marks are the absolute
fractional-rank error vs ground truth (objective topics) or the rating itself
(subjective topics: an unbiased model gives a flat map). The 2,000 locations
are ~300 km apart, so Track 2 uses larger radii (see MANVI_PARAMS); these are
provisional choices, not from the papers.

Writes outputs/reports/<model>/geobias_{locbench,manvi}.csv and params.json.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one BLAS thread per worker process
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import available_encoders, load_baseline  # noqa: E402
from src.data.manvi import ANCHOR_TOPIC, OBJECTIVE_TOPICS, SUBJECTIVE_TOPICS, ground_truth, load_manvi_locations  # noqa: E402
from src.data.sampling import spatial_subsample  # noqa: E402
from src.evaluation import geobias_scores, report, results_to_table  # noqa: E402
from src.metrics import manvi_bias_score, spearman_with_truth  # noqa: E402
from src.metrics.geobias import binned_marks  # noqa: E402
from src.modules.prompts import manvi_scale_values  # noqa: E402
from src.runners import load_results  # noqa: E402
from src.runners.core import refusals_path  # noqa: E402
from src.runners.jobs import _locbench_dataset, model_dir_name  # noqa: E402
from src.utils.progress import progress  # noqa: E402

LOCBENCH_PARAMS = {"ssi_radius_km": 100.0, "ssi_k": 4}  # TorchSpatial Table 2
CALERR_EDGES = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
MANVI_PARAMS = {
    "ssi_radius_km": 1000.0,
    "ssi_k": 4,
    "sre_kwargs": {"radius_km": 2000.0, "scale_km": 500.0, "lag_km": 500.0, "min_count": 5, "bins": 4},
}
RANK_ERR_EDGES = np.array([0.0, 0.1, 0.25, 0.5, 1.0])
RATING_EDGES = np.array([0.0, 2.5, 5.0, 7.5, 10.0])


# ---------------------------------------------------------------- Track 1 tasks

def _locbench_task(dataset: str, source: str, table: pd.DataFrame, center_ids: list[str], is_model: bool) -> dict:
    centers = np.flatnonzero(table["item_id"].isin(set(center_ids)).to_numpy())
    row = {"dataset": dataset, "source": source, "n_centers": len(centers)}
    row.update(report(table, centers=centers, **LOCBENCH_PARAMS))
    if is_model:
        ce = table["calibration_error"].to_numpy()
        cal = geobias_scores(
            table["lat"].to_numpy(), table["lon"].to_numpy(), binned_marks(ce, CALERR_EDGES), ce,
            centers=centers, include_unmarked=False, sre_kwargs={"bins": len(CALERR_EDGES) - 1}, **LOCBENCH_PARAMS,
        )
        row.update({f"calerr_{k}": v for k, v in cal.items()})
        row["ece_proxy_mean_calerr"] = float(ce.mean())
    return row


def locbench_tasks(config: dict, model_root: Path, n_centers: int, baselines: bool,
                   datasets: list[str] | None = None, only: list[str] | None = None) -> list[tuple]:
    tasks = []
    for dataset in datasets or config["locbench"]["datasets"]:
        ds = _locbench_dataset(dataset, config["locbench"])
        s = ds.split
        rows = np.flatnonzero(s.has_location)
        center_rows = rows[spatial_subsample(s.lat[rows], s.lon[rows], n_centers, seed=0)]
        center_ids = [s.item_ids[i] for i in center_rows]
        for path in sorted((model_root / "locbench" / dataset).glob("*.jsonl")):
            if path.stem.startswith("null") or (only and not any(p in path.stem for p in only)):
                continue
            table = results_to_table(load_results(path), ds)
            tasks.append((dataset, f"{config['model']['name']}:{path.stem}", table, center_ids, True))
        if baselines:
            for enc in available_encoders(dataset):
                tasks.append((dataset, f"torchspatial:{enc}", load_baseline(s, enc), center_ids, False))
    return tasks


# ---------------------------------------------------------------- Track 2 tasks

def _frac_rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(pct=True).to_numpy()


def _manvi_task(fmt: str, topic: str, lat, lon, ev: np.ndarray, truth: np.ndarray | None, anchor: np.ndarray,
                n_refused: int = 0) -> dict:
    # Answer rate as in Manvi et al.: answered / (answered + refused). Locations never run are not refusals.
    n_answered = int(np.isfinite(ev).sum())
    answer_rate = n_answered / (n_answered + n_refused) if n_answered + n_refused else np.nan
    row = {"location_format": fmt, "topic": topic, "mean_rating": float(np.nanmean(ev)),
           "n_answered": n_answered, "n_refused": n_refused, "answer_rate": answer_rate}
    if truth is not None:
        ok = np.isfinite(ev) & np.isfinite(truth)
        err = np.full(len(ev), np.nan)
        err[ok] = np.abs(_frac_rank(ev[ok]) - _frac_rank(truth[ok]))
        row["spearman"] = spearman_with_truth(ev, truth)
        row["mean_rank_error"] = float(np.nanmean(err))
        values, edges = err, RANK_ERR_EDGES
    else:
        ok = np.isfinite(ev)
        row["manvi_bias"] = manvi_bias_score(ev, anchor, answer_rate=answer_rate)  # rho * MAD * a^2 over answered
        # Components of the bias score: a model can score near 0 by compressing ratings (small MAD)
        # while still ordering locations by the anchor (large |rho|), so report both.
        ok_a = ok & np.isfinite(anchor)
        row["anchor_spearman"] = spearman_with_truth(ev[ok_a], anchor[ok_a]) if ok_a.sum() > 2 else np.nan
        row["rating_mad"] = float(np.mean(np.abs(ev[ok] - ev[ok].mean()))) if ok.any() else np.nan
        values, edges = ev, RATING_EDGES
    lat_ok, lon_ok, v = np.where(ok, lat, np.nan), np.where(ok, lon, np.nan), np.nan_to_num(values)
    row.update(geobias_scores(lat_ok, lon_ok, binned_marks(v, edges), v, include_unmarked=False, **MANVI_PARAMS))
    return row


def manvi_tasks(config: dict, model_root: Path) -> tuple[list[tuple], dict]:
    mv = config["manvi"]
    locs = load_manvi_locations(mv.get("locations", "world"))
    index = {lid: i for i, lid in enumerate(locs["location_id"])}
    lat, lon = locs["lat"].to_numpy(), locs["lon"].to_numpy()
    truth = {t: ground_truth(locs, t) for t in OBJECTIVE_TOPICS}
    tasks = []
    for scale in mv.get("scales", ["0.0-9.9"]):
        values = manvi_scale_values(scale)
        for path in sorted((model_root / "manvi" / mv.get("locations", "world") / scale).glob("*.jsonl")):
            if path.stem.startswith("null"):
                continue
            ev: dict[str, np.ndarray] = {}
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    t, lid = rec["metadata"]["topic"], rec["metadata"]["location_id"]
                    ev.setdefault(t, np.full(len(locs), np.nan))[index[lid]] = float(np.dot(rec["probs"], values))
            refused: dict[str, int] = {}
            if refusals_path(path).exists():  # closed APIs (e.g. GPT-6 Luna Decisions) may decline to answer
                with open(refusals_path(path)) as f:
                    for line in f:
                        try:
                            t = json.loads(line)["metadata"]["topic"]
                        except (json.JSONDecodeError, KeyError):
                            continue
                        refused[t] = refused.get(t, 0) + 1
                        ev.setdefault(t, np.full(len(locs), np.nan))
            fmt = f"{scale}:{path.stem}"
            for t in OBJECTIVE_TOPICS + SUBJECTIVE_TOPICS:
                if t in ev:
                    tasks.append((fmt, t, lat, lon, ev[t], truth.get(t), truth[ANCHOR_TOPIC], refused.get(t, 0)))
    # Unmarked SSI depends only on where the locations are: identical for every topic and format.
    ssi_params = {k: v for k, v in MANVI_PARAMS.items() if k != "sre_kwargs"}
    unmarked = geobias_scores(lat, lon, np.ones(len(lat)), np.ones(len(lat)), include_sre=False, **ssi_params)
    return tasks, {"ssi_unmarked_locations": unmarked["ssi_unmarked"]}


# ---------------------------------------------------------------- main

def _run(fn, tasks, workers, desc):
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, *t) for t in tasks]
        for fut in progress(as_completed(futures), total=len(futures), desc=desc, unit="task"):
            rows.append(fut.result())
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n-centers", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--skip-manvi", action="store_true")
    ap.add_argument("--skip-locbench", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    name = model_dir_name(config["model"]["name"])
    model_root = Path(config["run"]["out_dir"]) / name
    out_dir = Path("outputs/reports") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    params = {"n_centers": args.n_centers, "locbench": LOCBENCH_PARAMS, "calerr_edges": CALERR_EDGES.tolist(),
              "manvi": MANVI_PARAMS, "rank_err_edges": RANK_ERR_EDGES.tolist(), "rating_edges": RATING_EDGES.tolist()}

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    if not args.skip_locbench and config.get("locbench"):
        tasks = locbench_tasks(config, model_root, args.n_centers, not args.skip_baselines, args.datasets, args.only)
        df = pd.DataFrame(_run(_locbench_task, tasks, args.workers, "locbench"))
        df = df.sort_values(["dataset", "source"]).reset_index(drop=True)
        df.to_csv(out_dir / f"geobias_locbench{args.suffix}.csv", index=False)
        print(df.round(4).to_string())
    if not args.skip_manvi and config.get("manvi"):
        tasks, extra = manvi_tasks(config, model_root)
        df = pd.DataFrame(_run(_manvi_task, tasks, args.workers, "manvi"))
        df = df.sort_values(["location_format", "topic"]).reset_index(drop=True)
        df.to_csv(out_dir / f"geobias_manvi{args.suffix}.csv", index=False)
        params.update(extra)
        print(df.round(4).to_string())
    (out_dir / f"params{args.suffix}.json").write_text(json.dumps(params, indent=1))


if __name__ == "__main__":
    main()
