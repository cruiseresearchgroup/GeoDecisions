"""TorchSpatial location priors for decision models: P(y | I, x) ∝ P_dm(y | I) · P(y | x).

TorchSpatial (Wu et al., 2024) combines its image-only classifier with a separately trained location
encoder as ``pred = cnn_pred * net_prior`` (``eval_helper.compute_acc``), where ``net_prior`` is the
encoder's per-class sigmoid output. Here the image classifier is replaced by a decision model's
image-only prediction, restricted to each sample's candidate labels (all classes for fMoW/YFCC, the
top-20 shortlist for iNat).

    python scripts/torchspatial_prior.py prior   --dataset yfcc --encoder wrap
    python scripts/torchspatial_prior.py check   --dataset yfcc --encoder wrap
    python scripts/torchspatial_prior.py combine --dataset yfcc --encoder wrap --config configs/runs/<model>.yaml

prior    run the released checkpoint on every evaluation sample's coordinates and cache
         P(y | x) over its candidate labels in data/torchspatial/priors/<dataset>_<encoder>.npz
check    No Prior x cached prior must reproduce TorchSpatial's released per-sample hit@1
combine  write <setting>-coords-seed0-ts_<encoder>.jsonl next to a model's image-only results
         (same format; scripts/geobias_report.py scores it as one more result file)

TorchSpatial code and checkpoints (MIT) are downloaded once, pinned to commit ac7dbcf, into
data/torchspatial/. Samples without coordinates keep their image-only prediction, as in TorchSpatial.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_baseline  # noqa: E402
from src.data.datasets import LocBenchDataset  # noqa: E402
from src.data.locbench import load_locbench  # noqa: E402
from src.runners.jobs import model_dir_name  # noqa: E402

COMMIT = "ac7dbcf"
RAW = f"https://raw.githubusercontent.com/seai-lab/TorchSpatial/{COMMIT}"
CACHE = ROOT / "data" / "torchspatial"
CODE_FILES = ["utils.py", "models.py", "module.py", "SpatialRelationEncoder.py", "data_utils.py", "paths.py",
              "spherical_harmonics_ylm_numpy.py"]
# Released checkpoints (pre_trained_models/<dir>/<file>) for the encoders reported in the paper.
CHECKPOINTS = {
    ("wrap", "fmow"): "wrap/model_fmow_wrap_inception_v3_0.0050_64_0.0200000_1_512.pth.tar",
    ("wrap", "yfcc"): "wrap/model_yfcc_wrap_inception_v3_0.0010_64_0.0500000_1_512.pth.tar",
    ("wrap", "inat_2017"): "wrap/model_inat_2017_wrap_inception_v3_0.0050_64_0.1000000_1_512.pth.tar",
    ("wrap", "inat_2018"): "wrap/model_inat_2018_wrap_0.00316757_32_0.0001197_1_256_leakyrelu.pth.tar",
    ("Sphere2Vec-sphereC", "fmow"): "sphere2vec_sphereC/model_fmow_Sphere2Vec-sphereC_inception_v3_0.0050_64_0.0005000_1_512.pth.tar",
    ("Sphere2Vec-sphereC", "yfcc"): "sphere2vec_sphereC/model_yfcc_Sphere2Vec-sphereC_inception_v3_0.0005_64_0.0001000_1_512.pth.tar",
    ("Sphere2Vec-sphereC", "inat_2017"): "sphere2vec_sphereC/model_inat_2017_Sphere2Vec-sphereC_inception_v3_0.0001_32_0.0100000_1_1024.pth.tar",
    ("Sphere2Vec-sphereC", "inat_2018"): "sphere2vec_sphereC/model_inat_2018_Sphere2Vec-sphereC_0.0005_64_0.0010000_1_1024_leakyrelu.pth.tar",
    # At ac7dbcf the wrap and Sphere2Vec-sphereC iNat checkpoints are truncated upstream (no zip end record);
    # Space2Vec-theory's are complete, so it is the encoder used for all four datasets.
    ("Space2Vec-theory", "fmow"): "space2vec_theory/model_fmow_Space2Vec-theory_inception_v3_0.0200_64_0.0001000_360.000_1_512_BATCH8192_leakyrelu.pth.tar",
    ("Space2Vec-theory", "yfcc"): "space2vec_theory/model_yfcc_Space2Vec-theory_inception_v3_0.0100_64_0.0500000_360.000_1_256_BATCH4096_leakyrelu.pth.tar",
    ("Space2Vec-theory", "inat_2017"): "space2vec_theory/model_inat_2017_Space2Vec-theory_inception_v3_0.0200_64_0.1000000_360.000_1_256_BATCH4096_leakyrelu.pth.tar",
    ("Space2Vec-theory", "inat_2018"): "space2vec_theory/model_inat_2018_Space2Vec-theory_0.0200_64_0.0500000_360.000_1_512_BATCH4096_leakyrelu.pth.tar",
}


def _fetch(url: str, dst: Path) -> Path:
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dst)
    return dst


def _torchspatial():
    code = CACHE / f"code-{COMMIT}"
    for f in CODE_FILES:
        _fetch(f"{RAW}/main/{f}", code / f)
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    import utils as ts_utils  # TorchSpatial's main/utils.py

    return ts_utils


def _dataset(name: str) -> LocBenchDataset:
    return LocBenchDataset.from_config(name, shortlist_k=20 if name.startswith("inat") else None)


def _prior_path(dataset: str, encoder: str) -> Path:
    return CACHE / "priors" / f"{dataset}_{encoder}.npz"


def compute_prior(dataset: str, encoder: str, batch_size: int = 4096) -> Path:
    import torch

    ts = _torchspatial()
    ckpt = _fetch(f"{RAW}/pre_trained_models/{CHECKPOINTS[(encoder, dataset)]}", CACHE / "checkpoints" / CHECKPOINTS[(encoder, dataset)])
    net = torch.load(ckpt, map_location="cpu", weights_only=False)
    params = net["params"]
    params.setdefault("regress_dataset", [])  # set by TorchSpatial's trainer; classification datasets are not in it
    model = ts.get_loc_model(train_locs=None, params=params, spa_enc_type=params["spa_enc_type"],
                             num_inputs=params["num_feats"], num_classes=params["num_classes"],
                             num_filts=params["num_filts"], num_users=params["num_users"], device="cpu")
    model.load_state_dict(net["state_dict"])
    model.eval()

    ds = _dataset(dataset)
    s = ds.split
    if params["num_classes"] != s.num_classes:
        raise ValueError(f"checkpoint has {params['num_classes']} classes, {dataset} has {s.num_classes}")
    has_loc = s.has_location
    k = len(ds.candidates(0))
    prior = np.full((len(s), k), np.nan, dtype=np.float32)
    rows = np.flatnonzero(has_loc)
    dates = s.meta["date_c"].to_numpy(np.float32) if "date_c" in s.meta else np.zeros(len(s), np.float32)
    for start in range(0, len(rows), batch_size):
        idx = rows[start:start + batch_size]
        locs = np.stack([s.lon[idx], s.lat[idx]], axis=1)  # TorchSpatial order: (lon, lat)
        feats = ts.generate_model_input_feats(spa_enc_type=params["spa_enc_type"], locs=locs,
                                              dates=dates[idx], params=params, device="cpu")
        with torch.no_grad():
            p = model(feats).numpy()  # (batch, num_classes) sigmoid presence probabilities
        cand = np.stack([ds.candidates(i) for i in idx])
        prior[idx] = np.take_along_axis(p, cand, axis=1)
    out = _prior_path(dataset, encoder)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, item_ids=np.asarray(s.item_ids), prior=prior)
    print(f"{dataset}/{encoder}: prior for {len(rows)}/{len(s)} located samples over {k} candidates -> {out}")
    return out


def _load_prior(dataset: str, encoder: str) -> dict[str, np.ndarray]:
    z = np.load(_prior_path(dataset, encoder))
    return dict(zip(z["item_ids"].tolist(), z["prior"]))


def combine_probs(probs: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """P_dm(y|I) * P(y|x), renormalized; returns ``probs`` unchanged when the sample has no prior."""
    if not np.all(np.isfinite(prior)):
        return probs
    p = probs * prior
    total = p.sum()
    return p / total if total > 0 else probs


def check(dataset: str, encoder: str) -> None:
    """No Prior x cached prior vs TorchSpatial's released per-sample results for the same encoder."""
    ds = _dataset(dataset)
    s = ds.split
    probs = load_locbench(dataset, load_image_probs=True)  # TorchSpatial No Prior predictions, same rows
    z = np.load(_prior_path(dataset, encoder))
    prior = z["prior"]
    ours = np.zeros(len(s), dtype=bool)
    for i in range(len(s)):
        cand = ds.candidates(i)
        p = combine_probs(probs.image_probs_dense(i)[cand].astype(np.float64), prior[i].astype(np.float64))
        ours[i] = ds.label_indices[i] >= 0 and int(np.argmax(p)) == ds.label_indices[i]
    ref = load_baseline(s, encoder)["hit@1"].to_numpy().astype(bool)
    print(f"{dataset}/{encoder}: hit@1 ours {ours.mean():.4f} vs TorchSpatial {ref.mean():.4f}; "
          f"per-sample agreement {np.mean(ours == ref):.4f}")


def combine(dataset: str, encoder: str, config_path: str, setting: str) -> Path:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    d = Path(config["run"]["out_dir"]) / model_dir_name(config["model"]["name"]) / "locbench" / dataset
    prior = _load_prior(dataset, encoder)
    tag = encoder.replace("Sphere2Vec-", "s2v_").replace("Space2Vec-", "space2vec_").replace("+", "plus")
    written = None
    for src in sorted(d.glob(f"{setting}-*-seed*.jsonl")):
        if src.stem.count("-") != 2:  # skip derived files (calibrated, other priors)
            continue
        dst = src.with_name(f"{src.stem}-ts_{tag}.jsonl")
        n = 0
        with open(src) as fin, open(dst, "w") as fout:
            for line in fin:
                r = json.loads(line)
                p = combine_probs(np.asarray(r["probs"], dtype=np.float64), prior[r["item_id"]].astype(np.float64))
                r["probs"] = p.tolist()
                r["selected"] = int(np.argmax(p))
                r["raw"] = {"location_prior": f"torchspatial:{encoder}"}
                r["metadata"] = {**r["metadata"], "location_prior": encoder}
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
        print(f"{dataset}: {src.name} x {encoder} prior -> {dst.name} ({n} samples)")
        written = dst
    if written is None:
        raise FileNotFoundError(f"no {setting} results under {d}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prior", "check", "combine"])
    ap.add_argument("--dataset", required=True, choices=["fmow", "yfcc", "inat_2017", "inat_2018"])
    ap.add_argument("--encoder", required=True, choices=sorted({e for e, _ in CHECKPOINTS}))
    ap.add_argument("--config", help="model run config (combine)")
    ap.add_argument("--setting", default="image_only", help="results to combine (default: image_only)")
    args = ap.parse_args()
    if args.step == "prior":
        compute_prior(args.dataset, args.encoder)
    elif args.step == "check":
        check(args.dataset, args.encoder)
    else:
        if not args.config:
            raise SystemExit("combine needs --config")
        combine(args.dataset, args.encoder, args.config, args.setting)


if __name__ == "__main__":
    main()
