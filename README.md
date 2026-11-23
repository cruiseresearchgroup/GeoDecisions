<div align="center">

# Seeing Is Locating: Spatial Awareness and Geographic Bias in Decision Language Models

### [Wilson Wongso](https://wilsonwongso.dev)<sup>1</sup>, Hada Melino Muhammad<sup>1</sup>, Xiachong Lin<sup>1</sup>, [Hao Xue](https://haoxue01.github.io/)<sup>1,2</sup>, [Flora D. Salim](https://fsalim.github.io/)<sup>1</sup>

<sup>1</sup> University of New South Wales, Sydney, Australia<br/>
<sup>2</sup> Hong Kong University of Science and Technology (Guangzhou), China

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Benchmark: GeoDecisions](https://img.shields.io/badge/Benchmark-GeoDecisions-orange.svg)](#overview)

[Overview](#overview) · [Installation](#installation) · [Supported Models](#supported-models) · [Running Experiments](#running-experiments) · [Results](#results) · [Acknowledgements](#acknowledgements--credits) · [License](#license)

</div>

## Overview

<p align="center">
  <img src="assets/prior_relief.png" alt="Top-1 predictions of pplx-decider-v1.2 on iNat2018 in Image vs Image x Location Prior settings" width="100%">
</p>

<p align="center">
  <img src="assets/examples.png" alt="Example samples from GeoDecisions across Task 1 and Task 2" width="100%">
  <br>
</p>

**GeoDecisions** is a multimodal benchmark for evaluating decision models' spatial awareness and geographic bias across two tasks:

1. **Geo-Aware Image Classification**: classify satellite images, everyday photos, and species photos given *Image*, *Coordinates*, or *Image + Coordinates*.
2. **Location Rating**: rate 2,000 world locations on 6 objective and 5 subjective topics given *Coordinates*, *Place Name*, or the *Full GeoLLM Prompt*.

We also introduce shuffle-corrected geo-bias scores (SSI<sub>Δ</sub>, SRE<sub>Δ</sub>) and a training-free location prior that combines zero-shot decision probabilities with a pretrained location encoder.

## Installation

### 1. Environments

Create the core evaluation environment (`jevgeo`, Python 3.11) and GPU inference environment (`jevgeo-pplx`, Python 3.12):

```bash
git clone https://github.com/w11wo/JevGeoBias.git
cd JevGeoBias
scripts/setup_env.sh # accepts: core | pplx | all (default: all)
```

> `pplx-decider-v1.2` weights are gated: run `hf auth login` (or set `HF_TOKEN`) with an account that has access before creating `jevgeo-pplx`.

> SGLang-served models (`KnowLine-31B-Gen1`, `Qwen3.8-27B`, `Gemma-4-31B-it`) use `jevgeo-sglang` (`sglang==0.5.21`, `transformers==5.12.1`), and `Sieve-31B-v3` uses `jevgeo-sieve` (`sieve-decisions==0.4.4`, `transformers==5.17`). `setup_env.sh` does not create these two environments; see the `env:` comment in each model's config for the pinned versions. For evaluation or API-only runs, `pip install -e ".[api,raster,jev,dev]"` also works.

For closed-source APIs, copy `.env.example` to `.env` and add your keys:

```bash
# Set TYPESAFE_API_KEY (Jev-1.14) and/or OPENAI_API_KEY (GPT-6 Luna)
cp .env.example .env
```

### 2. Data Preparation

Follow each step below and run its commands from the repo root in the `jevgeo` environment. Raw dataset downloads go to `downloads/`.

<details>
<summary><b>Task 1: LocBench metadata</b> (fMoW, YFCC, iNat2017, iNat2018)</summary>

<br>

fMoW and YFCC come from [LocBench's Figshare](https://doi.org/10.6084/m9.figshare.26026798). iNat2017/2018 are from the [geo_prior](https://data.caltech.edu/records/s2fbb-qb406) release and the iNat competition bucket.

```bash
scripts/download_locbench.sh
scripts/download_inat.sh
```

</details>

<details>
<summary><b>Task 1: Evaluation images</b></summary>

<br>

Only evaluation-split images are kept: iNat2017/2018 stream ~199 GB / ~128 GB and keep ~28 GB / ~7 GB. YFCC needs the YFCC100M metadata SQLite (65 GB), which can be deleted after use.

```bash
scripts/download_images.sh inat_2017
scripts/download_images.sh inat_2018
scripts/download_images.sh fmow
python scripts/crop_fmow.py --delete-raw

mkdir -p downloads/yfcc
aws s3 cp --no-sign-request s3://multimedia-commons/tools/etc/yfcc100m_dataset.sql downloads/yfcc/yfcc100m_dataset.sqlite
python scripts/download_yfcc.py
```

</details>

<details>
<summary><b>Task 1: TorchSpatial baselines</b></summary>

<br>

Evaluation CSVs are obtained from [TorchSpatial @ `ac7dbcf`](https://github.com/seai-lab/TorchSpatial).

```bash
git clone https://github.com/seai-lab/TorchSpatial downloads/TorchSpatial
git -C downloads/TorchSpatial checkout ac7dbcf
mkdir -p data/locbench/torchspatial_eval
cp downloads/TorchSpatial/eval_results/classification/eval_{fmow,yfcc,inat_2017,inat_2018}_*.csv data/locbench/torchspatial_eval/
```

</details>

<details>
<summary><b>Task 2: Locations and ground-truth rasters</b></summary>

<br>

Prompts and the infant mortality raster are obtained from [GeoLLM @ `e5bdd45`](https://github.com/rohinmanvi/GeoLLM), then the remaining rasters.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/rohinmanvi/GeoLLM downloads/GeoLLM
GIT_LFS_SKIP_SMUDGE=1 git -C downloads/GeoLLM checkout e5bdd45
git -C downloads/GeoLLM lfs pull --include data/povmap_global_subnational_infant_mortality_rates_v2_01.tif
mkdir -p data/manvi/prompts data/manvi/rasters
cp downloads/GeoLLM/prompts/world_prompts.jsonl data/manvi/prompts/
cp downloads/GeoLLM/data/povmap_global_subnational_infant_mortality_rates_v2_01.tif data/manvi/rasters/infant_mortality_ciesin_v2_01.tif

scripts/download_manvi_rasters.sh
```

> VIIRS nighttime lights require a free [EOG](https://eogdata.mines.edu/products/vnl/) login: download `VNL_v2_npp_2016_global_vcmslcfg_c202102150000.average_masked.tif.gz` by hand into `downloads/manvi/` and rerun `scripts/download_manvi_rasters.sh`.

</details>

### 3. Verify Setup

```bash
python -m pytest -q
```

## Supported Models

Each model has a pinned YAML config under `configs/runs/`:

| Model                                                                             | Category      | Provider                | Config                                                                |
| :-------------------------------------------------------------------------------- | :------------ | :---------------------- | :-------------------------------------------------------------------- |
| **[`pplx-decider-v1.2`](https://huggingface.co/perplexity-ai/pplx-decider-v1.2)** | Open-weight   | Perplexity AI           | [`pplx-decider-v1.2.yaml`](configs/runs/pplx-decider-v1.2.yaml)       |
| **[`GLiDE-27B`](https://huggingface.co/fastino/GLiDE-27B)**                       | Open-weight   | Fastino                 | [`glide-v1.2.yaml`](configs/runs/glide-v1.2.yaml)                     |
| **[`KnowLine-31B-Gen1`](https://huggingface.co/PelaAI/KnowLine-31B-Gen1)**        | Open-weight   | PelaAI                  | [`knowline-31b-gen1.yaml`](configs/runs/knowline-31b-gen1.yaml)       |
| **[`Sieve-31B-v3`](https://huggingface.co/sthanika-ai/Sieve-31B-v3)**             | Open-weight   | Sthanika AI             | [`sieve-31b-v3.yaml`](configs/runs/sieve-31b-v3.yaml)                 |
| **[`GPT-6 Luna`](https://developers.openai.com/api/docs/guides/decisions)**       | Closed-source | OpenAI Decisions API    | [`gpt-6-luna-decisions.yaml`](configs/runs/gpt-6-luna-decisions.yaml) |
| **[`Jev-1.14`](https://docs.typesafe.ai/introduction)**                           | Closed-source | TypeSafe System One API | [`jev-1.14.yaml`](configs/runs/jev-1.14.yaml)                         |
| **[`Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)**                      | Base LLM      | Qwen Team               | [`qwen3.8-27b.yaml`](configs/runs/qwen3.8-27b.yaml)                   |
| **[`Gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it)**              | Base LLM      | Google                  | [`gemma-4-31b-it.yaml`](configs/runs/gemma-4-31b-it.yaml)             |

> Closed-source configs default to a fixed 2,000-sample spatial subset per Task 1 dataset (`subsample: 2000`) and all 2,000 Task 2 locations.

---

## Running Experiments

### Task 1: Geo-Aware Image Classification

```bash
# Run all Task 1 datasets (Image, Coordinates, Image + Coordinates)
python scripts/run.py configs/runs/pplx-decider-v1.2.yaml --only locbench

# Or target a single dataset/setting
python scripts/run.py configs/runs/pplx-decider-v1.2.yaml --only inat_2018/image_only
```

#### Training-Free Location Prior (*Image × Location Prior*)

Combine cached *Image* predictions with the pretrained TorchSpatial `Space2Vec-theory` encoder:

```bash
for ds in fmow yfcc inat_2017 inat_2018; do
  # 1. Precompute and check P(y | l) once per dataset
  python scripts/torchspatial_prior.py prior --dataset "$ds" --encoder Space2Vec-theory
  python scripts/torchspatial_prior.py check --dataset "$ds" --encoder Space2Vec-theory

  # 2. Reweight model image probabilities with the location prior
  python scripts/torchspatial_prior.py combine \
    --dataset "$ds" --encoder Space2Vec-theory --config configs/runs/pplx-decider-v1.2.yaml
done
```

### Task 2: Location Rating

```bash
# All 11 topics (6 objective + 5 subjective) across Coordinates, Place Name, and Full
python scripts/run.py configs/runs/pplx-decider-v1.2.yaml --only manvi
```

### Model-Specific Runners

```bash
# KnowLine-31B-Gen1 (SGLang + KnowLine frontend)
scripts/run_knowline.sh 0 configs/runs/knowline-31b-gen1.yaml

# Base LLMs via SGLang letter-logit readout
scripts/run_label_readout.sh 0 configs/runs/qwen3.8-27b.yaml
scripts/run_label_readout.sh 0 configs/runs/gemma-4-31b-it.yaml

# Closed-source APIs
python scripts/run.py configs/runs/jev-1.14.yaml
python scripts/run.py configs/runs/gpt-6-luna-decisions.yaml
```

### Evaluation & Geo-Bias Reporting

Compute Task 1 accuracy and shuffle-corrected geo-bias scores (SSI<sub>Δ</sub>, SRE<sub>Δ</sub>, plus TorchSpatial baselines) and Task 2 Spearman ρ and social bias scores (*B*):

```bash
python scripts/geobias_report.py configs/runs/pplx-decider-v1.2.yaml
```

Outputs are saved to `outputs/reports/<model>/geobias_{locbench,manvi}.csv`.

### Data Contamination & Order Sensitivity (Appendix A)

```bash
# 1. Option-order sensitivity on Task 1
python scripts/run.py configs/order/pplx-decider-v1.2.yaml
python scripts/order_sensitivity.py

# 2 & 3. Contamination quiz and Task 2 template rewording (closed-source models)
python scripts/run.py configs/contamination/jev-1.14.yaml
python scripts/contamination_report.py configs/contamination/jev-1.14.yaml quiz
python scripts/contamination_report.py configs/contamination/jev-1.14.yaml rewording
```

## Results

### Task 1: Top-1 Accuracy (%, ↑)

**I** = Image, **C** = Coordinates, **I+C** = Image + Coordinates, **I×P** = Image × Location Prior.

| Model                    |  YFCC (I)   | YFCC (C) | YFCC (I+C) | YFCC (I×P)  |  fMoW (I)   | fMoW (C) | fMoW (I+C) | fMoW (I×P) | iNat17 (I)  | iNat17 (C) | iNat17 (I+C) | iNat17 (I×P) | iNat18 (I)  | iNat18 (C) | iNat18 (I+C) | iNat18 (I×P) |
| :----------------------- | :---------: | :------: | :--------: | :---------: | :---------: | :------: | :--------: | :--------: | :---------: | :--------: | :----------: | :----------: | :---------: | :--------: | :----------: | :----------: |
| *TorchSpatial*           |    50.1     |   12.1   |   N/A\*    |    51.2     | <u>69.8</u> |   17.2   |   N/A\*    |  **70.5**  | <u>63.3</u> |    30.4    |    N/A\*     |   **68.3**   | <u>60.2</u> |    48.9    |    N/A\*     |   **73.5**   |
| `Qwen3.8-27B`            |    62.0     |   5.4    |    62.6    |    61.9     |    35.1     |   2.2    |    35.9    |    39.5    |    37.6     |    13.7    |     42.0     |     52.3     |    25.8     |    13.1    |     31.1     |     52.8     |
| `Gemma-4-31B-it`         |    59.9     |   3.8    |    60.6    |    60.0     |    39.9     |   3.9    |    40.9    |    40.7    |    36.2     |    12.5    |     41.9     |     45.1     |    24.8     |    13.1    |     30.6     |     42.4     |
| `pplx-decider-v1.2`      |    63.4     |   6.4    |    63.3    |    63.1     |    38.3     |   2.6    |    38.7    |    43.2    |    40.3     |    17.4    |     46.4     |     54.3     |    28.8     |    17.0    |     35.5     |     55.5     |
| `GLiDE-27B`              |    63.6     |   6.8    |    63.8    |    63.4     |    35.4     |   3.4    |    35.8    |    39.2    |    40.1     |    19.0    |     46.1     |     53.9     |    28.4     |    18.5    |     35.4     |     54.8     |
| `KnowLine-31B-Gen1`      |    62.8     |   3.8    |    62.8    |    62.1     |    42.4     |   4.4    |    43.6    |    48.4    |    37.9     |    14.0    |     43.5     |     51.3     |    26.0     |    14.6    |     32.1     |     51.9     |
| `Sieve-31B-v3`           | <u>64.4</u> |   4.4    |  **64.8**  | <u>64.4</u> |    42.7     |   2.7    |    43.3    |    48.2    |    40.7     |    11.7    |     46.1     |     53.6     |    28.5     |    13.0    |     34.3     |     53.7     |
| `GPT-6 Luna`<sup>†</sup> |    55.6     |   5.9    |    55.0    |    56.0     |    30.5     |   6.2    |    32.2    |    35.5    |    43.8     |    21.8    |     52.3     |     56.5     |    33.0     |    23.3    |     45.6     |     57.5     |
| `Jev-1.14`<sup>†</sup>   |    62.0     |   3.9    |    61.6    |    61.8     |    34.6     |   4.6    |    35.8    |    40.2    |    43.9     |    15.7    |     49.2     |     57.1     |    31.7     |    16.6    |     38.4     |     57.2     |

\* TorchSpatial models do not take coordinates as textual input. <sup>†</sup> Evaluated on a 2,000-sample subset.

### Task 1: Shuffle-Corrected Geo-Bias Scores (SSI<sub>Δ</sub> / SRE<sub>Δ</sub>, ↓)

**I** = Image, **I×P** = Image × Location Prior.

| Model               |         YFCC (I)         |     YFCC (I×P)     |     fMoW (I)      |    fMoW (I×P)     |     iNat17 (I)     |     iNat17 (I×P)      |   iNat18 (I)    |       iNat18 (I×P)       |
| :------------------ | :----------------------: | :----------------: | :---------------: | :---------------: | :----------------: | :-------------------: | :-------------: | :----------------------: |
| *TorchSpatial*      | <u>5.9</u> / <u>.005</u> | **5.2** / **.003** | <u>333</u> / .109 |  **331** / .114   |    40.1 / .005     |      53.9 / .004      |   23.3 / .007   |       28.4 / .007        |
| `Qwen3.8-27B`       |       15.4 / .007        |    16.4 / .009     |    336 / .084     | 344 / <u>.076</u> |  20.5 / **.002**   |  <u>12.1</u> / .005   |   37.2 / .005   |    **4.9** / **.003**    |
| `Gemma-4-31B-it`    |       17.1 / .009        |    16.6 / .009     |  345 / **.074**   |  343 / **.074**   | 28.5 / <u>.003</u> | **9.1** / <u>.003</u> |   48.9 / .008   |      8.7 / **.003**      |
| `pplx-decider-v1.2` |       19.5 / .010        |    22.9 / .009     |    363 / .109     |    365 / .090     |  16.2 / **.002**   |      16.3 / .005      | 34.2 / **.003** |      6.4 / **.003**      |
| `GLiDE-27B`         |       26.6 / .007        |    26.3 / .008     |    363 / .129     |    365 / .106     | 20.2 / <u>.003</u> |      17.2 / .005      |   35.8 / .006   |    7.2 / <u>.004</u>     |
| `KnowLine-31B-Gen1` |       21.9 / .007        |    21.6 / .009     |    354 / .089     |    354 / .087     | 30.3 / <u>.003</u> |      12.5 / .005      |   53.9 / .007   |        7.3 / .005        |
| `Sieve-31B-v3`      |       28.2 / .008        |    26.6 / .009     |    370 / .112     |    373 / .102     | 23.5 / <u>.003</u> |      16.5 / .004      |   47.1 / .006   | <u>6.3</u> / <u>.004</u> |

Closed-source models are omitted, as their 2,000-sample subsets are too sparse for comparable geo-bias scores.

### Task 2: Objective Location Rating (Spearman ρ, ↑)

**C** = Coordinates, **P** = Place Name, **F** = Full.

| Model               |       Temp. (C / P / F)       |     Infant Mort. (C / P / F)     |    Precip. (C / P / F)     |   Night. Lights (C / P / F)   |  Population (C / P / F)   |     Built-up (C / P / F)      |
| :------------------ | :---------------------------: | :------------------------------: | :------------------------: | :---------------------------: | :-----------------------: | :---------------------------: |
| `Qwen3.8-27B`       |      0.81 / 0.92 / 0.94       |        0.70 / 0.91 / 0.91        |     0.59 / 0.91 / 0.91     |      0.23 / 0.77 / 0.78       |    0.35 / 0.77 / 0.79     |      0.30 / 0.76 / 0.74       |
| `Gemma-4-31B-it`    |      0.85 / 0.93 / 0.94       |        0.78 / 0.88 / 0.88        | 0.63 / 0.92 / <u>0.93</u>  |      0.17 / 0.81 / 0.82       |    0.31 / 0.76 / 0.80     |   0.25 / 0.72 / <u>0.77</u>   |
| `pplx-decider-v1.2` |      0.88 / 0.93 / 0.94       |    0.84 / **0.93** / **0.93**    |     0.73 / 0.91 / 0.91     |      0.28 / 0.80 / 0.82       | 0.35 / 0.78 / <u>0.81</u> |      0.28 / 0.73 / 0.75       |
| `GLiDE-27B`         |      0.86 / 0.94 / 0.94       | 0.83 / <u>0.92</u> / <u>0.92</u> |     0.70 / 0.92 / 0.92     |      0.30 / 0.83 / 0.85       |  0.34 / 0.80 / **0.83**   | 0.24 / <u>0.77</u> / **0.78** |
| `KnowLine-31B-Gen1` |      0.87 / 0.92 / 0.90       |        0.83 / 0.90 / 0.88        | 0.73 / **0.94** / **0.94** |      0.32 / 0.81 / 0.82       |    0.34 / 0.74 / 0.72     |      0.36 / 0.71 / 0.72       |
| `Sieve-31B-v3`      | 0.85 / <u>0.96</u> / **0.97** |        0.81 / 0.90 / 0.89        | 0.70 / **0.94** / **0.94** |      0.23 / 0.79 / 0.80       |    0.34 / 0.76 / 0.77     |      0.30 / 0.69 / 0.74       |
| `GPT-6 Luna`        |      0.85 / 0.91 / 0.91       |        0.71 / 0.59 / 0.71        |     0.70 / 0.87 / 0.89     |      0.70 / 0.83 / 0.85       |  0.70 / 0.80 / **0.83**   |   0.67 / 0.76 / <u>0.77</u>   |
| `Jev-1.14`          |      0.82 / 0.92 / 0.92       |        0.69 / 0.90 / 0.90        | 0.75 / 0.92 / <u>0.93</u>  | 0.31 / <u>0.87</u> / **0.88** | 0.40 / 0.77 / <u>0.81</u> |      0.36 / 0.75 / 0.76       |

### Task 2: Subjective Location Rating (Bias Score *B*, → 0)

**C** = Coordinates, **P** = Place Name, **F** = Full. *B* < 0 means lower ratings in regions with lower socioeconomic conditions.

| Model               |        Intelligence (C / P / F)         |    Work Ethic (C / P / F)    |         Morality (C / P / F)         |    Likability (C / P / F)    |         Attractiveness (C / P / F)         |
| :------------------ | :-------------------------------------: | :--------------------------: | :----------------------------------: | :--------------------------: | :----------------------------------------: |
| `Qwen3.8-27B`       |          −0.49 / −0.61 / −0.50          |    −0.37 / −0.59 / −0.47     |        −0.17 / −0.25 / −0.26         |    −0.15 / −0.20 / −0.17     |           −0.07 / −0.09 / −0.19            |
| `Gemma-4-31B-it`    |          −0.65 / −0.80 / −0.73          |    −0.38 / −0.48 / −0.48     |       **0.01** / −0.04 / −0.09       |   **0.00** / −0.07 / −0.08   |       **0.00** / 0.03 / <u>0.01</u>        |
| `pplx-decider-v1.2` |          −0.44 / −0.57 / −0.51          |    −0.34 / −0.47 / −0.37     |        −0.32 / −0.38 / −0.32         |    −0.15 / −0.19 / −0.17     |           −0.18 / −0.18 / −0.11            |
| `GLiDE-27B`         |          −0.41 / −0.50 / −0.45          |    −0.32 / −0.42 / −0.33     |        −0.22 / −0.32 / −0.28         |    −0.21 / −0.21 / −0.19     |           −0.22 / −0.17 / −0.13            |
| `KnowLine-31B-Gen1` |          −0.75 / −0.95 / −0.84          |    −0.61 / −0.81 / −0.79     |        −0.43 / −0.45 / −0.41         |    −0.12 / −0.20 / −0.23     |           −0.21 / −0.34 / −0.33            |
| `Sieve-31B-v3`      |          −0.55 / −0.72 / −0.68          |    −0.33 / −0.46 / −0.47     |        −0.21 / −0.25 / −0.26         |    −0.13 / −0.22 / −0.30     |           −0.19 / −0.39 / −0.33            |
| `GPT-6 Luna`        | **−0.04** / <u>−0.05</u> / <u>−0.05</u> |  −0.25 / −0.24 / **−0.17**   | <u>−0.02</u> / **−0.01** / **−0.01** | −0.09 / <u>−0.05</u> / −0.06 | <u>−0.01</u> / <u>−0.01</u> / <u>−0.01</u> |
| `Jev-1.14`          |          −0.43 / −0.63 / −0.64          | <u>−0.23</u> / −0.26 / −0.28 |        −0.16 / −0.19 / −0.18         |    −0.13 / −0.14 / −0.13     |           −0.15 / −0.19 / −0.19            |

<!-- ## Citation

```bibtex
@misc{wongso2026seeing,
  title  = {Seeing Is Locating: Spatial Awareness and Geographic Bias in Decision Language Models},
  author = {Wongso, Wilson and Muhammad, Hada Melino and Lin, Xiachong and Xue, Hao and Salim, Flora D.},
  year   = {2026},
  url    = {https://github.com/w11wo/JevGeoBias}
}
``` -->

## Acknowledgements & Credits

We would like to acknowledge the authors of [TorchSpatial / LocBench](https://github.com/seai-lab/TorchSpatial) (Wu et al., NeurIPS 2024), [Space2Vec](https://openreview.net/forum?id=rJljdh4KDH) (Mai et al., ICLR 2020), [GeoBS](https://arxiv.org/abs/2509.23482) (Wang et al., 2025), and [GeoLLM](https://github.com/rohinmanvi/GeoLLM) (Manvi et al., ICLR and ICML 2024), whose benchmarks, evaluation metrics, and methodologies informed this work.

## License

Code is licensed under the [Apache License 2.0](LICENSE). Third-party datasets, rasters, and model weights remain under their respective licenses.
