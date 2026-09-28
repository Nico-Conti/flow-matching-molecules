<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/unipi_emblem_dark.svg">
    <img src="assets/unipi_emblem.svg" alt="Università di Pisa" width="140">
  </picture>
</p>

<h1 align="center">Flow Matching for Molecular Generation</h1>

<p align="center">
  MSc thesis · Università di Pisa, Dipartimento di Informatica<br>
  <b>Author:</b> Nico Conti · <b>Supervisor:</b> Prof. Davide Bacciu · <b>Co-supervisor:</b> Luca Miglior
</p>

## Overview

The thesis addresses **molecular property targeting**: generating 2D molecular graphs
whose properties match a requested value while staying valid, unique and novel.

It develops **GuiDeFoG**, which adds property conditioning and
**classifier-free guidance (CFG)** to discrete flow matching on graphs (DeFoG).
A single network learns both the conditional and the unconditional clean-graph posterior,
because the property embedding is replaced by a learned null token with probability
`p_uncond` during training. At sampling time the two predictions are combined **in logit
space**, before the softmax:

```
logits = logits_uncond + s * (logits_cond - logits_uncond)
```

`s = 0` is unconditional, `s = 1` conditional, and `s > 1` over-guides (FreeGress convention).
Because the combination happens before normalisation, the guided posterior is a valid
categorical distribution at every guidance scale.

Experiments cover unconditional generation and single or joint property targeting on
**QM9** (dipole moment μ, HOMO energy), **ZINC-250k** and **MOSES** (logP, QED). They also
study how guidance strength, detailed-balance stochasticity (η) and step count affect
targeting error and coverage. Discrete-diffusion results (DiGress with FreeGress guidance)
are taken from the FreeGress paper and are not reimplemented here.

## Installation

Python ≥ 3.10. With [uv](https://docs.astral.sh/uv/) (lockfile included):

```bash
uv sync
```

or with pip:

```bash
python -m pip install -e .
```

Optional extras: `eval` (PySCF, needed to score DFT properties μ and HOMO),
`eval-opt` (adds geomeTRIC geometry optimisation), `eval-psi4` (Psi4 backend).

```bash
python -m pip install -e ".[eval]"
```

Pushing datasets or checkpoints to the Hugging Face Hub needs `HF_TOKEN` in a `.env`
file at the repository root. Loading the public cleaned datasets does not.

## Data

Each loader in `src/dataset/` returns canonical SMILES plus property targets. Graphs are
featurised on the fly as dense tensors: one-hot heavy atoms and four kekulised bond
classes (none, single, double, triple). Aromaticity is re-perceived by RDKit at decode time.

| Dataset   | Loader                      | Split                              | Targets |
|-----------|-----------------------------|------------------------------------|---------|
| QM9       | `dataset.qm9.load_qm9`      | random 75 / 15 / 10                | μ, α, HOMO, LUMO, gap, Cv (HOMO/LUMO/gap in eV) |
| ZINC-250k | `dataset.zinc.load_zinc`    | random 75 / 15 / 10                | logP, QED, SAS (RDKit) |
| MOSES     | `dataset.moses.load_moses`  | official train / test / test_scaffolds | logP, QED, SAS (RDKit) |
| GuacaMol  | `dataset.guacamol.load_guacamol` | official train / valid / test | logP, QED, SAS (RDKit) |

Cleaned datasets are cached on the Hub (`nico8771/qm9_clean`, `nico8771/zinc_neutral`,
`nico8771/moses_*`, ...). If the cache is unavailable, they are rebuilt from the original
source. `train.build_split` creates the splits used for training and evaluation.

## Usage

The modules in `src/` are importable after installation. Training uses the Python API,
for example a HOMO-conditioned DeFoG model on QM9:

```python
import train

train.train(
    dataset="qm9", method="defog",       # method: "defog" (discrete) | "fm_graph" (continuous)
    n_layers=5, epochs=500, batch_size=128,
    extra_features="rrwp",               # RRWP + cycle-count structural features
    cond_cols=("homo",), p_uncond=0.1,   # CFG conditioning + condition-dropout rate
    save_path="checkpoints/qm9_defog_cfg_homo.pt",
)
```

`devices=N` runs single-node DDP, `accum_steps` enables gradient accumulation and
`push_repo="user/name"` uploads checkpoints to the Hub. The best checkpoint is selected on
generative validity. Checkpoints store both EMA and live weights, so they can be used for
evaluation or to resume training.

Property targeting sweeps the guidance scale and reports MAE and coverage for each value of `s`:

```python
from checkpoint import load_checkpoint
from evaluate import evaluate_property_targeting

ck = load_checkpoint("checkpoints/qm9_defog_cfg_homo.pt", device="cuda")
results = evaluate_property_targeting(
    ck["model"], ck["size_sampler"], ck["atom_vocab"], ck["k_X"], ck["k_E"],
    targets=targets,                      # (n_targets, n_props) requested values
    cond_cols=ck["extra"]["cond_cols"],
    s_list=(0.0, 1.0, 3.0, 5.0),
    method=ck["method"], steps=500, distortion="polydec", eta=0.0,
    device="cuda",
)
```

logP and QED are scored with RDKit. μ and HOMO are recomputed with DFT
(B3LYP/6-31G\*, PySCF) on an RDKit/MMFF conformer; set `DFT_JOBS` to parallelise.

`evaluate.evaluate(...)` handles unconditional generation. It reports validity, uniqueness,
novelty and, when given a reference set, FCD. `scripts/` contains the stand-alone MOSES FCD
audits (see [`scripts/README.md`](scripts/README.md)).

## Repository layout

```
src/
  model.py            graph transformer with time, structural-feature and CFG conditioning
  methods/defog.py    discrete flow matching: loss, R* + detailed-balance sampler, logit-space CFG
  methods/fm_graph.py continuous (Gaussian) flow matching baseline, velocity-space CFG
  features.py         RRWP and cycle-count extra features
  train.py            training loop (EMA, DDP, grad accumulation, validity-based checkpointing)
  evaluate.py         VUN / FCD and property-targeting evaluation
  checkpoint.py       save / load / Hub push of checkpoints
  sizes.py            graph-size sampler from the training histogram
  dataset/            loaders, featurisation, filtering, metrics, RDKit / DFT property scoring
notebooks/            data round-trip and VUN sanity checks, unconditional FM / DeFoG demos
scripts/              MOSES FCD evaluation of published DeFoG samples and rescoring
report/               intermediate progress reports
```

Cluster launch scripts (`runs/`), tests and local working notes are not tracked in this
repository.

## Acknowledgements

This project builds on the following work:

- [DeFoG](https://github.com/manuelmlmadeira/DeFoG) (Qin et al., 2024)
- [DiGress](https://github.com/cvignac/DiGress) (Vignac et al., 2023)
- [CatFlow](https://arxiv.org/abs/2406.04843) (Eijkelboom et al., 2024)
- [FreeGress](https://github.com/Asduffo/FreeGress) (Ninniri et al., 2024)
- [torch-molecule](https://github.com/liugangcode/torch-molecule)
