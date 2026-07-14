<h1 align="center">Predicting Clinical Validation of CNS Drug Targets</h1>

<p align="center">
  <em>Predicting which nervous-system drug targets will survive the clinic, from the biology<br>
  of the target alone, with a model whose reasoning you can actually read.</em>
</p>

<p align="center">
  <a href="manuscript/predicting-cns-target-validation.pdf"><img alt="Read the manuscript" src="https://img.shields.io/badge/%F0%9F%93%84_manuscript-PDF-b31b1b"></a>
  <a href="https://youtu.be/2Urf21q4VdY"><img alt="Watch the overview" src="https://img.shields.io/badge/%E2%96%B6_overview-video-red"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green"></a>
  <img alt="Claude for Life Sciences Hackathon" src="https://img.shields.io/badge/Claude_for_Life_Sciences-Hackathon-8A63D2">
</p>

<p align="center">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="NVIDIA CUDA" src="https://img.shields.io/badge/CUDA-13-76B900?logo=nvidia&logoColor=white">
  <img alt="NVIDIA DGX Spark (GB10)" src="https://img.shields.io/badge/DGX_Spark-GB10-76B900?logo=nvidia&logoColor=white">
  <img alt="RAPIDS cuDF / cuML" src="https://img.shields.io/badge/RAPIDS-cuDF_%7C_cuML-7400FF?logo=nvidia&logoColor=white">
  <img alt="tested with pytest" src="https://img.shields.io/badge/tests-41_passing-0A9EDC?logo=pytest&logoColor=white">
  <img alt="linted with Ruff" src="https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black">
  <img alt="type-checked with Pyright" src="https://img.shields.io/badge/types-pyright-FFD43B">
</p>

<p align="center">
  <b>A submission to the Claude for Life Sciences Hackathon</b>, built in about five days<br>
  over a weekend alongside a full-time job. More on how it was built <a href="#about-this-project">below</a>.
</p>

---

Most drugs fail, and in the brain and nervous system they fail more than anywhere else in
medicine. The costliest mistake is made years before a trial begins: choosing the wrong
**target**, the specific protein a drug is designed to act on. This project asks a direct
question. Can you tell, from the biology of a target protein alone, whether pairing it with
a given disease will succeed or fail in the clinic? And can you do it with a model a person
can read, rather than a black box?

It answers yes to both, on a pan-CNS dataset of **6,843 target-indication pairs**, evaluated
under a leakage-controlled protocol that holds homologous proteins and clinical outcomes
apart, the shortcut that inflates most published scores.

> ### 📄 Read the manuscript: **[predicting-cns-target-validation.pdf](manuscript/predicting-cns-target-validation.pdf)**
>
> The compiled PDF is the primary write-up. Markdown source: `manuscript/manuscript.md`.
> LaTeX source: `manuscript/latex/`.

### At a glance

| | |
|---|---|
| **Task** | Given a *(target gene, indication)* pair, predict **clinically validated** (a drug on the target was approved) vs **failed** (efficacy failure or abandonment) |
| **Data** | 6,843 pairs · 1,786 proteins · 258 indications · 2,724 leakage-free features |
| **Scope** | All CNS disease (ATC class N; EFO/MONDO nervous-system + mental-disorder branches): neurodegeneration, neuroimmune/MS, psychiatry, epilepsy, pain |
| **Headline** | Ranks a true success above a failure **~85%** of the time (target-level ROC-AUC on the honest homolog-safe split); **0.68** within a single disease, the committee's actual task |
| **Twist** | A **fully transparent** model matches the opaque black box at **no measurable accuracy cost** |
| **Reproduce** | `python src/reproduce_headline.py` for the headline number in ~10 s on CPU |

## Video overview

A short walkthrough of the problem, the approach, and the result:

[![Watch the video overview](https://img.youtube.com/vi/2Urf21q4VdY/maxresdefault.jpg)](https://youtu.be/2Urf21q4VdY)

---

## About this project

**In one line:** most drugs fail, and the costliest mistake is made years before a trial
begins, choosing the wrong target (the specific protein a drug acts on) for a disease; this
project asks whether the clinical success of a target-disease pairing can be predicted from
the basic biology of the target protein alone, and answers it with a model whose reasoning a
person can actually read.

This repository is a submission to the **Claude for Life Sciences Hackathon**. The entire
study, from assembling a pan-CNS dataset of 6,843 target-disease pairings to training and
benchmarking nine model families, running the interpretability analysis, and writing the
manuscript, was built in about five days over a weekend alongside a full-time job.
That timeline is part of the point: the data assembly that would normally take a small team
months was carried out by an agentic workflow (Claude Science) that dispatched sub-agents in
parallel to mine approvals, trial outcomes, and programme discontinuations across many
sources, while keeping a full per-pair audit trail.

The problem is worst in the brain and nervous system, where success rates are the lowest in
medicine: two decades of Alzheimer's amyloid programmes failed one after another, and a
Huntington's programme to lower the very protein that causes the disease was halted in
late-stage trials despite the target being genetically certain. The model does not give a
yes-or-no verdict; it ranks candidates from more to less promising, ranking a true success
above a failure about 85% of the time on its honest primary split (against 50% for a coin
toss). The result that matters most is the comparison to existing tools: they report scores
above 90%, but on an easier test that lets a model lean on stand-ins for its answers; tested
the honest way (train on the past, predict which targets entered trials next), the
association scores those tools rely on barely beat a coin toss (~56%), while this model
avoids the shortcut and still reaches ~70% on that same fair footing, and delivers what the
high-scoring black boxes cannot: a prediction whose reasoning a person can read.

The scientific deliverable is the model and the leakage-controlled evaluation protocol; the
agent workflow is the method that made the scale feasible in the time available. A fuller
general-audience write-up is in the accompanying project description.

---

## Running on the DGX Spark (the CUDA-X playbook)

The embeddings, the model training, and the hyperparameter sweeps all ran on an
**NVIDIA DGX Spark** (GB10, aarch64), following the RAPIDS **CUDA-X data-science playbook**:
accelerate the whole pipeline on the GPU with as little code change as possible, and never
ship a result that silently fell back to the CPU.

Three layers of acceleration, each grounded in `src/`:

1. **Zero-code-change tabular acceleration.** `cuml.accel` is installed at the top of the
   training harness, transparently routing scikit-learn PCA, `StandardScaler`, and
   `LogisticRegression` to cuML on the GPU without rewriting the modeling code
   (`src/training_harness/cvs_train_harness.py`). The in-fold PCA-32 reduction of the 2,560-dim ESM-Cambrian
   embedding, applied inside every cross-validation fold, is the step this matters most for.
2. **GPU-native gradient boosting.** The XGBoost families train with `device=cuda` on a
   GPU-resident `QuantileDMatrix`, and the k-nearest-neighbour family uses cuML's
   `KNeighborsClassifier` (`src/interpretation_harness/cvs_interp_adapter.py`, `src/training_harness/cvs_sweep_job.py`). The small
   additive, linear, and rule-list families stay on CPU (no aarch64 GPU build) but are tiny.
3. **GPU feature search.** The groupby-aggregate feature exploration ran as a cuDF
   groupby-and-score search on the GPU (generate many candidate aggregates, keep what
   improves cross-validated score).

**Fail-loud, never silent-CPU.** A hazard on this hardware was a broken cuML import silently
falling back to CPU while the job still exited 0 (this happened once, a run finished exit 0
entirely on CPU). `src/training_harness/cvs_run_logging.py` provides a `gpu_preflight` guard that actually
executes a cuML KNN fit and an XGBoost `device=cuda` train before the real work, and aborts
with a non-zero exit if either did not run on the GPU. Every run also writes a `run_meta.json`
recording the image, GPU, driver, CUDA, and package versions, so any result traces back to the
exact environment that produced it.

The reproducible container image is
`rapidsai/base:26.08a-cuda13-py3.12`; the CPU-only `src/reproduce_headline.py` reruns the
headline number without any of this, landing within the CUDA-vs-CPU histogram margin.

---

## The two results

**1. Interpretability is free.** Nine model families were compared, from transparent by
construction to a fully opaque reference, on one frozen homolog-safe split. Six of the nine
are statistically tied in gene-level ROC-AUC (every pairwise difference smaller than the
seed-to-seed noise; largest gap among the six is 0.0125), and three of those six are
interpretable by construction. A transparent model can therefore be chosen here at no
measurable accuracy cost.

**2. The honest performance depends on the question asked.** On the primary homolog-safe
split (whole sequence-similarity clusters held out together), the reference model reaches:

| Question the score answers | Metric | Value |
|---|---|---|
| Rank any target against any other (cohort-wide) | target-level ROC-AUC | **0.85** |
| Rank individual target-indication pairs | pair-level ROC-AUC | 0.80 |
| **Rank candidate targets within one disease** (the committee task) | within-disease ROC-AUC | 0.68 |
| Label-permutation control | target-level ROC-AUC | 0.45 |

The three-number triple is the point: **0.85** is the headline for cohort-wide ranking, but
**0.68** is the honest expectation for the question a target-selection committee actually
faces. Both are reported rather than the optimistic in-sample 0.95 (random split).

Reproduce the headline number in ~10 seconds on CPU:

```bash
pip install -r requirements.txt
python src/reproduce_headline.py
```

Expected: `Homolog-safe (fold_seqclust) gene-level ROC-AUC : 0.85xx (reference 0.856)`.
(The GPU-trained artifact scores 0.856; a CPU rerun lands at ~0.854, the CUDA-vs-CPU
histogram difference, not a change in method.)

---

## The nine model families

Ordered by homolog-safe gene-level ROC-AUC. The tie band (six families statistically
indistinguishable within a small margin) is marked.

| Family | Transparency | ROC-AUC | In tie band |
|---|---|---|:--:|
| Nearest-neighbour | partial | 0.858 | yes |
| Neural additive model (NAM) | by construction | 0.856 | yes |
| Monotone GBM | partial | ~0.856 | yes |
| Elastic-net logistic | by construction | 0.851 | yes |
| Additive + interactions (GA2M/EBM) | by construction | 0.848 | yes |
| **XGBoost (opaque reference, measurement anchor)** | opaque | 0.845 | yes |
| Integer scorecard | by construction | 0.813 | no |
| Decision tree (CART) | by construction | 0.791 | no |
| Rule list (RIPPER) | by construction | 0.634 | no |

The **neural additive model** is recommended for deployment: it is interpretable by
construction, sits at the top of the tie band, and carries no accuracy penalty relative to
the black box. The opaque XGBoost reference is used as the measurement anchor throughout,
because showing that even the black box agrees with the readable account is stronger
evidence than showing it for the recommended model alone.

---

## What's here

```
predicting-cns-target-validation/
├── README.md                     ← you are here
├── requirements.txt, LICENSE, .gitignore
│
├── data/
│   ├── gold/                     ← THE modeling dataset (train on this)
│   │   ├── cns_cvs_gold_modeling_dataset.parquet     6,843 pairs × 2,735 cols
│   │   ├── cns_cvs_gold_modeling_dataset_manifest.csv  column → block + test-time source
│   │   └── DATA_CARD.md
│   ├── raw_lineage/              ← upstream layer, for provenance / re-derivation
│   │   ├── cns_model_matrix_full.parquet
│   │   ├── cns_labels_master.parquet
│   │   └── homology_edges.parquet
│   └── splits/
│       └── per_split_id_lists.json   frozen fold assignments + split-lock fingerprint
│
├── provenance/                   ← full data provenance & audit trail
│   ├── PROVENANCE.md              single provenance doc: features, labels, sources, versions
│   └── tables/                    machine-readable companions to PROVENANCE.md
│       ├── feature_provenance_contract.csv    leakage tier for every feature
│       ├── feature_computation_provenance.csv per-column source/tool/computation
│       └── (feature/label dictionaries, literature comparison, outcome layer)
│
├── results/                      ← released artifacts behind every figure and number
│   ├── training_harness/         final training run outputs
│   │   ├── metrics_by_split.csv        all metrics × all splits (manuscript anchor)
│   │   └── oof_predictions.parquet     out-of-fold scores
│   └── interpretation_harness/   comparison / interpretability run outputs
│       ├── metrics_by_split.csv        all metrics × all splits
│       ├── oof_predictions.parquet     out-of-fold scores
│       ├── glassbox_oof.parquet        out-of-fold scores (glass-box families)
│       ├── feature_ablation.csv        remove-and-retrain (ROAR) deltas
│       ├── friedman_conover_stats.json omnibus + pairwise tie statistics
│       ├── oof_score_agreement.json    cross-family rank-agreement matrix
│       ├── glassbox_models.json        native model representations (scorecard points, etc.)
│       ├── monotone_dirs.json          monotonicity directions
│       └── ebm_shapes.json             GA2M/EBM shape functions
│
├── models/
│   ├── cvs_gold_model.ubj         trained XGBoost reference artifact (deployable)
│   ├── cvs_gold_model_results.csv all metrics, all protocols
│   ├── model_card.md             reference model + nine-family benchmark
│   └── trained_models/           fitted estimators, all families x all splits
│       ├── models_training_harness.tar.gz        final training run
│       └── models_interpretation_harness.tar.gz  ablation / comparison run
│
├── src/                          ← code, grouped to mirror the two harnesses
│   ├── reproduce_headline.py     minimal, tested reproduction of the headline number
│   ├── verify_split_lock.py      recomputes the split-lock fingerprint from the gold data
│   ├── training_harness/         ← produces the final trained models + generalization table
│   │   ├── cvs_train_harness.py      runs all families x all splits, refits on full data
│   │   ├── cvs_sweep_job.py          hyperparameter sweep driver
│   │   └── cvs_run_logging.py        run-provenance + fail-loud GPU preflight
│   ├── interpretation_harness/   ← runs the glass-box comparison on the trained models
│   │   ├── run_interp_harness.py     driver for the nine-family comparison
│   │   ├── kernel.py                 leakage-safe model-comparison engine
│   │   ├── harness_kernel.py         thin alias of kernel.py (the driver loads it by this name)
│   │   ├── cvs_interp_adapter.py     builds the nine model factories + the split lock
│   │   ├── glassbox_sweep_job.py     glass-box family training/eval
│   │   └── feature_spec.json         frozen feature contract (blocks, PCA-k, splits, seed)
│   ├── tests/                    ← 41 tests: `python -m pytest src/tests`
│   │   ├── test_cvs_train_harness.py
│   │   ├── test_adapter.py
│   │   └── test_harness.py
│   └── README.md                 file roles, how to reproduce, lint posture
│
├── requirements.txt              deps (core reproduction; optional harness/GPU extras noted inline)
├── ruff.toml, pyrightconfig.json lint / type-check config
│
├── notebooks/                    EDA.ipynb (runnable, outputs saved)
├── manuscript/
│   ├── predicting-cns-target-validation.pdf   the manuscript (PDF)
│   ├── manuscript.md                          Markdown source (Methods included)
│   ├── figures/                               the 9 figures
│   └── latex/                                 main.tex + refs.bib + figures/ (compiles the PDF)
└── figures/                      supporting analysis figures
```

---

## The gold modeling dataset

`data/gold/cns_cvs_gold_modeling_dataset.parquet` (6,843 pairs × 2,735 columns), spanning
1,786 human proteins and 258 CNS indications:

- **3 identifiers:** `ensembl_gene_id`, `efo_id`, `gene_symbol`
- **5 label columns:** `label_consensus` (**canonical**, 1,570 validated / 5,273 failed),
  `label_consensus_v2`, `label_efficacy_gold` (adjudicated 565/96 high-confidence subset),
  `label_tier`, `label_confidence`
- **3 frozen fold columns:** `fold_seqclust` (**use this**, homolog-safe primary split),
  `fold_gene` (target-disjoint), `fold_disease` (indication-disjoint)
- **2,724 features** in 5 blocks, **all available before any clinical outcome** (zero
  leakage-tier features): protein-language-model embedding (2,560; ESM-Cambrian 6B,
  reduced to 32 PCs in-fold), sequence & gene properties (84), predicted GO (61), predicted
  structure (15), homology (4).

```python
import pandas as pd
df  = pd.read_parquet("data/gold/cns_cvs_gold_modeling_dataset.parquet")
man = pd.read_csv("data/gold/cns_cvs_gold_modeling_dataset_manifest.csv")
feature_cols = man[man.block.isin(
    ["esmc6b","intrinsic_84","predicted_go","structure","homology"])].column.tolist()
X, y = df[feature_cols], df["label_consensus"]

for k in range(5):                       # homolog-safe primary split
    train, test = df.fold_seqclust != k, df.fold_seqclust == k
    # fit on train (reduce the embedding block to PCA-32 IN-FOLD), score on test
```

See `data/gold/DATA_CARD.md` for the full data card.

---

## Frozen splits and the split lock

The fold assignments for all splits are frozen and keyed by stable biological identifier
(Ensembl gene, EFO indication), so the training harness and the evaluator provably share
the same folds. The homolog-safe primary split is bound to a fingerprint that both pipelines
verify before running:

```
split-lock fingerprint: sha256:ffa3855aef4eb3901b0e6fdc8e2e9d5cf1088921b039a3608c8da57c54432731
```

`data/splits/per_split_id_lists.json` carries the frozen per-fold identifier lists and this
fingerprint; `src/verify_split_lock.py` checks it.

---

## Data provenance & integrity

Every feature is traceable to a source database and computation
(`provenance/tables/feature_computation_provenance.csv`), and every feature's leakage tier is
recorded in `provenance/tables/feature_provenance_contract.csv`. The gold dataset was audited to
contain **zero** of the banned-tier features (labels, post-hoc outcome-derived,
temporal-confounded, realized-investment/survivorship) and zero popularity/count features.
This anti-leakage principle is central to the study: the gap between the leakage-controlled
numbers here and the higher figures reported elsewhere is a measure of how much apparent
accuracy in this literature depends on evaluation that does not hold homologues and outcomes
apart.

Every label is traceable to its winning evidence channel, tier, and confidence weight across
the mining channels (`provenance/PROVENANCE.md`).

**Integrity check** of the gold dataset:

```
cns_cvs_gold_modeling_dataset.parquet
  sha256: 1c3e15c5aa19524f8b8318e177f7106ba3bfcb86f9307e84e1b0ab907c29dc11
  shape:  6843 × 2735
  labels: label_consensus 1570 validated / 5273 failed; label_efficacy_gold 565 / 96
```

---

## Applicability domain

The model tolerates a **new target** within a known protein class and a **known target for a
new indication**, the two cases that cover most real CNS target decisions. It has no basis
for a genuinely **unprecedented protein class** (leave-one-protein-class-out ROC-AUC falls to
0.46, three of five classes below chance) and should abstain there. Scores are ranking
values, not calibrated probabilities: the recommended additive model is systematically
over-confident (a mid-range score should not be read as a literal probability of validation).

---

## Citation

If you use this dataset or model, please cite the accompanying manuscript
(`manuscript/manuscript.md`). Data sources and comparable published work are enumerated with
DOIs in `provenance/PROVENANCE.md`.
