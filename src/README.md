# src/: code

The code is grouped to mirror the two harnesses behind the study. Two entry-point
scripts sit at the top; the rest is organised into `training_harness/`,
`interpretation_harness/`, and `tests/`.

## Reproduction (run from the repo root, CPU, ~10 s)

    python src/reproduce_headline.py       # rebuilds the headline homolog-safe ROC-AUC
    python src/verify_split_lock.py        # recomputes the split-lock fingerprint from the gold data

Both read `data/gold/cns_cvs_gold_modeling_dataset.parquet` and need only the core
packages in the repo's top-level `requirements.txt`.

## Layout

    src/
      reproduce_headline.py     minimal, tested reproduction of the headline number
      verify_split_lock.py      recomputes the split-lock fingerprint from the gold data

      training_harness/         produces the final trained models + the generalization table
        cvs_train_harness.py       runs all families x all splits, refits on full in-distribution data
        cvs_sweep_job.py           hyperparameter-sweep driver
        cvs_run_logging.py         run-provenance logging + fail-loud GPU preflight

      interpretation_harness/   runs the nine-family glass-box comparison on the trained models
        run_interp_harness.py      the driver used on the GPU host
        kernel.py                  leakage-safe model-comparison engine (run_comparison(...))
        harness_kernel.py          thin alias of kernel.py (the driver loads the engine by this name)
        cvs_interp_adapter.py      builds the nine model factories + the split lock from the gold data
        glassbox_sweep_job.py      glass-box family training/eval
        feature_spec.json          frozen feature contract (block membership, PCA-k, splits, seed)

      tests/                    41 tests: python -m pytest src/tests

## How the harnesses connect

The two harnesses share one frozen feature contract (`feature_spec.json`) so the models
they build are directly comparable:

1. **`training_harness/`** trains every family under each split strategy and refits each on
   all in-distribution data, writing fitted estimators, out-of-fold predictions, and the
   per-split generalization table. Its outputs are in `model/trained_models/` and
   `results/training_harness/`.
2. **`interpretation_harness/`** loads those fitted models and runs the comparison layer:
   the Rashomon tie band, permutation control, remove-and-retrain ablation, and
   cross-family score agreement. Its outputs are in `results/interpretation_harness/`.

`run_interp_harness.py` is the exact driver used on the GPU host. It loads its code as
siblings in `interpretation_harness/` (`cvs_interp_adapter.py`, `harness_kernel.py`,
`feature_spec.json`) and expects its run inputs staged into the working directory it runs
from: `gold.parquet`, `manifest.csv` (both in `data/gold/`),
`metrics_by_split.csv` (in `results/training_harness/`), and `monotone_dirs.json` (in
`results/interpretation_harness/`). The full model families additionally need the optional
GPU/ML extras noted at the bottom of the repo's top-level `requirements.txt` (torch,
interpret, wittgenstein, and RAPIDS cuML/cuDF); on a CPU-only clone they are imported only
inside the factories that use them.

## Linting

    ruff check src            # config in repo-root ruff.toml
    pyright                   # config in repo-root pyrightconfig.json (basic mode)

pyright reports 0 errors; remaining items are advisory type-inference warnings on untyped
numpy/pandas/torch calls, expected for a research pipeline verified by its test suite.
