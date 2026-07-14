# results/

Frozen outputs of the two model harnesses on the homolog-safe primary split
(`fold_seqclust`) and the other three splits. Every figure and headline number in the
manuscript is reproducible from these files plus the gold dataset, with no GPU run needed.

The folder mirrors `model/trained_models/`: one subfolder per harness.

```
results/
  training_harness/          final training run (src/cvs_train_harness.py)
    metrics_by_split.csv       9 families x 4 splits: gene/pair/within-disease ROC, PR, p@20, permuted null
    oof_predictions.parquet    out-of-fold scores, one column per family x split (keys: ensembl_gene_id, efo_id)
  interpretation_harness/    interpretation / comparison run (src/run_interp_harness.py)
    metrics_by_split.csv       the same 9x4 grid from the interpretation run
    oof_predictions.parquet    its out-of-fold scores
    glassbox_oof.parquet       out-of-fold scores for the eight glass-box families
    glassbox_models.json       native representations: scorecard points, tree splits, rule list, GLM coefficients
    monotone_dirs.json         monotonicity directions applied to the monotone GBM
    ebm_shapes.json            GA2M / EBM per-feature shape functions
    feature_ablation.csv       remove-and-retrain (ROAR) delta per feature block
    friedman_conover_stats.json  omnibus Friedman + pairwise Conover behind the tie band
    oof_score_agreement.json   per-gene Spearman rank-agreement matrix across families (Figure 8)
```

## The two runs

Both harnesses fit the same nine model families across the same four frozen splits, so their
metric grids have the identical shape (36 rows: 9 families x 4 splits). They are two separate
runs, not copies: the interpretation harness resamples across folds and seeds, so its numbers
sit within the seed-to-seed noise band rather than matching the training run exactly (largest
gene-ROC difference across all 36 cells is 0.021).

| Subfolder | Source | Role |
| --- | --- | --- |
| `training_harness/` | `src/cvs_train_harness.py` | the final training run; families refit on full in-distribution data per split. This is the manuscript's performance anchor (XGBoost homolog-safe gene-ROC 0.845). Its fitted models are `model/trained_models/models_training_harness.tar.gz`. |
| `interpretation_harness/` | `src/run_interp_harness.py` | the interpretation / comparison run behind the Rashomon tie band, remove-and-retrain ablation, cross-family score agreement, and the interpretable native representations. Its fitted models are `model/trained_models/models_interpretation_harness.tar.gz`. |

**Quote one run or the other for a given number, never one value from each as a single
measurement.** Manuscript headline metrics come from `training_harness/`; the tie-band,
ablation, and agreement analyses come from `interpretation_harness/`.

## Joining predictions to labels

Both `oof_predictions.parquet` files carry stable per-row keys (`ensembl_gene_id`, `efo_id`)
that match the gold dataset and the split lock in `data/splits/per_split_id_lists.json`, so
predictions, folds, and labels join unambiguously. Each family x split has its own
`oof_<split>_<family>` score column.
