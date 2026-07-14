# Model Card: CNS drug-target clinical-validation models

## What it does
Scores a **(target, indication) pair** for clinical viability in CNS disease: will a drug
acting on this protein be **validated** (approved / positive efficacy) or **fail** (efficacy
failure / abandonment)? It scores from the biology of the target protein alone, using nothing
about whether the target has already succeeded or failed. Intended as a **triage tool**: rank
candidates so spend avoids programmes unlikely to validate.

This card covers the **deployable reference model** (`cvs_gold_model.ubj`, XGBoost) and the
**nine-family benchmark** behind the paper's central result: a fully transparent model matches
the opaque reference at no measurable accuracy cost.

## Data
| | |
|---|---|
| Rows | 6,843 (target, indication) pairs, 1,786 proteins, 258 indications |
| Label | `label_consensus`: 1,570 validated (22.9%) / 5,273 failed |
| Features | 2,724, all known before clinical outcome (no post-hoc / survivorship / popularity; zero leakage-tier columns) |
| Feature blocks | ESM-Cambrian 6B embedding (2,560, PCA-32 in-fold), genome-intrinsic (84), predicted-GO (61), AlphaFold structure (15), homology (4) |
| Frozen folds | `fold_seqclust` (homolog-safe, primary), `fold_gene` (target-disjoint), `fold_disease` (indication-disjoint) |

## Reference model
- **Artifact:** `cvs_gold_model.ubj`, XGBoost `binary:logistic`, refit on all rows.
- **Trained on:** NVIDIA GB10 (`device=cuda`, GPU-resident `QuantileDMatrix`).
- **Hyperparameters:** max_depth 6, learning_rate 0.03, num_boost_round 400 (fixed), subsample 0.8, colsample_bytree 0.7, min_child_weight 5, reg_lambda 2.0.
- **Scores are rankings, not calibrated probabilities.** Use for ordering, not as P(success).

## Performance
Four frozen splits, each a deployment scenario. Values are the XGBoost reference; the tied
families track within the Rashomon band.

| Split | What's held out | target ROC-AUC | within-disease ROC-AUC |
|---|---|:--:|:--:|
| Random | nothing grouped (optimistic upper bound) | 0.95 | - |
| Target-disjoint (`fold_gene`) | whole genes | 0.86 | 0.68 |
| **Homolog-safe (`fold_seqclust`)** | whole sequence-similarity clusters | **0.85** | **0.68** |
| Indication-disjoint (`fold_disease`) | whole indications (hardest) | 0.68 | - |

**Homolog-safe is the number to trust.** Random and even target-disjoint are optimistic
because a held-out gene can still resemble a training gene; holding out whole homology
clusters removes that bridge. Report the triple, not a single figure:

- Target-level ROC-AUC **0.85**, pair-level **0.80**, within-disease **0.68**.
- Within-disease **0.68** is the honest expectation for the committee's real question (rank
  candidates *for one disease*), so it is stated alongside the flattering 0.85.
- Permutation control **0.45**: the signal is real, not leakage.

**Literature context.** Comparable tools report higher headline numbers under more lenient
evaluation (Han 2022 0.914 ungrouped; OTRec 2025 0.950 target-disjoint with disease retained).
On a genuinely prospective footing an association-score baseline reaches ~0.56 (OTRec temporal,
a not-yet-peer-reviewed preprint, indicative). A deliberately feature-restricted lower-bound
variant of this model clears that floor at ~0.70; the full model reaches 0.85 on its primary
honest split. Details in `provenance/tables/literature_comparison.csv`.

## The nine model families
Homolog-safe split, ordered by gene-level ROC-AUC. Six are statistically tied (largest gap
0.0125, within seed-to-seed noise); three of the six are interpretable by construction.

| Family | Transparency | gene ROC-AUC | Tie band |
|---|---|:--:|:--:|
| Nearest-neighbour | partial | 0.858 | yes |
| Neural additive model (NAM) | by construction | 0.856 | yes |
| Monotone GBM | partial | ~0.856 | yes |
| Elastic-net logistic | by construction | 0.851 | yes |
| Additive + interactions (GA2M / EBM) | by construction | 0.848 | yes |
| **XGBoost (opaque reference)** | opaque | 0.845 | yes |
| Integer scorecard | by construction | 0.813 | no |
| Decision tree (CART) | by construction | 0.791 | no |
| Rule list (RIPPER) | by construction | 0.634 | no |

**Deploy the neural additive model:** interpretable by construction, top of the tie band, no
accuracy penalty. The opaque XGBoost is kept only as the measurement anchor, showing the black
box agrees with the readable account is stronger evidence than reporting the readable model alone.

## Intended use and limitations
- **Use for:** triage / prioritisation of CNS target-indication pairs at the target-choice stage.
- **Do not use for:** non-CNS indications (trained on ATC-N / nervous-system EFO only); single-programme go/no-go without human review; as the sole basis for an investment decision.
- **Applicability domain:** handles a new target in a *known* protein class, and a known target for a new indication. It has no basis for an unprecedented protein class (leave-one-protein-class-out ROC-AUC 0.46, three of five classes below chance) and should abstain there.
- **Label noise:** only the gold-661 subset is efficacy-adjudicated; the broader label mixes phase-advancement and investment-signal channels (down-weighted by tier / confidence).
- **Ascertainment:** association evidence is present mostly for already-studied targets and is down-weighted by design; causal-evidence coverage is uneven across CNS areas.

## Reproducibility
- **Files:** fitted estimators for every family x split in `models/trained_models/` (two tarballs, training-harness run + interpretation-harness run, see its README). Interpretable native representations in `results/interpretation_harness/glassbox_models.json` and `ebm_shapes.json`; out-of-fold predictions in `results/`.
- **Fixed rounds, not early stopping.** In this grouped, small-positive regime early stopping underfits (measured 0.788 vs 0.909 on the same target-disjoint split); fixed `num_boost_round=400` is the documented choice.
- **In-fold PCA-32** of the embedding (fit on train only) prevents embedding leakage.
- **Split lock:** every pipeline verifies the `fold_seqclust` fingerprint (`sha256:ffa3855a...`) before running; recompute with `src/verify_split_lock.py`.
- **Metrics:** ROC-AUC and PR-AUC co-primary (PR-AUC for the imbalance), precision@k as the decision-facing metric. F1 is not used, triage cares about precision at the top of the ranking, not a symmetric threshold.
