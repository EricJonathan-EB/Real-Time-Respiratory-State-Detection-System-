#  Multimodal Real-Time Respiratory State Detection System 

Detecting breath-hold (**BH**) and deep-breathing (**DB**) episodes against normal breathing (**NB**) from wearable ECG using HRV/EDR-derived features — moving from a hand-engineered feature pipeline through classical ML up to graph neural networks, all evaluated with subject-level (LOSO) cross-validation.

Signal source: Zephyr BioHarness ECG (250 Hz), annotated against a breath-protocol reference (`DBInfo.xlsx`).

---

## Pipeline Overview

```
1. Feature Extraction   →  2. Baseline Model   →  3. Feature Selection
                                                          │
                                                          ▼
                              5. Traditional Models  ←  4. Optimized Feature Test
                                                          │
                                                          ▼
                                                6. Graph Neural Networks
```

Each stage consumes the CSV produced by the previous one. Window size (WS15 / WS20 / WS30 seconds) is a variable explored throughout — most later experiments converge on **WS20**.

---

## 1. Feature Extraction — `1_feature_extraction`

Turns raw ECG + annotation files into a labelled, windowed feature table.

- **Windowing:** sliding window over the ECG stream (`WINDOW_SEC` ∈ {15, 20, 30}, `STEP_SEC = 1`, `FS = 250 Hz`)
- **Labelling:** overlap-based against BH/DB annotations
  - ≥80% overlap → labelled BH or DB
  - ≤20% overlap with either → labelled NB
  - 20–80% (transition/buffer zone) → dropped
- **Raw features (12):** `Mean_RR, SDNN, RMSSD, pNN50, CV_RR, LF, HF, LF_HF_Ratio, EDR_Mean_Amp, EDR_Std_Amp, EDR_Peak_to_Peak, Shannon_Entropy`
- **Derived features:** each raw feature expanded into `_Delta` (change from personal baseline) and `_Ratio` (normalized to personal baseline) variants, giving the full engineered feature set
- **Output:** `BH_DB_Features_WS{15,20,30}.csv` — one row per window, per subject, with `Subject`, `Category`, and all raw + derived features

## 2. Baseline Model Test — `2_baseline_model_test`

First pass at classification on the **full** feature set (20 active features after dropping dead LF variants).

- Random Forest, class-weighted for imbalance (`{NB: 1, BH: 20, DB: 20}`)
- **LOSO (Leave-One-Subject-Out)** cross-validation — every subject held out once as the test set
- Metrics: accuracy, precision/recall/F1 (per class), ROC-AUC, PR-AUC, confusion matrix
- Establishes the performance floor before any feature pruning

## 3. Feature Selection — `3_feature_selection`

Statistically prunes the feature set per window size.

- **Paired Wilcoxon signed-rank test** (α = 0.05), each intervention (BH, DB) vs. the NB baseline, aggregated per subject
- Drops features with no significant separation from NB, plus explicitly dead features (`LF_*`, `LF_HF_Ratio_*`)
- Cross-checks remaining features for multicollinearity (VIF)
- **Output:** an optimized ~8-feature list (`G_LIST`) per window size, replacing the 20-feature baseline set:
  - Primary BH detectors: `pNN50_Ratio`, `pNN50_Delta`, `HF_Ratio`
  - Secondary HRV/ANS: `RMSSD_Ratio`, `Mean_RR_Ratio`, `Shannon_Entropy_Ratio`
  - DB isolators: `EDR_Std_Amp_Ratio`, `EDR_Mean_Amp_Delta`

## 4. Optimized Feature Test — `4_optimized_features_test`

Re-runs the same RF + LOSO evaluation as stage 2, but on the reduced 8-feature `G_LIST` from stage 3 — validating that the pruned set matches or beats the 20-feature baseline with a much smaller, less collinear feature space.

## 5. Traditional Models — `5_traditional_models`

Deeper, model-specific tuning on the optimized feature set, run across four classical classifiers: **Random Forest, SVM, XGBoost, LightGBM**.

Progressive refinements layered on top of plain LOSO:
- **Personalized Bayesian / NB calibration** — per-subject probability recalibration against their own normal-breathing baseline
- **Youden-J thresholding** — optimal decision threshold from OOB/CV probabilities rather than a flat 0.5 cutoff
- **Schmitt hysteresis** — prevents label flicker between adjacent windows
- **Winsorising** — per-fold outlier clipping (P1–P99) fit only on training data, avoiding leakage
- **Minimum-duration enforcement** — clinically-informed floor on episode length (BH ≥ 5s, DB ≥ 3s)
- **Hyperparameter tuning** for XGBoost/LightGBM variants
- Comparison plots: AUPRC across model iterations, t-SNE visualization of the feature space, per-model evolution plots

## 6. Graph Neural Networks — `6_GNN_complete`

Reframes each subject's windowed features as a graph and applies GNNs instead of tabular classifiers, testing two graph constructions × two architectures × three loss functions.

**Graph construction:**
- **Temporal edge (T-)** — windows connected to their temporal neighbors (sequence-aware)
- **Feature similarity (FS-)** — windows connected via k-NN in feature space

**Architectures:**
- **GCN** (`GCNConv`) — standard graph convolution
- **GIN** (`GINConv`) — Graph Isomorphism Network, stronger expressive power

**Loss functions, evaluated for each graph × architecture combination:**
- Cross-Entropy (CE)
- Focal Tversky Loss (FTL) — tuned for the BH/DB minority classes
- Combo Loss (CE + FTL)

Shared clinical safeguards carried over from stage 5: 8-feature optimized list, per-fold winsorising, minimum-duration enforcement, balanced batching (`NeighborLoader` + `WeightedRandomSampler`), and full LOSO evaluation. Comparison plots included for loss evolution and cross-configuration performance (GCN vs. GIN, temporal vs. feature-similarity, per loss function).

---

## Evaluation Protocol (consistent throughout)

- **LOSO cross-validation** — one subject fully held out per fold, across all six stages
- **3-class problem:** `NB = 0`, `BH = 1`, `DB = 2`, with heavy class weighting for the minority BH/DB classes
- **No leakage** — scaling, winsorising, and calibration are always fit on training folds only

## Repo Structure

```
1_feature_extraction.md          # raw ECG → windowed feature CSV
2_baseline_model_test.md         # RF baseline, full feature set
3_feature_selection.md           # Wilcoxon-based feature pruning
4_optimized_features_test.md     # RF re-test on pruned features
5_traditional_models.md          # RF / SVM / XGBoost / LightGBM, calibrated
6_GNN_complete.md                # T-GCN / T-GIN / FS-GCN / FS-GIN, 3 losses each
```
