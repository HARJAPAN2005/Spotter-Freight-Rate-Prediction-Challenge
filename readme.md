# Spotter Freight Rate Prediction — Machine Learning Assessment

> Machine learning solution to forecast freight load posted rates (`posted_rate`) across commercial freight corridors, using a geospatial-temporal pipeline with a Rate-Per-Mile L1 ensemble, Out-of-Time validation, and official scorer verification.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scorer Status](https://img.shields.io/badge/scorer-passed-success.svg)](#scorer-verification)
[![OOT R² Score](https://img.shields.io/badge/OOT%20R%C2%B2-0.827-green.svg)](#out-of-time-validation-results)
[![OOT MAE](https://img.shields.io/badge/OOT%20MAE-%24111-brightgreen.svg)](#out-of-time-validation-results)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Data Exploration & Quality Findings](#data-exploration--quality-findings)
3. [Validation Methodology & Anti-Leakage Split](#validation-methodology--anti-leakage-split)
4. [Feature Engineering](#feature-engineering)
5. [Model Selection & Loss Formulation](#model-selection--loss-formulation)
6. [Out-of-Time Validation Results](#out-of-time-validation-results)
7. [Fixed December Scenario Analysis](#fixed-december-scenario-analysis)
8. [Limitations](#limitations)
9. [Quickstart & Run Instructions](#quickstart--run-instructions)
10. [Repository Layout](#repository-layout)
11. [Deliverables Checklist](#deliverables-checklist)
12. [Loom Video Presentation Guide](#loom-video-presentation-guide)

---

## Executive Summary

This repository trains a LightGBM ensemble on 48,000 historical freight loads (Jan–Oct 2025) and predicts posted rates for 12,000 validation loads (Nov–Dec 2025) plus a fixed 31-day Lexington → Fort Wayne scenario.

Key design decisions:
- **Target reformulation**: Rate Per Mile (RPM = `posted_rate / distance`) with L1 loss to reduce sensitivity to extreme rate records.
- **Out-of-Time split**: Jan–Aug train, Sep–Oct holdout — no random K-Fold to avoid temporal leakage.
- **Coordinate-based geography**: avoids failure on the 8 cities in validation that were never seen in training.
- **Out-of-Distribution features dropped**: `month`, `quarter`, `dayofyear`, and year-level sine/cosine features were removed because the validation window (Nov–Dec) lies outside the training range (Jan–Oct) for those dimensions.

**OOT result summary (Sep–Oct 2025 holdout):**

| Metric | Naive Baseline | Tri-Ensemble |
| --- | ---: | ---: |
| RMSE ($) | ~1,060 | ~635 |
| MAE ($) | ~477 | ~111 |
| Median AE ($) | ~418 | ~36 |
| R² | ~0.61 | ~0.83 |

*Note: R² and RMSE are sensitive to the ~0.78% of training records where rpm falls outside [0.5, 6.0]. These records appear to be data-quality anomalies; the values above include them.*

---

## Data Exploration & Quality Findings

Five data quality issues were identified during EDA:

| Issue | Train | Val | Fix |
| :--- | :---: | :---: | :--- |
| **Negative weights** | 292 rows | 145 rows | `abs(weight)` before any imputation |
| **Missing weight** | 300 rows (0.63%) | 165 rows (1.38%) | Imputed 31,000 lbs median + `weight_isna` flag |
| **Missing market_index** | 374 rows (0.78%) | 249 rows (2.08%) | Daily mean lookup across observed dates + `market_index_isna` flag |
| **RPM outliers (possible corruptions)** | 373 rows (0.78%): 270 high (rpm > 6.0), 103 low (rpm < 0.5) | — | Flagged in comments; model sees actual targets. Low-rpm rows cannot be explained by "emergency surcharges" — they may be test records or miskeyed distances. High-rpm rows include some genuine short-haul premiums. Neither category is dropped; a downstream audit is warranted. |
| **Unseen cities in validation** | 0 | 8 new cities (Laredo, Charlotte, Knoxville, Jackson, Norfolk, Chicago, Allentown, San Diego) | Coordinate-based geo features (Haversine, tortuosity) generalise to any lat/lon pair. No city-ID encoding used. |

**Note on `quote_signal`**: `quote_signal` is present in both train and validation. Its correlation with actual rpm is ~0.05, so it is treated as a noisy auxiliary signal. The model uses it as an additional feature but it should not be described as a "rate estimate."

---

## Validation Methodology & Anti-Leakage Split

### Why Not Random K-Fold?
Freight rates have strong temporal structure: seasonal agricultural cycles, macroeconomic tightening, and fuel-cost shocks. Random K-Fold allows future months to appear in the training fold, producing inflated CV scores that collapse in production.

### Out-of-Time (OOT) Split

```
|<------- TRAIN (38,477 loads) ------->| OOT HOLDOUT (9,523) | VALIDATION TARGET (12,000) |
   Jan 2025                 Aug 2025     Sep–Oct 2025           Nov–Dec 2025
```

1. **OOT Train**: Jan 1 – Aug 31, 2025 (38,477 loads, 80.2%)
2. **OOT Holdout**: Sep 1 – Oct 31, 2025 (9,523 loads, 19.8%)
3. **Final predictions**: Nov 1 – Dec 31, 2025 (12,000 validation loads)

**Caveat — single split**: A single temporal split has higher variance than repeated time-series cross-validation. An ideal next step is an expanding-window or rolling-origin CV over multiple cut points.

---

## Feature Engineering

The pipeline creates **29 predictive features** (reduced from an earlier 35 after removing features that are out-of-distribution for Nov–Dec validation):

**Removed features (OOD on validation)**:
`month`, `quarter`, `dayofyear`, `sin_dayofyear`, `cos_dayofyear` — training covers Jan–Oct only; splits learned from those values cannot generalise to Nov–Dec.

**Retained features by domain:**

1. **Geospatial**: `pickup_lat/lon`, `delivery_lat/lon`, `haversine`, `tortuosity`, `lat_diff`, `lon_diff`, `mid_lat`, `mid_lon`
2. **Temporal** (generalise across all months): `day`, `dayofweek`, `is_weekend`, `is_month_end`, `sin_dayofweek`, `cos_dayofweek`
3. **Payload**: `weight_filled`, `weight_isna`, `weight_tier`, `ton_miles`, `distance`
4. **Equipment**: `eq_Dry Van`, `eq_Flatbed`, `eq_Reefer`
5. **Market / broker signals**: `quote_signal`, `est_base`, `market_index_filled`, `market_index_isna`, `market_adj_base`, `quote_x_market`

---

## Model Selection & Loss Formulation

### Rate Per Mile + L1 Loss

Freight spot rate distributions are right-skewed (~0.78% of records have rpm outside [0.5, 6.0]). Standard L2 (MSE) loss inflates predictions toward outliers. Predicting **RPM** and optimising with **L1 (MAE)** loss targets the conditional median, which is substantially more robust.

### Tri-Ensemble

```
prediction = 0.70 × (L1-RPM model × distance)
           + 0.15 × (L2-RPM model × distance)
           + 0.15 × (L1-Direct model)
```

The L2 component partially corrects for the conditional-median's tendency to underpredict high-rate loads.

---

## Out-of-Time Validation Results

Holdout: 9,523 loads from Sep 1 – Oct 31, 2025.

| Model | Objective | RMSE ($) | MAE ($) | Median AE ($) | R² |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **Naive baseline** (median rpm × distance) | — | ~1,060 | ~477 | ~418 | ~0.61 |
| XGBoost Direct | L2 | 758 | 207 | 98 | 0.753 |
| XGBoost RPM | L2 | 710 | 186 | 84 | 0.783 |
| LightGBM Direct | L2 | 656 | 174 | 76 | 0.815 |
| LightGBM RPM | L2 | 650 | 164 | 69 | 0.819 |
| **Tri-Ensemble (final)** | 70% L1-RPM + 15% L2-RPM + 15% L1-Direct | **~635** | **~111** | **~36** | **~0.827** |

*Metrics are affected by the ~0.78% outlier records in the data. Exact numbers vary slightly with the feature set version.*

---

## Fixed December Scenario Analysis

**Scenario**: Lexington → Fort Wayne | 360 miles | Dry Van | 32,000 lbs | one prediction per day, Dec 1–31, 2025.

### Scorer Chart

![Fixed December 2025 Predicted Load Rate](scorer_results/candidate_december.png)

### What the chart shows

The model captures a **weekly day-of-week pattern** learned across all training months:

| Day of week | Approx predicted rate |
| :--- | ---: |
| Monday | ~$811 |
| Tuesday | ~$819 |
| Wednesday | ~$825 |
| Thursday | ~$822 |
| Friday | ~$820 |
| Saturday | ~$817 |
| Sunday | ~$813 |

The pattern repeats weekly throughout December. There is **no meaningful late-December surge**: day ≥ 25 averages $818.1 versus $818.4 for the rest of the month. The model has no ability to learn December-specific seasonality because December did not appear in the training data.

---

## Limitations

1. **December absolute level is uncertain.** The model was trained on Jan–Oct data. The December mean prediction (~$818) may differ from the true market level; without Dec 2024 or Dec 2025 actuals, the absolute level cannot be validated.

2. **Single temporal split.** One OOT holdout has higher variance than a rolling-origin CV. The metrics should be treated as indicative, not definitive.

3. **~0.78% corrupted labels.** Training records with rpm < 0.5 or rpm > 6.0 inflate RMSE and affect R². A data audit with the source system is warranted before production use.

4. **`quote_signal` interpretation.** Its correlation with actual rpm is ~0.05. It contributes as a noisy auxiliary feature, not as a rate estimate.

5. **Negative weights.** 292 training / 145 validation records had negative weight values. These are treated as sign-flip recording errors and corrected with `abs(weight)`. If some represent legitimate missing-weight sentinel values, the imputation approach may need revisiting.

---

## Quickstart & Run Instructions

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/HARJAPAN2005/Spotter-Freight-Rate-Prediction-Challenge.git
cd Spotter-Freight-Rate-Prediction-Challenge

# Install dependencies
python -m pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
python train_predict.py
```

This will:
- Print a data quality report
- Report OOT metrics and lift over naive baseline
- Save `validation_predictions.csv` (12,000 rows)
- Save `data/december_chart_inputs.csv` (filled; original input is untouched)

### 3. Verify with Official Scorer
```bash
python score.py --predictions validation_predictions.csv \
                --december-predictions data/december_chart_inputs.csv
```

Expected output:
```
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```

---

## Repository Layout

```
.
├── train_predict.py           # Main training & inference pipeline
├── generate_report.py         # PDF / DOCX report generator
├── score.py                   # Spotter-provided scorer (unmodified)
├── requirements.txt
├── validation_predictions.csv # 12,000 final predictions (output)
├── december-chart-inputs.csv  # Original input — never overwritten
├── data/
│   ├── december_chart_inputs.csv   # Filled predictions (scorer input)
│   └── ...                         # Other data files
├── scorer_results/
│   └── candidate_december.png
├── research/                  # Exploratory analysis scripts (not part of
│   └── *.py                   # the submission pipeline; kept for transparency)
├── Freight_Rate_ML_Assessment_Report.pdf
├── Freight_Rate_ML_Assessment_Report.docx
└── LOOM_SCRIPT.md
```

---

## Deliverables Checklist

- [x] GitHub repository with clean pipeline code, dependencies, and run instructions.
- [x] `validation_predictions.csv` — exactly 12,000 rows, columns `load_id,predicted_rate`, all values positive.
- [x] `data/december_chart_inputs.csv` — 31 rows with `predicted_rate` filled.
- [x] `scorer_results/candidate_december.png` — verified via `score.py` with 0 errors.
- [x] `Freight_Rate_ML_Assessment_Report.pdf` — validation approach, benchmark table, limitations, December chart.
- [x] `Freight_Rate_ML_Assessment_Report.docx` — companion DOCX.
- [x] `LOOM_SCRIPT.md` — timestamped 2–3 minute presentation script.

---

## Loom Video Presentation Guide

See [`LOOM_SCRIPT.md`](LOOM_SCRIPT.md) for a timestamped script covering:

1. **0:00–0:35** — Key data exploration findings & the 5 data quality issues
2. **0:35–1:10** — Why OOT split, not K-Fold; what the split looks like
3. **1:10–1:55** — Model choice: RPM + L1 ensemble; benchmark vs naive baseline
4. **1:55–2:30** — Code walkthrough (`engineer_features`, ensemble, December output fix)
5. **2:30–2:50** — December chart: weekly pattern, honest uncertainty about absolute level
