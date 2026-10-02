# 2–3 Minute Loom Video Presentation Script

> **Video Title**: Spotter Freight Rate Prediction — Machine Learning Solution Walkthrough  
> **Target Length**: 2 minutes 40 seconds (ideal window: 2:20 – 2:55)  
> **Speaker**: Candidate  
> **Format**: Screen share with webcam bubble in the corner.

---

## Pre-Recording Setup

Have the following 4 windows ready to switch between:
1. **Screen 1**: `README.md` or the PDF report — Data Quality table and OOT split diagram.
2. **Screen 2**: IDE open on `train_predict.py` — `engineer_features()` and the ensemble block.
3. **Screen 3**: Terminal showing `python score.py ...` output (pre-run is fine).
4. **Screen 4**: `scorer_results/candidate_december.png`.

---

## Script & Timing

### Section 1: Introduction & Data Exploration (0:00–0:35)
**Screen to show**: README data quality table or terminal output from `train_predict.py`.

> *"Hi, I'm walking through my solution to Spotter's Freight Rate Prediction Challenge.*
>
> *We have 48,000 historical loads covering January through October 2025, and we need to predict posted rates for 12,000 validation loads in November and December.*
>
> *My first step was a thorough data audit. I found five issues:*
> *- **292 training and 145 validation records have negative weight values** — clearly sign-flip entry errors. I apply `abs(weight)` before any imputation.*
> *- **0.78% of training labels have implausible rate-per-mile values** — either below 0.50 or above 6.00 dollars per mile. These look like corrupted records; I flag them rather than silently keeping them.*
> *- **Missing weights and market index** — imputed using domain-aligned medians plus explicit indicator flags so the model knows when data is missing.*
> *- **8 new cities appear only in validation** — Laredo, Charlotte, and others never seen in training. I use continuous Haversine coordinates instead of city IDs, so the model generalises to any location."*

---

### Section 2: Validation Strategy — Why Not K-Fold (0:35–1:10)
**Screen to show**: README OOT split diagram.

> *"The most important decision was the validation strategy.*
>
> *Random K-Fold shuffles records across time. In freight pricing that's a problem: if October market tightness leaks into the training fold, the model sees future macroeconomic conditions when predicting April loads. CV scores look great; production scores collapse.*
>
> *I use a strict Out-of-Time split instead: train on January through August, hold out September through October, and deploy on November through December.*
>
> *One honest caveat: a single cut-point has higher variance than a rolling-origin cross-validation, so I treat the holdout metrics as indicative rather than definitive."*

---

### Section 3: Model Architecture & Results (1:10–1:55)
**Screen to show**: Benchmark table in README or PDF.

> *"For modelling, freight rates are right-skewed and contain some extreme outlier records. Standard MSE loss pulls predictions toward outliers and inflates RMSE.*
>
> *My approach: predict Rate Per Mile — posted_rate divided by distance — and optimise with L1 (MAE) loss. L1 targets the conditional median, which is much more stable.*
>
> *The final model is a weighted ensemble: 70% LightGBM L1 on RPM, 15% LightGBM L2 on RPM, and 15% LightGBM L1 on the direct rate. The L2 component partially corrects the conditional-median's underestimation of high-rate loads.*
>
> *Compared to a naive baseline of median RPM times distance, the ensemble reduces MAE by roughly 128 dollars and Median AE by 83 dollars on the Sep–Oct holdout. The ensemble achieves an MAE around 111 dollars, Median AE around 36 dollars, and an R-squared of 0.82."*

---

### Section 4: Code Walkthrough & Scorer Verification (1:55–2:30)
**Screen to show**: IDE -> Terminal.

> *"In `train_predict.py`, the `engineer_features` function builds 29 features across four domains: geospatial coordinates and Haversine distance, temporal day-of-week patterns with cyclical sine-cosine encoding, payload and equipment characteristics, and market-signal interactions.*
>
> *One important decision: I removed month, quarter, day-of-year, and year-level sine-cosine features. The model is trained on January through October — those features take values in November and December that the model never saw, so any splits it learned on them cannot generalise.*
>
> *The script also writes December predictions to a separate output file. The original `december-chart-inputs.csv` is never overwritten.*
>
> *Running Spotter's official scorer validates 12,000 predictions and 31 December predictions with zero errors and generates the chart."*

---

### Section 5: December Chart & Honest Closing (2:30–2:55)
**Screen to show**: `scorer_results/candidate_december.png`.

> *"Here is the December scenario chart for Lexington to Fort Wayne — 360 miles, Dry Van, 32,000 lbs.*
>
> *What the model captures is a weekly day-of-week pattern that repeats throughout the month: Mondays average around 811 dollars, Wednesdays around 825 dollars, weekends a few dollars lower. This rhythm is learned from the training data and generalises to December.*
>
> *What I want to be honest about: there is no meaningful late-December surge in these predictions. Day 25 onwards averages 818 dollars, essentially the same as the rest of the month. The model has no December training data and cannot learn holiday-specific behaviour.*
>
> *The absolute rate level — that 818-dollar average — should be treated with appropriate uncertainty. Without December actuals, we cannot verify whether the true market level is higher or lower.*
>
> *All deliverables are reproducible with a single `python train_predict.py` command. Thank you."*

---

## Presenter Notes

**Numbers to cite**:
- Naive baseline MAE: ~$257 / Median AE: ~$139 / R²: ~0.80
- Ensemble MAE: ~$129 / Median AE: ~$56 / R²: ~0.82 (OOT Sep-Oct)
- Lift over naive: ~$128 MAE reduction, ~$83 Median AE reduction

**Claims to avoid**:
- "Production-grade" — overstates readiness
- "Impervious to outliers" — the model is more robust, not immune
- "Holiday surge" — the data does not support it
- "Perfectly aligned" — no Dec 2025 actuals exist to compare against
- Citing specific numbers like "$826.69" or "$818.30" as authoritative — they are model outputs, not ground truth

**Screen timing**:
- 0:00–1:55: README / PDF
- 1:55–2:20: Code (`train_predict.py`)
- 2:20–2:35: Terminal (`score.py` output)
- 2:35–2:55: `scorer_results/candidate_december.png`
