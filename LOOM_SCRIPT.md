# Loom Video Script — Spotter Freight Rate Prediction Challenge
## Full Precision Guide: Exact Words + Exact Screen Location

> **Target length**: 2 min 30 sec – 2 min 55 sec  
> **Format**: Screen share (full screen, no window chrome) + webcam bubble bottom-right  
> **Font size in IDE**: bump to 16pt so code is readable on a small Loom player  
> **Pre-run everything** before hitting Record — terminal output should already be visible

---

## Pre-Recording Checklist

Before you start Loom, have these four things open and **ready to switch to instantly**:

| Tab | What's open | How to get there |
| --- | --- | --- |
| **A** | `readme.md` rendered in a Markdown previewer (VS Code preview, or GitHub.com repo page) | Open the README tab in the repo on github.com |
| **B** | `train_predict.py` in your IDE | Scroll so line 65 (`def engineer_features`) is visible at the top of the screen |
| **C** | Terminal showing the **already-completed** output of `python train_predict.py` | Run it before recording; scroll to the top of the output |
| **D** | `scorer_results/candidate_december.png` open in an image viewer (full screen) | Double-click the file |

---

## SECTION 1 — Introduction & Data Quality (0:00 – 0:38)

### Screen to show: Tab A — README, scrolled to "Data Exploration & Quality Findings" table (line 56–62)

**Speak exactly this:**

> "Hi, I'm walking through my solution to Spotter's Freight Rate Prediction Challenge.
>
> I had 48,000 historical freight loads from January through October 2025, and the goal is to predict posted rates for 12,000 validation loads in November and December.
>
> [*point to the table header row*]
> My first step was a thorough data audit. I found five issues.
>
> [*point to row 1 — Negative weights*]
> First — 292 training records and 145 validation records have a **negative weight**. That's a sign-flip recording error, not a sentinel. The fix is one line: `abs(weight)` before any imputation.
>
> [*point to row 4 — RPM outliers*]
> Second — about 0.78% of training labels have an implausible rate per mile. 270 records are above six dollars per mile, and 103 are below fifty cents. The low-rpm ones can't be explained by any normal freight scenario — they're likely miskeyed distances or test records. I flag both groups and leave them in; the model sees their actual targets, so nothing is silently hidden.
>
> [*point to rows 2 and 3 — Missing weight, Missing market_index*]
> Third and fourth — missing weight and missing market index, under one percent each. Standard imputation with domain-aligned medians plus explicit indicator flags so the model knows when data was missing.
>
> [*point to row 5 — Unseen cities*]
> And fifth — the validation set introduces eight cities never seen in training: Laredo, Charlotte, San Diego, and others. A model using categorical city IDs fails completely on those. Mine uses continuous Haversine coordinates instead, so geography generalises to any lat/lon pair."

---

## SECTION 2 — Validation Strategy (0:38 – 1:08)

### Screen to show: Tab A — README, scroll up slightly to "Validation Methodology & Anti-Leakage Split" (line 68–84)

**Speak exactly this:**

> "Now the most important architectural decision: the validation strategy.
>
> [*point to the ascii diagram — the three blocks*]
> I use a strict Out-of-Time split. Train on January through August — that's 38,477 loads. Hold out September and October — 9,523 loads. Never touch November and December until final inference.
>
> [*point to the 'Why Not Random K-Fold' heading*]
> Why not random K-Fold? Because freight rates have strong temporal structure — fuel-cost shocks, seasonal tightening. If you shuffle, October's market conditions bleed into the April training fold. CV scores look great. Then you deploy and they collapse.
>
> [*gesture at the bottom caveat line — 'single split'*]
> I'm also honest about the limitation here: a single cut-point has higher variance than a rolling-origin cross-validation. These numbers are indicative, not definitive. An ideal next step would be multiple expanding windows."

---

## SECTION 3 — Model & Benchmark Results (1:08 – 1:48)

### Screen to show: Tab A — README, scroll to "Out-of-Time Validation Results" table (line 127–134)

**Speak exactly this:**

> "Here are the benchmark results on the Sep–Oct holdout.
>
> [*point to the Naive baseline row*]
> I always start with a naive baseline — median training RPM times distance, no ML at all. That gives a MAE of about $477 and a Median AE of $418.
>
> [*trace down the table row by row*]
> XGBoost with direct rate, XGBoost with Rate Per Mile, LightGBM direct, LightGBM RPM — each step is better.
>
> [*point to the bottom row — Tri-Ensemble*]
> The final model is a weighted ensemble: 70% LightGBM L1 on Rate Per Mile, 15% LightGBM L2 on Rate Per Mile, 15% LightGBM L1 on the direct rate. That gives a MAE around $111 and a Median Absolute Error around $36 — meaning half of all predictions are within $36 of the actual rate. R-squared is about 0.827.
>
> [*pause briefly*]
> Why Rate Per Mile with L1 loss? Freight rates are right-skewed. Standard L2 loss gets pulled toward outlier records. Predicting rate-per-mile and optimising with L1 targets the conditional median — which is much more stable against those 0.78% corrupted labels I mentioned earlier."

---

## SECTION 4 — Code Walkthrough (1:48 – 2:22)

### Screen: Switch to Tab B — `train_predict.py`

**Step 4a — scroll so line 119–130 (Payload / negative weight fix) is centred on screen**

> "Let me show you the two most important code decisions.
>
> [*point to line 122: `d['weight_abs'] = d['weight'].abs()`*]
> Right here — line 122 — this is the negative weight fix. `abs()` is called *before* the `.fillna()` on line 124. If you do it in the wrong order, some negative weights silently become 31,000 lbs instead of their corrected absolute value."

**Step 4b — scroll up to line 133 (Temporal section comment)**

> "[*point to the comment on line 133: 'month / quarter / dayofyear / sin-cos-year deliberately excluded'*]
>
> And here — line 133 — I deliberately excluded month, quarter, day-of-year, and the year-level sine-cosine features. The training data covers January through October. In November and December those features take values the model has never seen. Any splits it learned on them are meaningless for generalisation. I kept only day-of-week and day-of-month, which repeat identically every week and every month."

**Step 4c — switch to Tab C — Terminal output, scroll to STEP 1 output block**

> "[*point to the 'Naive baseline' output line*]
> This is the live terminal output. You can see the naive baseline printed first, then the ensemble — and below it, the explicit lift calculation. There's no ambiguity about how much the model actually added over a trivial predictor.
>
> [*scroll down to STEP 4 / the NOTE block*]
> And here at the bottom of step four — the December note — which I'll come back to in a moment."

---

## SECTION 5 — Scorer Verification & December Chart (2:22 – 2:52)

### Screen: Switch to Tab D — `scorer_results/candidate_december.png` (full screen)

**Step 5a — stay on the chart**

> "[*gesture broadly at the chart*]
> This chart is produced entirely by Spotter's unmodified `score.py`. What you see is a weekly pattern that repeats every seven days throughout December.
>
> [*trace the repeating wave shape with your finger/cursor*]
> Monday dips to around $811. Wednesday peaks around $825. Weekends come back down to $813. That rhythm is something the model learned from all ten training months — it genuinely generalises to December.
>
> [*pause — then speak clearly*]
> What I want to be explicit about: there is **no meaningful late-December surge**. I checked — day 25 onwards averages $818, the rest of the month averages $818. Identical. And the model couldn't have learned December seasonality anyway, because December wasn't in the training data. The absolute rate level here — around $795 to $812 — should be treated with appropriate uncertainty."

**Step 5b — switch back to Tab C — Terminal, scroll to the very end showing the scorer command**

> "[*point to the scorer command at the bottom*]
> Running Spotter's official scorer — `python score.py` — confirms exactly 12,000 validation predictions and 31 December predictions pass with zero errors.
>
> Everything is reproducible from a single command: `python train_predict.py`. The repo, the PDF report, the CSV — all linked in the README. Thank you."

---

## Exact Numbers to Cite (memorise these)

| Fact | Number |
| --- | --- |
| Training loads | 48,000 |
| OOT Train window | Jan 1 – Aug 31, 2025 (38,477 loads) |
| OOT Holdout window | Sep 1 – Oct 31, 2025 (9,523 loads) |
| Validation loads | 12,000 (Nov–Dec 2025) |
| Negative weight records | 292 train / 145 val |
| RPM outliers in training | 373 rows (0.78%): 270 high, 103 low |
| Naive baseline MAE | ~$257 |
| Naive baseline Median AE | ~$139 |
| Ensemble MAE (OOT) | ~$129 |
| Ensemble Median AE (OOT) | ~$56 |
| Ensemble R² (OOT) | ~0.824 |
| Dec Monday avg | ~$811 |
| Dec Wednesday avg | ~$825 |
| Dec day ≥25 avg vs rest | $818 vs $818 (no surge) |

---

## Hard Don'ts — Things Never to Say

| Don't say | Why |
| --- | --- |
| "Holiday surge" or "holiday closures" | Data contradicts it: day ≥25 == rest of month |
| "Production-grade" | Overstates readiness; single-split validation |
| "Impervious to outliers" | It's more robust, not immune |
| "Perfectly aligned with historical rates" | No Dec actuals exist to compare against |
| "quote_signal predicts rate" | Correlation with rpm is only ~0.05 |
| Cite exact Dec absolute level as ground truth | It's a model output, not a verified market rate |

---

## Screen-Switch Timing Summary

| Clock | Screen | Key visual to point at |
| --- | --- | --- |
| 0:00 – 0:38 | Tab A: README data quality table | Each row of the 5-issue table |
| 0:38 – 1:08 | Tab A: README OOT split section | ASCII timeline diagram + caveat line |
| 1:08 – 1:48 | Tab A: README benchmark table | Naive baseline row → Tri-Ensemble row |
| 1:48 – 1:57 | Tab B: `train_predict.py` line 122 | `d['weight_abs'] = d['weight'].abs()` |
| 1:57 – 2:10 | Tab B: `train_predict.py` line 133 | Comment: "month / quarter... deliberately excluded" |
| 2:10 – 2:22 | Tab C: Terminal output | Naive baseline line → Lift line → Dec NOTE |
| 2:22 – 2:44 | Tab D: December chart | Weekly wave pattern + no surge |
| 2:44 – 2:52 | Tab C: Terminal end | scorer command, "0 errors" |
