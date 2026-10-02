# 2-3 Minute Loom Video Presentation Script

> **Video Title**: Spotter Freight Rate Prediction - Machine Learning Architecture & Results  
> **Target Length**: 2 minutes 30 seconds (ideal window: 2:15 - 2:50)  
> **Speaker**: Candidate  
> **Format**: Screen share with webcam bubble in the corner.

---

## Pre-Recording Setup & Visuals
Have the following 4 tabs or windows ready to switch between:
1. **Screen 1**: `README.md` or PDF report showing the Out-of-Time split and benchmark table.
2. **Screen 2**: Code in IDE showing `train_predict.py` (specifically `engineer_features()` and the model ensemble).
3. **Screen 3**: Terminal showing the successful run of `python score.py ...`.
4. **Screen 4**: `scorer_results/candidate_december.png` (the December prediction chart).

---

## Video Script & Timing Breakdown

### Section 1: Introduction & Data Exploration (0:00 – 0:35)
**Screen to Show**: Terminal / EDA summary or `README.md` Data Quality table.

> *"Hi everyone, today I'm walking through my end-to-end machine learning solution for Spotter's Freight Rate Prediction Challenge.*  
>  
> *We're tasked with predicting the final posted rate (`posted_rate`) across commercial trucking corridors using 48,000 historical development loads and forecasting across 12,000 validation loads.*  
>  
> *During exploration, I identified three critical data quality findings:*  
> *First, **missing payload weights** (0.63% of train, 1.38% of val). Instead of naive mean imputation, we analyzed equipment distributions and imputed missing values with domain-aligned medians (31,000 lbs) alongside an explicit boolean indicator `weight_isna` to preserve tree capacity.*  
> *Second, **missing market index** (around 1-2%). Because `market_index` is a daily macroeconomic indicator with minimal intra-day variance, we reconstructed missing values via a daily calendar lookup table.*  
> *Third, the validation set introduced **8 brand new cities** never seen in training—like Laredo and San Diego. Categorical city IDs would fail here, so our pipeline uses continuous Great-Circle coordinates, Haversine distances, and route tortuosity for 100% zero-shot geographic generalization."*

---

### Section 2: Validation Strategy & Anti-Leakage Split (0:35 – 1:10)
**Screen to Show**: `README.md` Validation Split diagram / PDF Section 2.

> *"Now, the validation strategy was the most critical architectural decision.*  
>  
> *Standard random K-Fold cross-validation randomly shuffles rows across time. In freight logistics, that causes severe **look-ahead leakage**—a model trained on October freight tightness and diesel levels would predict April shipments. That produces an artificially inflated CV score that collapses in production.*  
>  
> *To mimic real production deployment, I designed a strict **Out-of-Time (OOT) temporal validation split**:*  
> *- We train on **January 1 through August 31, 2025** (38,477 loads).*  
> *- We validate on **September 1 through October 31, 2025** (9,523 future loads).*  
>  
> *This rigorously tests model generalization across seasonal market regime shifts and gives us an authentic estimate of future performance."*

---

### Section 3: Model Architecture & Loss Formulation (1:10 – 1:55)
**Screen to Show**: Benchmark Table in `README.md` or PDF.

> *"Looking at model selection, freight spot rates have significant positive skew with rare high-dollar surge loads. Standard L2 (MSE) loss gets pulled aggressively by these extreme outliers, skewing predictions upwards and destroying median error.*  
>  
> *To solve this, I introduced two key modeling innovations:*  
> *1. **Target Reformulation**: We predict **Rate Per Mile (RPM = posted_rate / distance)** rather than raw rate, normalizing variance across corridor lengths.*  
> *2. **L1 (MAE) Loss Objective**: By optimizing with `regression_l1`, the model targets the conditional median rate per mile, making it resilient to outlier noise.*  
>  
> *Our production model is a **Tri-Ensemble**:*  
> *- 70% LightGBM L1 on Rate Per Mile*  
> *- 15% LightGBM L2 on Rate Per Mile*  
> *- 15% LightGBM L1 Direct on Posted Rate*  
>  
> *In our Out-of-Time benchmark, this ensemble slashed MAE from $207 down to **$110.94**, achieved a **Median Absolute Error of just $35.78**, and delivered an **$R^2$ of 0.8270** with a MAPE under 5%."*

---

### Section 4: Code Walkthrough & Verification (1:55 – 2:30)
**Screen to Show**: IDE with `train_predict.py` open $\rightarrow$ Terminal showing `score.py` output.

> *"Let's take a quick look at the code structure in `train_predict.py`.*  
>  
> *In `engineer_features()`, we build 35 predictive signals across four domains: Great-circle Haversine distances and route tortuosity, calendar dynamics including weekend and month-end flags with continuous sine/cosine cyclical day-of-year encodings, payload ton-miles, and broker quote-by-market cross-interactions.*  
>  
> *The script trains the ensemble, generates all 12,000 predictions for `validation_predictions.csv`, and forecasts the 31 December scenario inputs.*  
>  
> *Over in the terminal, running Spotter's official evaluation:*  
> `python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv`  
> *validates both files with zero errors and generates our final chart."*

---

### Section 5: December Prediction Chart & Closing (2:30 – 2:50)
**Screen to Show**: `scorer_results/candidate_december.png`.

> *"Finally, here is the generated December prediction chart for the fixed Lexington to Fort Wayne corridor (360 miles, Dry Van, 32,000 lbs).*  
>  
> *The model accurately captures true freight market dynamics:*  
> *You can see the clear weekly rhythm—rates peak mid-week during Wednesday and Thursday dispatch windows (~$826.69), soften over weekends (~$808-$812), and show a noticeable surge in late December during holiday capacity tightening.*  
>  
> *The overall average rate is $818.30 (~$2.27/mile), perfectly aligning with historical corridor economics.*  
>  
> *All deliverables—the clean repo, CSV predictions, executive PDF/DOCX reports, and code—are complete and reproducible with a single command. Thank you!"*

---

## Presenter Delivery Tips
1. **Pacing**: Speak at a steady, confident pace. Don't rush; let the metrics speak for themselves.
2. **Key Numbers to Emphasize**:
   - Out-of-Time MAE: **$110.94**
   - Median Absolute Error: **$35.78** (half of all loads predicted within $35)
   - $R^2$: **0.8270**
   - MAPE: **4.73%**
3. **Smooth Screen Switching**:
   - Tab 1 (0:00 - 1:55): Readme / PDF report
   - Tab 2 (1:55 - 2:15): Code (`train_predict.py`)
   - Tab 3 (2:15 - 2:30): Terminal (`score.py` execution)
   - Tab 4 (2:30 - 2:50): `scorer_results/candidate_december.png`
