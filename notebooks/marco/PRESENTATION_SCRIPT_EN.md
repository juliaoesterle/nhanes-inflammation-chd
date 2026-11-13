# Solo Presentation Script & Q&A Guide (English · Simple & Clear)
**ETH Zürich — *Foundations of Data Science* (`376-1983-00L`) · Mid-Term Assessment (Thursday)**  
**Format:** Strictly **10 minutes** presentation + **5 minutes** Q&A & feedback · **Speaker:** Solo (`we`-perspective, no teammate names)  
**Target Pace:** ~`7.5–8.0` minutes (~`1,000` words) — relaxed, easy to follow, plenty of time buffer.

---

## Slide-by-Slide Spoken Script

### Slide 1 — 1. Research Topic [`0:00–1:00`]
> "Good morning everyone. Today I'm presenting our group project on predicting coronary heart disease and heart attacks.
>
> Usually, doctors estimate heart-disease risk using standard factors like age, blood pressure, diabetes, smoking, and cholesterol. However, cardiovascular studies—for example Ridker et al. (2002) and Adamstein et al. (2021)—show that chronic inflammation in the body also drives heart disease independently of cholesterol.
>
> Two common inflammation blood markers are **hs-CRP** and the **neutrophil-to-lymphocyte ratio (NLR)**—and NLR is practically free because it comes from any routine blood count. So using recent US health survey data (NHANES 2021–2023), we asked two simple questions:
> **First, are NLR and hs-CRP linked to heart disease on their own? And second, do they actually improve machine learning predictions when added on top of standard clinical risk factors?**"

### Slide 2 — 2. Overview of Features & Labels [`1:00–2:15`]
> "Let's look at what we predict and which inputs we use:
> - **Our Label (`chd_mi`):** A simple yes/no outcome—`1` if a person reported a doctor diagnosis of coronary heart disease or a heart attack, and `0` otherwise. About **7.4%** of our final sample has the condition. Because this comes from a one-time survey, it captures *existing* disease, not future first-time heart attacks—a key detail we will come back to.
> - **Our Features (`13` total, added step by step):**
>   - First, basic demographics: age and sex.
>   - Second, socioeconomic & body info: BMI, income, and race/ethnicity.
>   - Third, standard clinical risk factors: smoking, diabetes, blood pressure, total and HDL cholesterol, and kidney function.
>   - Finally, our two inflammation markers: NLR and hs-CRP, which we log-transform because their values are strongly right-skewed.
>
> One deliberate choice: we kept **cholesterol and blood-pressure medications out** of our prediction models so the models can't cheat by seeing that someone is already being treated."

### Slide 3 — 3. Preprocessing Steps [`2:15–3:30`]
> "Our whole team works from one shared preprocessing script so every model sees the exact same data:
> 1. **Cleaning raw errors:** NHANES contains non-response codes like 'Refused' or 'Don't know', occasional corrupted strings, and technical zero artifacts from SAS files. We automatically checked every column against the official codebook metadata and turned invalid entries into clean missing values.
> 2. **Step-by-step filtering (`~12,000 → 4,544` adults):** Starting from roughly 12,000 survey participants, we kept adults aged 20 or older with a valid heart-disease answer, measured NLR and hs-CRP, and complete clinical covariates—leaving **`4,544` people**. Keeping complete cases ensures all models are compared fairly on the exact same patients.
> 3. **Checking for selection bias:** Since income was missing in about 13% of eligible adults, we checked whether dropped participants differed from kept ones. Excluded people had slightly lower average income and higher minority representation, **but almost identical heart-disease rates (~7%)**. So this mainly limits broad generalizability rather than biasing our model comparisons. All scaling and categorical encoding run strictly inside cross-validation to avoid data leakage."

### Slide 4 — 4. Analysis Plan [`3:30–4:45`]
> "How do we analyze the data? Before looking at any results, we split the `4,544` adults into a stratified **80% training set** (`~3,600` people) and an untouched **20% test set** (`~900` people), both with the same `7.4%` heart-disease rate. We used 5-fold cross-validation inside the training set to tune everything cleanly instead of cutting away another validation slice from our relatively small number of positive cases.
>
> To be sure our answer doesn't depend on one specific algorithm, we test four complementary methods from the lecture:
> - **Logistic Regression** to test whether NLR and hs-CRP have statistically significant links on their own.
> - **Lasso and Ridge Regression** as linear models built to handle correlated risk factors.
> - **k-Nearest Neighbors** to predict from similar neighboring patients.
> - **Random Forest** (`300` decision trees) to automatically capture non-linear patterns and feature interactions—plus we inspect which variables it actually uses.
>
> For our remaining 20%, we plan to test imputing missing income instead of dropping rows, check excluding extreme acute inflammation spikes (`hs-CRP > 10`), and look into Nested Cross-Validation."

### Slide 5 — 5a. Preliminary Results: Link vs. Prediction Gain [`4:45–6:15`]
> "Now to our preliminary results—and here is our most interesting finding: **a clear gap between statistical significance and real prediction gain.**
>
> On the **left chart** (our logistic regression): **NLR stays statistically linked to heart disease** even after adjusting for all risk factors and cholesterol medication—one standard deviation higher NLR means **18% higher odds (`OR = 1.18, p = 0.011`)**. By contrast, **hs-CRP loses its effect completely** once standard risk factors enter (`OR ≈ 1.01`).
>
> Now look at the **right chart**—does adding NLR and hs-CRP actually help our machine learning models predict better? **Barely at all.** Standard clinical factors already push all four models to roughly **0.88 ROC-AUC**. Adding both inflammation markers only raises AUC by **0.003 to 0.004**—virtually zero, with every confidence interval crossing zero. Whether linear, sparse, neighbor-based, or a Random Forest, every model gives the exact same negative answer."

### Slide 6 — 5b. Preliminary Results: Calibration & Feature Importance [`6:15–7:25`]
> "Looking closer at how the models behave reveals two helpful insights:
> - **Simpler linear models match complex trees:** Random Forest gets **0.887 ROC-AUC**, tying Ridge (**0.884**), while Ridge actually scores slightly better on Precision-Recall (**0.49 vs. 0.43**) and probability calibration (**Brier score 0.052 vs. 0.056**, where guessing the average gives `0.068`). That tells us Random Forest isn't missing big hidden non-linear interactions—the signal is mostly smooth and additive.
> - **Why is total cholesterol the #1 feature in Random Forest?** Look at the feature importance bar chart: **total cholesterol and age dominate**, while NLR adds almost nothing and hs-CRP sits near zero. Why does cholesterol rank first when heart-disease patients here actually have *lower* cholesterol (**~160 vs. ~191**)? Because of **reverse causation from statins**: roughly **80% of heart-disease patients take cholesterol-lowering drugs**, compared to only **25%** of healthy adults! The tree learns 'low cholesterol + older age' as a sign of already diagnosed and treated patients. When we drop cholesterol as a check, AUC drops noticeably."

### Slide 7 — 6. Takeaway & Questions for You [`7:25–8:10`]
> "To wrap up: **NLR has a statistically significant link to heart disease, but neither NLR nor hs-CRP meaningfully improves prediction beyond routine clinical risk factors**—and cross-sectional lab values strongly reflect ongoing medication.
>
> Going into our final 20%, we have **three quick questions** where we'd love your feedback:
> 1. **Missing Income (`~13%`):** Since excluded people have slightly lower average income, should we keep complete-case as our main result and add MICE imputation only as a robustness check, or use imputation as our main pipeline?
> 2. **Survey Weights:** NHANES includes survey weights for US population estimates. Should we apply them only in logistic regression for odds ratios, or also inside our ML models?
> 3. **Small Test Set (`67` cases):** Because we run several sensitivity checks on ~`900` test patients, would you recommend switching to **Nested Cross-Validation** for the final report?
>
> Thank you—happy to hear your thoughts and feedback!"

---

## Simple Q&A Backup Cheat Sheet (Ready 15-Second Answers)

- **Q1: "Isn't a near-zero improvement (`+0.003` to `+0.004` AUC) a weak or failed result?"**  
  > "Not at all—showing consistently across four very different ML methods that popular blood markers don't improve prediction on top of standard factors (`~0.88` AUC) is a classic, honest scientific finding. Just like we learned in class, a significant p-value (`0.011`) doesn't mean strong predictive power."

- **Q2: "Why didn't you balance class weights in Random Forest if only 7.4% have heart disease?"**  
  > "We tested `balanced_subsample` separately: it pushed predicted probabilities far away from the true 7.4% rate and ruined calibration (Brier got worse than guessing the baseline average), without improving AUC gain. Keeping unweighted probabilities lets all four models be compared fairly, and we still get **87% sensitivity** by adjusting the decision threshold (`0.088`) using only training folds."

- **Q3: "Why does Random Forest use low cholesterol to spot heart disease?"**  
  > "Because NHANES is cross-sectional: people who already had a heart attack get put on statins (~80% vs. ~25% of controls), which lowers their measured cholesterol (~160 vs. ~191). That's why we also ran models without cholesterol to confirm how much treatment status drives that signal."

- **Q4: "Why did Ridge beat Random Forest on Precision-Recall (`0.49` vs. `0.43`)?"**  
  > "With 13 clean tabular predictors and ~270 positive training cases, the underlying risk pattern is fairly smooth and additive—so regularized linear Ridge avoids overfitting slightly better than deep tree splits."

- **Q5: "Aren't you reusing the test set when running extra checks without cholesterol?"**  
  > "Fair point—while tuning and thresholds never touched the test set, evaluating extra sensitivity variants on 67 test cases widens uncertainty. That's exactly why we're asking you whether Nested CV is better for our final 20%."
