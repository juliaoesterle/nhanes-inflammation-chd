# Systemic Inflammatory Biomarkers and Coronary Heart Disease in NHANES 2021–2023: Independent Association and Incremental Machine Learning Discrimination

**Course Project — Foundations of Data Science (ETH Zürich, Dataset 5)**  
*Authors: Marco et al.*

---

## 1. Title and Abstract

### Title
**Systemic Inflammatory Biomarkers and Coronary Heart Disease in NHANES 2021–2023: Independent Association and Incremental Machine Learning Discrimination**

### Abstract
**Background.** Chronic low-grade vascular inflammation is central to atherosclerosis. The neutrophil-to-lymphocyte ratio (NLR) and high-sensitivity C-reactive protein (hs-CRP) are inexpensive, widely accessible inflammatory biomarkers. We investigated whether NLR and hs-CRP are independently associated with prevalent coronary heart disease or myocardial infarction (CHD/MI) beyond established cardiovascular risk factors, and whether they incrementally improve risk discrimination across diverse machine learning algorithms.

**Methods.** We conducted a cross-sectional study of 5,538 non-institutionalized US adults from NHANES 2021–2023 with complete biomarker profiles (412 cases of self-reported doctor-diagnosed CHD/MI, 7.4% prevalence). Every analytical step followed a pre-specified protocol evaluating three candidate options. Missing covariates (12.7% missing family income) were handled using median imputation with dedicated missingness indicators, strictly fitted within cross-validation folds to prevent data leakage. Independent associations were evaluated using multivariable logistic regression (reporting odds ratios [OR] per 1 standard deviation of log-transformed biomarkers; co-primary significance threshold $\alpha = 0.025$ via Bonferroni correction). Discriminative performance was evaluated across six algorithms taught in the course (unpenalized logistic regression, penalized elastic net, CART decision trees, $k$-nearest neighbors, RBF support vector machines, and random forests) using repeated nested cross-validation (15 outer test folds, 5 inner tuning folds). Model comparisons utilized the Nadeau & Bengio resampled paired $t$-test.

**Results.** After adjusting for established clinical risk factors (age, sex, race/ethnicity, income, BMI, smoking, diabetes, hypertension, total and HDL cholesterol, and eGFR), higher NLR was independently associated with CHD/MI ($\text{OR} = 1.20$ per SD, 95% CI: 1.07–1.34, $p = 0.002$; H1a supported). Conversely, hs-CRP was not independently associated ($\text{OR} = 0.96$ per SD, 95% CI: 0.84–1.08, $p = 0.48$; H1b not supported); its crude association was completely accounted for by body mass index. Established clinical risk factors discriminated prevalent CHD/MI accurately (baseline logistic regression ROC-AUC = 0.861). Adding both inflammatory biomarkers yielded negligible incremental discrimination ($\Delta\text{AUC} = +0.0013$, 95% CI: −0.0009 to +0.0035, $p = 1.00$; H1c not supported), with zero improvement in Brier calibration score ($\Delta\text{Brier} = -0.0002$). Standalone biomarkers exhibited poor discriminative ability (AUC = 0.631 for both markers, 0.506 for hs-CRP alone). No complex machine learning algorithm significantly outperformed logistic regression (SVM $\Delta\text{AUC} = +0.0041, p = 0.74$; Random Forest $\Delta\text{AUC} = -0.0041$; Decision Tree $\Delta\text{AUC} = -0.0575, p = 0.002$). Permutation importance confirmed that age and cholesterol dominated predictions, while hs-CRP ranked last. Findings remained consistent across five pre-specified sensitivity analyses.

**Conclusion.** In a representative US adult sample, elevated NLR is independently associated with prevalent coronary heart disease, whereas hs-CRP reflects underlying adiposity rather than independent vascular risk. Neither biomarker incrementally enhances individual risk discrimination beyond conventional risk factors, irrespective of algorithmic complexity.

---

## 2. Introduction

Ischaemic heart disease remains the primary contributor to global morbidity and mortality (World Health Organization, 2024). Established modifiable and non-modifiable risk factors—including advanced age, male sex, cigarette smoking, elevated arterial blood pressure, diabetes mellitus, dyslipidaemia, and impaired renal function—form the foundational bedrock of international clinical risk scoring tools such as the Pooled Cohort Equations (Goff et al., 2014), SCORE2 (SCORE2 Working Group, 2021), and the AHA PREVENT equations (Khan et al., 2024).

Beyond standard lipid profiles, biological evidence demonstrates that atherogenesis is fundamentally an inflammatory pathology of the vascular wall (Libby, 2002). The landmark CANTOS clinical trial confirmed that targeted anti-inflammatory therapy with canakinumab reduces recurrent cardiovascular events without lowering circulating cholesterol, providing definitive causal evidence for the inflammatory hypothesis of coronary disease (Ridker et al., 2017).

Consequently, circulating inflammatory biomarkers have generated substantial interest for cardiovascular risk assessment. High-sensitivity C-reactive protein (hs-CRP), an acute-phase reactant synthesized by hepatocytes under interleukin-6 stimulation, correlates with coronary events in large cohort studies (Emerging Risk Factors Collaboration, 2010). However, large-scale prospective meta-analyses indicated that hs-CRP adds minimal discrimination beyond conventional risk scores ($\Delta\text{C-index} = +0.0039$; ERFC, 2012), and Mendelian randomization studies have failed to demonstrate a causal role, suggesting that hs-CRP acts as a reactive marker rather than a direct vascular atherogen (C Reactive Protein CHD Genetics Collaboration, 2011). In parallel, the neutrophil-to-lymphocyte ratio (NLR)—a simple leukocyte index combining the innate active inflammatory response (neutrophils) and adaptive immunological regulatory homeostasis (lymphocytes)—has emerged as a readily accessible predictor of incident atherothrombotic events in randomized trial cohorts (Adamstein et al., 2021). Its diagnostic utility in broad, unselected general population surveys remains to be fully clarified.

A crucial methodological question in contemporary clinical data science is whether machine learning (ML) models can extract predictive value from inflammatory biomarkers that classical statistical regressions miss. Modern machine learning algorithms (such as support vector machines, decision trees, and ensemble forests) can theoretically capture subtle non-linear thresholds and complex biological interactions. However, empirical studies often report overly optimistic conclusions due to methodological flaws, including label leakage, biased feature selection, and the fragility of single train-test splits. Indeed, as demonstrated in our diagnostic experiments, the performance of identical models on the same dataset can fluctuate between AUC 0.80 and 0.92 across 200 random splits simply due to partition variability.

### Research Questions and Pre-specified Hypotheses
This study addresses two primary epidemiological hypotheses and one secondary machine learning hypothesis using the NHANES 2021–2023 survey:

* **Primary Association Hypotheses:**
  * **H1a:** Higher NLR is independently associated with prevalent CHD/MI after multivariable adjustment for established clinical risk factors ($\alpha = 0.025$).
  * **H1b:** Higher hs-CRP is independently associated with prevalent CHD/MI after multivariable adjustment for established clinical risk factors ($\alpha = 0.025$).
* **Secondary Discrimination Hypothesis:**
  * **H1c:** Adding NLR and hs-CRP significantly improves the discriminative performance of established risk factors ($\Delta\text{AUC} \ge 0.01$), and complex machine learning algorithms outperform standard logistic regression.

Because NHANES is a cross-sectional survey without prospective follow-up, our outcome is *prevalent* rather than incident disease. Therefore, this study investigates statistical association rather than prospective causal prediction. Testing hypothesis H1c represents a necessary preliminary condition: a biomarker that fails to discriminate existing coronary disease is highly unlikely to justify clinical implementation for prospective primary prevention.

---

## 3. Methods

### 3.1 Data Source and Study Population
Data were derived from the National Health and Nutrition Examination Survey (NHANES) 2021–2023, administered by the National Center for Health Statistics (NCHS) of the Centers for Disease Control and Prevention (CDC). NHANES uses a stratified, multistage probability cluster design to produce a representative sample of the civilian, non-institutionalized US population. Protocols were approved by the NCHS Research Ethics Review Board, and all participants provided documented informed consent.

The course raw dataset contained 11,933 participants and 763 variables, with deliberately injected data corruption. The study cohort was established following strict protocol exclusion criteria:
1. Restriction to adult participants aged $\ge 20$ years ($n = 7,809$), as cardiovascular medical history questionnaires are administered only from age 20.
2. Complete responses to doctor-diagnosed coronary heart disease and heart attack questions ($n = 7,773$).
3. Attendance at the Mobile Examination Center (MEC) examination ($n = 6,037$).
4. Complete laboratory measurements for both inflammatory biomarkers (NLR and hs-CRP), yielding the final analytic sample of **$n = 5,538$ participants** containing **412 cases of CHD/MI (7.4% prevalence)**.

```
Total NHANES Enrolled (N = 11,933)
       │
       ▼  Excluded age < 20 years (n = 4,124)
Adult Participants (n = 7,809; 559 cases)
       │
       ▼  Excluded missing CHD/MI responses (n = 36)
Answered Medical Questionnaire (n = 7,773; 559 cases)
       │
       ▼  Excluded non-attendance at MEC (n = 1,736)
Examined in Mobile Center (n = 6,037; 451 cases)
       │
       ▼  Excluded missing NLR or hs-CRP (n = 499)
Final Analytic Sample (N = 5,538; 412 cases, 7.4% prevalence)
```
*Figure 1. Sequential participant flow diagram for the analytic cohort.*

### 3.2 Variable Definitions & Preprocessing

**Primary Outcome.** Prevalent coronary heart disease was defined as an affirmative self-reported doctor diagnosis of coronary heart disease or myocardial infarction (`MCQ160C == 1` or `MCQ160E == 1`), yielding 412 cases. Sensitivity definitions included isolated myocardial infarction ($n = 237$), isolated CHD ($n = 297$), and a broader composite incorporating angina pectoris (`MCQ160D == 1`, $n = 457$).

**Inflammatory Biomarkers.** Serum hs-CRP was measured by immunoturbidimetry (mg/L). Complete blood counts provided automated differential percentages; NLR was calculated as the ratio of neutrophils to lymphocytes (`LBXNEPCT / LBXLYPCT`). Both raw biomarker distributions exhibited extreme positive skewness (skewness 4.66 for NLR, 7.78 for hs-CRP). Natural log-transformation normalized distributions (skewness reduced to 0.15 and 0.23, respectively; verified by Q-Q plots). Biomarker effects are reported per 1 standard deviation of the log-transformed marker (corresponding to multiplicative factors of 1.58 for NLR and 3.15 for hs-CRP on the original measurement scale).

**Established Cardiovascular Risk Factors.** Covariates were chosen strictly from established risk scoring systems (PCE, SCORE2, PREVENT, QRISK3, INTERHEART): age, sex, race/ethnicity, family income-to-poverty ratio (PIR), body mass index (BMI), smoking status (never, former, current), doctor-diagnosed diabetes, hypertension (diagnosed or measured $\text{SBP} \ge 140$ or $\text{DBP} \ge 90$ mmHg), total cholesterol, HDL cholesterol, and renal function via estimated glomerular filtration rate (eGFR). eGFR was calculated using the official race-free 2021 CKD-EPI creatinine equation (Inker et al., 2021). Post-diagnostic treatments (statins, anti-hypertensives) were omitted from the primary baseline model to prevent post-treatment collider bias.

### 3.3 Methodological Decision Protocol (Experiments E1–E4)

All analytical decisions evaluated three competing options against pre-specified decision rules:

* **Experiment E1: Missing Data Strategy.** Covariates exhibited low missingness ($< 3\%$) with the exception of income PIR (12.7% missing). A logistic model for the missingness was consistent with Missing at Random (MAR; missingness was associated with ethnicity and education, but not with heart disease: $\text{OR} = 1.03, p = 0.87$). We compared:
  * *Option A:* Listwise deletion (complete cases, dropping 975 patients / 70 cases).
  * *Option B:* Dropping the income variable.
  * *Option C:* Median imputation with a dedicated binary missingness indicator (`income_missing`).
  * *Decision Rule:* Select Option C unless Option A or B improves out-of-fold AUC by $> 0.005$ on identical complete test cases.
  * *Outcome:* AUC on the same complete-case test persons was identical across all three strategies (logistic regression 0.867 vs. 0.867 vs. 0.867; random forest 0.865 vs. 0.866 vs. 0.868). Option C was chosen, preserving the full cohort ($n = 5,538$) and maintaining valid risk estimates for patients with missing demographic data.

* **Experiment E2: Feature Selection & Label Leakage Check.** We compared:
  * *Strategy S1:* Literature-guided clinical risk set (M2/M3).
  * *Strategy S2:* Naive Lasso regression across 342 raw survey variables.
  * *Strategy S3:* Hybrid domain-screened pool (39 candidate variables) with lasso selection.
  * *Outcome:* Strategy S2 achieved an apparent AUC of 0.923, but inspectable coefficients revealed severe **label leakage**: the algorithm selected post-diagnostic clinical consequences ("metal objects in body" [coronary stents], "advised to take aspirin", "congestive heart failure", "statin prescriptions"). 35% of its 20 strongest variables were consequences or treatment of the disease, and the selection was unstable (about 114 variables per fold, Jaccard similarity 0.47). Strategy S3 gained only +0.007 (AUC 0.870 vs. 0.862) with about 35 variables. Strategy S1 avoided circular post-diagnostic confounding and was selected.

* **Experiment E3: Class Imbalance Handling (7.4% Case Prevalence).** We tested: (1) natural prevalence, (2) balanced class weights, (3) random oversampling, (4) random undersampling, and (5) SMOTE.
  * *Outcome:* Resampling techniques produced negligible changes in ranking discrimination (AUC 0.856–0.863), but severely degraded probability calibration (Brier score doubled from 0.058 to 0.156; mean predicted risk inflated from 7.4% to 32%). Resampling was rejected for tree and linear models; natural prevalence was preserved, and decision thresholds were optimized post-hoc via Youden's $J$ statistic. Only the RBF support vector machine received balanced class weights, as unweighted SVMs collapsed towards the trivial majority class due to margin penalties.

* **Experiment E4: Evaluation Design & Split Variability.** Simulating 200 random 80/20 train/test splits revealed severe partition instability: test AUC fluctuated between 0.802 and 0.923 (middle 95% of splits: 0.831–0.894), and 29.5% of splits spuriously suggested negative marker utility. Consequently, repeated nested cross-validation was adopted as mandatory.

### 3.4 Machine Learning Modeling Zoo & Nested Cross-Validation

To test whether non-linear algorithms extract clinical utility from inflammatory biomarkers, six algorithms were trained:
1. **Unpenalized Logistic Regression (LR):** Classical reference baseline.
2. **Penalized Logistic Regression (Elastic Net, EN):** Tuning regularizer $C \in [0.001, 1.0]$ and mixing ratio $\ell_1 \in [0, 1.0]$ via the SAGA solver.
3. **CART Decision Tree (DT):** Tuning maximum tree depth (3–8) and minimum leaf size (20–100).
4. **$k$-Nearest Neighbors (kNN):** Tuning $k \in [50, 800]$, distance metrics (Manhattan, Euclidean), and uniform vs. distance weighting.
5. **Support Vector Machine (SVM):** Radial basis function (RBF) kernel tuning $C \in [0.03, 1.0]$ and $\gamma \in [0.002, 0.02]$, calibrated via Platt sigmoid scaling.
6. **Random Forest (RF):** 300 bootstrap trees tuning feature sampling (`max_features` $\in [\sqrt{p}, 0.33]$) and leaf size (10–30).

**Nested Validation Architecture.** We implemented a 5-fold stratified cross-validation repeated 3 times (15 outer test folds). Within each outer training split (80% of data), an inner 5-fold cross-validation optimized hyperparameters using negative log-loss (rewarding ranking and probability calibration simultaneously). Preprocessing transformations (median imputation, standardization, one-hot encoding) were encapsulated within Scikit-Learn pipelines and fitted strictly on outer training folds.

**Statistical Hypothesis Testing.** Pairwise model comparisons across the 15 outer folds were evaluated using the Nadeau & Bengio (2003) corrected resampled paired $t$-test:
$$\text{Var}(\bar{d}) = s^2 \left( \frac{1}{J} + \frac{n_{\text{test}}}{n_{\text{train}}} \right) = s^2 \left( \frac{1}{15} + \frac{1}{4} \right)$$
Adjustments for multiple comparisons across algorithms utilized the family-wise Bonferroni correction ($m = 6$).

---

## 4. Results

### 4.1 Study Cohort Characteristics (Table 1)

Among 5,538 participants, individuals with CHD/MI were significantly older (median 70 vs. 55 years, $\text{SMD} = 1.14, p < 10^{-78}$), more frequently male (64.6% vs. 43.5%, $\text{SMD} = 0.43$), and had higher prevalences of diabetes (37.4% vs. 12.2%, $\text{SMD} = 0.61$) and hypertension (78.2% vs. 42.2%, $\text{SMD} = 0.79$; Table 1).

**NLR was elevated in CHD/MI cases** (median 2.39 [IQR: 1.84, 3.29] vs. 1.96 [IQR: 1.49, 2.65], $\text{SMD} = 0.42$, Bonferroni-adjusted $p < 10^{-17}$). In sharp contrast, **hs-CRP showed virtually no difference between cases and controls** even unadjusted (median 1.97 [IQR: 0.90, 4.48] vs. 1.81 [IQR: 0.81, 4.18] mg/L, $\text{SMD} = 0.02$, Bonferroni-adjusted $p = 1.00$).

A key methodological finding was the **statin treatment paradox**: cases displayed paradoxically *lower* total cholesterol than non-cases (151.5 vs. 187.0 mg/dL, $\text{SMD} = -0.72, p < 10^{-48}$) because 78.2% of cases were receiving statin pharmacotherapy compared to 24.9% of controls.

**Table 1. Baseline Characteristics of the Analytic Cohort (NHANES 2021–2023).**

| Characteristic | Controls ($n = 5,126$) | Cases ($n = 412$) | SMD | Raw $p$-value | Bonferroni $p$ |
|---|---|---|---|---|---|
| Age, years | 55.0 [38.0, 67.0] | 70.0 [63.0, 77.2] | +1.14 | $6.4 \times 10^{-80}$ | $< 0.001$ |
| Male sex, $n$ (%) | 2,228 (43.5%) | 266 (64.6%) | +0.43 | $1.9 \times 10^{-16}$ | $< 0.001$ |
| Non-Hispanic White, $n$ (%) | 3,016 (58.8%) | 286 (69.4%) | +0.22 | $3.2 \times 10^{-5}$ | $< 0.001$ |
| Income PIR | 2.94 [1.48, 5.00] | 2.46 [1.36, 4.21] | −0.18 | $7.5 \times 10^{-4}$ | $0.010$ |
| Body Mass Index, kg/m² | 28.3 [24.6, 33.5] | 29.5 [25.8, 34.4] | +0.16 | $2.8 \times 10^{-4}$ | $0.004$ |
| Current cigarette smoker, $n$ (%) | 721 (14.1%) | 80 (19.4%) | +0.14 | $3.9 \times 10^{-3}$ | $0.055$ |
| Doctor-diagnosed diabetes, $n$ (%) | 624 (12.2%) | 154 (37.4%) | +0.61 | $4.3 \times 10^{-45}$ | $< 0.001$ |
| Hypertension, $n$ (%) | 2,119 (42.2%) | 320 (78.2%) | +0.79 | $8.7 \times 10^{-45}$ | $< 0.001$ |
| Total cholesterol, mg/dL | 187.0 [162.0, 216.0] | 151.5 [131.0, 181.2] | −0.72 | $1.3 \times 10^{-50}$ | $< 0.001$ |
| HDL cholesterol, mg/dL | 53.0 [44.0, 63.0] | 48.0 [41.0, 58.0] | −0.34 | $2.7 \times 10^{-10}$ | $< 0.001$ |
| eGFR, mL/min/1.73 m² | 94.5 [79.5, 107.1] | 74.8 [57.0, 93.0] | −0.86 | $1.5 \times 10^{-52}$ | $< 0.001$ |
| Statin medication, $n$ (%) | 1,274 (24.9%) | 316 (78.2%) | +1.26 | $4.1 \times 10^{-114}$ | $< 0.001$ |
| **Neutrophil-to-Lymphocyte Ratio (NLR)** | **1.96 [1.49, 2.65]** | **2.39 [1.84, 3.29]** | **+0.42** | **$2.6 \times 10^{-19}$** | **$< 0.001$** |
| **hs-CRP, mg/L** | **1.81 [0.81, 4.18]** | **1.97 [0.90, 4.48]** | **+0.02** | **$0.256$** | **$1.000$** |

*Values are median [IQR] for continuous variables and n (%) for categorical variables. $p$-values calculated via Mann-Whitney U test (continuous) and Pearson Chi-Square test (categorical). Bonferroni correction applied across all 14 baseline characteristics. SMD = difference of the means divided by the square root of the mean of the two group variances.*

### 4.2 Epidemiological Association Analysis (H1a & H1b)

In the primary multivariable model (Model 2), **NLR remained independently associated with prevalent CHD/MI** ($\text{OR} = 1.20$ per 1 SD of $\ln(\text{NLR})$, 95% CI: 1.07–1.34, $p = 0.002 < 0.025$; **H1a supported**; Table 2). Likelihood ratio testing confirmed that adding NLR provided significant statistical information ($\chi^2 = 9.83, df = 1, p = 0.002$).

Conversely, **hs-CRP showed no independent association with CHD/MI** ($\text{OR} = 0.96$ per 1 SD of $\ln(\text{hs-CRP})$, 95% CI: 0.84–1.08, $p = 0.48$; **H1b not supported**; LR-test $\chi^2 = 0.50, df = 1, p = 0.48$).

**Table 2. Multivariable Logistic Regression Association Models (Odds Ratios per 1 SD of Log-Marker).**

| Model Specification | NLR Odds Ratio (95% CI) | NLR $p$-value | hs-CRP Odds Ratio (95% CI) | hs-CRP $p$-value |
|---|---|---|---|---|
| Model 1: Demographics (Age, Sex) | 1.30 (1.17–1.44) | $< 0.001$ | 1.11 (0.99–1.24) | $0.066$ |
| **Model 2: Primary (+ Established Risk Factors)** | **1.20 (1.07–1.34)** | **$0.002$** | **0.96 (0.84–1.08)** | **$0.480$** |
| Model 3: Sensitivity (+ Statin Medication) | 1.21 (1.08–1.36) | $0.001$ | 0.99 (0.87–1.13) | $0.920$ |
| Model 2 Complete-Case Cohort ($n = 4,563$) | 1.17 (1.03–1.32) | $0.015$ | 0.96 (0.83–1.11) | $0.574$ |
| Model 2 Survey-Weighted (US Population) | 1.18 (1.05–1.33) | $0.008$ | 0.91 (0.81–1.02) | $0.100$ |

*Significance evaluated at Bonferroni-adjusted $\alpha = 0.025$ for co-primary exposures.*

**Confounder Decomposition.** Stepwise covariate addition revealed the mechanism behind the disappearance of the crude hs-CRP association: adding **BMI alone** to Model 1 caused the hs-CRP odds ratio to collapse from 1.11 to **1.00 (95% CI: 0.88–1.13)**. Circulating hs-CRP predominantly reflects excess adipose mass. Adding statin therapy strengthened the association slightly (OR 1.15), confirming that post-diagnostic treatment partially masks inflammatory levels. For NLR, no single covariate accounted for the relationship; the OR adjusted gradually from 1.30 to 1.20 across all risk factors.

Multicollinearity was ruled out (all variance inflation factors $< 2.2$). Non-linearity checks (quadratic terms: $p = 0.67$ for NLR, $p = 0.79$ for hs-CRP) and interaction terms with statin therapy ($p = 0.95$) or sex ($p = 0.22$) were non-significant. In the full multivariable model, the strongest risk factors were age ($\text{OR} = 2.55$ per SD), current smoking ($\text{OR} = 2.70$), hypertension ($\text{OR} = 2.05$), diabetes ($\text{OR} = 1.79$), and male sex ($\text{OR} = 1.74$). Total cholesterol had an inverse association ($\text{OR} = 0.58$ per SD), directly reflecting statin treatment.

### 4.3 Machine Learning Discrimination Performance (H1c)

In the 15-fold outer cross-validation, established clinical risk factors provided strong baseline discrimination (logistic regression M2 ROC-AUC = 0.861, 95% CI: 0.839–0.884; Table 3). Demographic variables alone (age and sex, M0) achieved an AUC of 0.801.

In contrast, **the inflammatory biomarkers alone (M4) demonstrated weak discrimination (AUC = 0.631)**. This limited signal was driven entirely by NLR ($\text{AUC} = 0.633$); hs-CRP alone discriminated no better than chance ($\text{AUC} = 0.506$, 95% CI: 0.462–0.550).

**Table 3. Repeated Nested Cross-Validation Performance and Added Marker Gain Across Six Algorithms.**

| Algorithm | Baseline M2 (AUC) | With Markers M3 (AUC) | $\Delta\text{AUC}$ M3 − M2 (Corrected 95% CI) | $\Delta\text{Brier}$ Score | Bonferroni $p$-value |
|---|---|---|---|---|---|
| **Logistic Regression (LR)** | 0.861 | 0.863 | **+0.0013** (−0.0009 to +0.0035) | −0.0002 | $1.000$ |
| Penalized Logistic Regression (EN) | 0.862 | 0.863 | **+0.0015** (−0.0008 to +0.0038) | −0.0002 | $1.000$ |
| CART Decision Tree (DT) | 0.808 | 0.805 | **−0.0030** (−0.0123 to +0.0064) | −0.0001 | $1.000$ |
| $k$-Nearest Neighbors (kNN) | 0.859 | 0.863 | **+0.0039** (−0.0067 to +0.0144) | +0.0000 | $1.000$ |
| Support Vector Machine (RBF-SVM) | 0.867 | 0.867 | **+0.0003** (−0.0025 to +0.0030) | −0.0003 | $1.000$ |
| Random Forest (RF) | 0.856 | 0.859 | **+0.0022** (−0.0055 to +0.0099) | −0.0003 | $1.000$ |

*Reported metrics are means of 15 outer test folds. $p$-values and 95% confidence intervals calculated via Nadeau & Bengio (2003) corrected paired $t$-tests with Bonferroni multiplicity adjustment.*

Adding both inflammatory biomarkers to established risk factors yielded negligible discriminative gain across all six algorithms ($\Delta\text{AUC} = +0.0013$ for logistic regression; all Bonferroni-corrected $p = 1.000$; **H1c not supported**). For five of the six algorithms, the upper bound of the corrected 95% confidence interval excluded a gain of 0.010, confirming that an incremental benefit of clinical relevance can be ruled out with statistical confidence.

Furthermore, **no complex machine learning algorithm significantly outperformed unpenalized logistic regression** on the full predictor set M3 (SVM vs. LR: $\Delta\text{AUC} = +0.0041$, 95% CI: −0.0017 to +0.0099, Bonferroni $p = 0.74$; Random Forest vs. LR: $\Delta\text{AUC} = -0.0041, p = 1.00$). The CART decision tree performed significantly worse ($\Delta\text{AUC} = -0.0575, p = 0.002$).

At the optimal training-fold Youden threshold, logistic regression M3 identified 82.7% of CHD/MI cases at 75.0% specificity (Positive Predictive Value: 21.1%, Negative Predictive Value: 98.2%), which was virtually identical to baseline M2 (82.2% sensitivity, 75.1% specificity). Calibration was excellent for logistic regression, Elastic Net, and calibrated SVM, whereas Random Forest and kNN compressed probability predictions towards the center. Learning curves indicated performance plateaued beyond 2,500 training samples.

### 4.4 Model Interpretability & Sensitivity Analyses

Permutation feature importance across outer test folds gave similar predictive hierarchies across all algorithms: age was by far the most important feature (AUC drop 0.06–0.10, 0.18 for the decision tree), followed by total cholesterol (treatment artifact) and hypertension. NLR ranked between 6th and 9th among the 13 predictors; hs-CRP ranked last (13th of 13) in five of the six algorithms. Partial dependence plots showed predicted risk increasing monotonically with NLR (logistic regression: from 5.4% to 8.8% between the 5th and 95th percentile) and remaining flat to slightly decreasing across hs-CRP percentiles (8.0% to 6.5%). An optimal depth-3 decision tree split exclusively on age, diabetes, cholesterol, and renal function, omitting both inflammatory markers entirely.

Five pre-specified sensitivity analyses each changed one decision: complete cases instead of imputation ($n = 4,563$), adjustment for statins (Model 3), exclusion of possible acute inflammation (hs-CRP $> 10$ mg/L, $n = 5,132$), survey weights with design-based standard errors, and an outcome including angina pectoris (457 cases). The adjusted OR of NLR remained stable between 1.14 and 1.21 and significant at $\alpha = 0.025$ in all five, while hs-CRP remained non-significant (OR 0.91 to 0.99). The incremental $\Delta\text{AUC}$ of logistic regression stayed below 0.002 in all prediction variants (statins entered as an additional predictor instead of the survey weights).

---

## 5. Discussion

### 5.1 Principal Findings
In a representative nationwide sample of 5,538 US adults, higher neutrophil-to-lymphocyte ratio (NLR) demonstrated a modest, statistically significant association with prevalent coronary heart disease ($\text{OR} = 1.20$ per SD, $p = 0.002$) that was independent of established cardiovascular risk factors. In contrast, high-sensitivity C-reactive protein (hs-CRP) showed no independent association ($\text{OR} = 0.96, p = 0.48$), with its unadjusted relationship entirely confounded by body mass index. Crucially, neither biomarker incrementally enhanced individual risk discrimination beyond conventional risk factors ($\Delta\text{AUC} = +0.0013$), and flexible machine learning algorithms failed to outperform standard logistic regression.

### 5.2 Decoupling Epidemiological Association from Clinical Discrimination
A central finding of this investigation is the apparent discordance between statistical association and clinical discrimination: NLR was highly significant in multivariable regression ($p = 0.002$), yet added negligible value to risk classification ($\Delta\text{AUC} = +0.001$).

This discrepancy is well-grounded in statistical theory (Pepe et al., 2004). An odds ratio of 1.20 per standard deviation indicates that systemic neutrophil activation is biologically correlated with coronary disease, shifting the risk distributions between diseased and non-diseased individuals. However, as demonstrated by Pepe and colleagues, a biomarker requires an odds ratio of at least 3.0 (and frequently $> 5.0$) to meaningfully alter individual risk ranking and classification metrics. Because conventional clinical risk factors (primarily age, sex, blood pressure, and smoking) already achieve an AUC of 0.861, the incremental variance explained by a marker with an OR of 1.20 is insufficient to reclassify patients across decision thresholds. The likelihood-ratio test and the area under the ROC curve answer fundamentally different clinical questions: likelihood-ratio tests evaluate whether a biomarker contributes non-redundant information to the model parameters, whereas ROC-AUC assesses whether that information is potent enough to separate individuals who have the disease from those who do not.

### 5.3 Biological and Clinical Context
Our findings regarding hs-CRP align closely with large prospective epidemiological consortia and genetic studies. The Emerging Risk Factors Collaboration (2012) demonstrated that adding hs-CRP to conventional risk factors yielded an almost imperceptible C-index gain of 0.0039 across 52 prospective cohorts. Furthermore, Mendelian randomization studies have conclusively shown that genetically elevated CRP levels do not confer increased coronary risk, demonstrating that hepatic CRP synthesis is a downstream reactive phenomenon rather than an etiological driver (CCGC, 2011). Our confounder decomposition provides empirical insight into why observational studies frequently report crude CRP associations: adjusting for BMI alone completely eliminated the association, demonstrating that in general cross-sectional populations, hs-CRP serves primarily as a surrogate marker of excess visceral adiposity.

Conversely, NLR captures the balance between active innate inflammation (granulocyte activation) and adaptive immune senescence (lymphocyte depletion). Our finding of an independent NLR association aligns with clinical trial analyses (Adamstein et al., 2021). However, our machine learning evaluation confirms that this biological relationship does not translate into improved diagnostic discrimination in the general population.

### 5.4 Why Complex Machine Learning Did Not Outperform Logistic Regression
Proponents of machine learning frequently hypothesize that complex algorithms can discover intricate interactions or non-linear risk thresholds that linear regressions miss. In our benchmark, the RBF support vector machine, random forest, and Elastic Net failed to achieve statistically significant gains over unpenalized logistic regression.

This empirical result is consistent with the comprehensive systematic review by Christodoulou et al. (2019), which analyzed 71 clinical prediction studies and found no performance benefit of machine learning over logistic regression. When a clinical problem features moderate dimensional predictor spaces (13 established variables), predominantly monotonic risk trajectories, and well-characterized biological risk factors, flexible non-linear algorithms incur excess estimation variance without reducing model bias.

Our SVM experiment also highlights a critical lesson in machine learning diagnostics: in an unweighted initial run, the SVM collapsed toward predicting the majority class, achieving an apparent marker gain of $+0.008$. This spurious gain occurred purely because an uncalibrated, poorly performing model benefits indiscriminately from any additional input signal. Once class weights and proper Platt scaling were applied, the SVM marker gain dropped to $+0.0003$. Methodological artifacts can easily masquerade as scientific discoveries if rigorous calibration checks are omitted.

### 5.5 Strengths and Limitations

**Strengths.** This investigation adhered to a pre-specified decision protocol where every analytical choice compared three explicit options. We eliminated data leakage by executing all imputation, standardization, and tuning strictly within cross-validation folds. We exposed and avoided catastrophic label leakage in naive feature selection (post-diagnostic treatments and interventions). We accounted for partition variance through 15-fold repeated nested cross-validation and evaluated statistical significance using the Nadeau & Bengio resampled paired $t$-test. Finally, we confirmed our findings in five sensitivity analyses, each tied to one analytical decision.

**Limitations.**
1. *Cross-sectional study design:* Exposure and outcome were ascertained simultaneously. Consequently, reverse causality cannot be excluded: acute myocardial necrosis or post-infarction remodeling may elevate NLR, while post-diagnostic lifestyle modifications or medications alter biomarker concentrations. Prospective cohort designs are required to evaluate incident cardiovascular event prediction.
2. *Self-reported outcome:* The primary outcome relied on participant recall of physician diagnoses, which may omit undiagnosed subclinical coronary atherosclerosis or introduce non-differential misclassification.
3. *Treatment confounding:* Widespread statin therapy among cases (78.2%) created an inverted cholesterol relationship (total cholesterol $\text{OR} = 0.58$), although sensitivity analyses without cholesterol confirmed that marker findings were unaffected.
4. *Single biomarker assessment:* Biomarkers were measured from a single baseline blood draw, introducing within-person biological variability and regression dilution bias.
5. *Internal validation:* While cross-validation was rigorous, validation on external geographic cohorts was not performed.

---

## 6. Conclusion

In a nationally representative sample of US adults, the neutrophil-to-lymphocyte ratio (NLR) is modestly and independently associated with prevalent coronary heart disease, whereas high-sensitivity C-reactive protein (hs-CRP) is non-significant after accounting for body mass index. However, neither biomarker incrementally improves individual risk discrimination beyond established cardiovascular risk factors, and complex machine learning algorithms provide no discriminative advantage over classical logistic regression. These findings demonstrate that statistical association does not equate to clinical utility, and argue against the routine addition of these inflammatory biomarkers to standard cardiovascular risk screening tools.

---

## References

1. Adamstein NH, et al. The neutrophil–lymphocyte ratio and incident atherosclerotic events: analyses from five contemporary randomized trials. *European Heart Journal*, 2021;42(9):896–903.
2. C Reactive Protein Coronary Heart Disease Genetics Collaboration (CCGC). Association between C reactive protein and coronary heart disease: Mendelian randomisation analysis based on individual participant data. *BMJ*, 2011;342:d548.
3. Christodoulou E, et al. A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. *Journal of Clinical Epidemiology*, 2019;110:12–22.
4. Emerging Risk Factors Collaboration. C-reactive protein concentration and risk of coronary heart disease, stroke, and mortality: an individual participant meta-analysis. *The Lancet*, 2010;375(9709):132–140.
5. Emerging Risk Factors Collaboration. C-reactive protein, fibrinogen, and cardiovascular disease prediction. *New England Journal of Medicine*, 2012;367(14):1310–1320.
6. Goff DC Jr, et al. 2013 ACC/AHA guideline on the assessment of cardiovascular risk: a report of the American College of Cardiology/American Heart Association Task Force on Practice Guidelines. *Circulation*, 2014;129(25 Suppl 2):S49–S73.
7. Inker LA, et al. New creatinine- and cystatin C–based equations to estimate GFR without race. *New England Journal of Medicine*, 2021;385(19):1737–1749.
8. Khan SS, et al. Novel prediction equations for absolute risk assessment of total cardiovascular disease: The American Heart Association PREVENT Equations. *Circulation*, 2024;149(6):430–449.
9. Libby P. Inflammation in atherosclerosis. *Nature*, 2002;420(6917):868–874.
10. Nadeau C, Bengio Y. Inference for the generalization error. *Machine Learning*, 2003;52(3):239–281.
11. Pepe MS, et al. Limitations of the odds ratio in gauging the performance of a diagnostic, prognostic, or screening marker. *American Journal of Epidemiology*, 2004;159(9):882–890.
12. Platt JC. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*, 1999;10(3):61–74.
13. Ridker PM, et al. Antiinflammatory therapy with canakinumab for atherosclerotic disease (CANTOS). *New England Journal of Medicine*, 2017;377(12):1119–1131.
14. SCORE2 Working Group and ESC Cardiovascular Risk Collaboration. SCORE2 risk prediction algorithms: new models to estimate 10-year risk of cardiovascular disease in Europe. *European Heart Journal*, 2021;42(25):2439–2454.
15. World Health Organization. The top 10 causes of death. Fact sheet, 2024. Available from: https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death.

---

## Appendix A. Algorithm Hyperparameter Tuning Grids

| Algorithm | Course Lecture | Hyperparameter Search Grid (Inner 5-Fold CV, Optimized on Log-Loss) |
|---|---|---|
| Unpenalized Logistic Regression | W6–W7 | Fixed parameters; no hyperparameters tuned |
| Penalized Logistic Regression (Elastic Net) | W5–W6 | $C \in \{0.001, 0.01, 0.03, 0.1, 0.3, 1.0\}$; $\ell_1\text{-ratio} \in \{0.0 \text{ (Ridge)}, 0.5, 1.0 \text{ (Lasso)}\}$ |
| CART Decision Tree | W7 | $\text{max\_depth} \in \{3, 4, 5, 6, 8\}$; $\text{min\_samples\_leaf} \in \{20, 50, 100\}$ |
| $k$-Nearest Neighbors | W6, W8 | $k \in \{50, 100, 200, 400, 800\}$; weights $\in \{\text{uniform}, \text{distance}\}$; distance metric $\in \{\text{Manhattan}, \text{Euclidean}\}$ |
| Support Vector Machine (RBF) | W9 | $C \in \{0.03, 0.1, 0.3, 1.0\}$; $\gamma \in \{0.002, 0.005, 0.02\}$; balanced class weights; 3-fold Platt sigmoid calibration |
| Random Forest (300 Trees) | W8 | $\text{max\_features} \in \{\sqrt{p}, 0.33 \times p\}$; $\text{min\_samples\_leaf} \in \{10, 30\}$ |

---

## Appendix B. Full Multivariable Primary Model Coefficients (Model 2 + Both Inflammatory Biomarkers)

| Variable | Estimated Log-Odds ($\beta$) | Standard Error | Odds Ratio (95% CI) | Wald $z$-statistic | $p$-value |
|---|---|---|---|---|---|
| **$\ln(\text{NLR})$ (per 1 SD = 1.58-fold)** | **+0.194** | **0.059** | **1.21 (1.08–1.36)** | **+3.30** | **$0.0010$** |
| **$\ln(\text{hs-CRP})$ (per 1 SD = 3.15-fold)** | **−0.083** | **0.065** | **0.92 (0.81–1.05)** | **−1.27** | **$0.2038$** |
| Age (per 1 SD = 17.1 years) | +0.936 | 0.106 | 2.55 (2.07–3.14) | +8.83 | $< 0.001$ |
| Male sex | +0.554 | 0.128 | 1.74 (1.35–2.23) | +4.33 | $< 0.001$ |
| Mexican American (vs. NH White) | −1.050 | 0.362 | 0.35 (0.17–0.70) | −2.90 | $0.003$ |
| Other Hispanic (vs. NH White) | −0.030 | 0.207 | 0.97 (0.65–1.46) | −0.15 | $0.885$ |
| Non-Hispanic Black (vs. NH White) | −0.798 | 0.221 | 0.45 (0.29–0.69) | −3.61 | $< 0.001$ |
| Non-Hispanic Asian (vs. NH White) | +0.300 | 0.283 | 1.35 (0.78–2.36) | +1.06 | $0.290$ |
| Other / Multiracial (vs. NH White) | −0.342 | 0.246 | 0.71 (0.44–1.15) | −1.39 | $0.165$ |
| Income PIR (per 1 SD = 1.50) | −0.162 | 0.065 | 0.85 (0.75–0.97) | −2.49 | $0.013$ |
| Income missing indicator (`income_missing`) | −0.010 | 0.171 | 0.99 (0.71–1.38) | −0.06 | $0.930$ |
| Body Mass Index (per 1 SD = 7.2 kg/m²) | +0.157 | 0.068 | 1.17 (1.02–1.33) | +2.33 | $0.020$ |
| Former smoker (vs. Never) | +0.542 | 0.129 | 1.72 (1.34–2.22) | +4.20 | $< 0.001$ |
| Current smoker (vs. Never) | +0.993 | 0.171 | 2.70 (1.93–3.78) | +5.81 | $< 0.001$ |
| Doctor-diagnosed diabetes | +0.582 | 0.130 | 1.79 (1.39–2.31) | +4.48 | $< 0.001$ |
| Hypertension | +0.718 | 0.136 | 2.05 (1.57–2.68) | +5.28 | $< 0.001$ |
| Total cholesterol (per 1 SD = 42.3 mg/dL) | −0.545 | 0.070 | 0.58 (0.51–0.67) | −7.79 | $< 0.001$ |
| HDL cholesterol (per 1 SD = 14.6 mg/dL) | −0.094 | 0.073 | 0.91 (0.79–1.05) | −1.29 | $0.200$ |
| eGFR (per 1 SD = 21.0 mL/min/1.73 m²) | −0.223 | 0.067 | 0.80 (0.71–0.92) | −3.33 | $0.001$ |
| Constant ($\beta_0$) | −4.568 | 0.261 | — | −17.50 | $< 0.001$ |

*Model evaluated on $n = 5,538$ participants (412 cases); log-likelihood = −1154.2, Likelihood Ratio $\chi^2 = 824.1 (df = 19), p < 10^{-150}$. Continuous covariates standardized to zero mean and unit variance.*
