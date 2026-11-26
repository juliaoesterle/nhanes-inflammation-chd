# Technische Methodik und Entscheidungsmatrix (Decision Records D1–D18)

**ETH Zürich — Foundations of Data Science (Machine Learning Project)**  
*Detaillierte mathematische Formeln, Preprocessing-Pipelines und vollständige Entscheidungsnachweise.*

---

## 1. Mathematische Grundlagen & Formeln

### 1.1 Race-Free CKD-EPI 2021 eGFR Formel
Zur Bestimmung der geschätzten glomerulären Filtrationsrate (eGFR) ohne Rassenanpassung (Inker et al., *NEJM* 2021):
$$\text{eGFR} = 142 \times \min\left(\frac{S_{\text{cr}}}{\kappa}, 1\right)^\alpha \times \max\left(\frac{S_{\text{cr}}}{\kappa}, 1\right)^{-1.200} \times 0.9938^{\text{Age}} \times [1.012 \text{ falls weiblich}]$$
Wobei:
* Männer: $\kappa = 0.9, \alpha = -0.302$
* Frauen: $\kappa = 0.7, \alpha = -0.241$
* $S_{\text{cr}}$ ist das Serumkreatinin in mg/dL.

### 1.2 Nadeau & Bengio (2003) Resampled Paired t-Test
Da die Folds in wiederholter Kreuzvalidierung (Repeated CV) zu 75% dieselben Trainingspunkte teilen, sind die Test-Scores korreliert. Die korrigierte Varianz der mittleren Differenz $\bar{d}$ berechnet sich nach:
$$\text{Var}(\bar{d}) = s^2 \left( \frac{1}{J} + \frac{n_{\text{test}}}{n_{\text{train}}} \right)$$
Wobei:
* $J = K \times R = 5 \times 3 = 15$ Folds.
* $n_{\text{test}} / n_{\text{train}} = 1 / 4 = 0.25$.
* Standardfehler: $\text{SE}_{\text{korr}} = \sqrt{\text{Var}(\bar{d})}$.
* Teststatistik: $t = \frac{\bar{d}}{\text{SE}_{\text{korr}}} \sim t(df = 14)$.
* $95\%$-Konfidenzintervall: $\bar{d} \pm t_{0.975, 14} \times \text{SE}_{\text{korr}}$.

### 1.3 Odds Ratio für logarithmierten Prädiktor
Für einen Prädiktor $X$, der als $Z = \frac{\ln(X) - \mu_{\ln}}{\sigma_{\ln}}$ standardisiert wurde, lautet das Odds Ratio bezogen auf eine Standardabweichung:
$$\text{OR}_{\text{SD}} = \exp(\beta_Z)$$
Eine Zunahme um 1 SD auf der Log-Skala entspricht einer Multiplikation der Originalvariablen um den Faktor:
$$\text{Faktor} = \exp(\sigma_{\ln})$$
* Für NLR: $\sigma_{\ln} = 0.457 \rightarrow \text{Faktor} = \exp(0.457) = \mathbf{1.58}$.
* Für hs-CRP: $\sigma_{\ln} = 1.147 \rightarrow \text{Faktor} = \exp(1.147) = \mathbf{3.15}$.

### 1.4 Likelihood Ratio Test (LR-Test)
Zum statistischen Vergleich zweier geschachtelter Modelle (Restricted $M_R$ vs. Full $M_F$):
$$\text{LR} = 2 \times \left( \ell(\hat{\theta}_F) - \ell(\hat{\theta}_R) \right) \sim \chi^2(df_F - df_R)$$
Wobei $\ell$ die maximale Log-Likelihood des Modells darstellt.

### 1.5 Brier Score & Kalibrierung
$$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
Der Brier-Score misst die Genauigkeit probabilistischer Vorhersagen. Er zerfällt mathematisch in Schärfe (Refinement) und Kalibrierung (Reliability).

---

## 2. Vollständige Entscheidungsmatrix (D1–D18)

| Record | Entscheidungspunkt | Option A | Option B | Option C (Gewählt) | Entscheidungsregel & Kriterium | Ergebnis & Evidenz |
|---|---|---|---|---|---|---|
| **D1** | Forschungsfrage & Ziel | Reine Zukunftsprognose | Reines ML-Screening | **Assoziation primär, Diskrimination sekundär** | Beantworte Querschnittsfrage valide nach W1/W4 | Option C machbar und methodisch sauber |
| **D2** | Primäres Outcome | Nur Herzinfarkt (MI) | CHD + MI + Angina | **Selbstberichtete CHD oder MI** | Konsistenz mit ACC/AHA Hard CVD Definition | 412 Fälle (7.4%), Angina in Sensitivitätsanalyse |
| **D3** | Studienkohorte | Alle Befragten ($n=11'933$) | Nur vollständige Fälle ($n=4'563$) | **Erwachsene mit MEC & Biomarkern ($n=5'538$)** | Maximale Fallzahl ohne künstliche Deletion | Behält 975 Personen und 70 echte Fälle |
| **D4** | Biomarker-Skala | Rohwerte | Kategorien ($<1, 1-3, >3$) | **Logarithmische Transformation $\ln(x)$** | Schiefe $\le 0.5$ und linearer Q-Q-Plot | Schiefe sinkt von 4.7/7.8 auf 0.15/0.23 |
| **D5** | Referenz-Feature-Set | Vollautomatischer Lasso | Unüberwachte PCA | **Etablierte Guideline-Risikofaktoren (M2)** | S1 behalten, ausser Alternative ohne Leakage und > 0.005 AUC besser (KI > 0) | M2 vermeidet Behandlungslecks; M3 ergänzt Marker |
| **D6** | Datenlecks bei Prädiktoren | Alle Variablen zulassen | Blindes Filtern | **Strikter Ausschluss von Post-Diagnose-Variablen** | Keine Variablen, die Folgen der Diagnose sind | Schliesst Stents, Aspirin-Rat und Medikamente aus |
| **D7** | Evaluierungsdesign | Einfacher 80/20 Split | Standard 5-Fold CV | **Repeated Nested CV (5x3 Outer, 5 Inner)** | Stabilität über zufällige Splits nachweisen | E4 zeigt: 80/20 schwankt von 0.80 bis 0.92 |
| **D8** | Fehlende Kovariaten (E1) | Listwise Deletion | Spalte Einkommen löschen | **Median-Imputation + Missing-Indikator** | Option C, wenn Deletion nicht um $>0.005$ AUC überlegen | AUC auf denselben Testpersonen gleich (LR 0.867 / 0.867 / 0.867); Option C erhält 70 Fälle |
| **D9** | Klassen-Ungleichgewicht (E3) | SMOTE / Resampling | Klassengewichtung | **Keine Resampling-Korrektur (Ausser SVM)** | Korrektur nur bei $\Delta\text{AUC}>0.005$ & Brier stabil | Resampling verdoppelt Brier-Fehler; SVM erhält Weights |
| **D10** | Optimierungsmetrik Inner CV | Accuracy | ROC-AUC | **Negative Log-Loss (Kreuzentropie)** | Belohne Rangordnung UND Kalibrierung | Verhindert die Wahl fehlkalibrierter Hyperparameter |
| **D11** | Modellauswahl (ML Zoo) | Nur 1 komplexes Modell | Nur Lineare Regression | **Alle 6 ETH-Kursverfahren (W5–W9)** | Vollständige Kursabdeckung | LR, Penalized, Tree, kNN, SVM, RF |
| **D12** | SVM-Kalibrierung | Raw Distance Scores | Keine Wahrscheinlichkeiten | **Platt-Skalierung (CalibratedClassifierCV)** | Zulässige Wahrscheinlichkeiten nach W9 | Erzeugt kalibrierte Wahrscheinlichkeiten für SVM |
| **D13** | Modellvergleichstest | Unkorrigierter t-Test | Wilcoxon Signed-Rank | **Nadeau-Bengio korrigierter Paired t-Test** | Berücksichtigung der 75% Trainingsüberlappung | Vermeidet Typ-I-Fehler-Inflation |
| **D14** | Multiples Testen | Unkorrigiert ($p < 0.05$) | FDR / Benjamini-Hochberg | **Bonferroni-Korrektur** | Strikte Kontrolle des FWER bei co-primären Tests | $\alpha = 0.025$ für H1a/H1b; $m=6$ für ML-Vergleiche |
| **D15** | Entscheidungs-Schwellenwert | Feste 0.5 Schwelle | Schwellenwert auf Testfold | **Youden-Schwelle auf Trainingsfold** | Keine Schwellenwert-Optimierung auf Testdaten | Youden-Index maximiert Sensitivität + Spezifität |
| **D16** | Confounder-Modellierung | Nur Gesamt-Adjustierung | Bivariate Tests | **Schrittweise Zerlegung (Decomposition)** | Mechanismus der Dämpfung aufdecken | Isoliert BMI als Ursache für Wegfall von hs-CRP |
| **D17** | Nicht-Linearität & Interaktion | Splines (Nicht im Kurs) | Nicht geprüft | **Quadratische Terme / Interaktion** | Kurskonforme Verfahren (W7) | Kein Hinweis auf Nicht-Linearität (p = 0.67 / 0.79) oder Modifikation (lipidsenkende Medikamente p = 0.95, Geschlecht p = 0.22) |
| **D18** | Empfohlenes Modell | Komplexestes Modell | Bestes Test-Fold Modell | **Logistische Regression (Sparsamkeitsprinzip)** | Komplexes ML nur, wenn signifikant überlegen | Kein ML-Modell signifikant besser als LR |

---

## 3. Pipeline-Architektur & Leakage-Prävention

Die Scikit-Learn-Pipeline ist strikt so konstruiert, dass keinerlei Testinformationen in den Trainingsprozess gelangen:

```
[Rohdaten der Trainingsfolds]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Scikit-Learn ColumnTransformer (Fit auf Train-Fold)   │
│                                                        │
│ 1. Stetige Merkmale:                                   │
│    SimpleImputer(strategy='median')                   │
│    StandardScaler() (Z-Transformation)                 │
│                                                        │
│ 2. Kategoriale Merkmale:                               │
│    SimpleImputer(strategy='most_frequent')            │
│    OneHotEncoder(handle_unknown='ignore')              │
│                                                        │
│ 3. Fehlende Indikatoren:                               │
│    MissingFlagTransformer() (erzeugt income_missing)   │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Klassifikator (GridSearchCV mit neg_log_loss)          │
│ Hyperparameter-Tuning ausschliesslich auf Train-Fold   │
└────────────────────────────────────────────────────────┘
       │
       ▼
[Auswertung auf unberührtem Test-Fold]
```

---

## 4. Sensitivitätsanalysen (`06_association_analysis.py`)

Fünf Varianten, jede gehört zu einer Entscheidung oben:

| Variante | prüft | NLR: OR (95%-KI), p | hs-CRP: OR (95%-KI), p | ΔAUC LR |
|---|---|---|---|---|
| Hauptanalyse (Modell 2) | – | 1.20 (1.07–1.34), 0.002 | 0.96 (0.84–1.08), 0.48 | +0.0013 |
| Complete Cases (n = 4'563) | D8 fehlende Werte | 1.17 (1.03–1.32), 0.015 | 0.96 (0.83–1.11), 0.57 | +0.0011 |
| Modell 3 (+ lipidsenkende Medikamente) bzw. als Prädiktor | D6 Behandlung | 1.21 (1.08–1.36), 0.001 | 0.99 (0.87–1.13), 0.92 | +0.0009 |
| Ohne hs-CRP > 10 mg/L | D4 Ausreisser / akute Entzündung | 1.19 (1.05–1.34), 0.005 | 0.96 (0.82–1.12), 0.63 | +0.0005 |
| Survey-gewichtet | Repräsentativität (W3) | 1.18 (1.05–1.33), 0.008 | 0.91 (0.81–1.02), 0.10 | – |
| Outcome mit Angina | D2 Outcome | 1.14 (1.02–1.27), 0.019 | 0.96 (0.85–1.08), 0.50 | +0.0002 |

NLR bleibt in allen Varianten signifikant (α = 0.025), hs-CRP in keiner; der Marker-Gewinn bleibt überall unter 0.002.
