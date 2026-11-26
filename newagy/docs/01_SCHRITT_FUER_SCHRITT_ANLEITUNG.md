# Schritt-für-Schritt Anleitung: Von den Rohdaten zum Ergebnis

**ETH Zürich — Foundations of Data Science (Herbstsemester)**  
*Leitfaden fürs Team: Methodik, 3-Optionen-Entscheidungslogik und typische Fragen. Der Code dazu liegt in `newagy/scripts/` (Teil 1 auch einfach im Notebook `Final/nhanes_chd_analysis.ipynb`).*

---

## Inhaltsverzeichnis

1. [Einleitung & Grundprinzip der wissenschaftlichen Arbeit](#1-einleitung--grundprinzip)
2. [Schritt 1: Die präzise Forschungsfrage (W1)](#schritt-1-die-präzise-forschungsfrage)
3. [Schritt 2: Datenbereinigung & Kohorten-Flussdiagramm (W2, W3)](#schritt-2-datenbereinigung--kohorten-flussdiagramm)
4. [Schritt 3: Feature Engineering & Skalierung der Biomarker (W2, W3)](#schritt-3-feature-engineering--skalierung-der-biomarker)
5. [Schritt 4: Deskriptive Statistik & Baseline Table 1 (W3, Tutorial 4)](#schritt-4-deskriptive-statistik--baseline-table-1)
6. [Schritt 5: Fehlende Werte — Das Experiment E1 (W2)](#schritt-5-fehlende-werte--das-experiment-e1)
7. [Schritt 6: Feature Selection & Datenlecks — Das Experiment E2 (W4, W5, W6)](#schritt-6-feature-selection--datenlecks--das-experiment-e2)
8. [Schritt 7: Stichprobenumfang & Events-Per-Parameter (W3)](#schritt-7-stichprobenumfang--events-per-parameter)
9. [Schritt 8: Seltene Krankheiten & Klassen-Ungleichgewicht — Das Experiment E3 (W9)](#schritt-8-seltene-krankheiten--klassen-ungleichgewicht--das-experiment-e3)
10. [Schritt 9: Die Evaluierungsstrategie — Das Experiment E4 (W4, W5)](#schritt-9-die-evaluierungsstrategie--das-experiment-e4)
11. [Schritt 10: Die sechs Machine Learning Algorithmen (W5–W9)](#schritt-10-die-sechs-machine-learning-algorithmen)
12. [Schritt 11: Metriken für ungleiche Klassen (W9)](#schritt-11-metriken-für-ungleiche-klassen)
13. [Schritt 12: Statistischer Modellvergleich via Nadeau-Bengio (W3, Tutorial 4)](#schritt-12-statistischer-modellvergleich-via-nadeau-bengio)
14. [Schritt 13: Beantwortung der Hauptfrage (H1a & H1b via Logistischer Regression)](#schritt-13-beantwortung-der-hauptfrage-h1a--h1b)
15. [Schritt 14: Beantwortung der Zusatzfrage (H1c: Diskrimination via ML)](#schritt-14-beantwortung-der-zusatzfrage-h1c-diskrimination-via-ml)
16. [Schritt 15: Sensitivitätsanalysen & Robustheitsprüfung](#schritt-15-sensitivitätsanalysen--robustheitsprüfung)
17. [Schritt 16: Modell-Interpretierbarkeit & Feature Attribution (W7, W8)](#schritt-16-modell-interpretierbarkeit--feature-attribution)
18. [Schritt 17: Typische Fragen & mögliche Antworten](#schritt-17-typische-fragen--mögliche-antworten)

---

## 1. Einleitung & Grundprinzip

Es genügt nicht, einfach ein Modell zu trainieren und den besten Test-Score abzugeben. Wichtig sind uns drei Dinge:

1. **Systematische Begründung jeder Entscheidung:** Für jeden Schritt formulieren wir **3 sinnvolle Optionen**, definieren **ein explizites Kriterium (Entscheidungsregel)** und begründen die Wahl transparent.
2. **Kritische Selbstreflexion:** Wir decken methodische Fallstricke (z. B. Datenlecks durch Medikamente, Überanpassung, Kalibrierungsfehler) aktiv auf, anstatt sie zu vertuschen.
3. **Strikte Trennung von Assoziation und Vorhersage:** Eine Querschnittsstudie (NHANES) kann Zusammenhänge bei bestehenden Krankheiten messen, aber keine Zukunftsprognose liefern.

---

## Schritt 1: Die präzise Forschungsfrage

### Leitfrage & Ziel
Welche wissenschaftliche Frage können wir mit den Querschnittsdaten der NHANES 2021–2023 überhaupt empirisch valide beantworten?

### Kursbezug
* **W1 (Research Questions):** Eine Forschungsfrage muss *fokussiert*, *empirisch überprüfbar*, *machbar (feasible)* und *relevant (meaningful)* sein.
* **W4-Checkliste:** Passt die ML-Methode zur klinischen Fragestellung?

### Die drei Optionen
* **Option A: Künftige Herzinfarkte vorhersagen.**  
  *Kritik:* Völlig unmöglich. NHANES ist eine Querschnittsstudie (Snapshot). Es gibt keine Nachbeobachtung über die nächsten 10 Jahre.
* **Option B: Ein ML-Modell bauen, das herzkranke Patienten "erkennt".**  
  *Kritik:* Klinisch wenig relevant. Die Patienten wissen bereits, dass sie einen Herzinfarkt hatten.
* **Option C (Unsere Wahl): Epidemiologische Assoziation als Hauptfrage, ML-Diskrimination als methodische Vorprüfung.**  
  *Begründung:* Wir prüfen, ob zwei günstige Entzündungsmarker (NLR und hs-CRP) unabhängig von etablierten Risikofaktoren mit koronarer Herzkrankheit (CHD/MI) zusammenhängen. Als Zusatzfrage prüfen wir, ob sie die Trennschärfe (Diskrimination) verbessern. Wenn sie nicht einmal bestehende Fälle besser trennen, lohnt sich kein Einsatz in teuren Screening-Programmen.

### Hypothesen & Signifikanzniveau
* **H1a:** NLR ist unabhängig von Risikofaktoren mit CHD/MI assoziiert.
* **H1b:** hs-CRP ist unabhängig von Risikofaktoren mit CHD/MI assoziiert.
* **Multiple Testing:** Da zwei co-primäre Hypothesen geprüft werden, gilt nach **Bonferroni (Tutorial 4)**: $\alpha = 0.05 / 2 = 0.025$.
* **H1c (Sekundär):** NLR und hs-CRP verbessern die Klassifikation über etablierte Faktoren hinaus ($\Delta\text{AUC} \ge 0.01$).

---

## Schritt 2: Datenbereinigung & Kohorten-Flussdiagramm

### Leitfrage & Ziel
Wer befindet sich in unserer Stichprobe, wer wird ausgeschlossen und führt der Ausschluss zu Verzerrungen (Selection Bias)?

### Kursbezug
* **W3 (Population & Sampling, Repräsentativität)**
* **W2-Checkliste:** *What is missing? Where? Why?*

### Der Weg zur Studienkohorte (`flowchart.csv`)
1. **Ausgangspunkt:** 11'933 Teilnehmende in NHANES 2021–2023.
2. **Einschluss Kriterium 1 (Alter):** Nur Erwachsene ($\ge 20$ Jahre), da Herzfragen erst ab 20 gestellt werden $\rightarrow$ 7'809 Personen (559 Fälle).
3. **Einschluss Kriterium 2 (Outcome beantwortet):** 7'773 Personen (559 Fälle).
4. **Einschluss Kriterium 3 (MEC-Untersuchung):** Im mobilen Untersuchungszentrum untersucht $\rightarrow$ 6'037 Personen (451 Fälle).
5. **Einschluss Kriterium 4 (Biomarker gemessen):** Beide Entzündungsmarker (NLR und hs-CRP) vorhanden $\rightarrow$ **5'538 Personen (412 Fälle, 7.4% Prävalenz)**.

### Prüfung auf Selektionsverzerrung (Selection Bias)
Wir vergleichen die 499 ausgeschlossenen Personen ohne Blutprobe mit unserer Analysekohorte anhand der **Standardisierten Mittelwertdifferenz (SMD)**.  
*Ergebnis:* Ausgeschlossene Personen haben ein geringeres Einkommen ($\text{SMD} = -0.36$) und sind häufiger Nicht-Hispanische Schwarze. Dies dokumentieren wir offen als Limitation der externen Generalisierbarkeit.

---

## Schritt 3: Feature Engineering & Skalierung der Biomarker

### Leitfrage & Ziel
Wie transformieren wir schiefe Laborwerte sinnvoll, damit Regressions- und ML-Modelle stabil rechnen können?

### Kursbezug
* **W2 (Data Wrangling, Ausreisser)**
* **W3 (Verteilungsformen, Schiefe, Q-Q-Plots, Normalitätstests)**

### Die drei Optionen für die Marker-Skala
| Option | Schiefe NLR / hs-CRP | Beurteilung |
|---|---|---|
| **Option A: Rohwerte** | 4.66 / 7.78 | Extrem rechtsschief. Wenige Maximalwerte verzerren Gradienten und Distanzen. |
| **Option B: Log-Transformation $\ln(x)$** | **0.15 / 0.23** | **Nahezu perfekte Symmetrie. Normalverteilungsnah im Q-Q-Plot.** |
| **Option C: Klinische Kategorien** | – | z. B. CRP $<1, 1-3, >3$ mg/L. Verliert kontinuierliche Information. |

### Entscheidung & Interpretation
Wir wählen **Option B** für alle Modelle. Effekte werden pro Standardabweichung (SD) des logarithmierten Markers berichtet:
* Eine SD von $\ln(\text{NLR})$ entspricht einem Faktor von **1.58** auf der Originalskala (z. B. Anstieg des NLR von 2.0 auf 3.2).
* Eine SD von $\ln(\text{hs-CRP})$ entspricht einem Faktor von **3.15** auf der Originalskala (z. B. Anstieg von 1.8 auf 5.7 mg/L).

---

## Schritt 4: Deskriptive Statistik & Baseline Table 1

### Leitfrage & Ziel
Wie unterscheiden sich herzkranke und gesunde Teilnehmende, bevor wir Modelle anwenden?

### Kursbezug
* **W3 (Statistische Kennzahlen, nicht-parametrische Tests)**
* **Tutorial 4 (Mann-Whitney U, Chi-Quadrat, Bonferroni-Korrektur)**

### Die drei Optionen für den Gruppenvergleich
* **Option A: t-Test und Pearson-Korrelation:** Ungeeignet, da biologische Marker und Laborwerte stark rechtsschief sind.
* **Option B: Mann-Whitney U Test (stetig) und Chi-Quadrat-Test (kategorial):** Robust gegen Schiefe und Ausreisser. **(Unsere Wahl)**.
* **Option C: Nur Effektstärken (SMD) ohne Signifikanztests:** Nützlich, aber unvollständig.

### Die drei wichtigsten klinischen Entdeckungen in Table 1
1. **NLR ist bei Fällen deutlich erhöht:** Median 2.39 vs. 1.96 ($\text{SMD} = 0.42$, Bonferroni-$p < 10^{-17}$).
2. **hs-CRP unterscheidet sich bereits unadjustiert NICHT:** Median 1.97 vs. 1.81 mg/L ($\text{SMD} = 0.02, p = 0.256$, nach Bonferroni $p = 1.00$).
3. **Behandlungs-Paradoxon (Treatment Confounding):** Herzkranke haben *niedrigeres* Gesamtcholesterin (151.5 vs. 187.0 mg/dL, $\text{SMD} = -0.72$), wahrscheinlich weil **78.2%** von ihnen lipidsenkende Medikamente (Cholesterinsenker) einnehmen (im Vergleich zu nur 24.9% der Kontrollgruppe).

---

## Schritt 5: Fehlende Werte — Das Experiment E1

### Leitfrage & Ziel
Wie gehen wir mit fehlenden Werten um (insbesondere beim Familieneinkommen: 12.7% fehlend), ohne Information zu vernichten oder Modelle zu verzerren?

### Kursbezug
* **W2 (Missing Data Mechanismen: MCAR, MAR, MNAR)**
* Vorlesungsoptionen: Listwise Deletion, Spalte entfernen, Median-Imputation, Missing Indicator.

### Mechanismus-Prüfung
Ein logistisches Modell für "Einkommen fehlt" zeigt: Das Fehlen hängt von Bildung, Ethnie und Rauchen ab (nicht vom Alter), aber **nicht von der Herzkrankheit** ($\text{OR} = 1.03, p = 0.87$). Das passt zu **Missing at Random (MAR)**; MNAR lässt sich mit den Daten allein nie ganz ausschliessen.

### Die drei Optionen (Experiment E1)
| Strategie | Behaltene Personen | Test-AUC (LR) | Test-AUC (RF) | Beurteilung |
|---|---:|---|---|---|
| **Option A: Listwise Deletion (Complete Cases)** | 4'563 | 0.867 | 0.865 | Verliert 975 Personen (70 Fälle!). Verzerrt MAR-Daten. |
| **Option B: Spalte Einkommen löschen** | 5'538 | 0.867 | 0.866 | Verliert den bekannten Risikofaktor Sozioökonomie. |
| **Option C: Median-Imputation + Missing-Indikator** | **5'538** | **0.867** | **0.868** | **Erhält alle Fälle; Indikator nutzt die Information des Fehlens.** |

### Entscheidungsregel & Entscheidung
*Regel:* Wähle Option C, es sei denn, A oder B ist auf denselben Testpersonen um mehr als $\Delta\text{AUC} > 0.005$ überlegen.  
*Entscheidung:* **Option C gewählt.** Kein AUC-Vorteil für Deletion, dafür bleiben alle 412 Fälle in der Analyse!

---

## Schritt 6: Feature Selection & Datenlecks — Das Experiment E2

### Leitfrage & Ziel
Soll ein Algorithmus (z. B. Lasso) frei aus allen verfügbaren Umfragevariablen wählen, oder nutzen wir ein theoriegeleitetes Set?

### Kursbezug
* **W4, W6 (Feature Selection: Filter, Wrapper, Embedded)**
* **W5 (Lasso-Regularisierung)**
* **W4 (Label Leakage & Overfitting)**

### Die drei Strategien im Vergleich
* **Strategie S1 (Literatur/Guideline):** Die etablierten Faktoren grosser Risikoscores (PCE, SCORE2, PREVENT): Alter, Geschlecht, Ethnie, Einkommen, BMI, Rauchen, Diabetes, Bluthochdruck, Cholesterin, eGFR.
* **Strategie S2 (Naiver Lasso auf 342 Rohvariablen):** Vollautomatische Auswahl.
* **Strategie S3 (Hybrider Pool aus 39 fachlich gescreenten Variablen):** Vorauswahl nach Fachwissen, dann Lasso.

| Strategie | Test-AUC (10 Folds) | Variablen pro Fold | Stabilität (Jaccard) |
|---|---|---|---|
| S1 Literatur | 0.862 | 13 | 1.00 |
| S2 naiver Lasso | 0.923 | ≈ 114 | 0.47 |
| S3 hybrider Pool + Lasso | 0.870 | ≈ 35 | 0.86 |

### Die Entdeckung des massiven Datenlecks in S2
Der naive Lasso (S2) erreicht scheinbar eine fantastische AUC von **0.923**. Bei der Inspektion der Koeffizienten zeigt sich jedoch ein massives **Label Leakage (Post-Diagnosis-Verzerrung)**:
* Der Algorithmus wählt: *"Metallteile im Körper?"* (Stents!), *"Arzt riet zu Aspirin?"*, *"Herzinsuffizienz diagnostiziert?"*, *"Nimmt Cholesterinsenker?"*.
* Das sind keine Ursachen oder Prädiktoren, sondern **Folgen und Behandlungen der bereits erfolgten Diagnose**!

### Entscheidung
**Entscheidungsregel:** Wir behalten S1, ausser eine andere Strategie enthält keine Variablen, die Folge oder Behandlung der Krankheit sind, *und* ist mehr als 0.005 AUC besser als S1, mit einem korrigierten 95%-KI der gepaarten Differenz über 0. (Die Regel wurde nach dem ersten E2-Lauf formuliert, deshalb nennen wir sie nirgends "vorab festgelegt".)

| Vergleich mit S1 | ΔAUC (95%-KI) | ohne Leakage? | Regel erfüllt? |
|---|---|---|---|
| S2 naiver Lasso | +0.061 (0.050 bis 0.073) | nein (35 % der Top 20) | nein |
| S3 hybrider Pool + Lasso | +0.008 (−0.005 bis +0.020) | ja | nein (KI enthält 0) |

Wir wählen deshalb **S1 (Literatur-Set)**. Es beantwortet exakt unsere Forschungsfrage (*"über etablierte Faktoren hinaus"*), ist stabil über Folds und frei von Behandlungslecks.

---

## Schritt 7: Stichprobenumfang & Events-Per-Parameter

### Leitfrage & Ziel
Reicht unsere Fallzahl aus, um ein Modell mit 19 geschätzten Parametern ohne Überanpassung stabil zu schätzen?

### Kursbezug
* **W3-Checkliste:** *Was the sample size justified through statistical calculation?*
* **EPP/EPV-Regel (Events per Parameter):** Mindestens 10 bis 20 Fälle pro geschätztem Koeffizienten.

### Berechnung (`sample_size_epv.csv`)
Unser Modell M3 hat 19 Koeffizienten (inklusive Dummy-Variablen für Ethnie, Rauchen und Missing Indicator).
* Gesamtkohorte: 412 Fälle / 19 Parameter = **21.7 EPV**
* Äusserer Trainingsfold (80%): 329 Fälle / 19 Parameter = **17.3 EPV**
* Innerer Trainingsfold (64%): 263 Fälle / 19 Parameter = **13.9 EPV**
*Fazit:* Alle Splits erfüllen die EPV $\ge 10$ Anforderung der epidemiologischen Fachliteratur.

---

## Schritt 8: Seltene Krankheiten & Klassen-Ungleichgewicht — Das Experiment E3

### Leitfrage & Ziel
Verhilft eine Resampling-Methode (SMOTE, Oversampling, Klassengewichte) bei 7.4% Krankheitsprävalenz zu besseren Modellen?

### Kursbezug
* **W9 (Accuracy Paradox, Class Weights, Oversampling, Undersampling, SMOTE)**

### Das Genauigkeits-Paradoxon
Ein naiver Klassifikator, der einfach für jeden Menschen "gesund" vorhersagt, erreicht eine Accuracy von **92.6%**, übersieht aber jeden einzelnen herzkranken Patienten. Accuracy ist hier nutzlos.

### Ergebnisse des Experiments E3
| Methode | AUC (LR) | AUC (RF) | Brier Score | Mittleres vorhergesagtes Risiko |
|---|---|---|---|---|
| **Keine Korrektur** | **0.862** | **0.861** | **0.058** | **7.4% (Wahrheitsgetreu!)** |
| Klassengewichte (`balanced`) | 0.862 | 0.863 | 0.156 | 31.9% (Völlig verzerrt) |
| Oversampling | 0.861 | 0.860 | 0.156 | 31.9% |
| Undersampling | 0.856 | 0.858 | 0.164 | 32.6% |
| SMOTE | 0.860 | 0.857 | 0.151 | 30.0% |

### Erkenntnis & Entscheidung
Resampling-Methoden verbessern die Rangordnung (AUC) überhaupt nicht, **zerstören aber die Kalibrierung der Wahrscheinlichkeiten**: Das Modell sagt plötzlich Risiken von 30% voraus, obwohl die Grundrate 7.4% beträgt.  
*Entscheidung:* **Keine Resampling-Korrektur.** Um seltene Fälle zu fangen, optimieren wir stattdessen die Entscheidungsschwelle (Youden-Index).  
*Ausnahme RBF-SVM:* Bei der SVM bricht die Trennebene ohne Gewichte zusammen ($\text{AUC} = 0.76$). Die SVM erhält `class_weight='balanced'`, da ihre Wahrscheinlichkeiten erst nachträglich via Platt-Scaling kalibriert werden.

---

## Schritt 9: Die Evaluierungsstrategie — Das Experiment E4

### Leitfrage & Ziel
Reicht ein einfacher 80/20 Train-Test-Split, oder brauchen wir wiederholte verschachtelte Kreuzvalidierung?

### Kursbezug
* **W4 (Train/Test Split, Overfitting)**
* **W5, Tutorial 7 & 8 (Nested Cross-Validation)**

### Die "80/20-Lotterie" (Experiment E4)
Wir trainieren dasselbe Modell auf 200 verschiedenen zufälligen 80/20-Splits:
* Die Test-AUC schwankt zwischen **0.802 und 0.923** (mittlere 95 % der Splits: 0.831–0.894)!
* In **29.5%** der zufälligen Splits sieht es so aus, als ob die Entzündungsmarker die Vorhersage *verschlechtern* ($\Delta\text{AUC} < 0$).
* Ein einzelner Train/Test-Split ist reines Glücksspiel!

### Die Lösung: Repeated Nested Cross-Validation (5 Folds $\times$ 3 Repeats = 15 Folds)
1. **Äussere Schleife (Outer CV):** 5-fache stratifizierte Kreuzvalidierung, 3-mal wiederholt $\rightarrow$ 15 unabhängige Testfolds.
2. **Innere Schleife (Inner CV):** Innerhalb der 4 Trainingsfolds sucht ein 5-facher `GridSearchCV` nach den optimalen Hyperparametern (Optimierungsziel: `neg_log_loss`).
3. **Strikte Pipeline:** Alle Transformationen (Median, Skalierung, One-Hot) werden **ausschliesslich auf den Trainingsdaten des Folds gefittet**. Kein Test-Set Leakage!

---

## Schritt 10: Die sechs Machine Learning Algorithmen

### Kursbezug
Alle sechs im ETH-Kurs behandelten Lernverfahren (W5–W9):
1. **Logistische Regression (Baseline, W6–W7):** Unpenalisiert, maximal interpretierbar.
2. **Penalisierte Regression (W5–W6):** Elastic Net / Lasso / Ridge via SAGA-Solver (Tuning von $C$ und L1-Ratio).
3. **Entscheidungsbaum (CART, W7):** Gini-Index, Tuning von Tiefe (3–8) und Mindestblattgrösse.
4. **k-Nächste Nachbarn (kNN, W6, W8):** Tuning von $k$ (50–800), Distanzmetrik (Manhattan/Euklidisch) und Gewichtung.
5. **Support Vector Machine mit RBF-Kernel (W9):** Tuning von $C$ und $\gamma$, Platt-Skalierung für Wahrscheinlichkeiten.
6. **Random Forest (W8):** 300 Bäume, Bootstrap-Aggregierung, Tuning von `max_features` ($\sqrt{p}$ vs. 0.33) und Blattgrösse.

---

## Schritt 11: Metriken für ungleiche Klassen

* **ROC-AUC (Hauptmetrik):** Schwellenwert-unabhängiges Mass für die Trennschärfe (0.5 = Zufall, 1.0 = Perfekt).
* **PR-AUC (Precision-Recall AUC):** Besonders informativ bei seltenen Outcomes (Zufallslinie liegt bei der Prävalenz: 0.074).
* **Brier Score:** Mittlerer quadratischer Fehler der Wahrscheinlichkeiten (misst Diskrimination + Kalibrierung).
* **Youden-Schwelle:** Im *Trainingsteil* ermittelt als Maximum von $\text{Sensitivität} + \text{Spezifität} - 1$, danach auf den Testteil angewandt.

---

## Schritt 12: Statistischer Modellvergleich via Nadeau-Bengio

### Das Problem des Standard-t-Tests
Bei wiederholter Kreuzvalidierung teilen sich die Trainingsdaten verschiedener Folds zu 75% dieselben Datenpunkte. Die 15 Test-Ergebnisse sind daher **nicht statistisch unabhängig**. Ein gewöhnlicher gepaarter t-Test unterschätzt die Standardabweichung und liefert fälschlicherweise winzige p-Werte (Typ-I-Fehler).

### Die Nadeau & Bengio (2003) Varianz-Korrektur
Die korrigierte Varianz der mittleren Differenz $\bar{d}$ lautet:
$$\sigma^2_{\text{korrigiert}} = s^2 \left( \frac{1}{J} + \frac{n_{\text{test}}}{n_{\text{train}}} \right)$$
Bei $J = 15$ Folds und $n_{\text{test}}/n_{\text{train}} = 1/4 = 0.25$ vergrössert sich die Varianz um den Faktor $(1/15 + 0.25) = 0.317$ gegenüber $1/15 = 0.067$. Dies führt zu ehrlichen, konservativen Konfidenzintervallen.

---

## Schritt 13: Beantwortung der Hauptfrage (H1a & H1b)

### Regressionsergebnisse (`assoc_odds_ratios.csv`)
Odds Ratio pro 1 SD des log-Biomarkers im voll adjustierten Hauptmodell (Modell 2):

* **NLR (H1a):** $\text{OR} = \mathbf{1.20}$ (95%-KI: 1.07–1.34, $p = 0.002 < 0.025$).  
  $\rightarrow$ **Hypothese H1a bestätigt!** Pro typischer Erhöhung des NLR steigt die Chance auf CHD/MI unabhängig von Blutdruck, Rauchen, Alter und Cholesterin um 20%.
* **hs-CRP (H1b):** $\text{OR} = \mathbf{0.96}$ (95%-KI: 0.84–1.08, $p = 0.48$).  
  $\rightarrow$ **Hypothese H1b NICHT bestätigt!** Kein unabhängiger Zusammenhang.

### Die Confounder-Zerlegung: Warum verschwindet hs-CRP?
Im unadjustierten Modell 1 schien hs-CRP leicht assoziiert ($\text{OR} = 1.11, p = 0.066$). Fügen wir schrittweise einzelne Variablen hinzu, zeigt sich:
* **Allein die Hinzunahme des BMI senkt das OR von 1.11 auf 1.00 (0.88–1.13)!**
* *Klinische Erkenntnis:* hs-CRP misst im Wesentlichen nur das viszerale Fettgewebe (Adipositas) mit.

### Likelihood-Ratio-Test (`assoc_lr_test.csv`)
Verbessern die Marker die Modellanpassung signifikant?
* NLR allein hinzugefügt: $\chi^2 = 9.83, df = 1, p = 0.002$ (Signifikanter Informationsgewinn)
* hs-CRP allein hinzugefügt: $\chi^2 = 0.50, df = 1, p = 0.48$ (Kein Informationsgewinn)

---

## Schritt 14: Beantwortung der Zusatzfrage (H1c: Diskrimination)

### Ergebnisse der 15-Fold Kreuzvalidierung (`main_marker_gain.csv`)

| Algorithmus | Basis M2 (AUC) | Mit Markern M3 (AUC) | $\Delta\text{AUC}$ (95%-KI) | Bonferroni $p$ |
|---|---|---|---|---|
| **Logistische Regression** | 0.861 | 0.863 | **+0.0013** (−0.0009 bis +0.0035) | 1.00 |
| Elastic Net | 0.862 | 0.863 | **+0.0015** (−0.0008 bis +0.0038) | 1.00 |
| Entscheidungsbaum | 0.808 | 0.805 | **−0.0030** (−0.0123 bis +0.0064) | 1.00 |
| k-Nächste Nachbarn | 0.859 | 0.863 | **+0.0039** (−0.0067 bis +0.0144) | 1.00 |
| SVM (RBF) | 0.867 | 0.867 | **+0.0003** (−0.0025 bis +0.0030) | 1.00 |
| Random Forest | 0.856 | 0.859 | **+0.0022** (−0.0055 bis +0.0099) | 1.00 |

### Urteil zu H1c
**Hypothese H1c ist widerlegt.** Bei keinem einzigen Algorithmus bringt das Hinzufügen der beiden Biomarker einen klinisch relevanten Gewinn ($\Delta\text{AUC} \ge 0.01$). Für 5 von 6 Algorithmen schliessen die 95%-Konfidenzintervalle einen Zuwachs von 0.01 strikt aus!

### Schlägt ein komplexes ML-Verfahren die Logistische Regression?
* SVM erreicht eine minimale Steigerung von $+0.004$ gegenüber LR ($p = 0.74$, statistisch nicht signifikant).
* Der Entscheidungsbaum schneidet drastisch schlechter ab ($-0.058, p = 0.002$).
* *Fazit:* Nach dem Sparsamkeitsprinzip (Occam's Razor) bleibt die **einfache, unpenalisierte Logistische Regression das empfohlene Modell**.

---

## Schritt 15: Sensitivitätsanalysen & Robustheitsprüfung

Um zu zeigen, dass unsere Schlussfolgerungen nicht an einer einzelnen Entscheidung hängen, ändern wir jeweils eine Entscheidung (`sensitivity_association.csv`, `sensitivity_prediction.csv`). Fünf Varianten, jede gehört zu einer unserer Entscheidungen:
1. Complete Cases ($n = 4'563$, 342 Fälle; prüft E1): NLR OR 1.17 (1.03–1.32), CRP OR 0.96 (0.83–1.11).
2. Einschluss lipidsenkender Medikamente (Modell 3; prüft "Behandlung nicht im Hauptmodell"): NLR OR 1.21 (1.08–1.36), CRP OR 0.99 (0.87–1.13).
3. Ausschluss akuter Infektionen (hs-CRP $>10$ mg/L; prüft "Ausreisser behalten"): NLR OR 1.19 (1.05–1.34), CRP OR 0.96 (0.82–1.12).
4. Survey-Gewichte (repräsentativ für die US-Bevölkerung): NLR OR 1.18 (1.05–1.33), CRP OR 0.91 (0.81–1.02).
5. Inklusive Angina Pectoris im Outcome (prüft die Outcome-Definition): NLR OR 1.14 (1.02–1.27), CRP OR 0.96 (0.85–1.08).

*Ergebnis:* In allen fünf Varianten bleibt NLR signifikant (α = 0.025) und hs-CRP nicht signifikant. Der Marker-Gewinn ΔAUC der logistischen Regression bleibt in allen Varianten unter 0.002 (bei Variante 2 mit lipidsenkenden Medikamenten als zusätzlichem Prädiktor; für die Survey-Gewichte gibt es keine AUC-Variante).

---

## Schritt 16: Modell-Interpretierbarkeit & Feature Attribution

1. **Permutation Feature Importance:** Alter dominiert mit weitem Abstand (AUC-Abfall 0.06–0.10, beim Entscheidungsbaum 0.18). Auf Platz 2 folgt Cholesterin (invers, wahrscheinlich wegen der lipidsenkenden Behandlung). NLR belegt Rang 6–9 von 13; hs-CRP landet bei 5 von 6 Algorithmen auf dem **allerletzten Platz**.
2. **Partial Dependence Plots:** Das Risiko steigt monoton mit steigendem NLR; die Kurve für hs-CRP ist flach bis leicht fallend (logistische Regression: 8.0 % → 6.5 % vom 5. zum 95. Perzentil).
3. **Illustrativer Baum (Tiefe 3):** Teilt ausschliesslich nach Alter, Diabetes, Cholesterin und eGFR auf. Kein einziger Entzündungsmarker wird verwendet!

---

## Schritt 17: Typische Fragen & mögliche Antworten

Fragen, die in der Präsentation kommen könnten, und wie man sie beantworten kann:

### Frage 1: "Warum haben Sie die Entzündungsmarker logarithmiert?"
> **Antwort:** "Die Rohwerte von NLR und hs-CRP weisen extreme Rechtsschiefe auf (Schiefe 4.66 und 7.78). Wenige Extremwerte würden gradientenbasierte und distanzbasierte Verfahren wie kNN und SVM verzerren. Durch die Logarithmierung sinkt die Schiefe auf 0.15 und 0.23, was zu einer symmetrischen, normalverteilungsnahen Verteilung im Q-Q-Plot führt. Zudem erlaubt es die biologisch sinnvolle Interpretation als relatives Risiko: 1 SD entspricht einem festen Multiplikationsfaktor auf der Originalskala."

### Frage 2: "Warum hat Ihr Random Forest oder Ihre SVM die Logistische Regression nicht geschlagen?"
> **Antwort:** "Wir haben 13 Prädiktoren, 412 Fälle und vorwiegend monotone biologische Risikobeziehungen. Nicht-lineare Terme (quadratische Tests) und Interaktionen mit lipidsenkenden Medikamenten oder Geschlecht waren statistisch nicht signifikant. Komplexe Modelle wie Random Forest oder SVM haben eine höhere Modellvarianz, können hier aber keinen Bias abbauen, weil schlicht keine komplexe Interaktionsstruktur in den Daten vorliegt. Das deckt sich mit der Meta-Analyse von Christodoulou et al. (2019)."

### Frage 3: "Wie erklären Sie den scheinbaren Widerspruch: NLR ist statistisch hochsignifikant assoziiert (p = 0.002), verbessert aber die AUC um fast gar nichts (+0.0013)?"
> **Antwort:** "Assoziation und Diskrimination beantworten zwei mathematisch völlig verschiedene Fragen. Ein Odds Ratio von 1.20 pro SD bedeutet, dass NLR ein echtes statistisches Signal liefert (bestätigt durch den Likelihood-Ratio-Test, $\chi^2 = 9.83, p = 0.002$). Aber um die Trennung (AUC) zweier Gruppen spürbar zu verschieben, braucht ein Marker laut Pepe et al. (2004) Odds Ratios von 3.0 oder höher. Da Alter und etablierte Risikofaktoren bereits eine AUC von 0.861 erreichen, reicht das moderate Signal von NLR nicht aus, um Patienten in der Risikorangfolge umzureihen."

### Frage 4: "Warum haben Sie fehlende Einkommensdaten mit Median + Indikator ersetzt und nicht einfach gelöscht?"
> **Antwort:** "Listwise Deletion hätte 975 Personen und vor allem 70 echte Herzkranke (17% aller Fälle!) vernichtet. Unsere logistische Modellierung des Fehlens zeigte, dass das Fehlen vom Typ MAR (Missing at Random) ist. Im Experiment E1 haben wir gezeigt, dass Median-Imputation mit Missing-Indikator auf identischen Testfällen exakt dieselbe AUC (0.867) liefert wie Deletion, uns aber erlaubt, die volle Stichprobe von 5'538 Personen zu erhalten und auch Personen mit unvollständigen Angaben ein faires Risiko zuzuordnen."

### Frage 5: "Warum haben Sie trotz 7.4% Ungleichgewicht kein SMOTE oder Oversampling verwendet?"
> **Antwort:** "Im Experiment E3 haben wir empirisch nachgewiesen, dass Resampling zwar die Rangordnung (AUC) unverändert lässt, aber die Kalibrierung der Wahrscheinlichkeiten zerstört: Der Brier Score verdoppelte sich und das mittlere vorhergesagte Risiko stieg künstlich auf über 30%. Da für klinische Entscheidungen kalibrierte Wahrscheinlichkeiten entscheidend sind, haben wir die natürliche Prävalenz beibehalten und stattdessen den Klassifikationsschwellenwert über den Youden-Index optimiert."
