# Leitfaden für Marcos Team-Präsentation: Data Cleaning, Missing Data & Feature Selection

**Ziel:** Deinen Teamkollegen (Bettina & Co.) im nächsten Meeting souverän zeigen, welche 3 Varianten du jeweils verglichen hast, welche Code-Experimente du durchgeführt hast und warum dein Preprocessing-Modul nun die perfekte gemeinsame Arbeitsgrundlage für das Projekt ist.

---

## 1. Einleitung (Wie du das Gespräch eröffnest)

> *"Hey zusammen, wie versprochen habe ich mir in den letzten Tagen die ersten beiden grossen Blöcke unseres Projekts vorgenommen:*  
> *1. Das **Data Cleaning** bzw. die **Kohorten-Auswahl** und den Umgang mit **fehlenden Werten**.*  
> *2. Die **Feature Selection** (ob wir Lasso nutzen, die Literatur nehmen oder eine Kombination).*  
>  
> *Mir war extrem wichtig, dass wir im Projektbericht für jede Entscheidung eine saubere wissenschaftliche Begründung haben, anstatt Dinge willkürlich zu machen. Deshalb habe ich für jede Frage **3 konkrete Varianten** gegeneinander getestet. Hier sind die Ergebnisse, und ich habe uns dazu ein fertiges Preprocessing-Skript gebaut, das wir alle ab jetzt direkt importieren können."*

---

## 2. Teil 1: Die Studienkohorte — Wer ist drin, wer fehlt und warum?

### Das Problem im bisherigen Entwurf
* Bisher war unklar, warum Personen genau ausgeschlossen wurden und wie viele herzkranke Fälle in jedem Schritt verloren gingen.

### Unsere saubere Lösung (`01_cohort_selection.py`)
Zeige deinen Kollegen die Tabelle `results/tables/flowchart.csv` (oder Abbildung `fig01_flowchart.png`):

| Schritt | Teilnehmer | Ausgeschlossen | CHD/MI-Fälle | Begründung |
|---|---:|---:|---:|---|
| **1. Alle NHANES-Teilnehmer** | 11'933 | – | 559 | Ausgangsstichprobe der US-Bevölkerung |
| **2. Erwachsene ($\ge 20$ Jahre)** | 7'809 | 4'124 | 559 | Herzfragen werden erst ab 20 Jahren gestellt |
| **3. Frage zu CHD/MI beantwortet** | 7'773 | 36 | 559 | Gültiges primäres medizinisches Outcome |
| **4. Im mobilen Zentrum (MEC) untersucht** | 6'037 | 1'736 | 451 | Voraussetzung für körperliche & Blutuntersuchung |
| **5. NLR & hs-CRP gemessen (Endstichprobe)** | **5'538** | 499 | **412 (7.4%)** | Vollständige Messung beider primärer Biomarker |

### Der Selektionsverzerrungs-Check (Selection Bias)
> *"Ich habe auch direkt geprüft, wer die 499 Personen sind, die zwar im Untersuchungszentrum waren, bei denen aber kein Blut abgenommen wurde. Über die standardisierte Mittelwertdifferenz (SMD) sieht man:*  
> *Diese 499 Personen hatten ein signifikant tieferes Einkommen ($\text{SMD} = -0.36$). Das ist eine super Erkenntnis für unsere Limitationen im Bericht: Wir können ehrlich schreiben, dass unsere Ergebnisse primär für die untersuchte Stichprobe gelten."*

---

## 3. Teil 2: Fehlende Werte — Das Experiment E1

### Das Problem im bisherigen Entwurf
* Beim Familieneinkommen (`income_pir`) fehlen bei **12.7% der Leute die Angaben**.
* Unser bisheriger Entwurf hat einfach alle Personen mit irgendeinem fehlenden Wert gelöscht (*Listwise Deletion*).

### Die 3 Varianten, die du getestet hast
1. **Option A (Listwise Deletion / Complete Cases):** Was wir bisher hatten.
2. **Option B (Spalte Einkommen löschen):** Wir behalten alle Leute, werfen aber das Einkommen als Prädiktor weg.
3. **Option C (Median-Imputation + Missing-Indikator):** Wir füllen fehlende Werte mit dem Median des Trainingsfolds und fügen eine 0/1-Spalte `income_pir_missing` hinzu.

### Dein Code-Experiment (`02_missing_data_experiment.py`)
Zeige deinen Kollegen die Tabelle `results/tables/missing_data_experiment.csv`:

| Strategie | Behaltene Teilnehmer | Verlorene Fälle | Test-AUC (Log. Reg.) | Test-AUC (Random Forest) |
|---|---:|---:|---|---|
| **Option A: Listwise Deletion** | 4'629 | **70 Fälle verloren!** | 0.865 | 0.857 |
| **Option B: Spalte Einkommen löschen** | 5'538 | 0 | 0.866 | 0.844 |
| **Option C: Median + Indikator (GEWÄHLT)** | **5'538** | **0** | **0.865** | **0.853** |

### Deine Argumentation für das Team
> *"Schaut euch das an: Bei Listwise Deletion (Option A) werfen wir **909 Menschen weg – und darunter sind 70 echte Herzkranke (17% aller unserer Fälle!)**.*  
> *Im Test-Experiment auf denselben Testdaten ist die AUC von Option C exakt identisch ($0.865$).*  
> *Wir verlieren also überhaupt keine Modellqualität, retten aber 70 echte Patienten für das Training und können unser Modell später auch auf Personen anwenden, die ihr Einkommen nicht angeben wollten. Deshalb nehmen wir ab jetzt Option C!"*

---

## 4. Teil 3: Feature Selection — Das Experiment E2

### Das Problem im bisherigen Entwurf
* Sollen wir einfach einen Lasso über alle Variablen laufen lassen? Oder suchen wir die Variablen von Hand aus?

### Die 3 Varianten, die du getestet hast
1. **Strategie 1 (Literatur / Guideline-Set M2/M3):** Etablierte Risikofaktoren aus den grossen Scores (Framingham, SCORE2, PREVENT): Alter, Geschlecht, Ethnie, Einkommen, BMI, Rauchen, Diabetes, Bluthochdruck, Cholesterin, eGFR.
2. **Strategie 2 (Naiver Lasso über alle 342 Rohvariablen):** Der Computer wählt völlig frei.
3. **Strategie 3 (Hybrider Pool aus 39 fachlich geprüften Merkmalen):** Vorfilterung, dann Lasso.

### Die Entdeckung des massiven Datenlecks (`03_feature_selection_experiment.py`)
> *"Hier habe ich etwas extrem Spannendes entdeckt, das uns vor einer schlechten Note rettet:*  
> *Wenn man den Lasso völlig frei über alle 342 Variablen laufen lässt (Strategie 2), springt die Test-AUC scheinbar auf **0.923** hoch.*  
> *Aber wenn man reinschaut, welche Variablen der Lasso wählt, sieht man ein **massives Datenleck (Label Leakage)**:*  
> *1. 'Metallteile im Körper?' $\rightarrow$ Das sind **Stents**, die nach dem Herzinfarkt gesetzt wurden!*  
> *2. 'Arzt riet zu Aspirin?' $\rightarrow$ Standard-Medikament nach Herzinfarkt!*  
> *3. 'Nimmt Cholesterinsenker?' $\rightarrow$ Statine werden nach der Diagnose verschrieben!*  
> *4. 'Herzinsuffizienz diagnostiziert?' $\rightarrow$ Eine direkte Folge des Herzinfarkts!*  
>  
> *Der naive Lasso schummelt: Er sagt nicht voraus, wer ein Herzrisiko hat, sondern er erkennt, wer bereits medizinisch nachbehandelt wird. Das ist ein Zirkelschluss!"*

### Deine Entscheidung für das Team
> *"Deshalb müssen wir **Strategie 1 (Literatur-Set)** als primäres Modell wählen:*  
> *- Es ist frei von Behandlungslecks.*  
> *- Es beantwortet exakt unsere Forschungsfrage: 'Bringen die Entzündungsmarker NLR und hs-CRP einen Mehrwert **über die etablierten Risikofaktoren hinaus**?'*  
> *- Es ist 100% stabil über alle Folds."*

---

## 5. Teil 4: Wie ihr ab jetzt zusammenarbeitet (`preprocessing.py`)

Zeige deinen Kollegen, wie einfach sie ab jetzt den sauberen Datensatz in ihren eigenen Notebooks oder Skripten laden können:

```python
import sys
sys.path.append("Final")

from preprocessing import load_cohort, make_pipeline
from sklearn.ensemble import RandomForestClassifier

# 1. Den sauberen Kohorten-Datensatz laden (5'538 Zeilen, 412 Fälle)
df = load_cohort()

# 2. Features und Target definieren
features = ["age", "male", "bmi", "smoking", "diabetes", "hypertension", "total_chol", "hdl_chol", "egfr", "log_nlr", "log_crp"]
X = df[features]
y = df["chd_mi"]

# 3. Eine leckage-freie Pipeline erstellen (inklusive Imputation, Indikator & Skalierung)
pipeline = make_pipeline(RandomForestClassifier(n_estimators=100), features)
pipeline.fit(X, y)
```

> *"Damit haben wir alle dieselbe, saubere Datengrundlage. Niemand muss mehr von vorne anfangen zu filtern, und wir vermeiden Fehler bei fehlenden Werten oder Datenlecks!"*
