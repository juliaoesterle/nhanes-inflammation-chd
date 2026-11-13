# Übergabe: Stand und nächste Schritte (NewMarco)

Stand: 13.11.2026, Nachmittag.

## Ziel
`NewMarco/` ist eine **Referenz zum Lernen**. Marco baut die Analyse danach selbst neu auf: Die Logik kommt von ihm, beim Code hilft die KI. Deshalb gilt:
- Methoden so nah wie möglich am Kurs (Vorlesungen W1–W9 und Tutorials; W10–W14 sind noch nicht verfügbar).
- Für jede Entscheidung mehrere Optionen vergleichen, die beste begründen und damit weiterarbeiten.
- Alles Schritt für Schritt und laienverständlich erklären.

## Wo anfangen
1. **[docs/ANLEITUNG_SCHRITT_FUER_SCHRITT.md](docs/ANLEITUNG_SCHRITT_FUER_SCHRITT.md)**: 17 Schritte. Pro Schritt Ziel, Kursbezug, Optionen, Ergebnis, Entscheidung, «So machst du es selbst» (Python-Bausteine) und Prüffragen. Am Schluss eine Checkliste zum Nachbauen.
2. [docs/LAIEN_ERKLAERUNG.md](docs/LAIEN_ERKLAERUNG.md): die Kurzfassung für Nicht-Fachleute.
3. [report/REPORT_DRAFT.md](report/REPORT_DRAFT.md): Bericht nach Kursvorlage (Vorlage für den eigenen Bericht).
4. [docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md): technische Referenz (Entscheidungen D1–D18, alle Zahlen).

## Stand: kursnahe Version (alles gelaufen, alle Texte aktuell)
- **Forschungsfrage wie im Gruppen-README:** Hauptfrage = Assoziation (H1a NLR, H1b hs-CRP; Bonferroni α = 0.025), Zusatzfrage H1c = Diskrimination mit sechs Kursverfahren.
- **Ergebnis:**
  - H1a bestätigt: OR 1.20 (1.07–1.34), p = 0.002.
  - H1b nicht bestätigt: OR 0.96 (0.84–1.08), p = 0.48; BMI erklärt den groben Zusammenhang.
  - H1c nicht bestätigt: ΔAUC +0.0013 (−0.0009 bis +0.0035).
  - Gleiche Schlussfolgerungen in allen 13 Sensitivitätsanalysen.
- **Umstellung auf Kursmethoden:**
  - Fehlende Werte: die drei W2-Optionen (Deletion / Spalte löschen / Median + Indikator) statt iterativer und multipler Imputation.
  - Klassen-Ungleichgewicht: SMOTE (W9) ergänzt.
  - Assoziation: Bonferroni statt Holm; Wald- und Likelihood-Ratio-Test statt DeLong; quadratischer Term und Quartile statt Splines.
  - Gestrichen: Decision-Curve-Analyse, Stability Selection und Gradient Boosting.
  - Stichprobengrösse: «10 Fälle pro Parameter» statt Riley.
  - Modell 2 ist das Hauptmodell (Modell 3 mit Lipidsenkern nur als Sensitivität).
- **Alle Kritikpunkte aus dem «Professoren-Gutachten» behoben:**
  - Fragestellung und Hypothesen; Behandlungs-Leck beim Cholesterin (AUC ohne Cholesterin 0.849 statt 0.861); LR-Test statt DeLong.
  - Nachträgliche Schritte gekennzeichnet; E2 als Prüfung statt Entscheidung.
  - Bericht mit eigenem Literaturverzeichnis, durchnummerierten Abbildungen und einem Abstract von 226 Wörtern.
  - Formulierungen («sample», «adjusted»); vollständige Koeffiziententabelle; fixierte Versionen (`requirements.txt`).
- **Archiv:** Der erste, fehlerhafte SVM-Lauf liegt in `results/cv/*_previous_SVM.*` (Lehrbeispiel, siehe Anleitung Schritt 8).

## Umgebung / Stolpersteine
- Python: `/c/Users/Marco/anaconda3/envs/nhanes/python.exe`; Versionen in `requirements.txt`. Skripte aus `NewMarco/` starten.
- In Skripten mit statsmodels-Formeln `import config as cfg` (nicht `as C`, sonst patsy-Fehler).
- sklearn 1.9: unpenalisierte LR mit `C=np.inf`, Lasso mit `l1_ratio=1, solver="liblinear"`, SVM-Wahrscheinlichkeiten mit `CalibratedClassifierCV(SVC(), ensemble=False)`; `plot_tree(fontsize=...)` braucht eine ganze Zahl.
- Nur einen Algorithmus neu rechnen: `python 04_nested_cv.py SVM` (die alten Zeilen werden archiviert). Nicht zweimal hintereinander starten, sonst wird das Archiv überschrieben.
- Ein Modul wie `06_association.py` nicht einfach importieren: Der Logger überschreibt sonst dessen Log-Datei.
- Laufzeiten: 04 ≈ 60 Min., alle anderen Schritte je ≤ 3 Min.

## Offen (optional)
1. Literaturangaben vor einer Verwendung gegen die Originale prüfen.
2. Laien-Doku oder Anleitung als HTML-Seite (vorher Skill `artifact-design` laden).
3. Mit dem Team klären, wie die Erkenntnisse in die gemeinsame Arbeit einfliessen. Bisher ist nichts committet.
