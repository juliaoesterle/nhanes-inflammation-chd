"""
The six classifiers of the course (W5-W9) with their tuning grids, and the inner CV loop.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config as cfg


def make_model(algo, class_weight=None):
    """Returns (estimator, grid). Grid keys refer to the pipeline step 'clf'."""
    seed = cfg.RANDOM_SEED
    if algo == "LR":          # unpenalized, the classical form of clinical risk models
        return LogisticRegression(C=np.inf, max_iter=10_000, class_weight=class_weight), {}
    if algo == "L1":          # lasso, used for feature selection in E2
        return (LogisticRegression(l1_ratio=1.0, solver="liblinear", C=0.1, max_iter=5_000,
                                   class_weight=class_weight, random_state=seed),
                {"clf__C": [0.003, 0.01, 0.03, 0.1, 0.3]})
    if algo == "EN":          # ridge / elastic net / lasso
        return (LogisticRegression(solver="saga", l1_ratio=0.5, C=1.0, max_iter=20_000,
                                   class_weight=class_weight, random_state=seed),
                {"clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                 "clf__l1_ratio": [0.0, 0.5, 1.0]})
    if algo == "DT":
        return (DecisionTreeClassifier(criterion="gini", random_state=seed, class_weight=class_weight),
                {"clf__max_depth": [3, 4, 5, 6, 8],
                 "clf__min_samples_leaf": [20, 50, 100]})
    if algo == "kNN":
        return (KNeighborsClassifier(),
                {"clf__n_neighbors": [50, 100, 200, 400, 800],
                 "clf__weights": ["uniform", "distance"],
                 "clf__p": [1, 2]})
    if algo == "SVM":         # probabilities by Platt scaling on out-of-fold decision values
        cw = cfg.SVM_CLASS_WEIGHT if class_weight is None else class_weight
        return (CalibratedClassifierCV(SVC(kernel="rbf", class_weight=cw),
                                       method="sigmoid", cv=3, ensemble=False),
                {"clf__estimator__C": [0.03, 0.1, 0.3, 1.0],
                 "clf__estimator__gamma": [0.002, 0.005, 0.02]})
    if algo == "RF":
        return (RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=seed,
                                       class_weight=class_weight),
                {"clf__max_features": ["sqrt", 0.33],
                 "clf__min_samples_leaf": [10, 30]})
    raise ValueError(algo)


def fit_tuned(pipe, grid, X, y, seed):
    """Inner loop of the nested CV: grid search with log-loss on the training part only."""
    if not grid:
        return pipe.fit(X, y), {}
    inner = StratifiedKFold(cfg.N_INNER_SPLITS, shuffle=True, random_state=seed)
    search = GridSearchCV(pipe, grid, scoring=cfg.TUNING_SCORING, cv=inner, n_jobs=1,
                          refit=True, error_score="raise")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        search.fit(X, y)
    return search.best_estimator_, search.best_params_
