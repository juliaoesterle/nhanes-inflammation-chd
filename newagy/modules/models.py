"""
Machine Learning Model Zoo & Hyperparameter Grids.

Implements all six core algorithms taught in ETH Foundations of Data Science:
1. Logistic Regression (Unpenalized baseline, W6-W7)
2. Penalized Logistic Regression (Ridge / Lasso / ElasticNet, W5-W6)
3. Decision Tree (CART classifier, W7)
4. k-Nearest Neighbors (Instance-based learning, W6, W8)
5. Support Vector Machine (RBF kernel with Platt calibration and balanced weights, W9)
6. Random Forest (Ensemble bagging, W8)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config as cfg

# Algorithm identifiers
ALGO_LR = "LR"
ALGO_PENALIZED = "EN"
ALGO_DT = "DT"
ALGO_KNN = "KNN"
ALGO_SVM = "SVM"
ALGO_RF = "RF"

ALL_ALGORITHMS = [ALGO_LR, ALGO_PENALIZED, ALGO_DT, ALGO_KNN, ALGO_SVM, ALGO_RF]


def get_base_estimator(algo_name: str, random_state: int = cfg.RANDOM_SEED) -> BaseEstimator:
    """Instantiate the base estimator for the specified algorithm."""
    algo = algo_name.upper()

    if algo == ALGO_LR:
        # Standard unpenalized logistic regression
        return LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        )

    elif algo == ALGO_PENALIZED:
        # ElasticNet / Lasso / Ridge via saga solver
        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=2000,
            random_state=random_state,
        )

    elif algo == ALGO_DT:
        # CART Decision Tree
        return DecisionTreeClassifier(
            criterion="gini",
            random_state=random_state,
        )

    elif algo == ALGO_KNN:
        # k-Nearest Neighbors
        return KNeighborsClassifier(
            algorithm="auto",
        )

    elif algo == ALGO_SVM:
        # RBF Support Vector Machine with Platt sigmoid calibration
        # Note: class_weight='balanced' resolves rare-class collapse (course W9)
        base_svc = SVC(
            kernel="rbf",
            probability=False,
            class_weight="balanced",
            random_state=random_state,
        )
        return CalibratedClassifierCV(
            estimator=base_svc,
            method="sigmoid",
            cv=3,
        )

    elif algo == ALGO_RF:
        # Random Forest Ensemble
        return RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            random_state=random_state,
        )

    else:
        raise ValueError(f"Unknown algorithm: {algo_name}. Choose from {ALL_ALGORITHMS}")


def get_hyperparameter_grid(algo_name: str) -> Dict[str, list]:
    """
    Returns the hyperparameter tuning grid for inner cross-validation.
    Parameters are prefixed with 'classifier__' to target pipeline steps.
    """
    algo = algo_name.upper()

    if algo == ALGO_LR:
        # Baseline model has no hyperparameters to tune
        return {}

    elif algo == ALGO_PENALIZED:
        return {
            "classifier__C": [0.001, 0.01, 0.03, 0.1, 0.3, 1.0],
            "classifier__l1_ratio": [0.0, 0.5, 1.0],  # 0.0 = Ridge, 0.5 = ElasticNet, 1.0 = Lasso
        }

    elif algo == ALGO_DT:
        return {
            "classifier__max_depth": [3, 4, 5, 6, 8],
            "classifier__min_samples_leaf": [20, 50, 100],
        }

    elif algo == ALGO_KNN:
        return {
            "classifier__n_neighbors": [50, 100, 200, 400, 800],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2],  # 1 = Manhattan, 2 = Euclidean
        }

    elif algo == ALGO_SVM:
        return {
            "classifier__estimator__C": [0.03, 0.1, 0.3, 1.0],
            "classifier__estimator__gamma": [0.002, 0.005, 0.02],
        }

    elif algo == ALGO_RF:
        return {
            "classifier__max_features": ["sqrt", 0.33],
            "classifier__min_samples_leaf": [10, 30],
        }

    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
