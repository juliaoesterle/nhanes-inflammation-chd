"""
Preprocessing inside the CV folds: imputation, missing indicator, scaling, one-hot.
Everything is fitted on the training part only (it is part of the Pipeline).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as cfg
from models import make_model


class MissingFlag(BaseEstimator, TransformerMixin):
    """0/1 column '<name>_missing' (course W2: the missingness can carry information)."""

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        return pd.DataFrame({f"{c}_missing": X[c].isna().astype(float) for c in self.columns_},
                            index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_missing" for c in self.columns_], dtype=object)


def split_types(features):
    categorical = [f for f in features if f in cfg.CATEGORICAL_VARS]
    binary = [f for f in features if f in cfg.BINARY_VARS]
    continuous = [f for f in features if f not in categorical and f not in binary]
    indicators = [f for f in features if f in cfg.INDICATOR_VARS]
    return continuous, binary, categorical, indicators


def make_preprocessor(features, drop_first=False, indicators=True):
    """Two steps: (1) median / most frequent imputation + missing flag,
    (2) scaling of the continuous variables + one-hot of the categorical ones."""
    cont, binary, cat, ind = split_types(features)
    if not indicators:
        ind = []
    steps = [("cont", SimpleImputer(strategy="median"), cont),
             ("bin", SimpleImputer(strategy="most_frequent"), binary),
             ("cat", SimpleImputer(strategy="most_frequent"), cat),
             ("ind", MissingFlag(), ind)]
    impute = ColumnTransformer([s for s in steps if s[2]], remainder="drop",
                               verbose_feature_names_out=False)
    impute.set_output(transform="pandas")

    encode_steps = []
    if cont:
        encode_steps.append(("scale", StandardScaler(), cont))
    if cat:
        encode_steps.append(("onehot", OneHotEncoder(drop="first" if drop_first else None,
                                                     handle_unknown="ignore", sparse_output=False), cat))
    encode = ColumnTransformer(encode_steps, remainder="passthrough", verbose_feature_names_out=False)
    encode.set_output(transform="pandas")
    return [("impute", impute), ("encode", encode)]


def make_pipeline(features, algo, class_weight=None, indicators=True):
    """Preprocessing + classifier. Unpenalized LR drops the first category (no collinearity)."""
    clf, grid = make_model(algo, class_weight)
    pre = make_preprocessor(features, drop_first=(algo == "LR"), indicators=indicators)
    return Pipeline(pre + [("clf", clf)]), grid
