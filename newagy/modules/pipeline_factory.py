"""
Preprocessing Pipeline Factory Module.

Constructs strict, leakage-free scikit-learn preprocessing pipelines.
All imputation statistics (medians), standardization parameters (means and SDs),
and one-hot encodings are fitted strictly on training folds within cross-validation.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as cfg


class MissingFlagTransformer(BaseEstimator, TransformerMixin):
    """
    Constructs a binary indicator (0.0 / 1.0) flagging whether values in the
    specified column were missing prior to imputation (course W2: 'missingness holds information').
    """

    def __init__(self, target_cols: List[str] | None = None):
        self.target_cols = target_cols or cfg.INDICATOR_VARS

    def fit(self, X: pd.DataFrame, y=None):
        self.cols_present_ = [c for c in self.target_cols if c in X.columns]
        return self

    def transform(self, X: pd.DataFrame):
        df_out = pd.DataFrame(index=X.index)
        for col in self.cols_present_:
            df_out[f"{col}_missing"] = X[col].isna().astype(float)
        return df_out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_missing" for c in self.cols_present_], dtype=object)


def categorize_feature_types(features: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Partition feature list into continuous, binary, categorical, and indicator subsets."""
    categorical = [f for f in features if f in cfg.CATEGORICAL_VARS]
    binary = [f for f in features if f in cfg.BINARY_VARS]
    continuous = [f for f in features if f not in categorical and f not in binary]
    indicators = [f for f in features if f in cfg.INDICATOR_VARS]
    return continuous, binary, categorical, indicators


def build_preprocessor(
    feature_list: List[str],
    imputation_strategy: str = "median",
    include_missing_indicators: bool = True,
    scale_continuous: bool = True,
    drop_first_category: bool = False,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer tailored to the specified features.
    
    Parameters:
    -----------
    feature_list: list of column names
    imputation_strategy: 'median', 'mean', or 'most_frequent'
    include_missing_indicators: whether to append missing indicator flags for income
    scale_continuous: whether to standardize continuous variables (z-score)
    drop_first_category: whether to drop first level in one-hot encoding (useful for unpenalized regression)
    """
    continuous_cols, binary_cols, categorical_cols, indicator_cols = categorize_feature_types(feature_list)

    transformers = []

    # 1. Continuous pipeline: Impute -> (Optional StandardScale)
    if continuous_cols:
        cont_steps = [("imputer", SimpleImputer(strategy=imputation_strategy))]
        if scale_continuous:
            cont_steps.append(("scaler", StandardScaler()))
        transformers.append(("continuous", Pipeline(cont_steps), continuous_cols))

    # 2. Binary pipeline: Most frequent impute -> Pass through
    if binary_cols:
        bin_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ])
        transformers.append(("binary", bin_pipeline, binary_cols))

    # 3. Categorical pipeline: Most frequent impute -> One-Hot Encode
    if categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first" if drop_first_category else None, handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", cat_pipeline, categorical_cols))

    # 4. Missingness indicators
    if include_missing_indicators and indicator_cols:
        ind_pipeline = Pipeline([
            ("flags", MissingFlagTransformer(target_cols=indicator_cols)),
        ])
        transformers.append(("indicators", ind_pipeline, indicator_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def assemble_full_pipeline(
    estimator: BaseEstimator,
    feature_list: List[str],
    scale_continuous: bool = True,
    imputation_strategy: str = "median",
    include_indicators: bool = True,
    drop_first_category: bool = False,
) -> Pipeline:
    """Combines feature preprocessing with an estimator into a single Scikit-Learn Pipeline."""
    preprocessor = build_preprocessor(
        feature_list=feature_list,
        imputation_strategy=imputation_strategy,
        include_missing_indicators=include_indicators,
        scale_continuous=scale_continuous,
        drop_first_category=drop_first_category,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])
