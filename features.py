"""Cleaning and feature-engineering building blocks used by experiments.py.

Everything that learns from data (rare-category sets, tail quantiles) is fit on the
training fold only, so held-out rows are never touched by cleaning.
"""
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone

CAT_RAW = ["experience_level", "employment_type", "job_title", "company_location", "company_size"]
NUM_RAW = ["work_year", "remote_ratio"]
TARGET = "salary_in_usd"

CAT_ENG = ["experience_level", "employment_type", "company_size", "job_title_c", "location_c", "job_family", "region"]
NUM_ENG = ["work_year", "remote_ratio", "lvl"]

LEVEL = {"EN": 0, "MI": 1, "SE": 2, "EX": 3}
_EUROPE = {
    "GB", "DE", "FR", "ES", "NL", "PT", "IE", "IT", "PL", "GR", "CH", "AT", "BE", "SE", "DK", "FI", "NO", "LT",
    "EE", "LV", "RO", "HU", "CZ", "SI", "HR", "LU", "MT", "BA", "MK", "AL", "CY", "BG", "RS", "UA", "AD", "GI", "MD",
}


def job_family(title: str) -> str:
    t = title.lower()
    if re.search(r"manager|head|director|lead|principal", t):
        return "Leadership"
    if re.search(r"machine learning|\bml\b|\bai\b|nlp|computer vision|deep", t):
        return "ML/AI"
    if "scien" in t:
        return "Scientist"
    if "analy" in t or "bi " in t or "business intel" in t:
        return "Analyst"
    if "architect" in t:
        return "Architect"
    if "engineer" in t or "developer" in t:
        return "Engineer"
    return "Other"


def region(code: str) -> str:
    if code in ("US", "CA", "GB", "IN"):
        return code
    return "Europe (other)" if code in _EUROPE else "Rest of world"


class Engineer(BaseEstimator, TransformerMixin):
    """Collapse rare titles/locations (learned on train) and add job family, region and a level ordinal."""

    def __init__(self, min_title=10, min_location=10):
        self.min_title = min_title
        self.min_location = min_location

    def fit(self, X, y=None):
        t, c = X["job_title"].value_counts(), X["company_location"].value_counts()
        self.titles_ = set(t[t >= self.min_title].index)
        self.locations_ = set(c[c >= self.min_location].index)
        return self

    def transform(self, X):
        out = X[["experience_level", "employment_type", "company_size", "work_year", "remote_ratio"]].copy()
        out["job_title_c"] = X["job_title"].where(X["job_title"].isin(self.titles_), "Other")
        out["location_c"] = X["company_location"].where(X["company_location"].isin(self.locations_), "Other")
        out["job_family"] = X["job_title"].map(job_family)
        out["region"] = X["company_location"].map(region)
        out["lvl"] = X["experience_level"].map(LEVEL).astype(float)
        return out


class TrimFit(BaseEstimator, RegressorMixin):
    """Fit `base` on training rows whose target lies inside [lo, hi] quantiles. Prediction is untouched."""

    def __init__(self, base, lo=0.01, hi=0.99):
        self.base = base
        self.lo = lo
        self.hi = hi

    def fit(self, X, y):
        y = np.asarray(y)
        a, b = np.quantile(y, [self.lo, self.hi])
        keep = (y >= a) & (y <= b)
        self.n_trimmed_ = int((~keep).sum())
        self.model_ = clone(self.base).fit(X.loc[keep] if hasattr(X, "loc") else X[keep], y[keep])
        return self

    def predict(self, X):
        return self.model_.predict(X)


class CatBoostWrap(BaseEstimator, RegressorMixin):
    """CatBoost with the engineered categorical columns declared, clone-safe for scikit-learn."""

    def __init__(self, iterations=400, learning_rate=0.05, depth=5, l2_leaf_reg=3.0, random_seed=123):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed

    def fit(self, X, y):
        from catboost import CatBoostRegressor

        self.model_ = CatBoostRegressor(
            iterations=self.iterations, learning_rate=self.learning_rate, depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg, random_seed=self.random_seed, thread_count=1, verbose=0,
            allow_writing_files=False, cat_features=[c for c in CAT_ENG if c in X.columns],
        ).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)
