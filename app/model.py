"""Quantile salary model + data helpers for the web app."""
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parent.parent
CAT = ["experience_level", "employment_type", "job_title", "company_location", "company_size"]
NUM = ["work_year", "remote_ratio"]
FEATURES = CAT + NUM
TARGET = "salary_in_usd"
QUANTILES = {"low": 0.1, "mid": 0.5, "high": 0.9}  # 10-90 interval: ~77.5% out-of-fold coverage

LABELS = {
    "experience_level": {"EN": "Entry level", "MI": "Mid level", "SE": "Senior", "EX": "Executive"},
    "employment_type": {"FT": "Full time", "PT": "Part time", "CT": "Contract", "FL": "Freelance"},
    "company_size": {"S": "Small (<50)", "M": "Medium (50-250)", "L": "Large (>250)"},
    "remote_ratio": {0: "On-site", 50: "Hybrid", 100: "Fully remote"},
}
FEATURE_NAMES = {
    "experience_level": "Experience level",
    "employment_type": "Employment type",
    "job_title": "Job title",
    "company_location": "Company location",
    "company_size": "Company size",
    "work_year": "Year",
    "remote_ratio": "Remote ratio",
}
MIN_SUPPORT = 5  # rows needed before a category is considered well supported


def load_data() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "ds_salaries.csv")[FEATURES + [TARGET]]
    return df.drop_duplicates().reset_index(drop=True)


class SalaryModel:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.categories = {c: sorted(df[c].unique()) for c in CAT}
        X = self._frame(df)
        params = dict(
            n_estimators=250, learning_rate=0.04, num_leaves=8, min_child_samples=15,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=123, verbose=-1,
        )
        self.models = {
            k: LGBMRegressor(objective="quantile", alpha=q, **params).fit(X, df[TARGET])
            for k, q in QUANTILES.items()
        }

    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURES].copy()
        for c in CAT:
            X[c] = pd.Categorical(X[c], categories=self.categories[c])
        return X

    def support(self, row: dict) -> dict:
        """How many training rows share each categorical choice."""
        return {c: int((self.df[c] == row[c]).sum()) for c in ("job_title", "company_location", "experience_level")}

    def predict(self, row: dict) -> dict:
        X = self._frame(pd.DataFrame([row]))
        low, mid, high = (float(self.models[k].predict(X)[0]) for k in ("low", "mid", "high"))
        low, mid, high = sorted((low, mid, high))
        low = max(low, 0.0)  # a salary cannot be negative
        contrib = self.models["mid"].predict(X, pred_contrib=True)[0]
        base = float(contrib[-1])
        factors = sorted(
            (
                {"feature": f, "label": FEATURE_NAMES[f], "value": str(row[f]), "effect": float(v)}
                for f, v in zip(FEATURES, contrib[:-1])
            ),
            key=lambda d: abs(d["effect"]),
            reverse=True,
        )
        return {"low": low, "mid": mid, "high": high, "baseline": base, "factors": factors}

    def peers(self, row: dict) -> dict:
        """Observed salaries for similar people, widening the match until enough rows exist."""
        d = self.df
        steps = [
            ("same job title, experience level and company location", ["job_title", "experience_level", "company_location"]),
            ("same job title and experience level", ["job_title", "experience_level"]),
            ("same experience level and company location", ["experience_level", "company_location"]),
            ("same experience level", ["experience_level"]),
        ]
        for desc, cols in steps:
            m = d
            for c in cols:
                m = m[m[c] == row[c]]
            if len(m) >= MIN_SUPPORT:
                s = m[TARGET]
                return {
                    "match": desc, "n": int(len(m)),
                    "p25": float(s.quantile(0.25)), "median": float(s.median()), "p75": float(s.quantile(0.75)),
                }
        return {"match": "no comparable rows", "n": 0}
