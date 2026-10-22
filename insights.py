"""Business-facing summaries of the salary data, written to outputs/insights.json.

Used by the notebook and the web app's Insights tab:
  * pay premiums (percent, bootstrap intervals) from a log-salary linear model
  * pay bands (median and 25-75%) by job family, level and region
  * reliability: out-of-fold coverage of the 80% range by segment
Run from the repo root:  python insights.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from features import LEVEL, TARGET, job_family, region
from model_comparison import load
from tune_models import oof_intervals

OUT = Path("outputs")
LEVEL_NAMES = {"EN": "Entry", "MI": "Mid", "SE": "Senior", "EX": "Executive"}
N_BOOT = 300
SEED = 7


def prepared() -> pd.DataFrame:
    d = load(dedup=True)
    d["family"] = d.job_title.map(job_family)
    d["region"] = d.company_location.map(region)
    d["level"] = d.experience_level.map(LEVEL_NAMES)
    return d


def design(d: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=d.index)
    for lv in ("MI", "SE", "EX"):
        X[f"Level: {LEVEL_NAMES[lv]} vs Entry"] = (d.experience_level == lv).astype(float)
    for r in ("CA", "GB", "IN", "Europe (other)", "Rest of world"):
        X[f"Location: {r} vs US"] = (d.region == r).astype(float)
    for f in ("Scientist", "Analyst", "ML/AI", "Leadership", "Architect", "Other"):
        X[f"Role: {f} vs Engineer"] = (d.family == f).astype(float)
    X["Company: Small vs Medium"] = (d.company_size == "S").astype(float)
    X["Company: Large vs Medium"] = (d.company_size == "L").astype(float)
    X["Setup: Hybrid vs On-site"] = (d.remote_ratio == 50).astype(float)
    X["Setup: Remote vs On-site"] = (d.remote_ratio == 100).astype(float)
    X["Per year (2020-2023)"] = (d.work_year - d.work_year.min()).astype(float)
    return X


def premiums(d: pd.DataFrame) -> list:
    X, y = design(d), np.log(d[TARGET])
    point = LinearRegression().fit(X, y).coef_
    rng = np.random.default_rng(SEED)
    boots = np.empty((N_BOOT, X.shape[1]))
    for i in range(N_BOOT):
        idx = rng.integers(0, len(d), len(d))
        boots[i] = LinearRegression().fit(X.iloc[idx], y.iloc[idx]).coef_
    lo, hi = np.percentile(boots, [5, 95], axis=0)
    n = (X != 0).sum()
    return [
        {"factor": c, "pct": float(np.expm1(p) * 100), "lo": float(np.expm1(a) * 100), "hi": float(np.expm1(b) * 100),
         "n": int(n[c]), "significant": bool(a > 0 or b < 0)}
        for c, p, a, b in zip(X.columns, point, lo, hi)
    ]


def bands(d: pd.DataFrame, min_n=10) -> list:
    d = d.assign(market=np.where(d.region == "US", "US", "Outside US"))
    g = d.groupby(["family", "level", "market"])[TARGET]
    s = g.agg(n="count", median="median", p25=lambda x: x.quantile(.25), p75=lambda x: x.quantile(.75)).reset_index()
    s = s[s.n >= min_n]
    return [{"family": r.family, "level": r.level, "market": r.market, "n": int(r.n),
             "median": float(r["median"]), "p25": float(r.p25), "p75": float(r.p75)} for _, r in s.iterrows()]


def reliability(d: pd.DataFrame) -> list:
    lo, hi = oof_intervals(d)
    y = d[TARGET].to_numpy()
    d = d.assign(inside=(y >= lo) & (y <= hi), width=hi - lo)
    rows = []
    for col, label in (("region", "Location"), ("level", "Level"), ("family", "Role")):
        for k, g in d.groupby(col):
            rows.append({"group": label, "segment": str(k), "n": int(len(g)), "coverage": float(g.inside.mean()),
                         "median_width": float(g.width.median())})
    rows.append({"group": "All", "segment": "All rows", "n": int(len(d)), "coverage": float(d.inside.mean()),
                 "median_width": float(d.width.median())})
    return rows


def build() -> dict:
    d = prepared()
    return {
        "rows": int(len(d)),
        "median_salary": float(d[TARGET].median()),
        "years": [int(d.work_year.min()), int(d.work_year.max())],
        "premiums": premiums(d),
        "bands": bands(d),
        "reliability": reliability(d),
    }


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    data = build()
    (OUT / "insights.json").write_text(json.dumps(data, indent=1))
    for p in data["premiums"]:
        print(f"{p['factor']:<34} {p['pct']:+6.1f}%  [{p['lo']:+6.1f}, {p['hi']:+6.1f}]  n={p['n']}")
    print(len(data["bands"]), "bands;", len(data["reliability"]), "reliability rows")
