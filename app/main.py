"""FastAPI app: salary estimator + data explorer.

Run from the repo root:  python -m uvicorn app.main:app --reload
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .model import CAT, FEATURES, LABELS, MIN_SUPPORT, ROOT, TARGET, SalaryModel, load_data

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Data science salary estimator")

df = load_data()
model = SalaryModel(df)

GROUPABLE = ["job_title", "experience_level", "company_location", "company_size", "employment_type", "work_year", "remote_ratio"]


class Profile(BaseModel):
    job_title: str
    experience_level: str
    employment_type: str
    company_location: str
    company_size: str
    work_year: int
    remote_ratio: int


def label(feature: str, value) -> str:
    return LABELS.get(feature, {}).get(value, str(value))


@app.get("/api/options")
def options():
    out = {}
    for c in CAT:
        counts = df[c].value_counts()
        out[c] = [{"value": v, "label": label(c, v), "n": int(counts[v])} for v in model.categories[c]]
    out["work_year"] = [int(y) for y in sorted(df.work_year.unique())]
    out["remote_ratio"] = [{"value": int(v), "label": label("remote_ratio", int(v))} for v in sorted(df.remote_ratio.unique())]
    return {"options": out, "rows": int(len(df)), "min_support": MIN_SUPPORT}


@app.post("/api/predict")
def predict(p: Profile):
    row = p.model_dump()
    for c in CAT:
        if row[c] not in model.categories[c]:
            raise HTTPException(422, f"Unknown {c}: {row[c]!r}")
    if row["work_year"] not in set(df.work_year) or row["remote_ratio"] not in set(df.remote_ratio):
        raise HTTPException(422, "work_year or remote_ratio outside the training data")

    result = model.predict(row)
    support = model.support(row)
    warnings = []
    if support["job_title"] < MIN_SUPPORT:
        warnings.append(f"Only {support['job_title']} rows have this job title, so this estimate is mostly a guess.")
    if support["company_location"] < MIN_SUPPORT:
        warnings.append(f"Only {support['company_location']} rows are from this company location, so location effects are unreliable.")
    result["warnings"] = warnings
    result["peers"] = model.peers(row)
    for f in result["factors"]:
        f["value"] = label(f["feature"], row[f["feature"]])
    return result


@app.get("/api/explore")
def explore(
    group_by: str = "job_title",
    min_n: int = Query(10, ge=1),
    job_title: Optional[str] = None,
    experience_level: Optional[str] = None,
    company_location: Optional[str] = None,
    company_size: Optional[str] = None,
    employment_type: Optional[str] = None,
    work_year: Optional[int] = None,
    remote_ratio: Optional[int] = None,
):
    if group_by not in GROUPABLE:
        raise HTTPException(422, f"group_by must be one of {GROUPABLE}")
    d = df
    filters = dict(
        job_title=job_title, experience_level=experience_level, company_location=company_location,
        company_size=company_size, employment_type=employment_type, work_year=work_year, remote_ratio=remote_ratio,
    )
    for col, val in filters.items():
        if val is not None:
            d = d[d[col] == val]
    if d.empty:
        return {"rows": 0, "groups": [], "overall": None}

    g = d.groupby(group_by)[TARGET]
    stats = g.agg(n="count", median="median", mean="mean", min="min", max="max")
    stats["p25"] = g.quantile(0.25)
    stats["p75"] = g.quantile(0.75)
    stats = stats[stats.n >= min_n].sort_values("median", ascending=False)
    groups = [
        {
            "key": k.item() if hasattr(k, "item") else k,
            "label": label(group_by, k.item() if hasattr(k, "item") else k),
            **{c: float(r[c]) for c in ("median", "mean", "min", "max", "p25", "p75")},
            "n": int(r.n),
        }
        for k, r in stats.iterrows()
    ]
    s = d[TARGET]
    bins = np.histogram(s, bins=24)
    return {
        "rows": int(len(d)),
        "hidden_groups": int((g.count() < min_n).sum()),
        "groups": groups,
        "overall": {"median": float(s.median()), "p25": float(s.quantile(0.25)), "p75": float(s.quantile(0.75))},
        "histogram": {"counts": bins[0].tolist(), "edges": [float(e) for e in bins[1]]},
    }


DATA_DIR = Path(__file__).parent / "static" / "data"


def _data_file(name: str):
    path = DATA_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"{path.name} is missing. Run `python tools/export_web_data.py` first.")
    return FileResponse(path, media_type="application/json", headers={"Cache-Control": "no-cache"})


@app.get("/api/story")
def story():
    """Points, per-stage predictions and decile lines for the animated progression."""
    return _data_file("story")


@app.get("/api/models")
def models():
    """Model search table, decision and interval comparison."""
    return _data_file("models")


@app.get("/api/report")
def report():
    """Written report for every stage and section."""
    return _data_file("report")


@app.get("/api/insights")
def insights():
    """Premiums, pay bands and reliability, precomputed by `python insights.py`."""
    path = ROOT / "outputs" / "insights.json"
    if not path.exists():
        raise HTTPException(404, "outputs/insights.json is missing. Run `python insights.py` first.")
    return json.loads(path.read_text())


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
