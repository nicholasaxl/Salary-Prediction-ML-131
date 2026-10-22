"""Smoke test for the web app's data layer. Run from the repo root:  python tests/test_web_app.py

Checks that every endpoint answers, that the exported JSON is consistent with outputs/*.csv, and that the
written report text quotes the same numbers as the tables (so prose cannot drift from the charts).
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)
fails = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


# ---- endpoints
for url in ("/api/options", "/api/story", "/api/models", "/api/report", "/api/insights", "/", "/static/js/main.js", "/static/css/app.css"):
    check(client.get(url).status_code == 200, f"GET {url} -> 200")

r = client.post("/api/predict", json=dict(job_title="Data Scientist", experience_level="SE", employment_type="FT",
                                          company_location="US", company_size="M", work_year=2023, remote_ratio=0))
body = r.json()
check(r.status_code == 200 and 0 <= body["low"] < body["mid"] < body["high"], "predict returns an ordered, non-negative range")
check(client.post("/api/predict", json={**dict(job_title="Nope", experience_level="SE", employment_type="FT", company_location="US",
                                               company_size="M", work_year=2023, remote_ratio=0)}).status_code == 422, "unknown job title -> 422")
tiny = client.post("/api/predict", json=dict(job_title="Data Analyst", experience_level="EN", employment_type="FT",
                                             company_location="IN", company_size="S", work_year=2020, remote_ratio=0)).json()
check(tiny["low"] >= 0, "range never starts below $0")

# ---- story data vs tables
story, models, report = (client.get(f"/api/{n}").json() for n in ("story", "models", "report"))
stage = pd.read_csv("outputs/stage_results.csv").set_index(["family", "stage"])
check(len(story["yRaw"]) == 3755 and len(story["yDedup"]) == 2583 == len(story["keep"]), "story has 3,755 raw and 2,583 unique rows")
for tag in ("RF", "LGBM"):
    check(len(story["pred"][tag]) == 5 and len(story["pred"][tag][0]) == 3755 and all(len(a) == 2583 for a in story["pred"][tag][1:]),
          f"{tag}: stage 0 has raw length, stages 1-4 dedup length")
    for s in range(5):
        check(story["metrics"][tag][s]["rmse"] == round(stage.loc[(tag, s)].rmse), f"{tag} stage {s} RMSE equals stage_results.csv")
check(len(story["pred"]["BEST"][0]) == 2583, "best-model predictions have dedup length")
check(max(story["yRaw"]) * 100 <= 450_000, "salaries are in hundreds of dollars and within range")

# ---- report text quotes the table numbers
fmt = lambda v: f"${v:,.0f}"
for s in range(1, 5):
    txt = " ".join(report["sections"][f"stage{s}"]["happened"])
    for tag in ("RF", "LGBM"):
        check(fmt(stage.loc[(tag, s)].rmse) in txt, f"stage {s} report quotes {tag} RMSE {fmt(stage.loc[(tag, s)].rmse)}")
check(all(t in report["sections"] for t in ("stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "arena", "decision", "insights", "estimator", "explorer", "limits", "hero")),
      "report has a section for every stage and page area")
check(len(report["sections"]["insights"]["decisions"]) == 6, "insights report has six business decisions")
check(models["catboostTuned"] is False and not any(r["key"] == "tuned_CatBoost" for r in models["search"]), "CatBoost is default-only everywhere")
check(not re.search(r"under 1%", json.dumps(report)), "no stale hand-typed percentage claims in report text")

print("\n%d failed" % len(fails) if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
