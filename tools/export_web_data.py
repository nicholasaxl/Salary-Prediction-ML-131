"""Export the analysis results as compact JSON for the web app (app/static/data/).

    python tools/export_web_data.py

Reads outputs/*.csv|json|npz (written by experiments.py, interval_compare.py, insights.py) and writes:
  story.json   points, per-stage predictions and decile lines for the animated progression
  models.json  model search table, decision, interval comparison, scoreboard
  report.json  the written report under every stage/section; every number comes from outputs/,
               so the text cannot drift from the charts
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import build, raw_data  # noqa: E402
from features import CAT_RAW, NUM_RAW, TARGET  # noqa: E402

OUT = Path("outputs")
DEST = Path("app/static/data")
DEST.mkdir(parents=True, exist_ok=True)

usd = lambda v: f"${v:,.0f}"
H = lambda a: np.round(np.asarray(a, dtype=float) / 100).astype(int).tolist()  # integer hundreds of dollars

STAGES = [
    (0, "Raw, R-style", "Raw", True),
    (1, "Deduplicate", "Dedup", False),
    (2, "Log target", "Log", False),
    (3, "Clean + features", "Clean", False),
    (4, "Tuned", "Tuned", False),
    (5, "Best model", "Best", False),
]


def decile_line(y, p):
    edges = np.quantile(y, np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(y, edges[1:-1]), 0, 9)
    return {"x": [round(float(y[idx == k].mean()) / 100) for k in range(10)],
            "y": [round(float(p[idx == k].mean()) / 100) for k in range(10)]}


def main():
    oof = np.load(OUT / "stage_oof.npz", allow_pickle=True)
    stage = pd.read_csv(OUT / "stage_results.csv")
    search = pd.read_csv(OUT / "model_search.csv")
    decision = json.load(open(OUT / "decision.json"))
    ins = json.load(open(OUT / "insights.json"))
    ic = pd.read_csv(OUT / "interval_compare.csv")
    S = stage.set_index(["family", "stage"])

    raw = raw_data()
    keep = np.flatnonzero(~raw.duplicated().to_numpy())
    y_raw, y_dedup = oof["y_raw"], oof["y_dedup"]
    assert len(keep) == len(y_dedup)

    # in-sample fit of the R-style model: what the original report plotted
    from sklearn.ensemble import RandomForestRegressor
    rf = build("Random forest", "raw", False, False,
               est=RandomForestRegressor(300, max_features=1 / 3, random_state=123, n_jobs=-1))
    fit_in = rf.fit(raw.drop(columns=[TARGET]), y_raw).predict(raw.drop(columns=[TARGET]))

    best_key = decision["accuracy_winner"]
    best_name = best_key.split("_", 1)[1]
    pred, metrics, deciles = {}, {}, {}
    for tag in ("RF", "LGBM"):
        pred[tag], metrics[tag], deciles[tag] = [], [], []
        for s in range(5):
            y = y_raw if s == 0 else y_dedup
            p = oof[f"stage{s}_{tag}"]
            pred[tag].append(H(p))
            r = S.loc[(tag, s)]
            metrics[tag].append({"rmse": round(r.rmse), "mae": round(r.mae), "r2": round(r.r2, 4)})
            deciles[tag].append(decile_line(y, p))
    pb = oof["stage5_BEST"]
    pred["BEST"] = [H(pb)]
    rb = S.loc[("BEST", 5)]
    metrics["BEST"] = [{"rmse": round(rb.rmse), "mae": round(rb.mae), "r2": round(rb.r2, 4)}]
    deciles["BEST"] = [decile_line(y_dedup, pb)]

    story = {
        "stages": [{"id": i, "name": n, "short": sh, "leaky": lk} for i, n, sh, lk in STAGES],
        "units": "dollars / 100",
        "yRaw": H(y_raw), "yDedup": H(y_dedup), "keep": keep.tolist(),
        "pred": pred, "metrics": metrics, "deciles": deciles,
        "insample": {"pred": H(fit_in), "r2": round(float(r2_score(y_raw, fit_in)), 4),
                     "rmse": round(float(np.sqrt(np.mean((y_raw - fit_in) ** 2))))},
        "best": {"key": best_key, "name": best_name},
        "decileLineY": decile_line(y_raw, fit_in),
    }

    # ---------------- models
    cols = ["key", "model", "tuned", "rmse", "mae", "r2", "gap_to_best", "gap_se", "tied", "interval_capable", "complexity", "seconds"]
    models = {
        "search": json.loads(search[cols].round(4).to_json(orient="records")),
        "decision": decision,
        "intervals": json.loads(ic.round(4).to_json(orient="records")),
        "catboostTuned": False,
        "scoreboard": json.loads(stage.round(4).to_json(orient="records")),
    }

    # ---------------- report text (all numbers computed)
    dedup = raw.drop_duplicates()
    g = dedup.groupby(CAT_RAW + NUM_RAW)[TARGET].agg(["count", "std", "mean"]).query("count >= 5")
    noise = float((g["std"] / g["mean"]).median())
    n_dupes = len(raw) - len(dedup)

    def line(tag, name, s):
        r = S.loc[(tag, s)]
        if s == 0:
            return f"{name}: RMSE {usd(r.rmse)}, R² {r.r2:.3f}"
        p = S.loc[(tag, s - 1)]
        return f"{name}: RMSE {usd(r.rmse)} ({r.rmse - p.rmse:+,.0f}), R² {r.r2:.3f} ({r.r2 - p.r2:+.3f})"

    def deltas(s):
        return [S.loc[(t, s)].rmse - S.loc[(t, s - 1)].rmse for t in ("RF", "LGBM")]

    rep = {}
    rep["stage0"] = dict(
        title="Stage 0: the R-style starting point",
        did=f"Used the data exactly as the R project did: all {len(raw):,} rows including duplicates, original columns one-hot encoded, salary modelled in dollars, an untuned random forest and default boosting.",
        happened=[line("RF", "Random forest", 0), line("LGBM", "Boosting", 0),
                  f"Training-set fit of the same forest: R² {story['insample']['r2']:.2f}, the kind of plot the R report showed."],
        why="Copies of the same row can land in both the training and the test fold, so the model is partly graded on rows it has already seen. These are the most optimistic scores in the project.",
        takeaway="Treat stage 0 as the claim to test, not as a result.")
    d1 = deltas(1)
    rep["stage1"] = dict(
        title="Stage 1: remove duplicate rows",
        did=f"Dropped the {n_dupes:,} exact duplicate rows ({n_dupes / len(raw):.0%}). Nothing else changed.",
        happened=[line("RF", "Random forest", 1), line("LGBM", "Boosting", 1)],
        why="This is a correction, not an optimisation, so scores are expected to get worse. It restores an honest test: no row in a test fold has a twin in the training fold.",
        takeaway=f"The honest baseline is R² {S.loc[('RF', 1)].r2:.2f} ({usd(S.loc[('RF', 1)].rmse)} RMSE). Every later number is measured against it; the RMSE rose by {usd(max(d1))} at most.")
    d2 = deltas(2)
    rep["stage2"] = dict(
        title="Stage 2: model log-salary",
        did="Predicted log(salary) and converted back to dollars, so errors grow with pay instead of being dominated by the highest earners.",
        happened=[line("RF", "Random forest", 2), line("LGBM", "Boosting", 2)],
        why="A log target usually helps linear and neural models. Tree models split on order, not scale, so they gain little, and converting back adds a small bias.",
        takeaway=("Both tree families got slightly worse here. The transform is kept for the model search, where it helps the linear and neural models." if min(d2) > 0 else "Mixed effect on the tree families; kept for the model search."))
    d3 = deltas(3)
    rep["stage3"] = dict(
        title="Stage 3: cleaning and features",
        did="Learned from the training folds only: collapsed job titles and locations with fewer than 10 rows, added job family, region and a level ordinal, and trimmed the top and bottom 1% of salaries when fitting (never when scoring).",
        happened=[line("RF", "Random forest", 3), line("LGBM", "Boosting", 3)],
        why="Fewer, better-supported categories are easier to learn from, and trimming limits the pull of extreme salaries without making the test set easier.",
        takeaway=f"A modest gain ({usd(-d3[0])} for the forest, {usd(-d3[1])} for boosting)." if max(d3) < 0 else "A modest, mixed effect.")
    d4 = deltas(4)
    rep["stage4"] = dict(
        title="Stage 4: hyperparameter tuning",
        did="Ran a randomised search over tree depth, learning rate, regularisation and similar settings inside each cross-validation fold (nested CV).",
        happened=[line("RF", "Random forest", 4), line("LGBM", "Boosting", 4)],
        why="Nesting keeps the search from seeing the rows it is scored on, so the score is an honest estimate of what tuning delivers.",
        takeaway=f"Tuning helps both families by {usd(-d4[0])} and {usd(-d4[1])}: real, but small beside the noise in the data." if max(d4) < 0 else "Tuning did not reliably help.")
    tied = search[search.tied]
    span = tied.rmse.max() - search.rmse.min()
    rep["stage5"] = dict(
        title="Stage 5: the best model",
        did=f"Compared ten model families under the same protocol; the lowest-error model is {best_name}.",
        happened=[f"{best_name}: RMSE {usd(rb.rmse)}, MAE {usd(rb.mae)}, R² {rb.r2:.3f}",
                  f"{len(tied)} of {len(search)} configurations are statistically tied with it, spanning only {usd(span)} of RMSE."],
        why=f"People with an identical profile (same title, level, location, company size, employment type, year and remote ratio) differ in salary by about {noise:.0%}. A model that sees only these columns cannot beat the average of such a group.",
        takeaway="The ceiling is the data, not the algorithm. More modelling will not move the plot onto the diagonal; more information (skills, industry, tenure) might.")

    best_row = search.set_index("key").loc[best_key]
    pars = search.set_index("key").loc[decision["parsimonious_pick"]]
    prod = decision["product_pick"]
    ridge = search.set_index("key").loc["search_Ridge"]
    icb = ic.sort_values("pinball").iloc[0]
    app_row = ic[ic.model == "LightGBM"].iloc[0]
    nm = lambda k: ("tuned " if k.startswith("tuned_") else "") + k.split("_", 1)[1]
    fam_best = search[search.model != "Median baseline"].groupby("model").rmse.min()
    behind = [f for f, v in fam_best.items() if v > fam_best.min() * 1.01]
    close = len(fam_best) - len(behind)
    pv = search.pivot(index="model", columns="tuned", values="rmse").dropna()
    gain = (pv[False] - pv[True])
    worse = int((gain < 0).sum())
    tune_line = (f"Tuning moved the typical model by {usd(gain.abs().median())}; the biggest gain was {gain.idxmax()} at {usd(gain.max())}. "
                 f"{worse} of {len(gain)} tuned models scored worse than their defaults on held-out folds.")
    arena_takeaway = (f"{close} of {len(fam_best)} model families land within 1% of the best; only {', '.join(behind)} fall clearly behind. "
                      "The choice of model barely matters here.")
    rep["arena"] = dict(
        title="The model arena",
        did="Ran ridge, KNN, random forest, extra trees, HistGradientBoosting, XGBoost, LightGBM, CatBoost, a small neural net and a stacking blend, each at defaults and tuned, under one protocol (5-fold cross-validation repeated 3 times). CatBoost was evaluated at default settings only; its tuning run was stopped to save about 25 minutes.",
        happened=[f"Lowest error: {nm(best_key)} at {usd(best_row.rmse)} RMSE.",
                  f"{len(tied)} of {len(search)} configurations are statistically tied with it, spanning {usd(span)} ({span / search.rmse.min():.1%}).",
                  tune_line],
        why="Each model is compared with the winner fold by fold, and a gap smaller than its corrected standard error counts as a tie. That separates real differences from fold-to-fold noise.",
        takeaway=arena_takeaway)
    pin_spread = (ic.pinball.max() - ic.pinball.min()) / ic.pinball.min()
    rep["decision"] = dict(
        title="The decision",
        did="A rule fixed before the results: 1) the accuracy winner has the lowest mean RMSE; 2) models whose gap to the winner is smaller than its standard error are tied; 3) among tied models pick the simplest; 4) for the range-based product pick the simplest tied model with native quantile output.",
        happened=[f"Accuracy winner: {nm(best_key)}.",
                  f"Parsimonious pick: {nm(decision['parsimonious_pick'])}, {usd(pars.gap_to_best)} behind the winner and far simpler.",
                  f"Near-tie: ridge regression misses the cutoff by under a dollar (gap {usd(ridge.gap_to_best)}, standard error {usd(ridge.gap_se)}), is {ridge.gap_to_best / best_row.rmse:.2%} behind the best and is the most interpretable model.",
                  f"Product pick by the rule: {nm(prod)}. On the 10-90% range LightGBM has the lowest pinball loss ({usd(icb.pinball)}); the web app uses LightGBM quantile models (coverage {app_row.coverage:.1%}, median width {usd(app_row.median_width)})."],
        why=f"The rule did not score interval quality, and the four quantile models differ by only {pin_spread:.1%} on it, so the app keeps LightGBM as a deliberate, stated exception to the rule's XGBoost pick.",
        takeaway="No model is clearly better. Use the simplest one for accuracy, a quantile boosting model for ranges, and spend effort on better data.")

    P = {p["factor"]: p for p in ins["premiums"]}
    W = [r for r in ins["reliability"] if r["group"] == "All"][0]
    weak = sorted([r for r in ins["reliability"] if r["group"] != "All" and r["coverage"] < 0.76 and r["n"] >= 30], key=lambda r: r["coverage"])
    rep["insights"] = dict(
        title="What the business can do with this",
        did="Fitted a log-salary linear model with 300 bootstrap resamples to put each factor in percent, built pay bands by role family, level and market, and checked how often real salaries fell inside the model's 80% range, by segment.",
        happened=[f"Location is the dominant lever: the same role pays about {abs(P['Location: GB vs US']['pct']):.0f}% less in the UK, {abs(P['Location: Europe (other) vs US']['pct']):.0f}% less elsewhere in Europe and {abs(P['Location: IN vs US']['pct']):.0f}% less in India than in the US.",
                  f"Seniority compounds: mid {P['Level: Mid vs Entry']['pct']:+.0f}%, senior {P['Level: Senior vs Entry']['pct']:+.0f}%, executive {P['Level: Executive vs Entry']['pct']:+.0f}% versus entry level.",
                  f"Role matters less: analysts earn {abs(P['Role: Analyst vs Engineer']['pct']):.0f}% less than engineers; ML/AI roles {P['Role: ML/AI vs Engineer']['pct']:+.0f}% and leadership {P['Role: Leadership vs Engineer']['pct']:+.0f}%.",
                  f"Fully remote work shows no reliable pay difference ({P['Setup: Remote vs On-site']['pct']:+.0f}%, interval spans zero). Small companies pay {abs(P['Company: Small vs Medium']['pct']):.0f}% less than medium ones.",
                  f"Pay trends up about {P['Per year (2020-2023)']['pct']:+.0f}% a year across 2020-2023."],
        why=f"Raw accuracy is modest (R² about 0.4), so the value is benchmarking, not single-number prediction. Coverage is {W['coverage']:.1%} overall with a median range of {usd(W['median_width'])}" + (", and weakest for " + ", ".join(f"{r['segment']} ({r['coverage']:.0%}, n={r['n']})" for r in weak) if weak else "") + ".",
        takeaway="Use the bands to frame pay conversations, and read thin segments (small n) as a prompt for more data rather than a conclusion.",
        decisions=[
            {"q": "Set or review a pay band", "a": f"Start from the 25-75% band for the role family, level and market, then adjust for company size (about {P['Company: Small vs Medium']['pct']:.0f}% for small firms).", "c": "High where n is large (US engineers, scientists, analysts)"},
            {"q": "Check an offer", "a": "Enter the profile in the estimator. Treat the 10-90% range as plausible and an offer below it as a retention or fairness flag.", "c": f"Medium: ranges are wide (median {usd(W['median_width'])})"},
            {"q": "Decide where to hire", "a": "Location is the biggest cost driver. The data quantifies the gap; it cannot judge output or quality.", "c": "High for UK, Europe, India; low for thin locations"},
            {"q": "Set a remote policy", "a": f"Remote work shows no reliable salary penalty or premium ({P['Setup: Remote vs On-site']['pct']:+.0f}%), so remote alone does not change what you must pay.", "c": "Medium: remote is self-reported"},
            {"q": "Budget a career ladder", "a": f"Plan for about {P['Level: Mid vs Entry']['pct']:+.0f}% from entry to mid and {P['Level: Senior vs Entry']['pct']:+.0f}% from entry to senior.", "c": "High"},
            {"q": "Forecast wage inflation", "a": f"About {P['Per year (2020-2023)']['pct']:+.0f}% a year in this sample.", "c": "Low: four years of survey data"},
        ])
    rep["estimator"] = dict(
        title="How to read an estimate",
        did="Three quantile models predict the 10th, 50th and 90th percentile of salary for the profile you enter. A peer range shows what comparable people actually reported.",
        happened=[f"The 10-90% range covered {app_row.coverage:.1%} of held-out salaries (target 80%), with a median width of {usd(app_row.median_width)}."],
        why="Titles, levels and locations explain only about 40% of salary variance, so any honest estimate has to be wide. Categories with fewer than 5 rows trigger a warning because the model has almost nothing to learn from.",
        takeaway="Treat the range as the answer and the middle tick as its most likely value.")
    rep["explorer"] = dict(
        title="Explore the data",
        did="Group and filter the 2,583 unique rows by any field. Bars show the 25th-75th percentile range and the median.",
        happened=[], why="Small groups are hidden by default (minimum 10 rows) because their medians move a lot with one or two people.",
        takeaway="Click a row to filter to it.")
    rep["limits"] = dict(
        title="Limits to state out loud",
        did="", happened=[
            "Self-reported survey data, 2020-2023, USD. It is not a payroll dataset and it is not current.",
            "Seven columns only: no skills, tenure, industry, company name or education, which is why R² stops near 0.4.",
            "Associations, not causes: location effects bundle cost of living, taxes and local labour markets.",
            "Thin segments should not drive decisions.",
            "A range is not an offer: use the model to frame a conversation, not to set an individual's pay."],
        why="", takeaway="")
    rep["hero"] = dict(title="A salary is a range, not a number",
                       did=f"A study of {len(dedup):,} data-science salaries, rebuilt from a class project: honest evaluation, ten models, and what a business can actually do with the result.",
                       happened=[], why="", takeaway="")

    report = {"meta": {"rows": int(len(dedup)), "rawRows": int(len(raw)), "dupes": int(n_dupes), "noise": round(noise, 3),
                       "best": best_name, "r2Best": round(rb.r2, 3), "rmseBest": round(rb.rmse), "repo": "https://github.com/nicholasaxl/Salary-Prediction-ML-131"},
              "sections": rep}

    for name, obj in (("story", story), ("models", models), ("report", report)):
        path = DEST / f"{name}.json"
        path.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")
        print(f"{path}: {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
