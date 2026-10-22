"""Stage-by-stage progression and wider model search, all under one evaluation protocol.

Protocol: 5-fold CV repeated 3x. Cleaning (dedup excepted, which is a dataset-level fix) and
feature learning happen on training folds only; metrics and plots always use untouched held-out rows.

    python experiments.py stages     # stages 0-3 for random forest and LightGBM
    python experiments.py search     # all models, default and tuned, on the cleaned pipeline
    python experiments.py report     # assemble stage/model tables, decision and OOF predictions
    QUICK=1 python experiments.py ...  # tiny smoke-test settings, separate cache
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor,
                              StackingRegressor)
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from features import CAT_ENG, CAT_RAW, NUM_ENG, NUM_RAW, TARGET, CatBoostWrap, Engineer, TrimFit

warnings.filterwarnings("ignore")
QUICK = os.environ.get("QUICK") == "1"
OUT = Path("outputs")
CACHE = OUT / ("cache_quick" if QUICK else "cache")
CACHE.mkdir(parents=True, exist_ok=True)
SEED = 123
N_REPEATS = 1 if QUICK else 3
N_ITER = 2 if QUICK else 10
N_JOBS = 14
SKIP_TUNING = {"CatBoost"}  # default result only, noted in the notebook


# ---------------------------------------------------------------- data
def raw_data() -> pd.DataFrame:
    return pd.read_csv("ds_salaries.csv")[CAT_RAW + NUM_RAW + [TARGET]]


def dedup_data() -> pd.DataFrame:
    return raw_data().drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------- models
def estimators(seed=SEED):
    """name -> (estimator, scale_numeric, search_space or None). Single-threaded: parallelism is over folds."""
    return {
        "Median baseline": (DummyRegressor(strategy="median"), False, None),
        "Ridge": (RidgeCV(alphas=np.logspace(-2, 3, 20)), True, None),
        "KNN": (KNeighborsRegressor(n_neighbors=15, weights="distance"), True,
                {"model__n_neighbors": randint(5, 40)}),
        "Random forest": (RandomForestRegressor(300, min_samples_leaf=2, max_features=0.33, random_state=seed, n_jobs=1), False,
                          {"model__min_samples_leaf": randint(1, 15), "model__max_features": uniform(0.1, 0.7),
                           "model__max_depth": [None, 8, 12, 20]}),
        "Extra trees": (ExtraTreesRegressor(300, min_samples_leaf=3, max_features=0.5, random_state=seed, n_jobs=1), False,
                        {"model__min_samples_leaf": randint(1, 15), "model__max_features": uniform(0.1, 0.9)}),
        "HistGradientBoosting": (HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=8,
                                                               min_samples_leaf=15, random_state=seed), False,
                                 {"model__max_iter": randint(100, 500), "model__learning_rate": loguniform(0.01, 0.15),
                                  "model__max_leaf_nodes": randint(4, 32), "model__min_samples_leaf": randint(5, 40),
                                  "model__l2_regularization": loguniform(0.01, 30)}),
        "XGBoost": (XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.6,
                                 min_child_weight=3, random_state=seed, n_jobs=1), False,
                    {"model__n_estimators": randint(150, 700), "model__learning_rate": loguniform(0.01, 0.15),
                     "model__max_depth": randint(2, 7), "model__min_child_weight": randint(1, 12),
                     "model__subsample": uniform(0.6, 0.4), "model__colsample_bytree": uniform(0.4, 0.6),
                     "model__reg_lambda": loguniform(0.1, 30)}),
        "LightGBM": (LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.6, random_state=seed,
                                   n_jobs=1, verbose=-1), False,
                     {"model__n_estimators": randint(150, 700), "model__learning_rate": loguniform(0.01, 0.15),
                      "model__num_leaves": randint(4, 40), "model__min_child_samples": randint(5, 40),
                      "model__subsample": uniform(0.6, 0.4), "model__colsample_bytree": uniform(0.4, 0.6),
                      "model__reg_lambda": loguniform(0.1, 30)}),
        "CatBoost": (CatBoostWrap(iterations=400, learning_rate=0.05, depth=5, random_seed=seed), None,
                     {"model__iterations": randint(150, 500), "model__learning_rate": loguniform(0.02, 0.15),
                      "model__depth": randint(3, 8), "model__l2_leaf_reg": loguniform(1, 20)}),
        "MLP": (MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, early_stopping=True, max_iter=600,
                             random_state=seed), True,
                {"model__alpha": loguniform(1e-4, 1e-1), "model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
                 "model__learning_rate_init": loguniform(1e-3, 1e-2)}),
        "Stacking": (StackingRegressor(
            [("ridge", RidgeCV(alphas=np.logspace(-2, 3, 20))),
             ("rf", RandomForestRegressor(300, min_samples_leaf=2, max_features=0.33, random_state=seed, n_jobs=1)),
             ("lgbm", LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=15,
                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=seed,
                                    n_jobs=1, verbose=-1))],
            final_estimator=RidgeCV(), cv=5, n_jobs=1), True, None),
    }


# Models that can natively output a 10-90% range (quantile loss); used in the decision rule.
INTERVAL_CAPABLE = {"LightGBM", "XGBoost", "CatBoost", "HistGradientBoosting"}
COMPLEXITY = {"Median baseline": 0, "Ridge": 1, "KNN": 2, "Extra trees": 3, "Random forest": 3, "HistGradientBoosting": 4,
              "LightGBM": 4, "XGBoost": 4, "CatBoost": 4, "MLP": 5, "Stacking": 6}


def build(name: str, prep: str, log: bool, trim: bool, est=None, scale=None):
    """prep='raw' (original columns, one-hot) or 'eng' (collapsed + engineered features)."""
    base_est, base_scale, _ = estimators()[name]
    est = est if est is not None else base_est
    scale = base_scale if scale is None else scale
    steps = []
    if prep == "eng":
        steps.append(("eng", Engineer()))
    cat, num = (CAT_ENG, NUM_ENG) if prep == "eng" else (CAT_RAW, NUM_RAW)
    if name != "CatBoost":
        steps.append(("pre", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            ("num", StandardScaler() if scale else "passthrough", num)])))
    steps.append(("model", clone(est)))
    pipe = Pipeline(steps)
    if trim:
        pipe = TrimFit(pipe)
    if log:
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe


def space_for(name: str, trim: bool, log: bool):
    space = estimators()[name][2]
    if space is None:
        return None
    prefix = ("regressor__" if log else "") + ("base__" if trim else "")
    return {prefix + k: v for k, v in space.items()}


# ---------------------------------------------------------------- evaluation
def run_cv(key: str, df: pd.DataFrame, model, space=None, force=False) -> dict:
    path = CACHE / f"{key}.npz"
    if path.exists() and not force:
        return dict(np.load(path, allow_pickle=True))
    X, y = df.drop(columns=[TARGET]), df[TARGET].to_numpy()
    splits = list(RepeatedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=SEED).split(X))

    def one(tr, te):
        est = clone(model)
        if space is not None:
            est = RandomizedSearchCV(est, space, n_iter=N_ITER, cv=KFold(3, shuffle=True, random_state=SEED),
                                     scoring="neg_root_mean_squared_error", random_state=SEED, n_jobs=1)
        est.fit(X.iloc[tr], y[tr])
        return te, est.predict(X.iloc[te])

    t0 = time.time()
    results = Parallel(n_jobs=N_JOBS)(delayed(one)(tr, te) for tr, te in splits)
    preds = np.zeros((N_REPEATS, len(y)))
    rmse, mae, r2 = [], [], []
    for i, (te, p) in enumerate(results):
        preds[i // 5, te] = p
        rmse.append(root_mean_squared_error(y[te], p))
        mae.append(mean_absolute_error(y[te], p))
        r2.append(r2_score(y[te], p))
    out = dict(preds=preds, y=y, rmse=np.array(rmse), mae=np.array(mae), r2=np.array(r2), seconds=time.time() - t0)
    np.savez(path, **out)
    print(f"{key:<34} RMSE {np.mean(rmse):>9,.0f} MAE {np.mean(mae):>9,.0f} R2 {np.mean(r2):.3f}  ({out['seconds']:.0f}s)", flush=True)
    return out


# ---------------------------------------------------------------- phases
def phase_stages():
    raw, dedup = raw_data(), dedup_data()
    r_rf = RandomForestRegressor(300, max_features=1 / 3, random_state=SEED, n_jobs=1)  # R's randomForest defaults
    lg = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=SEED, n_jobs=1, verbose=-1)
    for tag, name, est in (("RF", "Random forest", r_rf), ("LGBM", "LightGBM", lg)):
        run_cv(f"stage0_{tag}", raw, build(name, "raw", False, False, est=est))
        run_cv(f"stage1_{tag}", dedup, build(name, "raw", False, False, est=est))
        run_cv(f"stage2_{tag}", dedup, build(name, "raw", True, False, est=est))
        run_cv(f"stage3_{tag}", dedup, build(name, "eng", True, True, est=est))


def phase_search(names=None):
    dedup = dedup_data()
    for name in estimators():
        if names and name not in names:
            continue
        run_cv(f"search_{name}", dedup, build(name, "eng", True, True))
        space = space_for(name, trim=True, log=True)
        if name in SKIP_TUNING:
            print(f"skipping tuning for {name} (default only: tuning takes ~25 min and the default is already tied for best)", flush=True)
        elif space is not None:
            run_cv(f"tuned_{name}", dedup, build(name, "eng", True, True), space=space)


def load_cached(key):
    p = CACHE / f"{key}.npz"
    return dict(np.load(p, allow_pickle=True)) if p.exists() else None


def summarize(key, r):
    return {"key": key, "rmse": r["rmse"].mean(), "rmse_sd": r["rmse"].std(), "mae": r["mae"].mean(),
            "r2": r["r2"].mean(), "r2_sd": r["r2"].std(), "seconds": float(r["seconds"])}


def decide(rows: dict):
    """Fixed rule: best mean RMSE; 'tied' = paired gap smaller than its corrected standard error."""
    best_key = min(rows, key=lambda k: rows[k]["rmse"].mean())
    best = rows[best_key]["rmse"]
    k = len(best)
    table = []
    for key, r in rows.items():
        d = r["rmse"] - best
        se = d.std(ddof=1) * np.sqrt(1 / k + 1 / 4) if k > 1 and key != best_key else 0.0
        gap = float(d.mean())
        table.append({"key": key, "model": key.split("_", 1)[1], "tuned": key.startswith("tuned_"),
                      "rmse": float(r["rmse"].mean()), "mae": float(r["mae"].mean()), "r2": float(r["r2"].mean()),
                      "gap_to_best": gap, "gap_se": float(se), "tied": bool(key == best_key or gap < se),
                      "interval_capable": key.split("_", 1)[1] in INTERVAL_CAPABLE,
                      "complexity": COMPLEXITY[key.split("_", 1)[1]], "seconds": float(r["seconds"])})
    tied = [t for t in table if t["tied"]]
    simplest = min(tied, key=lambda t: (t["complexity"], t["tuned"], t["rmse"]))
    capable = [t for t in tied if t["interval_capable"]]
    product = min(capable, key=lambda t: (t["complexity"], t["tuned"], t["rmse"])) if capable else None
    return table, {"accuracy_winner": best_key, "parsimonious_pick": simplest["key"],
                   "product_pick": product["key"] if product else None, "n_tied": len(tied)}


def phase_report():
    stage_rows, oof = [], {}
    dedup, raw = dedup_data(), raw_data()
    oof["y_raw"], oof["y_dedup"] = raw[TARGET].to_numpy(), dedup[TARGET].to_numpy()
    for tag, name in (("RF", "Random forest"), ("LGBM", "LightGBM")):
        for s, label in ((0, "0 Raw (R-style, leaky)"), (1, "1 Deduplicate"), (2, "2 Log target"), (3, "3 Clean + features")):
            r = load_cached(f"stage{s}_{tag}")
            stage_rows.append({"family": tag, "stage": s, "label": label, **summarize(f"stage{s}_{tag}", r)})
            oof[f"stage{s}_{tag}"] = r["preds"][0]
        r = load_cached(f"tuned_{name}")
        stage_rows.append({"family": tag, "stage": 4, "label": "4 Tuned", **summarize(f"tuned_{name}", r)})
        oof[f"stage4_{tag}"] = r["preds"][0]

    rows = {}
    for name in estimators():
        for prefix in ("search", "tuned"):
            r = load_cached(f"{prefix}_{name}")
            if r is not None:
                rows[f"{prefix}_{name}"] = r
    table, decision = decide(rows)
    best_key = decision["accuracy_winner"]
    best = load_cached(best_key)
    stage_rows.append({"family": "BEST", "stage": 5, "label": f"5 Best: {best_key}", **summarize(best_key, best)})
    oof["stage5_BEST"] = best["preds"][0]
    for key, r in rows.items():
        oof[key] = r["preds"][0]
    oof["insample_note"] = np.array(["see notebook"])

    pd.DataFrame(stage_rows).to_csv(OUT / "stage_results.csv", index=False)
    pd.DataFrame(table).sort_values("rmse").to_csv(OUT / "model_search.csv", index=False)
    (OUT / "decision.json").write_text(json.dumps({**decision, "n_repeats": N_REPEATS}, indent=2))
    np.savez_compressed(OUT / "stage_oof.npz", **oof)
    print(pd.DataFrame(stage_rows)[["family", "stage", "label", "rmse", "mae", "r2"]].round(3).to_string(index=False))
    print(pd.DataFrame(table).sort_values("rmse")[["key", "rmse", "r2", "gap_to_best", "gap_se", "tied"]].round(3).to_string(index=False))
    print(decision)


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    only = sys.argv[2:] or None
    if phase in ("stages", "all"):
        phase_stages()
    if phase in ("search", "all"):
        phase_search(only)
    if phase in ("report", "all"):
        phase_report()
